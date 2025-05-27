import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import (
    dequantize_module_weight,
    gather_params_ctx,
    get_bnb_param_type,
    skip_init_on_device,
)
from peft.utils.other import transpose

from .config import LoraConfig
#  ------------------------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

'''
    Version:    peft==0.15.1
    Path:       peft/tuners/lora/layer.py

    
    Related code's processing chain:
    
        peft//tuners/lora/model.py:
        "
            @staticmethod
            def _create_new_module(lora_config, adapter_name, target, **kwargs):
                ...
                    ...
                        for key, custom_cls in lora_config._custom_modules.items():  # get lora_config._custom_modules
                            if isinstance(target_base_layer, key):                   # _custom_modules
                                # In the LoTA_QAF_main.py, config._custom_modules = {nn.Linear: CustomLoraLinear}
                                    so, key==nn.Linear; custom_cls==CustomLoraLinear
                                custom_params = getattr(lora_config, "custom_config", {})
                                all_kwargs = {**kwargs, **custom_params}
                                new_module = custom_cls(target, adapter_name, **all_kwargs)

                                # gptq weights
                                from gptqmodel.nn_modules.qlinear import BaseQuantLinear
                                if isinstance(target_base_layer, BaseQuantLinear):
                                    target.qweight = target_base_layer.qweight
                                break  # Prevent multiple applications of different adapters to the same layer
        "

        peft//utils/others.py:
        "
            def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
                is_gptq_quantized = True  # I have discussed this in LoTA_QAF_main.py
        "
'''


class IntLinear(nn.Linear):  # TODO Ternary Adapter Init.
    def __init__(self, in_features, out_features, bias=False, zero_init=False):
        super(IntLinear, self).__init__(in_features, out_features, bias=bias)
        with torch.no_grad():
            if zero_init:
                self.weight.data = torch.randint(0, 1, self.weight.size(), dtype=torch.bfloat16)
            else:
                #
                # self.weight.data = torch.randint(-1, 2, self.weight.size(), dtype=torch.bfloat16)
                nn.init.kaiming_normal_(self.weight)
                delta = 0.75 * torch.mean(torch.abs(self.weight))
                self.weight.data = torch.where(self.weight > delta, 1.0, 
                                    torch.where(self.weight < -delta, -1.0, 0.0)).to(torch.bfloat16)
                #


import triton
import triton.language as tl
@triton.jit
def unified_threshold_kernel(
    ab_matrix_ptr,          # Pointer to input AB matrix (e.g., bfloat16)
    markers_ptr,            # Pointer to output markers (e.g., bfloat16)
    threshold,              # Threshold (scalar, e.g., bfloat16)
    packed_max_mask_ptr,    # Pointer to packed uint8 max mask
    packed_min_mask_ptr,    # Pointer to packed uint8 min mask
    n_elements,             # Total number of elements in AB matrix
    # Note: Using linear indexing, so n_rows and n_cols are not needed
    # They might be needed for complex 2D mask logic, but linear indexing is sufficient here
    BLOCK_SIZE: tl.constexpr, # Triton block size
    use_masks=1,              # Flag to use masks (1 for yes, 0 for no)
):
    # 1. Calculate element offsets for the current instance (linear indexing)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 2. Create mask for boundary checking
    elements_mask = offsets < n_elements

    # 3. Load data from AB matrix
    #    Ensure other=0.0 matches the type semantics of AB matrix or markers
    #    If AB is bfloat16 and markers is bfloat16, 0.0 is appropriate
    x = tl.load(ab_matrix_ptr + offsets, mask=elements_mask, other=0.0)
    # x = x.to(tl.float32) # Promoting to float32 for computation might be more stable
    # threshold_f32 = threshold.to(tl.float32) TODO No need to change dtype at all

    # 4. Calculate basic threshold conditions
    cond_pos = x > threshold
    cond_neg = x < -threshold

    # 5. Initialize allow flags (default to allow)
    allow_increase = tl.full(cond_pos.shape, 1, dtype=tl.int1) # Using Triton's boolean type (int1)
    allow_decrease = tl.full(cond_neg.shape, 1, dtype=tl.int1)

    # 6. If use_masks is True (equals 1), unpack and apply masks
    if use_masks == 1:
        # Calculate byte indices and bit positions in the packed tensor
        packed_indices = offsets // 8
        bit_positions = offsets % 8

        # Create mask for accessing the packed tensor (based on whether offsets are valid)
        # Note: When loading uint8, the mask is still based on the original elements_mask
        # tl.load handles boundaries; we just need to ensure packed_indices are within a reasonable range (guaranteed by elements_mask)
        # Load packed bytes (if mask is invalid, other=0 means bits are 0)
        packed_bytes_max = tl.load(packed_max_mask_ptr + packed_indices, mask=elements_mask, other=0)
        packed_bytes_min = tl.load(packed_min_mask_ptr + packed_indices, mask=elements_mask, other=0)

        # Extract the corresponding bits (1 or 0)
        # (packed_byte >> bit_position) & 1
        max_bits = (packed_bytes_max >> bit_positions) & 1
        min_bits = (packed_bytes_min >> bit_positions) & 1

        # Update allow flags (allow only if bit is 1)
        # Convert uint8 0/1 to Triton's int1 (True/False)
        allow_increase = max_bits.to(tl.int1)
        allow_decrease = min_bits.to(tl.int1)

    # 7. Combine conditions
    #    Note: In Triton, & is logical AND
    final_cond_pos = cond_pos & allow_increase
    final_cond_neg = cond_neg & allow_decrease

    # 8. Calculate the final result (1, -1, 0)
    #    tl.where(condition, true_value, false_value)
    result = tl.where(final_cond_pos, 1, tl.where(final_cond_neg, -1, 0))

    # 9. Convert the result back to the target type and store
    #    Ensure the type of result matches the type pointed to by markers_ptr
    #    Here, it is assumed that markers is bfloat16
    output_dtype = markers_ptr.dtype.element_ty
    tl.store(markers_ptr + offsets, result.to(output_dtype), mask=elements_mask)


class ThresholdFunction_pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, AB_Matrix, threshold, packed_weights_int_max, packed_weights_int_min, weight_shape):

        markers = torch.empty_like(AB_Matrix, dtype=torch.bfloat16)
        n_elements = AB_Matrix.numel()
        # Set Triton tuning parameters
        BLOCK_SIZE = 1024 # Can be adjusted and tuned as needed
        grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Calculate Grid size
        threshold_bfloat16 = torch.tensor(threshold, dtype=torch.bfloat16).item()
        unified_threshold_kernel[grid_size](
            AB_Matrix,              # Input AB matrix (pointer implicitly passed)
            markers,                # Output markers (pointer implicitly passed)
            threshold_bfloat16,     # Threshold (bfloat16 scalar)
            packed_weights_int_max,
            packed_weights_int_min,
            n_elements,             # Total number of elements
            BLOCK_SIZE=BLOCK_SIZE,  # Block size (compile-time constant)
            use_masks=1,         
        )
        return markers

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def pack_bool_tensor(bool_tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs the input torch.bool tensor into a torch.uint8 tensor.
    Each uint8 value represents 8 boolean values.
    """
    if not bool_tensor.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if bool_tensor.dtype != torch.bool:
        raise ValueError("Input tensor must be of type torch.bool.")

    numel = bool_tensor.numel()
    # Use triton.cdiv or math.ceil to calculate the packed size
    # packed_size = triton.cdiv(numel, 8)
    packed_size = math.ceil(numel / 8)

    # Ensure the tensor is contiguous and flattened
    flat_bool = bool_tensor.flatten().contiguous()

    # If the number of elements is not a multiple of 8, pad with 0 (False)
    padding_size = packed_size * 8 - numel
    if padding_size > 0:
        padded_flat_bool = torch.nn.functional.pad(flat_bool, (0, padding_size), value=False)
    else:
        padded_flat_bool = flat_bool

    # Convert boolean to uint8 (True -> 1, False -> 0)
    int_values = padded_flat_bool.to(dtype=torch.uint8)

    # Reshape to have each row contain 8 values
    reshaped_int = int_values.reshape(-1, 8)

    # Create weights for bit operations (2^0, 2^1, ..., 2^7) -> [1, 2, 4, ..., 128]
    # Assuming the first boolean is the least significant bit (LSB)
    powers_of_2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bool_tensor.device)

    # Perform packing: Multiply each row of 8 values by the corresponding powers of 2 and sum
    # (N, 8) * (8,) -> (N, 8) --sum(dim=1)--> (N,)
    packed_tensor = torch.sum(reshaped_int * powers_of_2, dim=1, dtype=torch.uint8)

    # packed_tensor now contains the packed data, with size packed_size
    return packed_tensor


class CustomLoraLinear(nn.Module, LoraLayer):
    def __init__(self, base_layer, adapter_name: str, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0,
                 fan_in_fan_out: bool = False, init_lora_weights: Union[bool, str] = True, threshold: int = -1,
                 residual: bool = True, **kwargs,) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.adaptive_scaling = {}
        self.threshold = threshold
        self.residual = residual
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights,)


    def update_layer(self, adapter_name, r, lora_alpha, 
                     lora_dropout, init_lora_weights, use_rslora=False, use_dora: bool = False, lora_bias: bool = False):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if r > 0:
            self.lora_A[adapter_name] = IntLinear(self.base_layer.in_features, r, zero_init=False)
            self.lora_B[adapter_name] = IntLinear(r, self.base_layer.out_features, zero_init=True)
            _, self.sorted_indices = torch.sort(self.base_layer.g_idx.long()) 

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

        with torch.no_grad():
            if self.base_layer.bits in [2, 4, 8]:
                weight_int = torch.bitwise_and(torch.bitwise_right_shift(
                        torch.unsqueeze(self.base_layer.qweight, 1).expand(-1, self.base_layer.pack_factor, -1),
                        self.base_layer.wf_unsqueeze_neg_one).to(self.base_layer.dequant_dtype), self.base_layer.maxq)
                weight_int = weight_int.reshape(self.base_layer.in_features, self.base_layer.out_features)
            elif self.base_layer.bits == 3:
                weight = self.base_layer.qweight.reshape(
                    self.base_layer.qweight.shape[0] // 3, 3, 1, self.base_layer.qweight.shape[1]
                ).expand(-1, -1, 12, -1)
                weight = (weight >> self.base_layer.wf_unsqueeze_neg_one) & 0x7
                weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
                weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
                weight = weight & 0x7
                weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
                weight_int = weight.reshape(self.base_layer.in_features, self.base_layer.out_features)
            else:
                raise ValueError(f"Unsupported bits: {self.base_layer.bits}")
            #
            self.weights_int_max_bool = (weight_int != self.base_layer.maxq)  # True if not at max value
            self.weights_int_min_bool = (weight_int != 0)                     # True if not at min value
            #
            self.packed_weights_int_max = pack_bool_tensor(self.weights_int_max_bool)
            self.packed_weights_int_min = pack_bool_tensor(self.weights_int_min_bool)
            self.weight_shape = self.weights_int_max_bool.shape
        #
        del weight_int, self.weights_int_max_bool, self.weights_int_min_bool
        torch.cuda.empty_cache()


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        # result = self.base_layer(x, *args, **kwargs)
        # torch_result_dtype = result.dtype

        for active_adapter in self.active_adapters: 
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A_weights = self.lora_A[active_adapter].weight
            lora_B_weights = self.lora_B[active_adapter].weight
            AB_Matrix = torch.matmul(lora_A_weights.T, lora_B_weights.T).to(x.dtype).contiguous()

            markers = ThresholdFunction_pack.apply(
                AB_Matrix,
                self.threshold,
                self.packed_weights_int_max,  # Pass packed uint8 tensor
                self.packed_weights_int_min, 
                self.weight_shape     
            )

            if self.residual:
                '''Adjust residual to zero_plus (The offset facotr in Paper)'''
                AB_Matrix = AB_Matrix - markers.detach() * self.threshold

                # idx is the group number for each row (0-31), self.sorted_indices are the row indices sorted by group from 0 to 31, 
                # sorted_AB is the matrix sorted by group for easier calculation. 
                # <Rows and columns are just an example.>
                sorted_AB = AB_Matrix[self.sorted_indices]     
                # view 32, 64, 2048 => 32 groups, each with 64 elements, out_features
                sorted_AB = sorted_AB.view(self.base_layer.scales.shape[0], self.base_layer.group_size, self.base_layer.out_features).mean(dim=1) 
                # Mean divided by threshold, e.g., 25/32, and broadcast to the corresponding dimensions
                zero_plus = sorted_AB[self.base_layer.g_idx.long()]/self.threshold

                '''Calculate'''
                # markers contain -1, 0, 1, zero_plus is the remainder calculated by group mean. Both are applied according to scales and aligned with quantized weights
                # AB_mix = self.base_layer.scales[self.base_layer.g_idx.long()] * (markers + zero_plus).contiguous()
                AB_mix = self.base_layer.scales[self.base_layer.g_idx.long()] * ((self.decode_qweight(self.base_layer)-self.decode_zero(self.base_layer)[self.base_layer.g_idx.long()]).detach() + markers + zero_plus).contiguous()
            else:
                pass
            lora_out = F.linear(x, AB_mix.T)  
            # result = result + lora_out
        # result = result.to(torch_result_dtype)
        return lora_out


    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        pass  # Implementation in lota_merge.py

    def unmerge(self) -> None:
        pass

    def _move_adapter_to_device_of_base_layer(self, adapter_name):
        if hasattr(self.get_base_layer(), 'weight'):
            device = self.get_base_layer().weight.device
        else:
            device = self.get_base_layer().qweight.device
        self.lora_A[adapter_name].to(device)
        self.lora_B[adapter_name].to(device)



''' ### torch_version ###
class ThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, AB_Matrix, threshold, weights_int_max_bool, weights_int_min_bool):

        if weights_int_max_bool != None:
            marker_up = weights_int_max_bool & (AB_Matrix > threshold)
            marker_low = weights_int_min_bool & (AB_Matrix < -threshold)
            markers = torch.where(marker_up, torch.ones_like(AB_Matrix), torch.where(marker_low, -torch.ones_like(AB_Matrix), torch.zeros_like(AB_Matrix)))

        # ctx.save_for_backward(AB_Matrix, torch.tensor(threshold))
        return markers

    @staticmethod
    def backward(ctx, grad_output):
        # AB_Matrix, threshold = ctx.saved_tensors
        # grad_AB_Matrix = grad_output 
        return grad_output, None, None, None
'''