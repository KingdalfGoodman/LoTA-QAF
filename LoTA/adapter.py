import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import safetensors
import torch
import torch.nn.functional as F

from ..utils.logger import setup_logger
from .peft import LoraConfig
from .remote import resolve_path

log = setup_logger()
LORA_MERGED_WEIGHT_PATHS = [None, ""]
HF_ADAPTER_FILE_NAME = "adapter_model.safetensors"
HF_ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
HF_ADAPTER_WEIGHT_KEY_PREFIX = "base_model.model."
#  ------------------------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

'''
    Version:    gptqmodel==2.1.1.dev0
    Path:       gptqmodel/adapter/adapter.py

    
    Related code's processing chain:

        gptqmodel/nn_modules/qlinear/torch.py:
            "
                from ...adapter.adapter import Adapter, Lora, LTA

                class TorchQuantLinear(PackableQuantLinear):
                    SUPPORTS_ADAPTERS = [Lora, LTA]
                    ...
                    def _forward(self, x, x_dtype, out_shape):
                        if self.adapter:
                            if isinstance(self.adapter, LTA):
                                out = self.adapter.apply(x=x, out=out, base_layer=self)
                            else:  
                                out = self.adapter.apply(x=x, out=out)
            "

        gptqmodel/nn_modules/qlinear/torch.py:
            "
                from ...adapter.adapter import Adapter, Lora, LTA

                class TritonV2QuantLinear(TorchQuantLinear, TritonModuleMixin):
                    SUPPORTS_ADAPTERS = [Lora, LTA]
                    ...
                    def forward(self, x):
                        if self.adapter:
                            if isinstance(self.adapter, LTA):
                                out = self.adapter.apply(x=x, out=out, base_layer=self)
                            else:  
                                out = self.adapter.apply(x=x, out=out)
            "
'''



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
    # Note: We use linear indexing, so n_rows and n_cols are not needed
    # They might be needed for complex 2D mask logic, but linear indexing is sufficient here
    BLOCK_SIZE: tl.constexpr, # Triton block size
    use_masks=1,              # Flag to use masks (1 for yes, 0 for no)
):
    # 1. Calculate the element offsets for the current instance (linear indexing)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 2. Create a mask for boundary checking
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

        # Create a mask for accessing the packed tensor (based on whether offsets are valid)
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
        BLOCK_SIZE = 1024  # Can be adjusted and tuned as needed
        grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Calculate grid size
        threshold_bfloat16 = torch.tensor(threshold, dtype=torch.bfloat16).item()
        unified_threshold_kernel[grid_size](
            AB_Matrix,              # Input AB matrix (implicitly passes pointer)
            markers,                # Output markers (implicitly passes pointer)
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
    Pack the input torch.bool tensor into a torch.uint8 tensor.
    Each uint8 value represents 8 boolean values.
    """
    if not bool_tensor.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if bool_tensor.dtype != torch.bool:
        raise ValueError("Input tensor must be of type torch.bool.")
    import math

    numel = bool_tensor.numel()
    # Calculate the packed size using triton.cdiv or math.ceil
    # packed_size = triton.cdiv(numel, 8)
    packed_size = math.ceil(numel / 8)

    # Ensure the tensor is contiguous and flattened
    flat_bool = bool_tensor.flatten().contiguous()

    # If the number of elements is not a multiple of 8, pad it
    # Pad with 0 (False)
    padding_size = packed_size * 8 - numel
    if padding_size > 0:
        padded_flat_bool = torch.nn.functional.pad(flat_bool, (0, padding_size), value=False)
    else:
        padded_flat_bool = flat_bool

    # Convert boolean values to uint8 (True -> 1, False -> 0)
    int_values = padded_flat_bool.to(dtype=torch.uint8)

    # Reshape to have 8 values per row
    reshaped_int = int_values.reshape(-1, 8)

    # Create weights for bit operations (2^0 to 2^7) -> [1, 2, 4, ..., 128]
    # Assume the first boolean is the least significant bit (LSB)
    powers_of_2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bool_tensor.device)

    # Pack: Multiply each row's 8 values by corresponding powers of 2 and sum
    # (N, 8) * (8,) -> (N, 8) --sum(dim=1)--> (N,)
    packed_tensor = torch.sum(reshaped_int * powers_of_2, dim=1, dtype=torch.uint8)

    # packed_tensor now contains the packed data with size packed_size
    return packed_tensor


@dataclass
class LTA(Adapter):
    def __init__(self, rank: int, threshold: float, residual: bool = True, path: str = None, lora_A: torch.Tensor = None, lora_B: torch.Tensor = None):
        super().__init__(rank, path)
        self.threshold = threshold
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.cached_zp = None
        self.residual = residual

    def decode_qweight(self, base_layer):
        # Implementation logic from gptqmodel
        if base_layer.bits in [2, 4, 8]:
            weight_int = torch.bitwise_and(
                torch.bitwise_right_shift(
                    torch.unsqueeze(base_layer.qweight, 1).expand(-1, base_layer.pack_factor, -1),
                    base_layer.wf_unsqueeze_neg_one
                ).to(base_layer.dequant_dtype), 
                base_layer.maxq
            )
            weight_int = weight_int.reshape(base_layer.in_features, base_layer.out_features)
        elif base_layer.bits == 3:
            weight = base_layer.qweight.reshape(
                base_layer.qweight.shape[0] // 3, 3, 1, base_layer.qweight.shape[1]
            ).expand(-1, -1, 12, -1)
            weight = (weight >> base_layer.wf_unsqueeze_neg_one) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
            weight_int = weight.reshape(base_layer.in_features, base_layer.out_features)
        else:
            raise ValueError(f"Unsupported bits: {base_layer.bits}")
        return weight_int

    def encode_qweight(self, new_weight_int: torch.Tensor, base_layer, target_dtype, target_device) -> torch.Tensor:
        """
            Implementation logic from gptqmodel
            Pack new_weight_int (integer weights) back into qweight format.
            new_weight_int shape: (base_layer.in_features, base_layer.out_features)
        """
        
        pack_dtype_bits = torch.iinfo(target_dtype).bits # e.g., 32 for torch.int32

        if base_layer.bits in [2, 4, 8]:
            # pack_factor should be pre-calculated and stored in base_layer
            pack_factor = base_layer.pack_factor # Or pack_dtype_bits // base_layer.bits
            
            # Reshape new_weight_int to (N, pack_factor, out_features) for packing
            # where N = base_layer.in_features // pack_factor
            unpacked_shape = (
                base_layer.in_features // pack_factor,
                pack_factor,
                base_layer.out_features
            )
            weight_int_reshaped = new_weight_int.reshape(unpacked_shape)

            # Create shifts for packing: [0, bits, 2*bits, ...]
            shifts = torch.arange(0, pack_dtype_bits, base_layer.bits, device=target_device, dtype=torch.int32)
            shifts = shifts.view(1, -1, 1) # Shape (1, pack_factor, 1) for broadcasting

            # Apply shifts and sum (equivalent to OR for non-overlapping bits)
            # Ensure new_weight_int values are within [0, max_q] so shifted versions don't overflow `bits` space.
            shifted_weights = weight_int_reshaped << shifts
            qweight = torch.sum(shifted_weights, dim=1, dtype=target_dtype)

        elif base_layer.bits == 3:
            # This logic mirrors the packing part of PackableQuantLinear.pack()
            in_features, out_features = new_weight_int.shape
            
            # For 3-bit, GPTQ logic processes 32 input weights into 3 packed (32-bit) words.
            # Ensure in_features is compatible or pad.
            num_int_chunks_of_32 = (in_features + 31) // 32
            padded_in_features = num_int_chunks_of_32 * 32
            
            current_int_weight = new_weight_int
            if padded_in_features != in_features:
                padding_rows = padded_in_features - in_features
                padding = torch.zeros(padding_rows, out_features, dtype=new_weight_int.dtype, device=target_device)
                current_int_weight = torch.cat((new_weight_int, padding), dim=0)

            # Calculate shape of qweight
            # Each 32 current_int_weight rows map to 3 qweight rows.
            num_qweight_rows = padded_in_features * 3 // pack_dtype_bits
            qweight = torch.zeros((num_qweight_rows, out_features), dtype=target_dtype, device=target_device)

            i = 0 # Index for current_int_weight rows
            row_idx = 0 # Index for qweight rows
            while row_idx < num_qweight_rows:
                # Pack 32 int_weights into 3 qweight words (per column)
                # Block 1
                for j_offset in range(10):
                    qweight[row_idx, :] |= current_int_weight[i + j_offset, :] << (3 * j_offset)
                i += 10
                qweight[row_idx, :] |= current_int_weight[i, :] << 30 # Lower 2 bits of 11th int_weight
                row_idx += 1
                if row_idx >= num_qweight_rows: break # Important for non-exact multiples

                # Block 2
                qweight[row_idx, :] |= (current_int_weight[i, :] >> 2) & 1 # Upper 1 bit of 11th int_weight
                i += 1
                for j_offset in range(10):
                    qweight[row_idx, :] |= current_int_weight[i + j_offset, :] << (3 * j_offset + 1)
                i += 10
                qweight[row_idx, :] |= current_int_weight[i, :] << 31 # Lower 1 bit of 22nd int_weight
                row_idx += 1
                if row_idx >= num_qweight_rows: break

                # Block 3
                qweight[row_idx, :] |= (current_int_weight[i, :] >> 1) & 0x3 # Upper 2 bits of 22nd int_weight
                i += 1
                for j_offset in range(10):
                    qweight[row_idx, :] |= current_int_weight[i + j_offset, :] << (3 * j_offset + 2)
                i += 10
                row_idx += 1
        else:
            raise ValueError(f"Unsupported bits for encoding: {base_layer.bits}")
        return qweight


    def apply(self, x: torch.Tensor, out: torch.Tensor, base_layer) -> torch.Tensor:
        if x.dtype != self.lora_A.dtype and self.cached_zp is None:
            log.info.once(f"Adapter: Lora A/B auto changed from `{self.lora_A.dtype}` to `{x.dtype}` to match forward input dtype.")
            self.lora_A = self.lora_A.to(device=x.device, dtype=x.dtype)
            self.lora_B = self.lora_B.to(device=x.device, dtype=x.dtype)

        if self.cached_zp is None:
            weight_int = self.decode_qweight(base_layer)
            b_dtype = base_layer.qweight.dtype
            b_device = base_layer.qweight.device
            weights_int_max_bool = (weight_int != base_layer.maxq)  # True if not at max value
            weights_int_min_bool = (weight_int != 0)                # True if not at min value
            weight_shape = weights_int_max_bool.shape

            # Pack boolean masks and store
            packed_weights_int_max = pack_bool_tensor(weights_int_max_bool)
            packed_weights_int_min = pack_bool_tensor(weights_int_min_bool)
            del weights_int_max_bool, weights_int_min_bool
            torch.cuda.empty_cache()

            AB_Matrix = torch.matmul(self.lora_A, self.lora_B).to(x.dtype).contiguous()
            # print(f"A: {self.lora_A.shape}; B: {self.lora_B.shape}; AB: {AB_Matrix.shape}")
            # Move to CPU and clean memory
            self.lora_A = self.lora_A.to('cpu')  
            self.lora_B = self.lora_B.to('cpu')  
            torch.cuda.empty_cache()

            markers = ThresholdFunction_pack.apply(
                AB_Matrix,
                self.threshold,
                packed_weights_int_max, # Pass packed uint8 tensor 
                packed_weights_int_min, # Pass packed uint8 tensor 
                weight_shape          
            )
            del packed_weights_int_max, packed_weights_int_min
            torch.cuda.empty_cache()

            # 1. Apply markers to weight_int (Eq.5 in Paper)
            new_weight_int = weight_int + markers.to(weight_int.dtype)  
            del weight_int
            torch.cuda.empty_cache()

            # 2. Repack new_weight_int into qweight
            new_qweight = self.encode_qweight(new_weight_int, base_layer, b_dtype, b_device)

            # 3. Update base_layer.qweight
            # base_layer.qweight = new_qweight
            base_layer.qweight.copy_(new_qweight)
            del new_qweight
            torch.cuda.empty_cache()

            if self.residual:
                AB_Matrix = AB_Matrix - markers * self.threshold
                del markers
                torch.cuda.empty_cache()

                '''Adjust residual to zero_plus (The offset facotr in Paper)'''
                _, sorted_indices = torch.sort(base_layer.g_idx.long())
                sorted_AB = AB_Matrix[sorted_indices]
                sorted_AB = sorted_AB.view(base_layer.scales.shape[0], base_layer.group_size, base_layer.out_features).mean(dim=1)/self.threshold 
                self.cached_zp = sorted_AB.to(torch.bfloat16)
                del AB_Matrix, sorted_indices, sorted_AB, 
                torch.cuda.empty_cache()
                
        if  self.cached_zp != None:
            cached_zp_scale = (base_layer.scales[base_layer.g_idx.long()] * (self.cached_zp[base_layer.g_idx.long()])).to(x.dtype).contiguous()
            lora_out = F.linear(x, cached_zp_scale.T)
        else:
            lora_out = 0

        if out.dim() > x.dim() and out.shape[0] > 1:
            out_orgi_shape = out.shape
            out = out.view(-1, out.shape[-1])
            out.add_(lora_out)
            out = out.view(out_orgi_shape)
        else:
            out.add_(lora_out)

        return out

    @classmethod
    def name(cls) -> str:
        return "lora"

    def to_dict(self):
        return {
            "name": self.name(),
            "path": self.path,
            "rank": self.rank
        }


