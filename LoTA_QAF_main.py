import os
import shutil
import logging
logger = logging.getLogger(__name__)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRITON_CACHE_DIR"] = os.path.join(os.getcwd(), ".triton/cache")

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer
from transformers import HfArgumentParser, TrainingArguments

from data_print_save import *
from t_signSGD import tSignSGD
from peft.tuners.lora.layer import CustomLoraLinear, IntLinear
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from gptqmodel import GPTQModel, BACKEND
from gptqmodel.adapter.adapter import Lora, LTA
from gptqmodel.nn_modules.qlinear import BaseQuantLinear

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import torch
import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 512 

check_cuda()
print(f"Current PID: {os.getpid()}")
# DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["HF_HOME"] = "/home/ubuntu/lib/huggingface"
# os.environ["XDG_CACHE_HOME"] = "/home/ubuntu/lib/cache"
#  ------------------------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


'''
    mode 1: training; methods (LoRA, LoTA); datasets (alpaca, gsm8k, sql, viggo)

    mode 2: testing; automatically select model based on Adapter_Path_Name, automatically (LoRA, LoTA); 
            performance-recovery Lm-eval(mmlu); task-specific evalGSV(sql, viggo, gsm8k).
    {
        automatically set w_bits based on model path
        automatically set optimizer based on lota_qaf
        based on load_adapter:
            automatically set lota_qaf
            automatically set interval (under lota_qaf == True)  # Omega in Paper
            automatically load the corresponding quantized model
    }
'''


from typing import List, Any, Dict
from dataclasses import dataclass, field
@dataclass
class baseConfig:
    mode: int = field(default=0, metadata={"choices": [1, 2], "help": "MODE"})
    training_flag: bool = field(default=False, metadata={"help": "MODE"})

    '''LoTA-QAF''' 
    lota_qaf: bool = field(default=False, metadata={"help": "LoTA-QAF"})
    residual: bool = field(default=True, metadata={"help": "LoTA-QAF"})

    '''gptqModel'''
    quantized_model_dir: str = field(default=None, metadata={"help": "gptqModel"})
    w_bits: int = field(default=None, metadata={"help": "gptqModel"})
    group_size: int = field(default=None, metadata={"help": "gptqModel"})
    sym: bool = field(default=False, metadata={"help": "gptqModel"})

    ''' LoRA ''' 
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], metadata={"help": "LoRA"})
    lora_r: int = field(default=None, metadata={"help": "LoRA"})
    lora_alpha: int = field(default=None, metadata={"help": "LoRA"})
    lora_dropout: float = field(default=0.00, metadata={"help": "LoRA"})

    ''' baseModel ''' 
    pretrained: str = field(default=None, metadata={"help": "baseModel"})
    device_map: str = field(default="cuda:0", metadata={"help": "baseModel"})
    dtype: str = field(default="bfloat16", metadata={"help": "baseModel"})
    flash_attn: str = field(default="flash_attention_2", metadata={"help": "baseModel"})
    trust_remote_code: bool = field(default=True, metadata={"help": "baseModel"})


@dataclass
class trainingConfig:
    '''Key Training Parameters'''
    learning_rate: float = field(default=1e-5, metadata={"help": "training params"})   

    interval_point: int = field(default=48, metadata={"help": "LoTA-QAF"})   # TODO is Omega
    optimizer_type: str = field(default="paged_adamw_32bit", metadata={"choices": ["adamw_torch", "paged_adamw_32bit", "int_sign_sgd"], "help": "LoTA-QAF"})
    # The int_sign_sgd will auto choice by (baseConfig.lota_qaf == True)

    filter_ratio: float = field(default=0.95, metadata={"help": "LoTA-QAF"}) # TODO is Sigma_t. Here is discard 0.95 and choice top 0.05.
    min_grad: float = field(default=0.999, metadata={"help": "LoTA-QAF"})
    filter_upper: float = field(default=0.9999, metadata={"help": "LoTA-QAF"})

    '''Datasets'''
    training_data_name: str = field(default=None, metadata={"choices": ["alpaca", "gsm8k", "sql", "viggo"], "help": "Datasets"})
    train_ratio: float = field(default=1.0, metadata={"help": "Datasets"})

    num_train_epochs: int = field(default=0.0, metadata={"help": "Datasets"}) 
    max_steps: int = field(default=-1, metadata={"help": "Datasets"})              
    save_number: int = field(default=10, metadata={"help": "Datasets"})

    train_batch_size: int = field(default=64, metadata={"help": "Datasets"})  # All Experiments (Effective Batch Size == 64)
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Datasets"})
    source_max_len: int = field(default=1024, metadata={"help": "Datasets"})
    target_max_len: int = field(default=256, metadata={"help": "Datasets"})

    '''training params'''
    warmup_ratio: float = field(default=0.03, metadata={"help": "training params"})           
    weight_decay: float = field(default=0.0, metadata={"help": "training params"})            
    max_grad_norm: float = field(default=0.3, metadata={"help": "training params"})  # Will close when using LoTA-QAF. and have auto logic in get_training_args.
    lr_scheduler_type: str = field(default="constant", metadata={"help": "training params"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "training params"})

    logging_steps: int = field(default=1, metadata={"help": "training params"})
    save_strategy: str = field(default="steps", metadata={"help": "training params"})
    adapter_path: str = field(default="./adapter", metadata={"help": "training params"})        

    '''ckpt'''
    adapter_ckpt: str = field(default="none", metadata={"help": "Adapter_ckpt_path"})
    debug_number: int = -1


@dataclass
class evalConfig:
    '''lm-eval'''
    load_adapter: str = field(default="none", metadata={"help": "lm-eval"})  # "none" is gptqModel without adapters. Don't have baseModel(16-bit) + Adapter.
    auto_gptq: str = field(default="gptq", metadata={"choices": ["none", "gptq"], "help": "lm-eval"})  # "none" is test baseModel(16-bit)
    output_path: str = field(default="./eval_results", metadata={"help": "lm-eval"})
    load_ada_interval: int = field(default=48, metadata={"help": "lm-eval"}) # Should set Omega in test, but we have auto logic, read interval from path. Keep here, for print and check.

    tasks: List[str] = field(default_factory=lambda: [None], metadata={"help": "lm-eval"})
    ft_dataset_name: str = field(default="none", metadata={"choices": ["gsm8k", "sql", "viggo", "none"], "help": "ft_datasets"})
    eval_batch_size: int = field(default=16, metadata={"help": "lm-eval"})      
    num_fewshot: int = field(default=5, metadata={"help": "lm-eval"})  # This won't work for GSV ft_dataset_name.
    device: str = field(default="cuda", metadata={"help": "lm-eval"})


class ModelConfigManager:
    def __init__(self, base_args: baseConfig, training_args: trainingConfig, eval_args: evalConfig):
        self.base_args = base_args
        self.training_args = training_args
        self.eval_args = eval_args

    def get_model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "torch_dtype": self.base_args.dtype if self.eval_args.auto_gptq == 'none' else torch.bfloat16,
            "device_map": self.base_args.device_map,
            "trust_remote_code": self.base_args.trust_remote_code,
            "attn_implementation": self.base_args.flash_attn
        }
        return base_kwargs

    def get_lora_config(self) -> LoraConfig or None:
        if self.base_args.lora_r == 0:
            return None

        config = LoraConfig(
            r=self.base_args.lora_r,
            lora_alpha=self.base_args.lora_alpha,
            lora_dropout=self.base_args.lora_dropout,
            target_modules=self.base_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.base_args.lota_qaf:
            '''We implement LoTA_QAF by [peft.tuners.lora.layer import CustomLoraLinear]'''
            config._custom_modules = {BaseQuantLinear: CustomLoraLinear}
            config.custom_config = {
                "residual": self.base_args.residual,             # About offset factor. And always True in our experiments.
                "threshold": self.training_args.interval_point,  # Omega in Paper.
            }
        return config

    def get_training_args(self) -> TrainingArguments:
        # checkpoint_path = None if self.training_args.adapter_ckpt=='none' else self.training_args.adapter_ckpt
        self.training_args.temp_save_path = get_Ada_path(self.base_args, self.training_args, is_temp=True)
        
        import math  # auto ckpt number settings.
        real_batch = self.training_args.train_batch_size * self.training_args.gradient_accumulation_steps
        if self.training_args.num_train_epochs == 1.0:
            steps_dict = {"gsm8k": math.ceil(((7473/real_batch)/self.training_args.save_number)/5.0)*5, 
                          "viggo": math.ceil(((5103/real_batch)/self.training_args.save_number)/5.0)*5,
                          "sql": math.ceil(((30000/real_batch)/self.training_args.save_number)/5.0)*5,
                          "alpaca": 20, "flan": 20}
            save_steps = next(steps_dict[key] for key in steps_dict if key == self.training_args.training_data_name)
        else:
            save_steps = math.ceil((self.training_args.max_steps/self.training_args.save_number)/5.0)*5

        return TrainingArguments(
            num_train_epochs=self.training_args.num_train_epochs,
            max_steps=self.training_args.max_steps if self.training_args.num_train_epochs == 0.0 else -1,

            save_strategy=self.training_args.save_strategy,
            save_steps=save_steps,
            output_dir=self.training_args.temp_save_path, 

            bf16=(self.base_args.dtype == "bfloat16"),
            optim=self.training_args.optimizer_type if self.training_args.optimizer_type != "int_sign_sgd" else "adamw_torch", 
            # Resolve the error 'int_sign_sgd is not a valid OptimizerNames, please select one of ['adamw_torch', ...]', 
            # TODO while the actual selection is in the logic of IntCustomTrainer.

            learning_rate=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
            max_grad_norm= None if self.base_args.lota_qaf else self.training_args.max_grad_norm, # LoTA-QAF shouldn't set max_grad_norm.
            lr_scheduler_type=self.training_args.lr_scheduler_type,
            warmup_ratio=self.training_args.warmup_ratio,

            per_device_train_batch_size=self.training_args.train_batch_size,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            
            logging_steps=self.training_args.logging_steps,
            gradient_checkpointing=self.training_args.gradient_checkpointing,
            label_names=['input_ids', 'labels', 'attention_mask'],  # Or remove_unused_columns=False. Handling specific details within the library file. KEEP HERE!!!

            resume_from_checkpoint=None if self.training_args.adapter_ckpt=='none' else self.training_args.adapter_ckpt,
        )


class IntCustomTrainer(Trainer):
    def __init__(self, base_args: baseConfig, training_args,  *args, filter_ratio=0.95, min_grad=0.999, filter_upper=0.9999, **kwargs):
        self.base_args = base_args
        self.training_args = training_args

        self.filter_ratio = filter_ratio
        self.min_grad = min_grad
        self.filter_upper = filter_upper

        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.training_args.optimizer_type == "int_sign_sgd":
            int_params = []
            for name, module in self.model.named_modules():
                if isinstance(module, IntLinear):
                    int_params.extend(module.parameters())
            print(f"find {len(int_params)} INT module")
            self.optimizer = tSignSGD(params=self.model.parameters(), int_params=int_params,
                                        threshold_ratio=self.filter_ratio, min_grad=self.min_grad, filter_upper=self.filter_upper)  # self.model.parameters(), , **optimizer_kwargs

            datasets_steps = {"alpaca": 300, "gsm8k": 117, "sql": 469, "viggo": 80,}   # in 64 batch_size. TODO It needs to be set based on your training number and batch.
            scheduler_steps = datasets_steps[self.training_args.training_data_name]
            self.optimizer.create_scheduler(num_training_steps=scheduler_steps)
            print(f"\nScheduler_Steps: {scheduler_steps}\n")

        else:
            return super().create_optimizer()


def QAF(base_args, training_args, config_manager, tokenizer):
    model = GPTQModel.load(base_args.quantized_model_dir, **config_manager.get_model_kwargs(), backend=BACKEND.AUTO_TRAINABLE)  
    model.optimize() 

    ''' kbit: freeze parameters, establish gradient flow, use grad_checkpoint, 
              and change precision to float32 if quantization is not specified. (NEED!!! hardcoded is_gptq_quantized = True. in prepare_model_for_kbit_training)'''
    model = prepare_model_for_kbit_training(model)

    lora_config = config_manager.get_lora_config()
    model = get_peft_model(model, lora_config)  
    model.print_trainable_parameters()          

    '''dataset'''
    dataset = prepare_dataset(data_name=training_args.training_data_name, tokenizer=tokenizer, train_ratio=training_args.train_ratio,
                              source_max_len=training_args.source_max_len, target_max_len=training_args.target_max_len,debug_number=training_args.debug_number) 
    check_data_simple(training_args.training_data_name, dataset, tokenizer)

    '''traing config'''
    training_args_obj = config_manager.get_training_args()
    trainer = IntCustomTrainer(
        base_args=base_args,
        training_args=training_args,
        model=model,
        args=training_args_obj,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),

        filter_ratio=training_args.filter_ratio,
        min_grad=training_args.min_grad,
        filter_upper=training_args.filter_upper
        
    )
    model.config.use_cache = False  # Ban this one, in training.
    torch.set_float32_matmul_precision('high')

    '''training and saving'''
    start_time = timePrint(end=False)
    torch.cuda.reset_peak_memory_stats()

    if training_args_obj.resume_from_checkpoint != "none":
        trainer.train(resume_from_checkpoint=training_args_obj.resume_from_checkpoint)  # ckpt
    else:
        trainer.train()
    trainer.save_model()

    runtime = timePrint(end=True, time_1=start_time)
    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Max GPU mem usage: {max_memory:.2f} MB")
    
    final_save_path = get_Ada_path(base_args, training_args, runtime, max_memory)
    try:
        shutil.move(training_args.temp_save_path, final_save_path)
        print(f"Training completed, Adapter has been saved to: {final_save_path}")
    except Exception as e:
        print(f"Failed to move model:{e}")

    del model
    torch.cuda.empty_cache()


def gptq_load(base_args, eval_args, config_manager, ada_flag):
    back = BACKEND.TRITON if base_args.w_bits != 3 else BACKEND.TORCH

    if ada_flag:
        if base_args.lota_qaf:
            adapter = LTA(path=eval_args.load_adapter, rank=base_args.lora_r, threshold=eval_args.load_ada_interval, residual=base_args.residual)
        else:
            adapter = Lora(path=eval_args.load_adapter, rank=base_args.lora_r, group_size=base_args.group_size)
    
        model = GPTQModel.load(base_args.quantized_model_dir, **config_manager.get_model_kwargs(), adapter=adapter, backend=back) 
        print(f"Successfully loaded GPTQ model adapter from {eval_args.load_adapter}")
    else:
        model = GPTQModel.load(base_args.quantized_model_dir, **config_manager.get_model_kwargs(), backend=back) 

    if eval_args.tasks == [None]:
        model.optimize()  # It seems there's a compatibility conflict with lm-eval. We suggest testing MMLU without it enabled.
    return model



import re
import gc
import pprint
import itertools
from copy import deepcopy
torch._dynamo.config.capture_scalar_outputs = True 
# Captures scalar outputs in the graph, similar to Triton's .item(), for unified handling despite a larger graph.

# torch._dynamo.config.disable = True               
# os.environ["TRITON_NO_COMPILE"] = "1"             


def auto_bit_group(base_args):
    match = re.search(r'int(\d+)_(\d+)', base_args.quantized_model_dir)
    base_args.w_bits, base_args.group_size = int(match.group(1)), int(match.group(2))


def auto_lora_params(model_size, base_args):
    lora_r_map = {"8B": 64, "14B": 64, "32B": 32, "70B": 32}
    base_args.lora_r, base_args.lora_alpha = lora_r_map[model_size], lora_r_map[model_size]*2


if __name__ == "__main__":
    set_seed(42)
    parser = HfArgumentParser((baseConfig, trainingConfig, evalConfig))
    base_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    model, lm, tokenizer = None, None, None

    if base_args.mode == 1:                                   
        '''AUTO params config.'''
        base_args.training_flag = True
        if base_args.lota_qaf:
            training_args.optimizer_type = "int_sign_sgd"

        auto_bit_group(base_args)
        match = re.search(r'(\d+B)', base_args.quantized_model_dir)
        model_size = match.group(1)  
        auto_lora_params(model_size, base_args)

        config_manager = ModelConfigManager(base_args, training_args, eval_args)

        ''' training '''
        params_print((base_args, "baseConfig"), (training_args, "trainingConfig"))

        tokenizer = AutoTokenizer.from_pretrained(base_args.pretrained) 
        if tokenizer.pad_token is None: 
            tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
        QAF(base_args, training_args, config_manager, tokenizer)


    elif base_args.mode == 2:
        ''' AUTO params config. '''
        base_args.training_flag = False
        if eval_args.load_adapter != "none": 
            # LoTA params AUTO.
            base_args.lota_qaf ="lota" in "/".join(eval_args.load_adapter.split("/")[-2:]).lower()
            if base_args.lota_qaf:
                match = re.search(r'lota_(\d+)_', "/".join(eval_args.load_adapter.split("/")[-2:]).lower())           
                # match = re.search(r'LoTA_(\d+)_', eval_args.load_adapter)           
                eval_args.load_ada_interval = int(match.group(1))   # AUTO set load_ada_interval Omega from adatper_path

            # Model Choice AUTO.
            group_size_map = {"8B": 64, "14B": 64, "32B": 128, "70B": 128}
            pretrained_map = {"8B": "llama_3.1_8B_Instruct", "14B": "qwen_2.5_14B_Instruct", "32B": "qwen_2.5_32B_Instruct", "70B": "llama_3.3_70B_Instruct"}

            match = re.search(r'(\d+B)_int(\d+)', "/".join(eval_args.load_adapter.split("/")[-2:]))
            # match = re.search(r'(\d+B)_int(\d+)', eval_args.load_adapter)
            model_size = match.group(1)              # Get model size，e.g. "8B" or "70B"
            base_args.w_bits = int(match.group(2))   # Get bit，e.g. "4"

            pre = pretrained_map.get(model_size)
            base_args.pretrained = f"/your_path/models/{pre}"  

            base_args.group_size = group_size_map.get(model_size)
            base_args.quantized_model_dir = f"/your_path/quant_models/{model_size}_instruct/int{base_args.w_bits}_{base_args.group_size}_asym"

            auto_lora_params(model_size, base_args)

        config_manager = ModelConfigManager(base_args, training_args, eval_args)

        ''' eval '''
        params_print((base_args, "baseConfig"), (eval_args, "evalConfig"))

        # load model
        if eval_args.auto_gptq == "gptq":
            ada_flag = False if eval_args.load_adapter == "none" else True
            model = gptq_load(base_args, eval_args, config_manager, ada_flag)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_args.pretrained, **config_manager.get_model_kwargs()) # "none" is 16-bit load

        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        torch.set_float32_matmul_precision("high")

        if eval_args.ft_dataset_name == "none": 
            lm = HFLM(pretrained=model, dtype=base_args.dtype, batch_size=eval_args.eval_batch_size,
                        device=eval_args.device, trust_remote_code=base_args.trust_remote_code)
            start_time = timePrint(end=False)
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=eval_args.tasks,
                num_fewshot=eval_args.num_fewshot,
                batch_size=eval_args.eval_batch_size,
            )
            timePrint(end=True, time_1=start_time)

            if eval_args.output_path:
                save_results(results, base_args, eval_args)
        else:
            from evalGSV import evalgsv
            tokenizer = AutoTokenizer.from_pretrained(base_args.pretrained) 
            evalgsv(model, tokenizer, base_args, eval_args)  # , subsample=eval_args.subsample

    # Clear
    if model is not None:
        del model
        print("Deleted model object.")
    if lm is not None:
        del lm
        print("Deleted HFLM object.")
    if tokenizer is not None:
        del tokenizer
        print("Deleted tokenizer object.")
    model, lm, tokenizer = None, None, None
    torch.cuda.empty_cache()




