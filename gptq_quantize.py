import os
import logging
logger = logging.getLogger(__name__)
os.environ["TRITON_CACHE_DIR"] = os.path.join(os.getcwd(), ".triton/cache")

from transformers import AutoTokenizer, HfArgumentParser
from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.adapter.adapter import Lora, LTA
from data_print_save import *

import torch
check_cuda()
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#  ------------------------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


from typing import List, Any, Dict
from dataclasses import dataclass, field, replace
@dataclass
class ModelConfig: 
    '''gptqModel'''
    w_bits: int = field(default=4, metadata={"help": "Weight quantization bits"})  
    group_size: int = 64
    sym: bool = False
    pretrained: str = field(default="your_path", metadata={"help": "Path to pretrained model"})
    quantized_model_save: str = "../quant_models"

    gptqing_flag: bool = True
    device_map: str = field(default="cuda:0", metadata={"help": "Device map configuration"})
    dtype: str = field(default="bfloat16", metadata={"help": "Data type for model"})
    flash_attn: str = field(default="flash_attention_2")
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code"})


class ModelConfigManager:
    def __init__(self, args: ModelConfig):
        self.args = args

    def get_model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.args.device_map,
            "trust_remote_code": self.args.trust_remote_code,
            "attn_implementation": self.args.flash_attn if self.args.gptqing_flag == False else "eager"
        }
        return base_kwargs


def gptqing(args, config_manager):
    quant_config = QuantizeConfig(bits=args.w_bits, group_size=args.group_size, sym=args.sym,)
    model = GPTQModel.load(args.pretrained, **config_manager.get_model_kwargs(), quantize_config=quant_config)

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(1024))["text"]

    model.quantize(calibration_dataset, batch_size=1)  
    model.save(args.quantized_model_save)


if __name__ == "__main__":  
    set_seed(42)
    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]  
    config_manager = ModelConfigManager(args) 
    gptqing(args, config_manager)
    torch.cuda.empty_cache()

