import os
import json
from datetime import datetime
import torch
import time
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import PreTrainedTokenizer
from typing import Dict
import dataclasses
import logging
IGNORE_INDEX = -100
#  ------------------------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


def get_Ada_path(base_args, training_args, runtime=None, max_memory=None, is_temp=False):
    timestamp = datetime.now().strftime("%d_%H%M")
    data_name = training_args.training_data_name

    if base_args.lota_qaf:
        middle_part = f"_LoTA_{training_args.interval_point}_{training_args.filter_ratio:.3f}_{training_args.min_grad}"
    else:
        middle_part = f"LoRA_{training_args.learning_rate:.0e}"

    if is_temp: 
        adapter_name = f"temp_int{base_args.w_bits}{middle_part}_{data_name}_{timestamp}"
    else:      
        runtime_str = str(runtime).split('.')[0] 

        import re
        match = re.search(r'(\d+B)', base_args.quantized_model_dir)
        model_size = match.group(1) 
        adapter_name = f"{model_size}_int{base_args.w_bits}{middle_part}_{data_name}_{runtime_str}_{int(max_memory)}_{timestamp}"

    save_path = os.path.join(training_args.adapter_path, adapter_name)  # e.g. "./adapter/8B_int4_LoTA_48_0.950_0.999_alpaca_..."

    return save_path


''' eval '''
def save_results(results, base_args, eval_args):
    os.makedirs(eval_args.output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")                           
    results_without_samples = {k: v for k, v in results.items() if k != 'samples'}  

    model_name = '_'.join((base_args.pretrained if eval_args.auto_gptq=='none' else eval_args.load_adapter).split('/')[-2:])
    # model_name = os.path.basename(base_args.pretrained if eval_args.auto_gptq=='none' else eval_args.load_adapter)

    full_results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "arguments": {
            "base_args": vars(base_args),  
            "eval_args": vars(eval_args)   
        },
        "evaluation_results": results_without_samples
    }

    results_file = os.path.join(eval_args.output_path, f"{model_name}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


''' base '''
def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def timePrint(end=False, time_1=None):
    if end == False:                  
        time_1 = datetime.now()
        print(f"||{'‾' * 50} Start: {time_1} {'‾' * 50}||")         # for .out in HPC
        logging.info(f"||{'‾' * 50} Start: {time_1} {'‾' * 50}||")  # for .err
        return time_1                   
    else:                              
        time_2 = datetime.now()
        print(f"||{'-' * 50} END: {time_2} {'-' * 50}||")
        logging.info(f"||{'-' * 50} END: {time_2} {'-' * 50}||")
        if time_1 is not None:
            runtime = time_2 - time_1
            print(f"||{'_' * 50} RUNNING: {runtime} {'_' * 50}||")
            logging.info(f"||{'_' * 50} RUNNING: {runtime} {'_' * 50}||")
        return runtime  
 

def params_print(*args_with_names):
    print(f"|{'‾' * 25} Model configuration parameters: {'‾' * 25}|")
    logging.info(f"|{'‾' * 25} Model configuration parameters: {'‾' * 25}|")
    for args, name in args_with_names:
        print(f"\n[{name}]")
        logging.info(f"\n[{name}]")
        current_section = None
        for f in dataclasses.fields(args):
            field_name = f.name
            value = getattr(args, field_name)
            section = f.metadata.get("help", "Other")
            if section != current_section:
                print(f"\n  {section}:")
                logging.info(f"\n  {section}:")
                current_section = section
            print(f"    - {field_name}: {repr(value)}")
            logging.info(f"    - {field_name}: {repr(value)}")
    print(f"|{'-' * 100}|")
    logging.info(f"|{'-' * 100}|")


''' Datasets '''
VIGGO_PROMPT = ''.join([
    "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. ",
    "This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. ",
    "The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
    "\n\n### Target sentence:\n{target}"
])


def prepare_dataset(data_name: str, tokenizer: PreTrainedTokenizer, train_ratio: float = 1.0, source_max_len=1024, target_max_len=256, debug_number=-1) -> DatasetDict:
    # 1. load
    if data_name == 'alpaca':
        dataset = load_dataset('json', data_files="your_path/data/alpaca/alpaca_data_cleaned.json", split=f"train", num_proc=32)
        source_fields = ["instruction", "input"]  
        target_field = "output" 
    elif data_name == 'gsm8k':
        dataset = load_from_disk("your_path/data/gsm8k_train")
        source_fields = "question"
        target_field = "answer"
    elif data_name == 'sql':
        dataset = load_dataset('json', data_files="your_path/data/sql/sql_train.jsonl", split="train", num_proc=32)
        source_fields = "messages"  # Will be processed dynamically
        target_field = "messages"
    elif data_name == 'viggo':
        dataset = load_from_disk("your_path/data/viggo_train")
        source_fields = "target"
        target_field = "meaning_representation"

    if debug_number != -1:
        print(f"####################################-debug_number: {debug_number}-####################################")
        dataset = dataset.select(range(debug_number))

    # 2. split
    if train_ratio == 1.0:
        train_dataset = dataset
        val_dataset = Dataset.from_dict({"input_ids": [], "labels": [], "attention_mask": []})
    else:
        split_dataset = dataset.train_test_split(test_size=float(1 - train_ratio), seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']

    # 3. preprocess
    # print(tokenizer.chat_template)
    def preprocess_fn(example):
        if data_name == "sql":      # Extract input and output from messages
            source_text = example[source_fields][0]['content'].strip()
            target_text = example[target_field][1]['content'].strip()
        else:
            if data_name == "viggo":                  # Apply ViGGO-specific prompt template
                source_text = VIGGO_PROMPT.format(target=example[source_fields]).strip()
            elif isinstance(source_fields, list):     # process  source_fields = ["inputs", "task"]
                source_text = "\n".join([example[field] for field in source_fields]).strip()
            else:
                source_text = example[source_fields].strip()
            target_text = example[target_field].strip()

        # Truncation to source and target in token-level
        source_tokens = tokenizer.encode(source_text, max_length=source_max_len, truncation=True)
        target_tokens = tokenizer.encode(target_text, max_length=target_max_len, truncation=True)
        source_text = tokenizer.decode(source_tokens, skip_special_tokens=True)
        target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)

        conversation = [
            {"role": "system", "content": ""},             
            {"role": "user", "content": source_text},     
            {"role": "assistant", "content": target_text}  
        ]
        # tokenize=True, add_generation_prompt=False, add_special_tokens=True
        model_inputs = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,                
            add_generation_prompt=False,  
            add_special_tokens=True,            # !!!  Right process of BOS/EOS  !!!
            max_length=source_max_len + target_max_len + 256,  
            truncation=True,
            return_dict=True,
        )

        # 3. input_ids and labels
        input_ids = model_inputs["input_ids"]
        labels = list(input_ids)
        attention_mask = model_inputs["attention_mask"]

        # 4. calculate prompt_len
        temp_prompt_tokens = tokenizer.apply_chat_template(
            conversation[:2],           # only usr, align to conversation.
            tokenize=True,
            add_generation_prompt=True, # Include prompt tokens to calculate full prompt length
            add_special_tokens=True,    # Consistent with main call, ensure BOS etc. are included
            return_dict=True,
        )
        prompt_len = len(temp_prompt_tokens['input_ids'])   

        ''' Align Check '''
        if input_ids[:prompt_len] != temp_prompt_tokens['input_ids']:
            logging.info(f"Warining token dismatch: {input_ids[:prompt_len]} != {temp_prompt_tokens['input_ids']}")
        labels = [IGNORE_INDEX] * prompt_len + labels[prompt_len:] 
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    # 4. apply
    train_dataset = train_dataset.map(preprocess_fn, remove_columns=list(train_dataset.column_names), num_proc=16)
    val_dataset = val_dataset.map(preprocess_fn, remove_columns=list(val_dataset.column_names), num_proc=16)

    # 5. reture DatasetDict
    return DatasetDict({'train': train_dataset, 'validation': val_dataset})


def check_data_simple(data_name, dataset, tokenizer):
    print("\n\nDataset used: ", str(data_name))    
    if isinstance(dataset, DatasetDict):
        print("Dataset type is DatasetDict, containing the following keys:", list(dataset.keys()))
    else:
        print("Dataset type does not meet expectations!")
        exit(1)

    # Print sample counts
    print(f"Number of training samples: {len(dataset['train'])}")
    print(f"Number of validation samples: {len(dataset['validation'])}")

    # Print the first 3 samples from the training set
    print("\nFirst few training samples:")

    lenth_data = 3000
    number = 1
    print(f"{lenth_data} {lenth_data} {lenth_data}")

    for i in range(min(number, len(dataset['train']))):  # Prevent insufficient samples
        sample = dataset['train'][i]
        print(f"Sample {i+1}:")
        print("input_ids:", sample['input_ids'][:lenth_data])  # Print first 10 tokens
        print("labels:", sample['labels'][:lenth_data])
        # Decode input_ids to view text content
        decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print("Decoded input_ids:", decoded_input[:lenth_data])  # Print first 100 characters

    # Print the first 3 samples from the validation set
    print("\nFirst few validation samples:")
    for i in range(min(number, len(dataset['validation']))):  # Prevent insufficient samples
        sample = dataset['validation'][i]
        print(f"Sample {i+1}:")
        print("input_ids:", sample['input_ids'][:lenth_data])
        print("labels:", sample['labels'][:lenth_data])
        decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print("Decoded input_ids:", decoded_input[:lenth_data])


