import os
import os.path
import json
import random
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from typing import Dict

IGNORE_INDEX = -100
#  ------------------------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

'''
    The eval script is further implemented based on HALO's test code (About gsm8k, sql, viggo).
    ref -- HALO: Hadamard-Assisted Lossless Optimization for Efficient Low-Precision LLM Training and Fine-Tuning
    ref -- https://github.com/IST-DASLab/HALO
'''

def is_correct(pred, label, dataset_name):
    pred, label = pred.strip(), label.strip()
    if dataset_name == 'gsm8k':
        return pred.split('####')[-1].strip() == label.split('####')[-1].strip()
    elif dataset_name == 'viggo':
        pred_fn = pred.split('(')[0]
        true_fn = label.split('(')[0]
        if pred_fn != true_fn:
            return False 
        fn = true_fn
        pred_attrs = sorted([p.strip() for p in pred.replace(fn, '').strip('()').split(',')])
        label_attrs = sorted([l.strip() for l in label.replace(fn, '').strip('()').split(',')])
        return len(pred_attrs) == len(label_attrs) and all([p == l for p, l in zip(pred_attrs, label_attrs)])
    elif dataset_name == 'sql':
        return pred == label


def batch_eval(model, tokenizer, dataset_name, batch):
    preprocess_fn = globals()[f"{dataset_name}_preprocessing_function"]

    chat_text = [preprocess_fn({'input': inp, 'output': ''}, tokenizer)['prompt'] for inp in batch]

    model_inputs = tokenizer(chat_text, return_tensors="pt", padding=True).to("cuda")
    generated_ids = model.generate(
        **model_inputs,
        max_length=512, do_sample=False, num_beams=1, top_p=None, temperature=None,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id
    )
    results_raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    generated_ids = generated_ids[..., model_inputs['input_ids'].shape[-1]:]
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    results = [res.strip() for res in results]

    return results, results_raw


@torch.no_grad()
def evalgsv(model, tokenizer, base_args, eval_args, subsample=None):
    model.eval()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    dataset_name = eval_args.ft_dataset_name

    if dataset_name == 'gsm8k':
        dataset = load_from_disk("your_path/data/gsm8k_test")
        dataset = dataset.map(
            lambda example: {'inp': example['question'], 'label': example['answer']})
        
    elif dataset_name == 'sql':
        dataset = load_dataset('json', data_files="your_path/data/sql/sql_valid.jsonl", split="train")
        dataset = dataset.map(
            lambda example: {'inp': example['messages'][0]['content'], 'label': example['messages'][1]['content'],}, remove_columns=['messages'])
            
    elif dataset_name == 'viggo':
        dataset = load_from_disk('your_path/data/viggo_test')
        dataset = dataset.map(
            lambda example: {'inp': example['target'], 'label': example['meaning_representation']})
    else:
        assert False, f"Unknown dataset {dataset_name}"

    if subsample is not None:
        assert type(subsample) == int, "subsample must be an integer"
        assert subsample <= len(dataset), "subsample must be less than the dataset size"
        assert subsample > 0, "subsample must be greater than 0"
        dataset = dataset.select(random.sample(range(len(dataset)), subsample))

    dataloader = DataLoader(dataset, batch_size=eval_args.eval_batch_size, shuffle=False)

    results_get = []
    results_raw = []
    labels_all = []
    for batch in tqdm(dataloader):
        preds, preds_raw = batch_eval(model, tokenizer, dataset_name, batch['inp'])
        labels = batch['label']
        results_get.extend([is_correct(pred, label, dataset_name) for pred, label in zip(preds, labels)])
        results_raw.extend(preds_raw)
        labels_all.extend(labels)
    acc = sum(results_get) / len(results_get)
    save_rs = (
        acc,
        results_raw[:3],  
        results_get[:3],  
        labels_all[:3]    
    )
    save_results(save_rs, base_args, eval_args)
    

def save_results(save_rs, base_args, eval_args): 
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M") 
    os.makedirs(eval_args.output_path, exist_ok=True)

    results_acc, results_raw, results_get, labels_all = save_rs
    model_name = '_'.join((base_args.pretrained if eval_args.auto_gptq=='none' else eval_args.load_adapter).split('/')[-2:])

    full_results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "arguments": {
            "base_args": vars(base_args),  
            "eval_args": vars(eval_args)   
        },
        "acc": results_acc,
        "results_raw": results_raw, 
        "results_get": results_get,    
        "labels_all": labels_all       
    }

    results_file = os.path.join(eval_args.output_path, f"{model_name}_{results_acc}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


def chat_preprocess(text_prompt, text_response, tokenizer):
    prompt = tokenizer.apply_chat_template(
        conversation=[
            # {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text_prompt},
            # {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        tokenize=False, add_generation_prompt=True, add_special_tokens=True
    )

    prompt_response = tokenizer.apply_chat_template(
        conversation=[
            # {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text_prompt},
            {"role": "assistant", "content": text_response},
        ],
        tokenize=False, add_generation_prompt=False, add_special_tokens=True
    )

    for i in range(len(prompt)):
        assert prompt[i] == prompt_response[i], f"Prompt mismatch at index {i}: {prompt[:i]} != {prompt_response[:i]}"
    response = prompt_response[len(prompt):]

    return {
        "prompt": prompt, "response": response,
    }


def gsm8k_preprocessing_function(inp: Dict, tokenizer) -> Dict:
    try:
        if 'input' not in inp:
            inp['input'] = inp['question']
        if 'output' not in inp:
            inp['output'] = inp['answer']
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e
    return chat_preprocess(inp['input'], inp['output'], tokenizer)


def sql_preprocessing_function(inp: Dict, tokenizer) -> Dict:
    """Split out prompt/response from text."""
    try:
        if 'input' not in inp:
            inp['input'] = inp['messages'][0]['content']
        if 'output' not in inp:
            inp['output'] = inp['messages'][1]['content']
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from 'text'={inp['text']}") from e
    return chat_preprocess(inp['input'], inp['output'], tokenizer)


def viggo_preprocessing_function(inp: Dict, tokenizer) -> Dict:
    try:
        if 'input' not in inp:
            inp['input'] = inp['target']
        if 'output' not in inp:
            inp['output'] = inp['meaning_representation']
        prompt = VIGGO_PROMPT.format(target=inp['input'])
        response = inp['output']
    except Exception as e:
        raise ValueError(f"Unable to extract prompt/response from {inp}") from e
    return chat_preprocess(prompt, response, tokenizer)


VIGGO_PROMPT = ''.join([
    "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. ",
    "This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. ",
    "The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
    "\n\n### Target sentence:\n{target}"
])



