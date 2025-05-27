# LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

This repository contains the code for LoTA-QAF, a novel fine-tuning method for quantized Large Language Models (LLMs). It enables the lossless merging of ternary adaptation weights and the adjustment of all quantized weights.
LoTA-QAF combines:
* Custom-designed Ternary Adaptation (TA) that aligns ternary weights with the quantization grid to adjust quantized weights.
* A TA-based mechanism for the lossless merging of adaptation weights.
* Ternary Signed Gradient Descent (t-SignSGD) for updating TA weights.

## 1. File Structure

-   **Core Logic:**
    -   `LoTA_QAF_main.py`: The main script for training LoTA-QAF and performing evaluations (using lm-eval for MMLU and `evalGSV.py` for GSM8K, SQL, and ViGGO).

-   **LoTA Components (located in the `LoTA/` directory):**
    -   `LoTA/layer.py`: Contains `CustomLoraLinear`, where Ternary Adaptation is implemented, used for training.
    -   `LoTA/adapter.py`: Provides the `LTA` (Lossless Ternary Adaptation) classes for loading trained Ternary Adaptation during inference and evaluation.
    -   `LoTA/lota_merge.py`: Includes the logic for merging Ternary Adaptation weights into the quantized model weights.

-   **Optimizer:**
    -   `t_signSGD.py`: Implementation of the Ternary Signed Gradient Descent (t-SignSGD) optimizer used for training Ternary Adaptation.

-   **Utility Modules:**
    -   `data_print_save.py`: A collection of utility functions for preparing datasets (e.g., Alpaca, GSM8K, SQL, ViGGO), printing configurations, and saving experimental results, etc.
    -   `evalGSV.py`: A custom evaluation script designed for Task-Specific such as GSM8K, SQL, and ViGGO.
    -   `gptq_quantize.py`: A script used for quantizing models using the GPTQModel library, preparing them for QAF.

## 2. Quick Start

### Hardware Requirements:
* CUDA Version: 12.2 (Recommended).

### Software Dependencies:
The LoTA-QAF implementation is built upon specific versions of key libraries:
* `peft==0.15.1`
* `gptqmodel==2.1.1.dev0`

It is recommended to install packages using a virtual environment.
```bash
pip install -r requirements.txt
# For detailed versioning of all dependencies, please refer to the environment.yml file.
````

### Basic Usage

The main script `LoTA_QAF_main.py` operates in two modes: Training (mode 1) and Evaluation (mode 2).

**Common Base Parameters (`baseConfig`):**

  * `--mode`: `1` for training, `2` for evaluation.
  * `--pretrained`: Path to the base pre-trained model (e.g., `/your_path/models/llama_3.1_8B_Instruct`).
  * `--quantized_model_dir`: Path to the quantized model directory (e.g., `/your_path/quant_models/8B_instruct/int4_64_asym`). 

**Mode 1:**

For training, you'll primarily use `trainingConfig` arguments alongside `baseConfig`.

```bash
python LoTA_QAF_main.py \
    --mode 1 \
    --pretrained "/your_path/models/llama_3.1_8B_Instruct" \
    --quantized_model_dir "/your_path/quant_models/8B_instruct/int4_64_asym" \
    --lota_qaf True \

    --training_data_name "alpaca" \
    --adapter_path "your_path/adapter_output" \

    --interval_point 48 \   # Omega            for LoTA-QAF
    --filter_ratio 0.95 \   # Sigma_t          for LoTA-QAF, here 0.95 is discard 0.95 and select top 0.05. 
    --min_grad 0.999 \      # Effective range 0.95-0.999 in 0-80% of epochs. [Refer in "Baselines and Hyper-parameters" of the paper. The naming is not ideal and has not been updated yet.]
    --filter_upper 0.9999   # 0.999-0.9999     in 20-100% epoch

    --max_steps 300 \
    --save_number 5 \
    --train_batch_size 64 \
    --gradient_accumulation_steps 1 \
```


**Mode 2:**

For evaluation, you'll use `evalConfig` arguments. Parameters like `pretrained`, `quantized_model_dir`, `w_bits`, `group_size`, `lora_r`, `lora_alpha`, and `lota_qaf` are often automatically inferred from the `--load_adapter` path if an adapter is being evaluated.

```bash
# Example 1: Evaluate a GPTQ model with a LoTA-QAF adapter on MMLU
python LoTA_QAF_main.py \
    --mode 2 \
    --load_adapter "/path/to/your/trained_lota_adapter/8B_int4_LoTA_48_0.950_0.999_alpaca_..." \
    --tasks "mmlu" \
    --num_fewshot 5 \
    --eval_batch_size 16 \
    --output_path "./eval_results" \
    # --auto_gptq "gptq" # Default for loading adapter with GPTQ model

# Example 2: Evaluate a GPTQ model with a LoTA-QAF adapter on a task-specific dataset (e.g., gsm8k)
python LoTA_QAF_main.py \
    --mode 2 \
    --load_adapter "/path/to/your/trained_lota_adapter/8B_int4_LoTA_48_0.950_0.999_gsm8k" \
    --output_path "./eval_results_gsv" \
    --ft_dataset_name "gsm8k" \
    --eval_batch_size 64 \
    --auto_gptq "gptq"
```

  * **Key Evaluation Parameters (`evalConfig`):**
      * `--load_adapter`: Path to the trained adapter to load. Set to `"none"` to evaluate the base model without an adapter. Many parameters like `lota_qaf`, `w_bits`, `group_size`, `pretrained` model path, and `quantized_model_dir` will be auto-configured based on this path.
      * `--auto_gptq`: Use `"gptq"` to load a GPTQ quantized model (with or without adapter). Use `"none"` to load a 16-bit model (typically for evaluating a base 16-bit model without an adapter).
      * `--tasks`: List of tasks for lm-eval (e.g., "mmlu").
      * `--ft_dataset_name`: For task-specific evaluation using `evalGSV.py` (e.g., "gsm8k", "sql", "viggo"). If not "none", this evaluation type is chosen over lm-eval.
      * `--num_fewshot`: Number of few-shot examples for lm-eval.
      * `--eval_batch_size`: Batch size for evaluation.
      * `--output_path`: Directory to save evaluation results.

**Automatic Parameter Configuration:**
The script includes logic to automatically determine several parameters, especially in evaluation mode (`mode 2`) when `--load_adapter` is specified. This includes:

  * `w_bits`, `group_size` from `quantized_model_dir` (training) or `load_adapter` path (evaluation).
  * `lora_r`, `lora_alpha` based on model size.
  * `lota_qaf` and `load_ada_interval` (***Omega***) based on the `load_adapter` path structure.
  * `pretrained` model path and `quantized_model_dir` based on model size and quantization bits inferred from the `load_adapter` path. **Note:** You will need to update the placeholder `/your_path/` in the script (`base_args.pretrained = f"/your_path/models/{pre}"` and `base_args.quantized_model_dir = f"/your_path/quant_models/{model_size}_instruct/int{base_args.w_bits}_{base_args.group_size}_asym"`) to your actual model paths for this auto-configuration to work correctly.

## 3. License and Version

  - **Version**: 2025.05.15
  - **License**: MIT License

Copyright (c) 2025 Kingdalfgoodman@foxmail.com.

This project is licensed under the MIT License.

```
```
