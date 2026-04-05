---
title: Llama-3.2-3B Code Generator LoRA Fine-tuned
emoji: 💻
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.23.3"
python_version: "3.10"
app_file: app.py
pinned: false
models:
  - meta-llama/Llama-3.2-3B
  - parthtamu/QLoRA-Finetuning
datasets:
  - sahil2801/CodeAlpaca-20k
---

# Llama-3.2-3B · CodeAlpaca LoRA Adapter

A LoRA adapter fine-tuned on [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
for instruction-following code generation tasks. Built on top of
[meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) with
4-bit NF4 quantization via `bitsandbytes`. Only **~1% of parameters** are
trainable — the rest of the base model is frozen.

---

## Model Details

| Field            | Value                                      |
|------------------|--------------------------------------------|
| **Base Model**   | meta-llama/Llama-3.2-3B                    |
| **Adapter Type** | LoRA (via PEFT)                            |
| **Task**         | Instruction-following code generation      |
| **Language**     | English                                    |
| **Author**       | Parth Deshmukh                             |
| **Date**         | April 2026                                 |

---

## Training Configuration

| Config               | Value                                           |
|----------------------|-------------------------------------------------|
| **LoRA Rank (r)**    | 8                                               |
| **LoRA Alpha**       | 16                                              |
| **LoRA Dropout**     | 0.05                                            |
| **Target Modules**   | `q_proj`, `v_proj`                              |
| **Quantization**     | 4-bit NF4 (`bitsandbytes` BitsAndBytesConfig)   |
| **Compute dtype**    | float16                                         |
| **Batch size**       | 2 (+ gradient accumulation steps = 4)           |
| **Mixed Precision**  | fp16                                            |
| **Hardware**         | Google Colab T4 GPU (16GB VRAM)                 |
| **Experiment Tracking** | MLflow + Weights & Biases                  |

---

## Dataset

- **Name:** [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- **Size:** ~20,000 code instruction samples
- **Split:** 90/10 train/test (~18,000 train, ~2,000 test)
- **Columns:** `instruction`, `input`, `output`
- **Prompt format:**
Instruction:
{instruction}

Input:
{input}

Response:
{output}

---

## Evaluation Results

Evaluated on **200 held-out test samples** from CodeAlpaca-20k using 4-bit
quantized inference. Metrics computed with `evaluate` (ROUGE-L) and
`bert_score` (BERTScore-F1).

| Model                              | ROUGE-L | BERTScore-F1 |
|------------------------------------|---------|--------------|
| Base (Llama-3.2-3B, no adapter)    | 0.3303  | 0.7835       |
| **Fine-tuned (this adapter)**      | **0.5458**  | **0.8856**   |
| **Delta**                          | **+0.2155 (+65.2%)** | **+0.1021 (+13.0%)** |

> ROUGE-L of 0.5458 is at the top of the competitive range for fine-tuned
> code generation models (0.43–0.55), confirming that LoRA fine-tuning
> successfully taught the model consistent instruction-following and code
> formatting behaviour.

---

## How to Use

Load the base model with 4-bit quantization, then apply this adapter using
PEFT's `PeftModel.from_pretrained()`.

**Prompt format:**
Instruction:
Write a Python function that reverses a string.

Input:
Response:
text

**Inference parameters used during evaluation:**
- `max_new_tokens`: 200
- `do_sample`: False
- `repetition_penalty`: 1.1
- `pad_token_id`: tokenizer.eos_token_id

---

## Repository Structure
llm-lora-finetuning/
├── app.py # Gradio demo (CPU + GPU modes)
├── requirements.txt # Dependencies for HF Spaces
├── README.md # This file
└── .github/
└── workflows/
└── sync_to_hf_spaces.yml # CI/CD — auto-syncs GitHub → HF Spaces

---

## Limitations

- Trained for only 1–3 epochs on 18k samples — may struggle with highly
  complex or multi-file code tasks
- Optimised for single-instruction, single-response code generation;
  not designed for multi-turn conversation
- Performance measured on CodeAlpaca-style prompts; may degrade on very
  different prompt formats
- Base model is 3B parameters — larger models (7B+) would likely achieve
  higher absolute scores

---

## Intended Use

✅ Learning and experimentation with LoRA fine-tuning  
✅ Generating short-to-medium code snippets from natural language instructions  
✅ Portfolio demonstration of PEFT / QLoRA techniques  

❌ Not intended for production code generation  
❌ Not suitable for security-sensitive or mission-critical applications  

---

## Project

This adapter was built as part of a 7-day end-to-end LLM fine-tuning project
covering LoRA/QLoRA concepts, dataset preparation, training, evaluation,
deployment, and CI/CD.

**GitHub:** [github.com/0-Parth-D/llm-lora-finetuning](https://github.com/0-Parth-D/llm-lora-finetuning)  
**Adapter on Hub:** [parthtamu/QLoRA-Finetuning](https://huggingface.co/parthtamu/QLoRA-Finetuning)