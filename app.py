from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
from peft import PeftModel
import torch
import os
from huggingface_hub import login

# ─────────────────────────────────────────────
# LOGIN TO HUGGINGFACE
# ─────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

repo_path = "parthtamu/QLoRA-Finetuning"
base_model_name = "meta-llama/Llama-3.2-3B"

# ─────────────────────────────────────────────
# GPU MODE (Colab / HuggingFace Spaces with GPU)
# Uncomment this block and comment out CPU MODE below
# Requires: bitsandbytes, CUDA GPU
# ─────────────────────────────────────────────

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )
# _base = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,
#     device_map="auto"            # auto-assigns to GPU
# )

# ─────────────────────────────────────────────
# CPU MODE (HuggingFace Spaces free CPU tier)
# Slower (~2-5 min/response) but works without a GPU
# Comment this block out if switching to GPU MODE above
# ─────────────────────────────────────────────

_base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,   # float16 keeps memory lower on CPU
    device_map="cpu"
)

# ─────────────────────────────────────────────
# SHARED: tokenizer + adapter (same for both modes)
# ─────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
finetuned_model = PeftModel.from_pretrained(_base, repo_path)

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def run_inference(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cpu")   # change to .to("cuda") when using GPU MODE
    # ).to("cuda")   # change to .to("cpu") when using CPU MODE

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    generated_only = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_only, skip_special_tokens=True).strip()


def chat(message, history):
    prompt = f"### Instruction:\n{message}\n\n### Input:\n\n\n### Response:\n"
    return run_inference(finetuned_model, tokenizer, prompt)

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

demo = gr.ChatInterface(
    fn=chat,
    title="Llama-3.2-3B Code Generator — LoRA Fine-tuned",
    description=(
        "Fine-tuned on CodeAlpaca-20k using QLoRA (rank=8, alpha=16). "
        "Ask it to write, explain, or fix code. "
        "⚠️ Running on CPU — responses may take a few minutes."
    ),
    examples=[
        "Write a Python function that reverses a linked list.",
        "Write a SQL query to find the top 5 customers by total purchase amount.",
        "Write a JavaScript function that debounces another function.",
    ],
    cache_examples=False,
)

# For Colab testing: use demo.launch(share=True)
# For HuggingFace Spaces: use demo.launch() — Spaces handles server config

demo.launch()

# This for Colab:
# demo.launch(share=True)