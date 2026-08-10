import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

p = argparse.ArgumentParser()
p.add_argument('--model', default='Qwen/Qwen3-1.7B')
p.add_argument('--train', required=True)
p.add_argument('--output', required=True)
p.add_argument('--epochs', type=float, default=2.0)
p.add_argument('--max-steps', type=int, default=-1)
p.add_argument('--max-length', type=int, default=2048)
a = p.parse_args()

if not torch.cuda.is_available():
    raise SystemExit('QLoRA script expects an NVIDIA CUDA GPU. Use a GPU machine/Colab/Kaggle/cloud instance.')

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    a.model,
    quantization_config=quant,
    device_map='auto',
    torch_dtype=dtype,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_ds = load_dataset('json', data_files=a.train, split='train')

peft = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules='all-linear',
)

cfg = SFTConfig(
    output_dir=a.output,
    num_train_epochs=a.epochs,
    max_steps=a.max_steps,
    max_length=a.max_length,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy='epoch',
    gradient_checkpointing=True,
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    report_to='none',
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    processing_class=tokenizer,
    peft_config=peft,
)
trainer.train()
trainer.save_model(a.output)
tokenizer.save_pretrained(a.output)
print(f'Saved LoRA adapter to {a.output}')
