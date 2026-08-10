import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p=argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--adapter', required=True); p.add_argument('--output', required=True); a=p.parse_args()
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, device_map='auto')
model = PeftModel.from_pretrained(base, a.adapter)
merged = model.merge_and_unload()
merged.save_pretrained(a.output, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.output)
print(f'Saved merged model to {a.output}')
