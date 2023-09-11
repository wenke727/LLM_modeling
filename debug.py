
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import torch
from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer


repo_id = "./ckpt/llama-2-13b-chat-hf"
model = LlamaForCausalLM.from_pretrained(repo_id, device_map='auto', torch_dtype=torch.float16, load_in_8bit=False)
model = model.eval()
tokenizer = LlamaTokenizer.from_pretrained(repo_id, use_fast=False)


tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: Introduce the history of Shenzhen\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)