import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer
from utils.misc import vis_model_stucture, set_logger

logger = set_logger('./log.log')

repo_id = "./ckpt/llama-2-13b-chat-hf"
model = LlamaForCausalLM.from_pretrained(repo_id, device_map='auto')
model = model.eval()
tokenizer = LlamaTokenizer.from_pretrained(repo_id, use_fast=False)

vis_model_stucture(model)

tokenizer.pad_token = tokenizer.eos_token
# input_ids = tokenizer(['<s>Human: Introduce the history of Shenzhen\n</s><s>Assistant: '], 
                    #   return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
input_ids = tokenizer(['<s>Human: Who are you?\n</s><s>Assistant: '], 
                      return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')  
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

# GenerationMixin().generate: transformers/generation/utils.py GenerationMixin
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)