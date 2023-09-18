import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

from chatglm2 import ChatGLMForConditionalGeneration, ChatGLMTokenizer

token_id = "./chatglm2/tokenizer"
repo_id = "./ckpt/chatglm2-6b-hf"

tokenizer = ChatGLMTokenizer.from_pretrained(token_id, trust_remote_code=True)
model = ChatGLMForConditionalGeneration.from_pretrained(repo_id).cuda()

# from bigmodelvis import Visualization
# Visualization(model).structure_graph()

# 创建一些输入数据
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to('cuda')

response, history = model.chat(tokenizer, "你好，介绍下你自己", history=[])
print(response)
