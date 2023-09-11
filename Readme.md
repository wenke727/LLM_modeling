# LLaMA2 代码学习

## 代码

> copy from [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/), date: 2023-09-11

![13b 模型框架](.fig/LLaMa_code_framework.png)

步骤：

1. 复制代码至 `src`
2. `src` 中将 `...` -> `transformers.`


## 准备

### 准备权重


```bash
cd ./ckpt
ln -s /root/share/LLama2/llama-2-13b-chat-hf ./llama-2-13b-chat-hf
```

