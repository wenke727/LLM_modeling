# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- /



## [0.0.4] - 2023-09-18

### Changed

- module

  - attention
    - repeat_kv： num_key_value_heads 在维度上的重复是逐个相间的， AAABBB…XXX
    - LlamaAttention：总结创新点

  - model
    - 增加 logger 功能
    - forward 注释
      - position_ids 作用：仅作用于 RoPE, 用于位置编码，主要是KV Cache 的引入，使得顺序需要使用绝对位置表示

## [0.0.3] - 2023-09-12

### Added

- utils.misc
  - vis_model_stucture
  - set_logger

### Changed

- llama/module
  - attetnion: 注释`维度`和`前向过程`
  - decorder layer: 增加层数
  - mlp: 注释`维度`和`前向过程`
  - model: 给 decoder layer 增加层数
  - position embedding: 注释`维度`和`前向过程`

## [0.0.2] - 2023-09-11

### Changed

- LLaMA 模块化
  - LlamaModel
  - decode layer
  - attention
  - mlp
  - norm
  - rope
  - misc

## [0.0.1] - 2023-09-11

### Added

- `llama` 代码
  - 从 [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/) 库中复制代码，时间为 2023-09-11
  - `...` -> `transformers.` (transformers==4.29.1)

- debug.py
  用于 debug 的代码入口

- ckpt
  - `llama-2-13b-chat-hf` 模型权重，convert_llama_weights_to_hf

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/Author/Repository/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/Author/Repository/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/Author/Repository/releases/tag/v0.0.1