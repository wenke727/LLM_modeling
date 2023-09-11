# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- /

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