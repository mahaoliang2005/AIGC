# Qwen Bilingual V1

小组在 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28) 上对 1,800 条中英文合成规则判断样本做了 QLoRA 微调。样本覆盖边界条件、否定和复合条件。微调后的适配器合并到基础模型权重，用 llama.cpp b5400 转换为 BF16 GGUF。该文件对应正式评测使用的 `qwen-bilingual-v1:bf16`。 [正式实验的模型来源](../../results/formal-qwen-bilingual-v1-cloud/model-provenance.json)

上游模型按 Apache License 2.0 分发，随附[上游许可](LICENSE)。这里发布的适配器和 GGUF 含小组做的上述修改。

## LoRA 适配器

从 [LoRA 适配器 Release](https://github.com/mahaoliang2005/AIGC/releases/tag/qwen-bilingual-v1-adapter) 下载 `qwen-bilingual-v1-lora.tar.gz` 和对应校验文件。压缩包包含训练得到的适配器权重、PEFT 配置和分词器。加载适配器时显式使用上面的基础模型版本，配置中的本地模型路径和训练环境相关。

## 完整 BF16 模型

从 [BF16 GGUF Release](https://github.com/mahaoliang2005/AIGC/releases/tag/qwen-bilingual-v1-bf16) 下载全部分卷、校验文件和 `Modelfile` 到同一空目录。将所有分卷依编号拼回原文件，然后将它导入 Ollama。

```sh
sha256sum -c SHA256SUMS
cat qwen-bilingual-v1-bf16.gguf.part-* > qwen-bilingual-v1-bf16.gguf
sha256sum -c MODEL.sha256
ollama create qwen-bilingual-v1:bf16 -f Modelfile
```

在 macOS 上，可将两条 `sha256sum -c` 命令换为 `shasum -a 256 -c`。合并后的文件大小为 15,237,852,960 字节。
