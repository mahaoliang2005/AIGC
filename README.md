# 代码生成的中英文提示词评测

在 HumanEval 和 HumanEval+ 上比较代码模型对中英文题面的表现。实验翻译函数文档字符串中的题目说明，沿用原函数签名、示例、英文答题指令和测试。每个正式模型包含 164 道题、两种题面和每种题面 3 次采样，共 984 次生成。

## 云服务器实验结果

| 模型 | 正式报告 |
| --- | --- |
| Qwen2.5-7B-Instruct，微调前 BF16 | [基础模型结果](results/formal-qwen25-7b-instruct-bf16-cloud/report.md) |
| Qwen Bilingual V1，微调后 BF16 | [微调模型结果](results/formal-qwen-bilingual-v1-cloud/report.md) |

[微调前后对照报告](results/formal-qwen25-7b-instruct-bf16-cloud/comparison.md)包含按题配对的通过率差值和 95% 区间。各次运行的目录包含题目清单、模型配置、原始回复和逐条判分。`pilot-` 目录保存各模型的 8 题试跑。`formal-` 目录保存独立的全量运行。

[模型说明和下载方式](models/qwen-bilingual-v1/README.md)

## 运行评测

在仓库根目录运行。需要 Python 3.9 或更新版本以及 Docker。使用本机模型时，还需要 Ollama。

```sh
docker pull python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c
python3 -B -m prompt_language_comparison --output results/single-task-demo.jsonl
python3 -B -m unittest discover -s tests -v
```

单题演示使用固定回复，不调用模型。判题在禁用网络、只读且无项目目录挂载的容器里执行。完整题库、翻译审校和数据许可见[数据说明](prompt_language_comparison/data/SOURCES.md)。

导入微调版后，用发布的运行配置开始一轮新的试跑及全量评测。

```sh
config=results/formal-qwen-bilingual-v1-cloud/run-config.json
python3 -B -m prompt_language_comparison.experiment run \
  --phase pilot --config "$config" --directory results/pilot-bilingual-new
python3 -B -m prompt_language_comparison.experiment run \
  --phase formal --config "$config" --directory results/formal-bilingual-new \
  --pilot-directory results/pilot-bilingual-new --workers 4 --generation-workers 1
```

使用相同的配置和目录重复运行会沿用已保存的生成和判分。用 `experiment report --directory <运行目录>` 可从现有记录重新生成报告。第三方 API 配置通过环境变量读取各自的凭据。
