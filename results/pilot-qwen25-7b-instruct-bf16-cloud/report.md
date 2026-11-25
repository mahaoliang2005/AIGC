# pilot 中英文提示词评测

计划 48 次生成，收到 48 份回复。

## HumanEval+

| 模型 | 题面 | 题数 | pass@1 (%) |
| --- | --- | ---: | ---: |
| qwen25-7b-instruct-base:bf16 | en | 8 | 75.00 |
| qwen25-7b-instruct-base:bf16 | zh | 8 | 87.50 |

| 模型 | 配对题数 | 中文 − 英文（pp） | 95% 区间（pp） |
| --- | ---: | ---: | --- |
| qwen25-7b-instruct-base:bf16 | 8 | 12.50 | 0.00 到 37.50 |

## HumanEval

| 模型 | 题面 | 题数 | pass@1 (%) |
| --- | --- | ---: | ---: |
| qwen25-7b-instruct-base:bf16 | en | 8 | 75.00 |
| qwen25-7b-instruct-base:bf16 | zh | 8 | 87.50 |

| 模型 | 配对题数 | 中文 − 英文（pp） | 95% 区间（pp） |
| --- | ---: | ---: | --- |
| qwen25-7b-instruct-base:bf16 | 8 | 12.50 | 0.00 到 37.50 |

评测使用项目自己的固定时间和内存限制。HumanEval 套件限时 3 秒，
HumanEval+ 套件限时 12 秒。官方 EvalPlus 的超时算法不同，性能相关失败可能有所差异。

HumanEval+ v0.1.10 的 HumanEval/32 自带参考实现有一条新增用例
未达到官方多项式残差阈值。本次评测保留该用例与原阈值。
