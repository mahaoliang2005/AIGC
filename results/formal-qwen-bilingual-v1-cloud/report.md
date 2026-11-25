# formal 中英文提示词评测

计划 984 次生成，收到 984 份回复。

## HumanEval+

| 模型 | 题面 | 题数 | pass@1 (%) |
| --- | --- | ---: | ---: |
| qwen-bilingual-v1:bf16 | en | 164 | 79.67 |
| qwen-bilingual-v1:bf16 | zh | 164 | 76.42 |

| 模型 | 配对题数 | 中文 − 英文（pp） | 95% 区间（pp） |
| --- | ---: | ---: | --- |
| qwen-bilingual-v1:bf16 | 164 | -3.25 | -7.93 到 1.22 |

## HumanEval

| 模型 | 题面 | 题数 | pass@1 (%) |
| --- | --- | ---: | ---: |
| qwen-bilingual-v1:bf16 | en | 164 | 83.13 |
| qwen-bilingual-v1:bf16 | zh | 164 | 79.67 |

| 模型 | 配对题数 | 中文 − 英文（pp） | 95% 区间（pp） |
| --- | ---: | ---: | --- |
| qwen-bilingual-v1:bf16 | 164 | -3.46 | -8.13 到 1.22 |

题面审校方式为一位代理撰写，独立代理对照英文逐题复核。

正式运行前核对的试跑：

| 批次 | 模型 | 题面 | HumanEval+ 通过 / 判分 |
| --- | --- | --- | ---: |
| pilot | qwen-bilingual-v1:bf16 | en | 19 / 24 |
| pilot | qwen-bilingual-v1:bf16 | zh | 21 / 24 |

评测使用项目自己的固定时间和内存限制。HumanEval 套件限时 3 秒，
HumanEval+ 套件限时 12 秒。官方 EvalPlus 的超时算法不同，性能相关失败可能有所差异。

HumanEval+ v0.1.10 的 HumanEval/32 自带参考实现有一条新增用例
未达到官方多项式残差阈值。本次评测保留该用例与原阈值。
