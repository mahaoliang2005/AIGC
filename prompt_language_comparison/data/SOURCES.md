# HumanEval/0 数据来源

`humaneval_0.json` 存放单题评测数据。`task_id`、`entry_point`、`prompt` 与 `test` 来自 OpenAI 原始题库。`plus_solution` 对应 EvalPlus 的 `canonical_solution`，`plus_input` 保留全部 999 组新增输入及原始顺序。该题的原始与加强结果均为布尔值。

两份完整归档已保存在本目录。下载后校验了 SHA-256，并确认两套题号都完整覆盖 `HumanEval/0` 至 `HumanEval/163`。

| 来源 | 本地文件 | 官方存档 | 官方存档 SHA-256 | 许可 |
| --- | --- | --- | --- | --- |
| OpenAI HumanEval | [HumanEval.jsonl.gz](HumanEval.jsonl.gz) | [下载](https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz) | `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` | [MIT](LICENSE-HumanEval.txt) |
| EvalPlus HumanEval+ v0.1.10 | [HumanEvalPlus-v0.1.10.jsonl.gz](HumanEvalPlus-v0.1.10.jsonl.gz) | [下载](https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz) | `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101` | [Apache 2.0](LICENSE-HumanEvalPlus.txt) |
