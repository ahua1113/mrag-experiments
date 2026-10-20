# DSE vs. ColPali：统一 ViDoRe benchmark

这套脚本让 DSE 与 ColPali 在完全相同的 query、页面语料和 qrels 上运行。默认评测：

- `vidore/docvqa_test_subsampled`：文档文字、表格等页面检索；
- `vidore/infovqa_test_subsampled`：信息图页面检索。

主指标是 ViDoRe 常用的 `nDCG@5`，同时输出 Recall@1/5/10、MRR@10、nDCG@10、编码速度和逐 query 的 Top-10 排名。

## 为什么不直接合并两个 demo

两个模型最好在各自已经验证过的 Python 环境里分别运行同一个脚本。这样既避免依赖冲突，也避免 4 GB GPU 同时加载两个模型。两次运行使用相同的 dataset 参数、随机种子和采样规则，结果仍严格可比。

## 1. 安装依赖

两个环境都安装：

```bash
pip install -r requirements-common.txt
```

ColPali 环境额外安装（已有可跳过）：

```bash
pip install -U "sentence-transformers[image]>=6.0.0"
```

DSE 环境额外安装（已有可跳过）：

```bash
pip install -U transformers accelerate bitsandbytes qwen-vl-utils
```

## 2. 先做等规模 smoke test

以下配置是“20 个 query、100 个页面”，每个 query 的正样本页面一定被保留，其余页面按 `seed=42` 固定采样。

在 ColPali 环境运行：

```bash   43
python benchmark_vidore.py \
  --model colpali \
  --datasets vidore/docvqa_test_subsampled vidore/infovqa_test_subsampled \
  --max-queries 20 \
  --corpus-size 100 \
  --batch-size 1 \
  --output-dir results_smoke \
  --cache-dir cache_smoke
```

在 DSE 环境运行（RTX 3050 Ti 4 GB）：

```bash
python benchmark_vidore.py \
  --model dse \
  --datasets vidore/docvqa_test_subsampled vidore/infovqa_test_subsampled \
  --max-queries 20 \
  --corpus-size 100 \
  --batch-size 1 \
  --dse-dimension 512 \
  --dse-max-image-patches 144 \
  --dse-gpu-memory 3.2GiB \
  --dse-cpu-memory 12GiB \
  --output-dir results_smoke \
  --cache-dir cache_smoke
```

Windows PowerShell 可将行尾的 `\` 换成反引号，或者把命令写成一行。

## 3. 对比 smoke test 结果

```bash
python compare_results.py results_smoke/*.json
```

先检查：

1. 两个模型对应数据集的 `queries` 与 `docs` 数量必须一致；
2. CSV 中 `is_relevant=1` 的页面是否能进入 Top-10；
3. DSE 是否出现 CUDA OOM；若有，先把 `--dse-max-image-patches` 降到 96，不要改变 ColPali 参数；
4. 缓存目录会按模型、数据集、query/doc IDs 和关键模型参数隔离，重跑会复用完整 shard。

## 4. 跑完整 V1 子集

移除 `--max-queries` 与 `--corpus-size`（它们默认都为 0，代表全量）：

```bash
python benchmark_vidore.py --model colpali --batch-size 1 --output-dir results_full --cache-dir cache_full
python benchmark_vidore.py --model dse --batch-size 1 --output-dir results_full --cache-dir cache_full
python compare_results.py results_full/*.json
```

`docvqa_test_subsampled` 和 `infovqa_test_subsampled` 各含 500 条 query。首次页面编码是最耗时步骤；脚本按 batch 保存 shard，中断后可从已有 shard 继续。

## 5. 公平性与报告建议

- 主结果先报告两个模型在相同 corpus 上的 `nDCG@5`，不要用各自 demo 的手写 query 结果作结论。
- 同时报告 embedding 配置：DSE 的 512 维与 144 patches 是效率配置，不等同于官方最大分辨率/1536 维结果。
- 速度比较应注明硬件，并把“文档离线编码”“query 在线编码”“相似度计算”分开。
- ColPali 是多向量 late interaction，DSE 是单向量点积；只比较 nDCG 不足以说明部署成本，后续应补充索引体积和 scoring latency。
- 若要复现官方 leaderboard，应在 smoke/full 本地实验稳定后，再迁移到当前官方推荐的 MTEB ViDoRe 评测流程。

## 输出

每个模型/数据集生成：

- `*.json`：汇总指标、配置、硬件和耗时；
- `*.csv`：逐 query 的 Top-K、分数、相关性和首个相关页面名次。

请保留 JSON 文件作为实验记录；CSV 用来分析失败案例。
