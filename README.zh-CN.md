# EngageGraph

基于面板数据的时序 next-item 推荐：可复现基准、四个参考模型、可部署 API 与仪表盘。

[![CI](https://github.com/cxy2696/app-engagement/actions/workflows/ci.yml/badge.svg)](https://github.com/cxy2696/app-engagement/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/cxy2696/app-engagement)](https://github.com/cxy2696/app-engagement/releases)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-4f46e5)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-0f766e)](LICENSE)
[![Data: research use](https://img.shields.io/badge/data-research%20use%20only-e05d44)](DATA_LICENSE.md)

[展示页](docs/showcase.md) · [技术报告](paper/engagegraph-report.pdf) · [评估报告](reports/evaluation-report.md) · [可复现性验证](docs/verification.md) · [Release](https://github.com/cxy2696/app-engagement/releases)

![EngageGraph 仪表盘](docs/assets/dashboard.png)

## 项目内容

EngageGraph 用一个周期拟合模型，对用户未使用过的物品排序，评估其在下一周期的采用情况。参考数据是 2024 年 1–2 月的假名化移动应用面板；引擎、评估、artefact 和 API 面向任意“用户 × 物品”交互表，应用只是其中一个领域配置。

原 notebook 的 ROC-AUC 0.9257 来自 2 月边、相似度选负样本和错误的 cosine 读出，不能用来估计下一周期排序质量。差异与复现见 [旧结果审计](reports/legacy-audit.md)。

## 快速开始

```bash
git clone https://github.com/cxy2696/app-engagement.git
cd app-engagement
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,als,plots]"      # MetaPath2Vec 需要 ".[graph]"

engagegraph data build --config configs/dataset.yaml --out data/processed
engagegraph benchmark --config configs/benchmark.yaml
engagegraph serve --artifact artifacts/runs/<run>/artifact --port 8000
```

仪表盘 <http://127.0.0.1:8000>，OpenAPI `/docs`，Prometheus 指标 `/metrics`。仪表盘状态可通过 `?user=...&k=10&min_percentile=0.9` 分享，每行可跳转相似物品。

快捷方式：

```bash
make demo && make serve
```

## 评估协议

1 月数据用于拟合全部映射、目录统计、流行度、相似度矩阵与模型参数；2 月只用于构造标签：物品在 1 月目录内、且该用户在 1 月未使用过，才算正例。评分时屏蔽 1 月已用物品。主指标为按用户宏平均的 NDCG@10；置信区间为 1,000 次用户级配对 bootstrap，模型比较使用 Holm 校正。

没有 1 月历史的用户有 10,568 人，作为冷启动切片单独报告；有 1 月历史但不足 3 个目录物品的用户有 307 人，属于不合格样本，也单独计数，不混入个性化分母。完整协议见 [docs/evaluation.md](docs/evaluation.md) 与 [docs/methodology.md](docs/methodology.md)。

## 参考结果

<!-- RESULTS_TABLE_START -->
| Model | NDCG@10 (95% CI) | Recall@10 | MAP@10 | Coverage@10 | Novelty@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `itemknn` | **0.2685** [0.2660, 0.2710] | 0.1522 | 0.1516 | 0.013 | 1.107 |
| `popularity` | **0.2351** [0.2327, 0.2376] | 0.1347 | 0.1297 | 0.006 | 1.055 |
| `metapath2vec` | **0.1639** [0.1622, 0.1660] | 0.0933 | 0.0772 | 0.015 | 1.661 |
| `als` | **0.0993** [0.0974, 0.1013] | 0.0798 | 0.0505 | 0.317 | 4.830 |

Dataset fingerprint: `dca196daee55783bae1c` · Primary metric: `ndcg@10` · Evaluation users: 28,286 · Cold-start users: 10,568. Paired bootstrap intervals (1,000 resamples) are computed over evaluation users.
<!-- RESULTS_TABLE_END -->

最强的是 item-kNN，而不是最初驱动仓库的图模型。ALS 与 MetaPath2Vec 的覆盖率/新颖度权衡也保留在报告中，没有通过调参隐藏。见 [评估报告](reports/evaluation-report.md)。

## 一个引擎，多个领域

`domain` 只提供领域词汇和可选的 serving blocklist，不进入模型或指标。

```yaml
data:
  train_path: data/your_domain/train.parquet
  test_path: data/your_domain/test.parquet
  user_col: customer_id
  item_col: sku
  weight_col: revenue
  domain:
    name: ecommerce
    user_label: customer
    item_label: product
    item_label_plural: products
    task_name: next-period product adoption ranking
    blocklist: []
```

可直接使用：[通用零售 CSV 示例](profiles/generic-interactions.yaml)、[电商模板](profiles/ecommerce.example.yaml)、[应用参考配置](configs/benchmark.yaml)。详见 [docs/generality.md](docs/generality.md)。

## 目录

```text
src/engagegraph/   数据、模型、评估、API、CLI
tests/             单元/集成/数学不变量测试
configs/           基准实验配置
profiles/          领域配置与模板
docs/              协议、数据/模型卡、部署、运维
paper/             LaTeX 技术报告与 PDF
reports/           评估、审计、验证报告
scripts/           数据获取、demo、报告、截图、验证
Jan-Feb-parquet/   原始聚合导出（仅研究复现）
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [Methodology](docs/methodology.md) | 任务、估计量、泄漏控制、统计 |
| [Algorithm notes](docs/algorithm.md) | item-kNN、百分位、决策规则 |
| [Evaluation](docs/evaluation.md) | 命令与指标定义 |
| [Reproducibility](docs/reproducibility.md) | 种子、指纹、环境 |
| [Verification](docs/verification.md) | 确定性、领域不变性、数学不变量 |
| [Data card](docs/data-card.md) | 来源、字段、质量、隐私 |
| [Model card](docs/model-card.md) | 适用范围、指标、局限 |
| [Generality](docs/generality.md) | 适配其他物品目录 |
| [Architecture](docs/architecture.md) | 组件与扩展点 |
| [API](docs/api.md) | 端点与示例 |
| [Deployment](docs/deployment.md) | Docker、代理、配置 |
| [Operations](docs/operations.md) | SLO、监控、重训练、事故 |
| [Showcase](docs/showcase.md) | 截图、链接、参考结果 |
| [Governance](docs/governance.md) | 变更控制与发布 |

## 数据与许可

代码使用 Apache-2.0。`Jan-Feb-parquet/` 中的数据不适用该许可，仅用于研究复现；详见 [DATA_LICENSE.md](DATA_LICENSE.md)。ID 虽为假名化，仍可能被再识别；禁止再识别或公开个人级记录。权重单位是从分布推断的毫秒，并非厂商 codebook 事实；见 [数据卡](docs/data-card.md)。

## 引用与维护

```bibtex
@software{engagegraph2026,
  author  = {Cui, Xinyi},
  title   = {EngageGraph: A Reproducible Benchmark and Toolkit for Temporal App-Engagement Prediction},
  year    = {2026},
  version = {2.3.0},
  url     = {https://github.com/cxy2696/app-engagement}
}
```

维护者：Xinyi Cui，[xc535@cornell.edu](mailto:xc535@cornell.edu)。问题与 PR 请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)；安全问题见 [SECURITY.md](SECURITY.md)；发布与变更策略见 [GOVERNANCE.md](GOVERNANCE.md)。
