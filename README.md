
# JournalMatch-Agent

> An evidence-grounded paper quality assessment and journal recommendation system built with LangChain, DeepSeek, BGE-M3, Chroma, SQLite and Cross-Encoder Reranking.

## 1. Overview

JournalMatch-Agent是一个面向计算机科学论文的本地期刊推荐系统。

用户输入论文 PDF 后，系统首先通过 DeepSeek 将论文转换为结构化 `PaperProfile`，提取研究方向、研究问题、方法、数据集、主要贡献、创新点、实验结果和局限性。

在此基础上，系统进一步执行 **Evidence-based Paper Quality Assessment**，通过预定义 Rubric 从以下五个维度评价论文：

- Novelty
- Methodology
- Dataset Quality
- Experimental Quality
- Conclusion Support

每个维度采用 **1～5 级离散评分 + Evidence + Concerns** 的方式进行可解释评价，而不是直接生成缺乏校准依据的 0～100 分数。

随后系统通过 BGE-M3 + Chroma 对本地期刊知识库进行语义召回，结合 SQLite 中的 CCF、JCR、中科院分区及 Impact Factor 完成结构化过滤，并使用 BGE-Reranker Cross-Encoder 对候选期刊进行精排。

最终，DeepSeek 只允许基于真实检索候选、论文质量评价和期刊 Scope 生成 Grounded Top-K Recommendation。

当前本地数据库包含约 **291 本计算机领域期刊**。
![示例图片](pdf1.png)

---

## 2. Core Architecture

```text
Paper PDF
   ↓
PyPDFLoader
   ↓
DeepSeek Structured Analysis
   ↓
PaperProfile
   ↓
Evidence-based Paper Quality Assessment
   │
   ├── Novelty
   ├── Methodology
   ├── Dataset Quality
   ├── Experimental Quality
   └── Conclusion Support
   ↓
BGE-M3
   ↓
Chroma Semantic Retrieval
   ↓
SQLite Structured Filtering
   │
   ├── CCF
   ├── JCR
   ├── CAS
   └── Impact Factor
   ↓
BGE-Reranker-v2-m3
   ↓
Reranked Candidate Journals
   ↓
DeepSeek Grounded Recommendation
   ↓
Top-K Journal Recommendations
```

Agent Mode:

```text
User
 ↓
LangChain Agent
 ↓
Intent Routing / Tool Calling
 ↓
├── Analyze Paper & Quality
├── Search Journals
├── Get Journal Details
└── Recommend Journals
        ↓
Deterministic Recommendation Pipeline
```

---

## 3. PaperProfile

论文首先被转换为结构化数据：

```text
title
abstract
keywords
research_fields
research_problem
methods
datasets
main_contributions
claimed_innovations
experimental_results
limitations
summary
```

使用 Pydantic Structured Output 的主要原因是让后续：

- Retrieval
- Quality Assessment
- Reranking
- Recommendation
- Evaluation

能够直接消费稳定的结构化字段，而不是解析自由格式的 LLM 文本。

---

## 4. Evidence-based Paper Quality Assessment

系统没有采用简单的：

```text
Quality Score = 87 / 100
Innovation Score = 92 / 100
```

因为这类 LLM 数字缺乏真实校准依据。

本项目采用 **Rubric-constrained Assessment**。

### Novelty

评价：

- 是否只是已有工作的复现
- 是否只是简单模块组合
- 是否存在明确的新方法、新模块或新机制
- 是否说明与已有工作的区别
- 创新点是否有实验支持

### Methodology

评价：

- 方法设计是否合理
- 技术路线是否完整
- 方法是否真正对应研究问题
- 模块之间逻辑是否清晰
- 方法描述与贡献是否一致

### Dataset Quality

评价的不是“哪个数据集更高级”，而是数据使用是否充分：

- 数据量是否合理
- 是否具有代表性
- 是否使用公开 Benchmark
- 数据划分是否合理
- 是否存在多数据集 / 外部验证
- 是否存在明显数据泄漏风险

### Experimental Quality

评价：

- Baseline 是否充分
- 是否与代表性方法 / SOTA 比较
- 是否存在 Ablation Study
- 是否使用多个评价指标
- 是否存在泛化、鲁棒性或统计实验

### Conclusion Support

评价：

> 论文最终结论是否真正受到实验和证据支持。

例如，如果论文声称具有很强的跨数据集泛化能力，但 PaperProfile 中只提供单数据集实验，则 Conclusion Support 不应得到高评价。

---

## 5. Assessment Output

每个维度输出：

```json
{
  "score": 4,
  "level": "Strong",
  "evidence": [
    "..."
  ],
  "concerns": [
    "..."
  ]
}
```

最终得到：

```text
Novelty
Methodology
Dataset Quality
Experimental Quality
Conclusion Support

Overall Maturity

Strengths
Weaknesses
```

### Important

这些评分属于：

**Rubric-based decision-support signals**

它们不是：

- Acceptance Probability
- 投稿成功率
- 同行评审最终分数
- 经过真实录用数据校准的预测

---

## 6. Hybrid Retrieval

期刊数据分成两类。

### Semantic Fields

用于 BGE-M3 Embedding：

```text
name
research_fields
keywords
aims_scope
```

### Structured Fields

由 SQLite 负责：

```text
ccf_rank
jcr_quartile
cas_quartile
impact_factor
```

因此系统采用：

```text
Semantic Retrieval
+
Structured Filtering
```

而不是将所有期刊条件交给向量搜索处理。

---

## 7. Two-stage Retrieval

### Stage 1 — BGE-M3

BGE-M3 作为 Bi-Encoder，负责高 Recall 候选召回：

```text
Paper Query
→ Embedding
→ Chroma
→ Semantic Candidates
```

### Stage 2 — BGE Reranker

`BAAI/bge-reranker-v2-m3` 作为 Cross-Encoder：

```text
Paper Query + Journal Scope
→ Relevance Score
→ Reranking
```

因此：

```text
BGE-M3
=
Recall

BGE-Reranker
=
Precision
```

---

## 8. Grounded Recommendation

最终 DeepSeek 输入：

```text
PaperProfile
+
PaperQualityAssessment
+
RerankedCandidates
```

DeepSeek 只能：

- 比较候选
- 调整最终排序
- 生成推荐理由
- 给出 Topic / Method / Scope Fit
- 分析投稿风险

不能：

- 自行增加期刊
- 修改 CCF/JCR/CAS/IF
- 根据模型记忆补充不存在的期刊数据
- 生成录用概率

结构化期刊事实始终来自 SQLite。

核心原则：

```text
Retrieval determines candidates.
Quality Assessment evaluates paper maturity.
LLM reasons over retrieved evidence.
```

---

## 9. Journal Dataset

当前知识库包含约 291 本计算机领域期刊。

核心字段：

```text
name
ccf_rank
research_fields
keywords
aims_scope
jcr_quartile
cas_quartile
impact_factor
```

---

## 10. Pipeline Mode

完整推荐：

```bash
python main.py data/papers/test_paper.pdf
```

增加约束：

```bash
python main.py data/papers/test_paper.pdf \
  --ccf A B \
  --jcr Q1 Q2 \
  --min-if 5
```

完整流程：

```text
PDF
→ PaperProfile
→ PaperQualityAssessment
→ Hybrid Retrieval
→ Cross-Encoder Rerank
→ Grounded Recommendation
```

---

## 11. Agent Mode

启动：

```bash
python scripts/run_agent.py
```

示例：

```text
Analyze this paper and evaluate its quality.

Recommend journals for paper.pdf, only CCF A or B.

Find CCF B journals related to retrieval augmented generation.

What is the JCR quartile of XXX?
```

Agent 只负责：

```text
Intent Understanding
+
Tool Routing
```

底层推荐仍然由 Deterministic Pipeline 完成。

---

## 12. Evaluation

项目提供 Offline Evaluation Framework：

### Retrieval

- Hit@K
- Recall@K
- Precision@K
- MRR
- nDCG@K

### Structured Filtering

- Constraint Satisfaction Rate
- Filter Leakage

### Recommendation

- Candidate Containment
- Metadata Faithfulness
- Structured Output Validity

正式 Gold Dataset 仍在人工整理中，因此项目不会填写未经真实实验验证的 Recall、MRR 或 nDCG 指标。

---

## 13. Key Design Decisions

### Why not directly ask the LLM to score the paper?

Unconstrained numerical LLM scores are difficult to calibrate.

Therefore this project uses:

```text
Rubric
+
Discrete 1–5 Assessment
+
Evidence
+
Concerns
```

instead of arbitrary 0–100 scores.

### Why SQLite + Chroma?

SQLite handles exact structured facts.

Chroma handles semantic similarity.

### Why BGE-M3 + Reranker?

Bi-Encoder provides efficient high-recall retrieval.

Cross-Encoder provides more precise candidate ranking.

### Why does DeepSeek not search journals directly?

Because the model may hallucinate or use outdated journal metadata.

The LLM only reasons over retrieved candidates.

### Why keep Pipeline Mode and Agent Mode separately?

Pipeline Mode provides:

- Stability
- Reproducibility
- Evaluation

Agent Mode provides:

- Natural-language interaction
- Intent routing
- Tool Calling

---

## 14. Limitations

1. 推荐范围受本地期刊数据库覆盖范围限制。

2. CCF/JCR/CAS/Impact Factor 依赖本地数据更新时间。

3. Paper Quality Assessment 基于 PaperProfile 中可观察证据，信息不足时可能无法完整评价。

4. 1～5 Rubric Score 不是同行评审分数，也不是录用概率。

5. 当前不会使用论文数据库自动验证“创新点是否全球首次提出”。

6. 扫描版纯图片 PDF 暂不支持 OCR。

7. LLM Recommendation 属于 Decision Support，不等同于正式同行评审意见。

8. 正式 Evaluation Gold Dataset 仍需人工维护。

---

## 15. Future Work

- 完成人工 Gold Evaluation Dataset
- Retrieval / Reranker Ablation Study
- Paper Quality Assessment Rubric Calibration
- 引入相关工作检索辅助 Novelty Verification
- 自动更新期刊 Scope 与分区数据
- Failure Analysis
- Journal Data Versioning
- Web UI
