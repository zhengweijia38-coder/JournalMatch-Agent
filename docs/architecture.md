# Architecture

The system keeps deterministic retrieval and recommendation separate from natural-language Agent routing.

```text
User PDF
  → Paper Layer: PyPDFLoader + DeepSeek PaperProfile extraction
  → Assessment Layer: five evidence-based 1-5 rubric dimensions
  → Retrieval Layer: BGE-M3 query embedding + Chroma semantic recall
  → Journal Data Layer: SQLite source-of-truth + exact structured filters
  → Reranking Layer: BGE-Reranker-v2-m3 cross-encoder
  → Recommendation Layer: DeepSeek reasoning over verified candidates
  → PipelineResult / CLI / JSON

Natural-language user
  → Agent Layer: LangChain create_agent + DeepSeek Tool Calling
  → Four read-only high-level Tools
  → Existing Paper, Retrieval, Journal Data, and Pipeline layers
```

## Paper Layer

Input: a user-specified text-based PDF path.

Output: `PaperProfile`, including title, research fields, keywords, problem, methods, contributions, results, limitations, and summary. Scanned image-only PDFs require OCR outside the current project.

## Assessment Layer

Input: the existing `PaperProfile`; this layer never re-reads the PDF.

Output: `PaperQualityAssessment`, containing novelty, methodology, dataset quality,
experimental quality, conclusion support, overall maturity, strengths, and weaknesses.
Every dimension has a 1-5 score, its matching qualitative level, profile-grounded
evidence, and concerns. Insufficient source information remains explicit instead of
being filled from model memory. The result is a recommendation aid, not an acceptance
probability or calibrated peer-review score.

After structured generation, deterministic grounding validation neutralizes dataset
status claims not present in the profile and conservatively caps Dataset Quality when
source, split, preprocessing, representativeness, and external-validation evidence are
all absent. The canonical object is validated again before entering the pipeline.

## Journal Data Layer

SQLite is the source of truth for journal identity, CCF rank, JCR quartile, CAS quartile, impact factor, research fields, keywords, and aims/scope. Imports are validated through the Pydantic `Journal` model.

Chroma is not the source of truth. It stores the semantic index generated from stable journal IDs and selected semantic text fields.

## Retrieval Layer

BGE-M3 embeds paper/topic queries and journal semantic text. Chroma provides candidate recall. Hybrid retrieval then reads current SQLite rows and applies strict CCF/JCR/CAS/impact-factor filters. Filters are never silently relaxed.

## Reranking Layer

BGE-Reranker-v2-m3 jointly scores query/journal-text pairs for the small retrieved candidate set. Its raw score is a relative ranking signal, not a probability.

## Recommendation Layer

DeepSeek evaluates the paper using both `PaperProfile` and its rubric-based
`PaperQualityAssessment`, then reasons only over verified reranked candidates. Quality
is an auxiliary signal and cannot override topical fit or introduce a new journal.
Journal metadata displayed to users is mapped back to SQLite-backed candidate objects.
The LLM cannot add an out-of-candidate journal.

## Evaluation Layer

Phase 7 provides Hit@K, Precision@K, Recall@K, MRR, nDCG@K, filtering checks, reranker before/after comparisons, and recommendation grounding checks. Reported benchmark metrics require a manually curated Gold Dataset; example cases are smoke data only.

The evaluation data-preparation path freezes each PDF as a Phase 1 `PaperProfile`, then joins it
with human-authored graded relevance labels through stable SQLite journal IDs. The builder performs
validation and deterministic serialization only: it does not run retrieval, reranking,
recommendation, Agent routing, quality assessment, or automatic Gold generation. The resulting
JSONL remains readable by the existing Phase 7 evaluators; their query is derived deterministically
from the stored profile.

## Agent Layer

The Agent is an intent router and Tool orchestrator. It selects one of four read-only capabilities: paper analysis, journal search, journal detail lookup, or complete recommendation. It does not control embeddings, Chroma internals, SQLite mutations, reranker batches, or recommendation validation.

The deterministic pipeline remains independently callable through `main.py`. Agent Mode does not replace it.
