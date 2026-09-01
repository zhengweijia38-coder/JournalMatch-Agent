# Human Recommendation Rubric

This rubric is for human evaluation of Phase 6 output. It is not automatically scored by DeepSeek, and it must remain separate from the automatic hard checks.

## Topic Fit Explanation (1–5)

- **1:** Topic claim is unsupported or contradicts the paper/candidate data.
- **2:** Mentions a broad field but misses the paper's central topic.
- **3:** Identifies the main topic and one relevant journal field or keyword.
- **4:** Explains multiple specific topic overlaps and any important mismatch.
- **5:** Gives a precise, balanced explanation grounded in both the paper and journal data.

## Scope Grounding (1–5)

- **1:** Does not use the supplied aims/scope or invents scope information.
- **2:** Makes only a generic scope assertion.
- **3:** Connects one paper contribution to a supplied scope element.
- **4:** Uses specific aims/scope evidence and acknowledges boundary conditions.
- **5:** Provides a precise, well-balanced scope analysis with no unsupported claims.

## Method Fit Explanation (1–5)

- **1:** Method claim is absent, incorrect, or invented.
- **2:** Names a method but does not connect it to the journal.
- **3:** Correctly connects one paper method to a journal keyword or scope element.
- **4:** Explains several relevant methods and notes methodological mismatches.
- **5:** Gives detailed, evidence-grounded method alignment without overclaiming.

## Evidence Use (1–5)

- **1:** Reasoning is unsupported or relies on model memory.
- **2:** Uses vague statements with little traceable evidence.
- **3:** Uses at least one concrete PaperProfile fact and one candidate fact.
- **4:** Consistently grounds major claims in supplied contributions, results, limitations, or scope.
- **5:** Every material claim is traceable, relevant, and balanced by limitations where appropriate.

## Hallucination (0/1)

- **0:** No candidate-external journal, fabricated metadata, probability, review-time claim, or unsupported factual claim is found.
- **1:** At least one such hallucination or unsupported factual claim is present.

Reviewers should record brief evidence for each score. Inter-reviewer disagreements should be discussed rather than averaged without inspection. A future LLM-as-a-Judge, if added, must be clearly labeled as auxiliary and must not replace this human rubric.
