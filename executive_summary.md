## HP-RAG Overview
Hyperlink-Positioned Retrieval-Augmented Generation (HP-RAG) augments contract question answering with a two-step pipeline: an LLM first filters a table-of-contents-aware SQLite store to select the most relevant sections, then a downstream answer LLM synthesizes a response from those curated passages. This architecture keeps retrieval grounded in document structure while preserving full-fidelity generation from the selected snippets. Unlike cosine-based vector retrieval, HP-RAG maintains precision even when questions are long or combine multiple tariff scenarios—conditions that typically flatten vector similarity scores.

## Task & Dataset
The task focuses on extracting tariff, duty, and price-adjustment obligations from long-form procurement and distribution agreements stored under <a href="data/contracts">data/contracts</a>. Each contract is a GPT-5 Pro-generated markdown document that mirrors realistic corporate legal language, complete with Incoterms, tariff-adjustment clauses, and schedules listing HS codes. The evaluation questions live in <a href="data/contracts/questions.jsonl">questions.jsonl</a>. Example prompts include:
1. “Compute our net exposure if the US raises Section 301 tariffs on HS 8538.90 from 25% to 35% effective next month… Who pays what and can we cancel?”
2. “If the US reduces tariffs on HS 8542.31 from 25% to 10% for Chinese origin, what price change applies to AR-IC-M4 for undelivered quantities with CIF $30/unit?”
3. “Identify all HS codes and baseline US duty rates from the Agreement for analytics.”
4. “EU imposes a retaliatory tariff that raises the composite duty on HS 9026.20 by +9% for three consecutive months. What remedies are available and who bears the duty?”
5. “India imposes an anti-dumping duty of 15% on HS 7007.19, raising composite duty from the 10% Baseline Composite Duty to 25%. For an undelivered order of 20,000 m² of GLT-SG-3.2 at CIF $8.10/m², compute Buyer’s incremental duty and whether repricing/cancellation rights apply.”

##Evaluation Approach
We rebuilt both HP-RAG and a FAISS vector baseline on the contract corpus and evaluated 40 tariff-focused questions via `scripts/run_evals.py`. Each prompt includes a GPT-5-generated ground-truth answer in `questions.jsonl`, and both retrievers are scored directly against that reference. Answer quality was assessed with three complementary similarity measures that appear in the results table:
- **answer_token_f1** – a token-overlap F1 score that rewards exact lexical agreement between model output and the gold answer, highlighting precision on numeric and legal phrases.
- **answer_embedding_similarity** – cosine similarity of OpenAI embeddings, capturing semantic alignment even when wording differs (useful for paraphrased duties or rights).
- **answer_llm_correctness** – an LLM-judged match score that qualitatively checks whether the generated narrative stays faithful to the reference answer.
Retrieval fidelity is measured with **context_precision**, **context_recall**, and **context_f1**, which compare retrieved passages against the ground-truth citations supplied in `reference_contexts`. We also log **avg_contexts**, **avg_tokens**, and **avg_latency_ms** to compare operational efficiency across retrievers.

## Results
The current question set skews toward relatively short prompts; richer, multi-part scenarios are still needed to fully demonstrate HP-RAG’s advantages over cosine-only retrieval.

| Metric | vector | hyperlink |
| --- | --- | --- |
| avg_contexts | 3.000 | 3.000 |
| avg_tokens | 225.800 | 160.800 |
| avg_latency_ms | 337.127 | 6340.677 |
| answer_token_f1 | 0.267 | 0.239 |
| answer_embedding_similarity | 0.821 | 0.774 |
| answer_llm_correctness | 0.467 | 0.390 |
| context_precision | 0.258 | 0.167 |
| context_recall | 0.490 | 0.331 |
| context_f1 | 0.317 | 0.210 |

### Potential Improvements
Optimizing HP-RAG may involve compressing TOC entries to strip repetitive boilerplate, lowering token usage during selection. Additional gains could come from adaptive `toc_limit` values, lightweight summarization of long chunks, selector-result caching, or tiered LLM usage (small model for TOC filtering, larger model for final answers). In domain-specific agentic workflows, HP-RAG can anchor iterative query loops while a deterministic calculator tool handles duty math, keeping the LLM focused on interpretation and policy reasoning.
