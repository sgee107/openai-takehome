# Vector Search Test & Metrics Plan

This document outlines how to evaluate and monitor retrieval quality, speed, and stability once your 300-item fashion dataset is indexed with text-embedding-3-small (1536-d).

## Goals
- Establish a repeatable offline evaluation you can run after re-indexing.
- Track a small set of health metrics over time (quality, latency, stability).
- Compare variants (normalization, distance, index params, chunking) safely.

## Dataset & Assumptions
- Corpus: 300 Amazon Fashion items in `data/amazon_fashion_sample.json`.
- Embeddings: text-embedding-3-small (1536-d). Recommend L2-normalized vectors at write and query time.
- Distance: cosine similarity (with normalization, equivalent to inner product ranking).
- Index: For 300 items, exact search is fast; ANN (pgvector ivfflat/HNSW) is optional but you can still measure ANN recall.

## Query Set (Seed)
Create a small, realistic query set you can reuse across runs. Start with 50–100 queries:
- Category: "summer dress", "hoodie", "ankle boots", "slim-fit jeans".
- Attributes: "red midi dress", "black leather jacket", "waterproof hiking boots", "v-neck t-shirt".
- Audience: "girls floral dress", "men's running shorts", "women's yoga leggings".
- Brands (if present): "Levi's jeans", "adidas running shoes".
- Use actual tokens found in titles, bullets, or categories in the JSON to reduce ambiguity.

Store this as a static list you check into the repo (e.g., `docs/queries_fashion.txt`).

## Ground-Truth Labels (Heuristic)
With limited data, build pragmatic labels from product metadata:
- Positive relevance (grade 3): exact category match AND ≥2 attribute matches (e.g., color + material + type).
- Medium relevance (grade 2): category match OR 2 attribute matches.
- Weak relevance (grade 1): 1 attribute match or brand-only match.
- Non-relevant (grade 0): otherwise.

Implementation tips (heuristic, not code):
- Tokenize query into normalized tokens; expand obvious synonyms ("sneakers" ↔ "running shoes", "hoodie" ↔ "hooded sweatshirt").
- From each product, extract candidate fields: `title`, `brand`, `category`, `color`, `material`, `bullets/description`.
- Count matches; assign graded labels 0–3. Save labels as `docs/labels.jsonl` with schema `{query_id, item_id, grade}` for reproducibility.
- Optionally add a handful of human-judged labels to spot-check the heuristic.

## Core Metrics
Quality (ranked retrieval):
- NDCG@K: quality with graded labels (recommended primary, K=10).
- Recall@K: coverage of relevant items (use grade ≥2 as relevant).
- Precision@K / HitRate@K: sanity checks for top-K quality.
- MRR@K: usefulness when a single best item exists.

Stability:
- Score distribution: mean and std of top-1/top-10 similarities per query.
- Norm distribution: mean and std of stored vector norms (should be ~1.0 if normalized).

ANN correctness (if using ANN):
- ANN Recall@K: overlap between ANN top-K and exact brute-force top-K on the same normalized vectors.

Latency:
- P50/P95 end-to-end query latency; index-only latency if measurable.

## Targets (Initial, Adjustable)
- NDCG@10 ≥ 0.85 on the labeled set (heuristic labels tend to be noisy; adjust after inspection).
- Recall@10 ≥ 0.95 vs exact baseline (if using ANN).
- P95 latency ≤ 150–250 ms for index-only at this scale (practically much faster for 300 docs).
- Stable similarity and norm distributions within ±5% week-over-week.

## Evaluation Procedure
1) Snapshot versions
- Record `dataset_version`, `model_version`, `index_params`, and `chunking_config` for traceability.

2) Build query embeddings
- Embed the fixed query set; L2-normalize.

3) Run retrieval
- For each query, run top-K search (K=10 or 20) with your chosen filter set (e.g., collection).
- Log `scores`, `ids`, and latency.

4) Compute exact baseline (gold ranking)
- For this 300-item corpus, compute exact cosine on normalized vectors in memory to produce gold top-K.
- If using ANN, compute ANN Recall@K as `|ANN∩Exact| / K`.

5) Join with labels
- For each query-result, look up graded label (0–3). Default to 0 if unlabeled.

6) Compute metrics
- NDCG@K, Recall@K, Precision@K, MRR@K per query; aggregate mean and 95% bootstrap CI.

7) Analyze segments
- Segment by query type (category/brand/attribute) and by product category to catch regressions.

8) Report & archive
- Save a run artifact: `reports/eval_{date}_{model}_{index}.json` containing inputs, metrics, and sample failures.

## Variants to Compare
- Normalization: normalized+cosine (baseline) vs unnormalized+cosine (control). Expect normalized to be more stable.
- Distance op: with normalization, cosine vs inner product should produce identical rankings; verify equivalence.
- Index params: ivfflat `lists`/`probes` or HNSW `M`/`ef_search`; chart Recall@K vs p95 latency.
- Chunking: whole-title vs title+bullets; 300–800 token chunks; overlap vs none.
- Filters: category/brand filters and their effect on metrics.

## Practical Tips
- With 300 items, exact search is cheap. Use it as your gold baseline even if you keep ANN for parity with production.
- Keep the query set stable; add new queries only with a new query-set version.
- Store model name and embedding dimension in a registry; never mutate existing embeddings when switching models—add new rows.
- If scores look compressed or drift over time, verify normalization, tokenization, and chunking haven’t changed silently.

## Reporting Template
Capture the following per run:
- Meta: `dataset_version`, `model_version`, `index_params`, `chunking_config`, `query_set_version`.
- Quality: mean NDCG@10, Recall@10, Precision@10, MRR@10 (+ 95% CI).
- ANN: ANN Recall@10 vs exact (if applicable).
- Latency: P50/P95/P99 for index-only and end-to-end.
- Stability: mean/std of top-1 and top-10 similarity; vector norm mean/std.
- Examples: Top 5 success examples and 5 failure cases with queries and items.

## Metric Definitions (Reference)
- Cosine similarity: `cos(q, d) = (q · d) / (||q|| · ||d||)`. With L2-normalized vectors, this is simply dot product.
- DCG@K: `sum_{i=1..K} (2^{rel_i} - 1) / log2(i + 1)` where `rel_i` is graded relevance (0–3).
- NDCG@K: `DCG@K / IDCG@K` where IDCG is DCG of ideal ranking.
- Recall@K: `(# relevant retrieved up to K) / (total relevant)`.
- Precision@K: `(# relevant retrieved up to K) / K`.
- MRR@K: `1 / rank_of_first_relevant` if ≤ K, else 0.
- ANN Recall@K: `|ANN_topK ∩ Exact_topK| / K`.

## Exact Baseline (Conceptual Pseudocode)
- Build a matrix `D` of shape `[N, 1536]` with normalized doc vectors.
- For each normalized query vector `q`: compute `scores = D @ q` (dot product) → cosine scores.
- Sort scores descending; take top-K as the gold ranking for metrics and ANN recall.

## Next Steps
- Finalize a query set and label file (heuristic + spot-checked) and check them into `docs/`.
- Add a lightweight eval runner that outputs the reporting template above and archives JSON artifacts per run.

---

Appendix A: Query Ideas for This Dataset
- "women's red summer dress", "black ankle boots leather", "men's slim-fit jeans", "girls floral skirt", "cotton crewneck t-shirt", "hooded sweatshirt fleece", "waterproof hiking boots", "running shoes breathable mesh", "down puffer jacket", "linen button-up shirt".

Appendix B: Troubleshooting
- Low ANN recall: increase `probes` (ivfflat) or `ef_search` (HNSW); ensure vectors are normalized for cosine/IP equivalence.
- Score drift: confirm same model/version and identical chunking; check for accidental double-normalization or missing normalization.
- Poor NDCG@10 but good Recall@10: consider re-ranking top-K with a cross-encoder or add attribute filters.
