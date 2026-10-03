# Pre-registration: L2 rerankers for product search (Amazon ESCI)

Written 2026-10-03, before any model was run on ESCI. Only data preparation had been run
(query sampling, label counts, catalog statistics); no retrieval or reranker output existed.

## Why this study

The first study (Amazon Reviews'23, repo [rec-l2-bench](https://github.com/bangadpurva/rec-l2-bench)) asked text rerankers to predict
future purchases. Two-thirds of in-pool purchases came from popularity, and every text
reranker lost to the behavioural L1 order. That task is collaborative-filtering dominated,
so it does not test what these models are built for. ESCI is query-to-product relevance
judged by people: a content task.

## Question

On the same frozen candidate lists, how do Qwen3-Reranker, Jev, Clef Flash and CLM-v0.1-8B
compare as L2 rerankers for product search, on precision, latency and cost?

## Data

- Amazon Shopping Queries Dataset (ESCI), official files (sha256-checked), **US locale,
  `small_version == 1`** (the Task 1 ranking subset).
- **Test:** 1,000 queries sampled from the test split (seed 20261003).
- **Validation:** 500 queries sampled from the train split, same seed. Used only for smoke
  tests and choosing fusion weights; **never** used as fine-tuning data.
- Labels: E (exact), S (substitute), C (complement), I (irrelevant). Gains for NDCG are the
  official Task 1 values: E 1.0, S 0.1, C 0.01, I 0.
- Product text: title, brand, colour, bullet points, description; HTML stripped; title and
  brand first so truncation never drops them.

## Pipeline (fixed)

1. **L1:** Qwen/Qwen3-Embedding-0.6B (revision `97b0c614…`), products embedded without a
   prompt and cut to 512 tokens; queries embedded with the instruction
   "Given a product search query, retrieve products that match what the shopper asks for".
   FAISS HNSW (M 32, efSearch 512) over all 1,215,854 US products, top-100.
   **Gate:** ANN overlap with exact search at 100 must be at least 95% on validation.
2. **Freeze:** lists and query cohorts are written once, hashed, and verified on every load.
3. **L2:** every model reranks the same lists with the same text: product cut to 256
   tokens, query to 64, one tokenizer (Qwen3-Embedding's). Template `esci_v1` (its hash is
   in every run manifest):
   - Qwen3-Reranker-0.6B (revision `e61197ed…`), instruction from `esci_v1`.
   - Jev `jev-1.13.0` and Clef Flash `@cf/cloudflare/clef-flash`: per-pair, one yes/no
     (`noul`) question, "Does this product exactly match what the shopper's search query
     asks for?", state headed "Search query" / "Candidate product"; backoff on 429/529 and
     network errors.
   - CLM-v0.1-8B via `contrastive-lm` `Engine.rank(query, candidates)`, one call per
     query, Qwen3-8B served by vLLM (pooling, max length 2,048).

Amendment (2026-10-03, still before any model run on ESCI): bge-reranker-v2-m3 was dropped
as a reference cross-encoder to keep the comparison to the four models of interest.

## Two settings

- **Retrieved (primary):** L1 top-100 from the full catalog. Realistic, but most retrieved
  products were never judged.
- **Judged (key secondary):** each query's ESCI-judged products (median 16), ordered by the
  same embedding similarity. Every candidate is labelled, so it is a clean head-to-head.

## Metrics and decision rule

- **Primary:** P@10, strict (relevant = E; unjudged counts as not relevant), retrieved
  setting, test.
- **Decision rule:** paired bootstrap over queries (10,000 resamples) of each model's P@10
  minus L1 order's; Holm correction across all non-baseline models in the table. A model
  beats L1 if its adjusted p < 0.05 and the mean difference is positive.
- **Key secondary:** P@5 (same rule), and P@10 in the judged setting (same rule, own Holm
  family).
- **Other secondary:** P@10/P@5 with E+S relevant; judged-only P@10/P@5 (unjudged removed
  before cutting at k); judged@10; NDCG@10 with official gains; L1 Recall@100 of E;
  per-query latency p50/p95 at fixed hardware and concurrency; USD per 1,000 queries from
  billed tokens.
- **Robustness rule:** with incomplete labels, a retrieved-setting difference between two
  models is called robust only if the judged-setting difference has the same sign.

## Baselines

L1 order (embedding similarity), random order within the list, and an oracle that orders by
ESCI gain (ceiling; uses labels). L1 order and oracle are not in the Holm family.

## Later tracks (not part of the primary result)

Fusion of rerankers with L1 order (weights from validation), alternative questions for the
decision models (for example a four-way E/S/C/I `choice` question), and fine-tuning
Qwen3-Reranker on ESCI training queries (excluding the 500 validation queries). Each will
be recorded here before its test run.
