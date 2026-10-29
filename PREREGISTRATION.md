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

1. **L1:** Qwen/Qwen3-Embedding-0.6B (revision `97b0c614…`, fp16 on GPU), products embedded
   without a prompt and cut to 512 tokens; queries embedded with the instruction
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
   - CLM-v0.1-8B via `contrastive-lm` `Engine.rank(query, candidates, instructions)`, with
     the same `esci_v1` question as `instructions` (added before CLM's first run), one call per
     query, Qwen3-8B served by vLLM (pooling, max length 2,048).

Amendment (2026-10-03, still before any model run on ESCI): bge-reranker-v2-m3 was dropped
as a reference cross-encoder to keep the comparison to the four models of interest.

## Two settings (co-primary)

- **Judged:** each query's ESCI-judged products (median 16), ordered by the same embedding
  similarity. Every candidate is labelled, so it is the clean head-to-head. Headline numbers
  come from this setting.
- **Retrieved:** L1 top-100 from the full catalog. The end-to-end view, but most retrieved
  products were never judged.

## Metrics and decision rule

- **Co-primary:** P@10, strict (relevant = E; unjudged counts as not relevant), on test, in
  both the judged and the retrieved setting.
- **Decision rule:** in each setting separately, paired bootstrap over queries (10,000
  resamples) of each model's P@10 minus L1 order's, Holm-corrected across all non-baseline
  models in that setting. A model **beats L1** only if it does so in **both** settings
  (adjusted p < 0.05 and positive mean difference in each). Requiring both is an
  intersection-union test, so no further correction across settings is needed.
- **Key secondary:** P@5 under the same rule.
- **Other secondary:** P@10/P@5 with E+S relevant; judged-only P@10/P@5 (unjudged removed
  before cutting at k); judged@10; NDCG@10 with official gains; L1 Recall@100 of E;
  per-query latency p50/p95 at fixed hardware and concurrency; USD per 1,000 queries from
  billed tokens.
- **Model-vs-model:** a difference between two rerankers is reported as robust only if it
  has the same sign in both settings.

### Amendment 2 (2026-10-03, after the L1 dry run, before any reranker or baseline output)

Originally the retrieved setting was the sole primary and the judged setting a key
secondary. The L1 dry run on test showed judged@10 = 0.248: three of every four products in
the retrieved top 10 have no ESCI label, and strict P@10 counts them as not relevant. Such a
large unlabelled share could swamp real differences between rerankers, so the judged setting
was made co-primary and a win over L1 now has to hold in both settings. Seen at the time of
this decision: only L1 diagnostics (ANN overlap@100 0.974; recall of E@100 0.498; queries
with an E in the top 100 0.806; judged@10 0.248). No reranker, baseline or label-dependent
ranking output existed.

### Amendment 3 (2026-10-04, after all test results except Clef were seen)

Added an exploratory head-to-head section to every report: a paired bootstrap of P@10 and P@5
between each pair of rerankers (Holm across the comparisons in that report, 95% percentile
CIs). It was added because the pre-registered rule only tests each model against L1 order,
and the results show a consistent Jev > Qwen3 ordering whose significance the paper needs.
These tests were not pre-registered and are reported as exploratory; no pre-registered
result changes.

### Amendment 4 (2026-10-04, after all other test results were seen; Clef never run on test)

Clef Flash is removed from the study. Cloudflare's free tier rate-limited every attempt, so no
test run was completed; its code path stays in the repository but no Clef result is reported.
No other model, setting, metric or rule changes.

### Amendment 5 (2026-10-29, after all test results were seen; in response to peer review)

Written before any of the runs below were started. Everything here is exploratory: the main
result tables, decision rules and frozen lists are unchanged, and no added analysis can change a
pre-registered result. The templates and model configurations named here are committed with this
amendment (`configs/templates/esci_v{1p,2,3,4,5}.yaml`, `configs/models/{qwen3_reranker_4b,qwen3_reranker_8b,prompted_qwen3_8b,bge_reranker_v2_m3}.yaml`).

1. **Disclosure of an interface asymmetry in the main runs.** Jev received the `esci_v1` question plus its
   true/false meanings; Qwen3-Reranker received the one-line `instruction` field of the same template (a shorter
   paraphrase of the question, without the meanings). This was true from the first run and is recorded in the
   template hash. Template `esci_v1p` gives Qwen3-Reranker exactly the text Jev reads.
2. **Prompt sensitivity (validation queries only; never test).** Templates `esci_v1p`, `esci_v2` (paraphrase),
   `esci_v3` (terse, question only, for every model), `esci_v4` (broad relevance, a deliberate wrong-criterion
   control) and a graded formulation `esci_v1p` + `esci_v5` whose score is P(exact) + P(at least an acceptable
   substitute), i.e. the expected grade on a 0-2 scale (no weights, nothing tuned). Run for Qwen3-Reranker-0.6B and
   Jev on ESCI judged validation (500 queries) and WANDS validation (80 queries). Reported: P@10 per model and the
   paired Jev minus Qwen3-Reranker difference with a 95% CI. No variant replaces `esci_v1` in the main results.
3. **Added reference rerankers (test, exploratory).** Qwen3-Reranker-4B, Qwen3-Reranker-8B, Qwen3-8B prompted
   pointwise with the Qwen3-Reranker yes/no format but no reranker training, and bge-reranker-v2-m3 (dropped in
   Amendment 1 before any run; reinstated here after results were seen, so it is not a pre-run selection). Each
   receives the `esci_v1` instruction (bge has no instruction channel). Reported: P@10, P@5, NDCG@10 and Jev minus
   model P@10 (paired bootstrap, Holm across the added models within each setting). Model revisions are resolved
   to the loaded commit hash and written to each run manifest.
4. **Score fusion with L1 (offline, exploratory).** z-score fusion `alpha * z(L2) + (1 - alpha) * z(L1 similarity)`
   (z within each query's candidates), alpha from {0, 0.1, ..., 1} chosen to maximise P@10 on the validation queries
   of the same setting (ties go to the larger alpha) and then applied unchanged to test; and reciprocal rank fusion
   with k = 60 (Cormack et al., 2009), nothing tuned. Needs validation and test runs of the same model.
5. **Interface controls for CLM (validation, ESCI judged).** Candidate-order shuffles (seeds 1, 2) of the list
   interface used on test, and CLM through a pointwise yes/no interface (state = query + product, options
   {"yes", "no"}, score = P(yes)), to check whether CLM's failure depends on scoring the whole list at once.
6. **Metrics and label analyses.** NDCG@10 (already listed above as a secondary metric) is now reported for the
   test sets, with paired bootstrap differences. Two additional exploratory descriptions: the label mix of each
   model's top 10 (Exact / Substitute / Complement / Irrelevant / unlabelled), and the ESCI results restricted to
   queries with at least one Exact label, as a like-for-like check against the WANDS cohort definition.
7. **Cost sensitivity.** Analytic only (`scripts/revision_cost.py`): one assumption varied at a time from the
   reported latencies and billed tokens. No new runs.

## Baselines

L1 order (embedding similarity), random order within the list, and an oracle that orders by
ESCI gain (ceiling; uses labels). L1 order and oracle are not in the Holm family.

## Later tracks (not part of the primary result)

Fusion of rerankers with L1 order (weights from validation), alternative questions for the
decision models (for example a four-way E/S/C/I `choice` question), and fine-tuning
Qwen3-Reranker on ESCI training queries (excluding the 500 validation queries). Each will
be recorded here before its test run.

## Frozen lists (2026-10-03, A40, commit 341dd44 code)

| File | sha256 |
|---|---|
| `pools_test_retrieved.parquet` | `aba1d03fff960f02dc071a929738ac224f169eb760c3b4efa67fb214abdb65b6` |
| `pools_test_judged.parquet` | `78e53c70bff8e07cdeade730784d8153feeb23fd0c456937cd44357e3fdc041a` |
| `eval_users_test_{retrieved,judged}.parquet` (same 1,000 queries) | `45f275f196d912e6855c8ba861553b9b72fb7adf4b3a234365210312d2023a03` |
| `pools_valid_retrieved.parquet` | `e920da0e6884592a90656168ec5853567356f95f99b3dbaf8014cb36fcc0a91b` |
| `pools_valid_judged.parquet` | `34f15714a5d9b889f846834baa33175cddfe846b32202b35c4079878d104b10c` |
| `eval_users_valid_{retrieved,judged}.parquet` (same 500 queries) | `5f373d89308861e2bcc915f252a6cbd9dc10ca179821d3df49d44637fa79f651` |

Test L1 at freeze: ANN overlap@100 0.974 (gate passed); recall of E@100 0.498; queries with an E
in the top 100 0.806; judged@10 0.248. Identical to the dry run.
