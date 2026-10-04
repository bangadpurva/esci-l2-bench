# Pre-registration: L2 rerankers for product search on WANDS (second dataset)

Written 2026-10-03, before any retrieval or reranker run on WANDS. Seen at the time: the raw
files and the output of `wands_prepare.py` (query and label counts below). Seen from ESCI
(`PREREGISTRATION.md`): every ESCI result to date, which is why this study exists.

## Why a second dataset

On ESCI only about 25% of the L1 top 10 carried a label, so the realistic (retrieved) setting
counted many correct but unlabelled products as wrong, and the clean comparison needed an
artificial judged-only list. WANDS labels about 220 products per query (median) from a 43K
catalog, so the retrieved top 100 should be mostly labelled. The question is whether the
ESCI ordering of rerankers holds when labels are close to complete.

## Data

- Wayfair WANDS (github.com/wayfair/WANDS), files sha256-checked by `scripts/wands_download.sh`:
  42,994 products, 480 queries, 233,448 labels (Exact / Partial / Irrelevant).
- Labels map to Exact -> E, Partial -> S, Irrelevant -> I. 1,467 duplicated (query, product)
  pairs are resolved by majority vote, ties toward the less relevant label (14 conflict).
- **Queries:** only the 379 with at least one Exact label; for the other 101, strict P@k is 0
  for every ranking and carries no signal. **Validation:** 80 of them (seed 20261003), used
  only for smoke tests. **Test:** the remaining 299.
- Product text: name, class, category path, description, then features; whitespace normalised.

## Pipeline (identical to ESCI unless noted)

1. **L1:** Qwen3-Embedding-0.6B (revision `97b0c614…`, fp16 on GPU), products cut to 512
   tokens, queries with the ESCI search instruction. **Exact** cosine search over all 42,994
   products (no ANN, so no gate), top 100. Frozen and hashed as for ESCI.
2. **L2:** the same four models, configurations and template `esci_v1` as ESCI:
   Qwen3-Reranker-0.6B, Jev `jev-1.13.0` (per pair), Clef Flash (per pair), CLM-v0.1-8B
   (one call per query, the same question as instructions). Product text cut to 256 tokens,
   queries to 64.

## Setting

**Retrieved only** (L1 top 100). The judged-only list used on ESCI is not scored here: WANDS
labels are dense enough that the retrieved list is itself mostly labelled, and judged lists
would be long (median about 220, up to several thousand products per query).

## Metrics and decision rule

- **Primary:** P@10, strict (relevant = Exact; unjudged = not relevant), test.
- **Decision rule:** paired bootstrap over queries (10,000 resamples) of each model's P@10
  minus L1 order's, Holm-corrected across all non-baseline models. A model beats L1 if its
  adjusted p < 0.05 and the mean difference is positive.
- **Robustness:** a result counts only if judged-only P@10 (unjudged removed before cutting
  at 10) has the same sign. This replaces ESCI's judged setting.
- **Key secondary:** P@5 under the same rule.
- **Other secondary:** P@10/P@5 with Exact + Partial relevant; judged@10; NDCG@10 with gains
  Exact 1, Partial 0.5, Irrelevant 0 (equivalent to the common 2/1/0 scale); L1 recall of
  Exact at 100; latency p50/p95; USD per 1,000 queries.
- **Cross-dataset claim:** a statement such as "model A ranks above model B on both
  datasets" requires the difference to have the same sign on ESCI (both settings) and WANDS.

### Amendment 3 (2026-10-04, after all test results except Clef were seen)

Added an exploratory head-to-head section to every report: a paired bootstrap of P@10 and P@5
between each pair of rerankers (Holm across the comparisons in that report, 95% percentile
CIs). It was added because the pre-registered rule only tests each model against L1 order,
and the results show a consistent Jev > Qwen3 ordering whose significance the paper needs.
These tests were not pre-registered and are reported as exploratory; no pre-registered
result changes.

## Baselines

L1 order, random order within the list, and the label oracle (ceiling), as for ESCI.

## Prepare output (seen before writing this)

Test: 299 queries, 158,229 labels, median 221 labels and 28 Exact per query; label shares
Partial 0.60, Irrelevant 0.27, Exact 0.13. Validation: 80 queries, 42,148 labels. All test
labels refer to products in the catalog.

## Frozen lists (2026-10-03, A40, commit 209c214 code)

| File | sha256 |
|---|---|
| `pools_test_retrieved.parquet` | `b87fb97c9fed154bdf0b0786ac00f73693221b9add9a233ddcf1aa3e44af63e6` |
| `eval_users_test_{retrieved,judged}.parquet` (same 299 queries) | `fba350c18cb141cf077bda43e14b15137feffaedb04defe24e910b39f8c91d54` |
| `pools_test_judged.parquet` (not scored) | `0860664f2bd27043e8dd06145f19c85b1db4afad08fe9e28714be49cbe419b3c` |
| `pools_valid_retrieved.parquet` | `dbca184a39b0167ac5de64a95d0545a6e8e66c9540f4eb85c3da275abb8c6a94` |
| `eval_users_valid_{retrieved,judged}.parquet` (same 80 queries) | `40f9c56a227ad8b5a0e13fb74b854b420456c34efd025b27bd53825939b31c09` |
| `pools_valid_judged.parquet` (not scored) | `bcbd65b77440b37704711fd6a30d11d2b27b3f6ffab81bb0834979dad7625e49` |

Test L1 at freeze (identical to the dry run): judged@10 0.870, judged@100 0.708; recall of Exact
at 100 0.624; queries with an Exact in the top 100 0.940. On ESCI the same figures were 0.248,
0.075, 0.498 and 0.806, which is the label-completeness gain this study was designed for.
