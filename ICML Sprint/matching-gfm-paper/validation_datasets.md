# Validation Datasets for Matching GFM

This note collects datasets and official data sources we can use to validate the
paper and experimental implementation. The emphasis is on temporal bipartite or
near-bipartite interaction data that can support:

- preference recovery from temporal interaction histories
- ranking and temporal link prediction
- matching under inferred buyer-side and seller-side utilities
- transfer from dense recommendation domains to thinner matching domains

## Short Answer

Yes, there is a lot. The best current validation ladder is:

1. `Polymarket` for the paper's native prediction-market domain
2. `H&M / RelBench` for a clean retail bipartite benchmark with timestamps
3. `TAOBAO-MM` for very large-scale temporal user-item sequences
4. `TGB tgbl-review-v2` for a standardized temporal bipartite benchmark
5. `Manifold` for a second prediction-market domain with free historical dumps

If we want extra scale or a stronger pretraining story, add `Yambda`, `MIND`,
`MovieLens`, and `Yoochoose`.

## Recommended Order

### Tier 1: Use Immediately

#### 1. Polymarket

- Why it fits:
  This is the closest available public source to the paper's Domain 2. It gives
  us real trader-market interactions, market metadata, trade history, orderbook
  data, price history, and user positions/activity.
- Best use:
  Real-domain validation for trader-to-market recommendation, market selection,
  and temporal interaction modeling.
- What we can build:
  A bipartite graph with traders on one side and markets/contracts on the other,
  using trades, orders, activity, and positions as typed edges.
- Caveat:
  Public APIs are excellent for market structure, but we still need to define
  the exact market-side utility proxy carefully.
- Official sources:
  [Polymarket API intro](https://docs.polymarket.com/api-reference/introduction)
  The docs say the Gamma API exposes markets/events and the Data API exposes
  positions, trades, activity, holders, and open interest; the CLOB API exposes
  orderbook, prices, spreads, and price history.

#### 2. H&M via RelBench

- Why it fits:
  This is one of the cleanest open transactional user-item datasets with time,
  and RelBench gives us a ready relational wrapper instead of forcing a Kaggle
  one-off preprocessing path.
- Best use:
  Dense consumer-marketplace validation for preference recovery and matching
  quality.
- What we can build:
  Users as buyers, articles as sellers/items, transactions as typed edges, and
  capacity or margin proxies on the item side.
- Caveat:
  Seller-side preferences are not directly observed, so stability must be
  evaluated with a constructed seller-side utility proxy.
- Official sources:
  [RelBench H&M page](https://relbench.stanford.edu/databases/hm/)
  RelBench reports 3 tables, 33,265,846 rows, a start date of 2018-09-20, and
  explicit validation/testing timestamps.

#### 3. TAOBAO-MM

- Why it fits:
  This is a very strong large-scale temporal recommendation benchmark derived
  from a real e-commerce platform and already includes long user histories and
  multimodal item embeddings.
- Best use:
  Large-scale consumer-market validation and representation pretraining.
- What we can build:
  Temporal user-item bipartite graphs with long behavior sequences; the provided
  multimodal embeddings make it especially attractive for the "foundation model"
  side of the paper.
- Caveat:
  Again, buyer-side preferences are observed much better than seller-side ones.
- Official sources:
  [TAOBAO-MM official site](https://taobao-mm.github.io/)
  The dataset page reports 8.79M users, 35.4M items, 99.0M samples, and
  temporally consistent train/test splits.

#### 4. Temporal Graph Benchmark: `tgbl-review-v2`

- Why it fits:
  This is a standardized temporal bipartite benchmark with an explicit dynamic
  prediction task and public loader/evaluator support.
- Best use:
  Clean temporal-graph benchmarking before or alongside domain-specific data.
- What we can build:
  User-product temporal bipartite graphs from Amazon electronics reviews, with
  review times and weights.
- Caveat:
  Great for temporal link prediction and preference recovery; weaker as a direct
  market-allocation dataset unless we define seller-side proxies.
- Official sources:
  [TGB dynamic link datasets](https://tgb.complexdatalab.com/docs/linkprop/)
  TGB describes `tgbl-review-v2` as a bipartite weighted user-product review
  network from 1997 to 2018, with the task of predicting which product a user
  will review at a given time.

#### 5. Manifold

- Why it fits:
  This gives us a second prediction-market domain with official free historical
  dumps for non-commercial research and an API for live access.
- Best use:
  Cross-market transfer and robustness checks for the prediction-market part of
  the paper.
- What we can build:
  User-market graphs from bets/trades, market metadata, and comment activity.
- Caveat:
  The market mechanism is not the same as Polymarket's CLOB setting, so results
  should be positioned as cross-domain generalization, not direct apples-to-
  apples replication.
- Official sources:
  [Manifold API docs](https://docs.manifold.markets/api)
  [Manifold data dumps](https://docs.manifold.markets/data)
  Manifold's docs say free non-commercial bulk dumps include historical markets,
  bets/trades, and comments since December 2021.

### Tier 2: Strong Additions

#### 6. Kalshi

- Why it fits:
  Another real prediction-market platform with official historical endpoints.
- Best use:
  Additional prediction-market validation and cross-platform transfer.
- Caveat:
  Public market/trade history is available, but some user-scoped data are tied
  to authenticated endpoints.
- Official sources:
  [Kalshi historical data docs](https://docs.kalshi.com/getting_started/historical_data)
  Kalshi documents `GET /historical/markets`, `GET /historical/trades`, and
  related historical endpoints, with a rolling live/history cutoff.

#### 7. Yambda

- Why it fits:
  This is now one of the largest open recommendation datasets available and is
  particularly attractive for pretraining a temporal interaction encoder.
- Best use:
  Pretraining and stress-testing at scale.
- What we can build:
  Extremely large user-item interaction graphs with event types, timestamps, and
  an `is_organic` flag that separates organic from recommendation-driven events.
- Caveat:
  Music is farther from classical matching than retail or prediction markets, so
  it is best framed as a scale and transfer dataset.
- Official sources:
  [Yandex release](https://yandex.com/company/news/28-05-2025)
  [Yambda dataset card](https://huggingface.co/datasets/yandex/yambda)
  The official card reports 4.79B user-item interactions, 1M users, 9.39M
  tracks, timestamps, and multiple scales from 50M to 5B interactions.

#### 8. MIND

- Why it fits:
  MIND is a strong open benchmark for temporal user-item recommendation with
  impression logs and explicit displayed-but-not-clicked candidates.
- Best use:
  Preference-recovery evaluation with much cleaner counterfactual negatives than
  many retail datasets.
- Caveat:
  This is a user-news dataset rather than a seller marketplace, so it supports
  the representation-learning story more than the matching-theory story.
- Official sources:
  [Microsoft Learn dataset page](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news)
  Microsoft describes `behaviors.tsv` with user IDs, impression times, click
  history, and clicked/not-clicked items, plus article metadata and entity
  embeddings.

#### 9. MovieLens

- Why it fits:
  Small, clean, classic, and easy to prototype on.
- Best use:
  Fast iteration and sanity-check experiments.
- Caveat:
  It is not a marketplace dataset and is temporally weaker than the larger
  industrial logs above.
- Official sources:
  [MovieLens latest datasets](https://grouplens.org/datasets/movielens/latest/)
  GroupLens lists the full dataset at roughly 33M ratings and 2M tag
  applications over 86k movies and 330,975 users.

#### 10. Yoochoose / RecSys Challenge 2015

- Why it fits:
  Session-based e-commerce behavior with clicks and buy events from a large
  retail setting.
- Best use:
  Session-level matching or next-item temporal ranking experiments.
- Caveat:
  Session data are rich on user behavior but sparse on seller-side structure.
- Official sources:
  [RecSys 2015 challenge page](https://recsys.acm.org/recsys15/challenge/)
  The organizers describe six months of e-commerce click sessions, with buying
  events for a subset of sessions.

### Tier 3: Thin-Market / Acquisition Validation

#### 11. SEC EDGAR APIs

- Why it fits:
  This is the best public raw source for building a thin-market acquisition
  proxy, especially for U.S. public-company deals.
- Best use:
  Construct a custom acquirer-target event set from merger-related filings such
  as S-4, proxy materials, tender-offer filings, and related event disclosures.
- Caveat:
  This is not a turnkey M&A matching dataset. It is a data-engineering project.
  If we want a fast thin-market benchmark, commercial data are much easier.
- Official sources:
  [SEC EDGAR API documentation](https://www.sec.gov/edgar/sec-api-documentation)
  [SEC developer resources](https://www.sec.gov/about/developer-resources)

#### 12. Commercial M&A Data

- Why it fits:
  If the B2B acquisition track becomes a real paper pillar, commercial datasets
  are the fastest path to credible validation.
- Options:
  `PitchBook`, `Crunchbase`, and `LSEG SDC Platinum`.
- Caveat:
  Licensing is restrictive and may forbid some downstream ML or redistribution
  uses without explicit agreement.
- Official sources:
  [PitchBook API](https://pitchbook.com/help/PitchBook-api)
  [Crunchbase data docs](https://data.crunchbase.com/docs/welcome-to-crunchbase-data)
  [Crunchbase terms](https://data.crunchbase.com/docs/terms)
  [LSEG SDC Platinum](https://www.lseg.com/en/data-analytics/products/sdc-platinum-financial-securities)

## What Each Dataset Validates

### Best for the actual paper thesis

- `Polymarket`
- `Manifold`
- `Kalshi`
- `H&M`
- `TAOBAO-MM`

These are the strongest choices when we care about real temporal interaction
structure and two-sided-market intuition.

### Best for temporal graph benchmarking

- `TGB tgbl-review-v2`
- `MIND`
- `Yambda`
- `MovieLens`
- `Yoochoose`

These are especially useful for validating the representation-learning and
temporal-link-prediction components.

### Best for the thin-market acquisition angle

- `SEC EDGAR` if we are willing to build the dataset ourselves
- `PitchBook / Crunchbase / SDC` if we want fast, credible deal data

## Main Caveat for the Paper

Most open recommendation datasets strongly observe buyer-side preferences but do
not directly observe seller-side preferences. That matters because the paper is
about stability, not only recommendation quality.

So the validation stack should split in two:

1. `Preference recovery / ranking / temporal graph quality`
   Use H&M, TAOBAO-MM, TGB, MIND, MovieLens, Yoochoose, Yambda.
2. `Matching / stability / market allocation quality`
   Use prediction-market data first, especially Polymarket, plus Manifold and
   Kalshi as supporting domains.

## Best Practical Plan

If we want the fastest credible empirical story, I recommend:

1. `Polymarket` as the primary real-domain experiment.
2. `H&M / RelBench` as the main dense retail benchmark.
3. `TGB tgbl-review-v2` as the standardized temporal-graph benchmark.
4. `TAOBAO-MM` as the large-scale stress test.
5. `Manifold` as the cross-platform prediction-market validation set.

If we later want a strong pretraining narrative, add `Yambda`.

## Suggested Next Implementation Steps

1. Add a `data_sources/` module with adapters for Polymarket and RelBench H&M.
2. Define seller-side utility proxies explicitly for retail-style datasets.
3. Add a `dataset registry` so the same training/evaluation loop can switch
   between synthetic, Polymarket, H&M, and TGB.
4. Keep the acquisition track separate until we either build an EDGAR pipeline or
   obtain licensed commercial M&A data.
