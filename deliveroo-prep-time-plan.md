# ML Project Plan: Restaurant Preparation Time Estimation

## Introduction

Deliveroo's delivery experience depends on precise coordination between food preparation and rider dispatch. Currently, preparation time is estimated as a simple rolling average of each restaurant's orders over the past month. This approach ignores the rich structure in the data — a 20-item Friday-evening pizza order will take far longer than a single weekday lunch salad, yet both receive the same estimate.

Inaccurate prep-time estimates have direct business impact: **overestimates** cause riders to wait idle at restaurants, increasing rider cost and reducing fleet efficiency; **underestimates** cause riders to arrive late, leading to cold food, poor customer experience, and lower retention. A better model improves all three sides of the marketplace — customers get hotter food, riders spend less time waiting, and restaurants face less pressure from premature rider arrivals.

## Goals / Non-goals

**Goals:**
- Build a predictive model for `prep_time_seconds` that significantly outperforms the current per-restaurant monthly average baseline.
- Capture the key drivers of preparation time: order complexity, restaurant characteristics, food type, temporal patterns, and restaurant busyness.
- Produce estimates that are well-calibrated — not just accurate on average, but reliable across the distribution (e.g. a predicted "15 minutes" should genuinely mean ~15 minutes, not "somewhere between 5 and 40").
- Deliver a model that is interpretable enough for restaurant partners to understand and trust (e.g. "your prep time increased because you had 3 concurrent large orders").

**Non-goals:**
- Predicting rider travel time or total delivery time (separate models).
- Real-time dynamic updating mid-preparation (future extension, discussed below).
- Replacing restaurant-side operational decisions (we predict, not prescribe).

## Proposed Approach

### Data and Feature Engineering

**Target variable:** `prep_time_seconds` (continuous, non-negative). From the distribution's properties — bounded below at zero, right-skewed, heavy-tailed — I expect something between a log-normal and a gamma distribution. I would begin by examining the empirical distribution and fitting candidate parametric forms.

**Feature engineering from available data:**

| Feature Group | Features | Rationale |
|---|---|---|
| **Order complexity** | `number_of_items`, `order_value_gbp`, value-per-item ratio | More items = longer prep. Value proxies for dish complexity (a 3-item order worth £45 likely involves more complex dishes than one worth £12). |
| **Food type** | One-hot or embedding of `type_of_food` | Pizza vs. sushi vs. salad have fundamentally different preparation processes and times. |
| **Restaurant identity** | `restaurant_id` as categorical (target-encoded or random effect) | Captures unobserved restaurant-level factors: kitchen size, staffing levels, equipment, operational efficiency. |
| **Geography** | `country`, `city` (categorical) | Cultural and operational differences (kitchen norms, staffing models). |
| **Temporal — time of day** | Hour-of-day from `order_acknowledged_at`, binned into shifts (breakfast/lunch/afternoon/dinner/late-night) | Non-linear demand patterns drive kitchen load. |
| **Temporal — calendar** | Day of week, month, public holidays (country-specific calendar lookup) | Weekend vs. weekday demand differs; seasonality matters. Holiday calendars must be country-specific (e.g. Ramadan, bank holidays, local festivals). |
| **Interactions** | `type_of_food` x `restaurant_id`, calendar x `country` | A "Turkish" restaurant's prep time for its core shawarma differs from its rarely-ordered pizza. Country-specific holidays and cultural patterns (Friday prayers, Shabbat, Carnival) interact with temporal features. |
| **Restaurant busyness (engineered)** | Count of concurrent orders at the restaurant at `order_acknowledged_at` time | The most important latent driver. Computed by counting overlapping [acknowledged, ready] windows per restaurant. A restaurant with 8 active orders will be slower than one with 1. |
| **Restaurant historical stats** | Median prep time (trailing window), order volume (trailing window), prep-time variance | Proxies for kitchen capacity and consistency. A jump in median prep time may signal a staffing change; a jump in volume may signal a new promotion. |

**Data processing steps:**
1. **Clean `prep_time_seconds`**: Remove clear outliers (negative values, implausibly short <30s or long >2hr prep times) — these likely indicate data entry errors or system issues.
2. **Parse timestamps**: Extract hour, day-of-week, month, and compute concurrent order counts per restaurant.
3. **Process `type_of_food`**: Standardise casing, consolidate near-duplicates (e.g. "Fish & Chips" / "Fish and Chips"), then encode. For high-cardinality, use target encoding or learn embeddings.
4. **Engineer busyness**: For each order, count how many other orders at the same restaurant had `order_acknowledged_at` before this order's acknowledged time AND `order_ready_at` after this order's acknowledged time.
5. **Train/validation/test split**: Temporal split (e.g. train on months 1–N-2, validate on N-1, test on N) to respect time ordering and avoid leakage.

### Modelling Approach

I propose an **iterative approach** with increasing complexity, where each stage is evaluated before proceeding:

**Stage 1 — Enhanced baselines:**
- Per-restaurant average segmented by time-of-day bucket and number-of-items bucket. This directly extends the current baseline and quantifies how much simple segmentation helps.
- Linear regression on the engineered features (log-transformed target). Fast, interpretable, establishes feature importance.

**Stage 2 — Gradient-boosted trees (primary candidate):**
- LightGBM or XGBoost regression on the full feature set. Handles non-linearity, interactions, and mixed feature types naturally.
- Advantages: fast training, handles missing values, built-in feature importance, strong empirical performance on tabular data.
- Quantile regression variant (predict 10th, 50th, 90th percentiles) to produce calibrated prediction intervals — critical for the dispatch problem (we may want to optimise for a specific quantile rather than the mean).

**Stage 3 — Mixed-effects model (restaurant as random effect):**
- Treats each restaurant's baseline prep time as drawn from a population distribution, with order-level features as fixed effects. Naturally handles the hierarchical structure (orders nested within restaurants) and provides well-calibrated uncertainty.
- Can be implemented via `statsmodels` mixed-effects or Bayesian hierarchical model (PyMC/Stan).

**Stage 4 (exploratory) — Survival analysis framing:**
- Model prep time as a time-to-event problem. This naturally handles the non-negative, right-skewed distribution and allows for **censored observations** (useful if we want to predict remaining time for an order already in progress).
- Cox proportional hazards or parametric survival models (Weibull, log-logistic) as baselines; neural survival models (e.g. mixture density networks with exponential components) for flexibility.
- This framing also enables a **dynamic prediction** extension: given that an order has already been in preparation for *t* minutes, what is the expected remaining time? This is operationally valuable for real-time rider dispatch.

**Feature selection and interactions:** At each stage, I would use permutation importance, SHAP values, and partial dependence plots to understand which features and interactions drive predictions. This guides both model refinement and business insight.

### Incorporating Human Expert Knowledge

Restaurant staff and experienced riders hold valuable tacit knowledge about prep times that our data cannot directly observe — a kitchen manager knows that a new hire slows service, that the oven is unreliable, or that a particular dish is deceptively complex. Riders learn which restaurants consistently run late. This knowledge is typically accurate in *direction* but poorly calibrated in *magnitude* (humans are good at "this will be slow" but bad at "this will take exactly 23 minutes"). The goal is to extract the directional signal while letting the model handle calibration.

**Approach 1 — Restaurant-provided prep-time estimates as a feature:**
- At order acknowledgement, restaurants already indicate an estimated prep time (or could be prompted to). Rather than using this as the prediction itself, feed it as an *input feature* to the model. The model learns how to debias it — e.g. if Restaurant X consistently says "15 minutes" but actually takes 22, the model learns that restaurant's optimism factor. This is a simple Bayesian prior update: the restaurant's estimate is the prior, the model adjusts it using historical data as the likelihood.

**Approach 2 — Structured elicitation of kitchen parameters:**
- Periodically ask restaurant managers to provide structured information the data cannot reveal: number of kitchen staff per shift, number of cooking stations, whether they batch-cook or cook-to-order, peak capacity (max simultaneous orders before quality degrades). These become slow-moving features in the model — updated monthly or on request, not per-order. This captures the "capital" and "staffing" latent variables from the causal diagram that we otherwise can only proxy through historical prep-time patterns.

**Approach 3 — Rider feedback as a real-time signal:**
- Riders who arrive at restaurants observe the kitchen state directly — they can see if the kitchen is overwhelmed, if orders are piling up, or if the restaurant is quiet. A lightweight feedback mechanism (e.g. "kitchen looked: calm / busy / overwhelmed" at pickup) provides a ground-truth busyness signal that complements our concurrent-order-count proxy. Over time, this builds a rider-sourced busyness index per restaurant that can be used as a real-time feature.

**Approach 4 — Bayesian prior from expert knowledge:**
- In the mixed-effects model (Stage 3), restaurant-level random effects can be initialised with informative priors derived from restaurant self-reported characteristics rather than starting from the population mean. A restaurant that reports 2 cooking stations and 1 chef gets a different prior than one reporting 6 stations and 4 chefs. This is especially valuable for **new restaurants** with little historical data — the cold-start problem — where expert-informed priors prevent the model from defaulting to a generic population average.

**Key principle:** Human knowledge enters as *priors and features*, never as the final prediction. The model's job is to calibrate human intuition against observed data. Over time, as the model accumulates more data for a restaurant, the data dominates and the human prior fades naturally (exactly as Bayesian updating prescribes). For new restaurants or unusual situations (first day of Ramadan, post-renovation reopening), the human signal is most valuable precisely when the model has least data.

### Model Evaluation

**Statistical metrics:**
- **Primary: Mean Absolute Error (MAE)** — directly interpretable in seconds, robust to outliers. Target: significant reduction vs. baseline monthly average.
- **Root Mean Squared Error (RMSE)** — penalises large errors more heavily, relevant because a 20-minute misestimate is operationally far worse than two 10-minute ones.
- **Quantile calibration** — for probabilistic models: does the 90th percentile prediction actually contain 90% of observations?
- Evaluate **per-segment** (by country, food type, restaurant volume tier, time of day) to ensure the model improves uniformly and doesn't sacrifice accuracy for low-volume restaurants.

**Business metrics:**
- **Rider wait time at restaurant** (primary business KPI): Simulate dispatch decisions using predicted prep times vs. actual. Reduction in mean and P90 rider wait time = direct cost saving.
- **Early rider arrival rate**: % of orders where rider arrives >5 min before food is ready (wasted rider time).
- **Late rider arrival rate**: % of orders where rider arrives >5 min after food is ready (cold food, poor CX).
- **Food-ready-to-pickup latency**: Time between food ready and rider pickup. Should decrease with better estimates.

**Measuring business impact:**
- **Offline evaluation**: Replay historical orders with the new model's predictions feeding into the dispatch simulator. Compare rider wait times, early/late rates against the baseline model.
- **Online evaluation (A/B test)**: Roll out to a random subset of restaurants/cities. Measure rider wait time, customer ratings, and rider efficiency (deliveries per hour) against the control group using the existing model. Statistical significance via standard two-sample tests with correction for clustering by restaurant.

**Iteration criteria:** Progress from Stage 1 → 2 → 3 → 4 only if the previous stage's improvements justify complexity. If Stage 2 (gradient-boosted trees) achieves <15% MAE improvement over Stage 1, investigate feature engineering before adding model complexity.
