## TRACK C: Polymarket Causal Analysis

### Context
Repository: polymarket (existing repo with pipeline/ and dashboard/ — ClickHouse data pipeline collecting market metadata, prices, trades, orderbook depth)

### What to Build

Add a causal inference layer on top of the existing correlational data pipeline. The goal: go beyond "what correlates with what in prediction markets" to "what CAUSES market movements."

### Architecture Extension

```
pipeline/
├── existing collection code...
├── causal/
│   ├── __init__.py
│   ├── dag_builder.py      # Automatic causal DAG construction from market data
│   ├── event_impact.py     # Causal impact analysis of real-world events on markets
│   ├── cross_market.py     # Cross-market causal relationships
│   ├── manipulation.py     # Detection of market manipulation (causal anomalies)
│   ├── information_flow.py # Information flow analysis between markets
│   └── counterfactual.py   # Counterfactual market analysis
│
dashboard/
├── existing dashboard code...
├── src/
│   ├── components/
│   │   ├── CausalGraph/        # Visualize causal relationships between markets
│   │   ├── ImpactAnalysis/     # Event impact visualization
│   │   ├── InformationFlow/    # Animated information flow between markets
│   │   ├── ManipulationAlert/  # Manipulation detection alerts
│   │   └── CounterfactualView/ # "What if" market analysis
```

### Core Analyses to Implement

**1. Cross-Market Causal Discovery**
- Use PC algorithm or GES to discover causal structure between related markets
- Example: does the "Will X be elected?" market CAUSE movements in "Will policy Y pass?" or vice versa?
- Granger causality as baseline, then upgrade to proper causal discovery
- Visualize discovered DAGs in dashboard

**2. Real-World Event Impact Analysis**
- Interrupted time series / causal impact analysis (Google's CausalImpact methodology)
- When a news event occurs, estimate its causal effect on market prices
- Separate genuine information incorporation from noise/manipulation
- Track: event → price change → volume change → spread change causal chain

**3. Information Flow Analysis**
- Transfer entropy between markets: which markets lead and which follow?
- Detect informed trading: does volume predict price changes (suggests information asymmetry)?
- Map the information topology: which markets are "source" markets and which are "derivative"?

**4. Market Manipulation Detection**
- Anomaly detection: identify price movements not explained by the causal model
- Wash trading detection: suspicious volume patterns
- Spoofing: orderbook patterns suggesting manipulation
- These are causal anomalies: events that violate the expected causal structure

**5. Counterfactual Analysis**
- "What would this market's price be if event X hadn't occurred?"
- Synthetic control method: construct counterfactual market trajectory using related markets
- Enable "what if" queries on the dashboard

### Libraries
- `dowhy` — Microsoft's causal inference library
- `causal-learn` — CMU's causal discovery
- `statsmodels` — interrupted time series, Granger causality
- `tfcausalimpact` or `causalimpact` — Google's CausalImpact
- `pgmpy` — Bayesian network / DAG operations

### Dashboard Components
- Interactive causal DAG visualization (D3.js force-directed graph)
- Event timeline with causal impact overlays
- Information flow animation (markets as nodes, edges show direction and strength of information flow)
- Manipulation alert panel
- Counterfactual simulator: select an event, see estimated market trajectory without it

### Implementation Priority
1. Granger causality matrix between top markets
2. CausalImpact analysis for major events
3. Transfer entropy / information flow
4. Causal discovery (PC algorithm) on market panel data
5. Dashboard visualizations for above
6. Manipulation detection
7. Counterfactual simulator

---
