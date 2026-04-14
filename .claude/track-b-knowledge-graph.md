## TRACK B: Knowledge Graph Architecture

### Context
Repository: knowledge-graph (existing repo with backend/frontend/contracts/docs structure)

### Architecture to Implement

The system implements a "Republic of AI Agents" based on Plato's Republic, mapped onto Pearl's causal hierarchy:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHILOSOPHER-KINGS (Human Layer)               │
│  - Hypothesis generation interface                               │
│  - Causal model specification (DAG editor)                       │
│  - Counterfactual reasoning (Pearl Level 3)                      │
│  - Dashboard for reviewing merchant reports & warrior results     │
├─────────────────────────────────────────────────────────────────┤
│                    MERCHANTS (Data Gathering Agents)              │
│                                                                   │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐  │
│  │   ONLINE MERCHANTS   │  │      OFFLINE MERCHANTS           │  │
│  │                      │  │      (Future: Embodied Robots)   │  │
│  │  - HuggingFace API   │  │                                  │  │
│  │  - Polymarket feed   │  │  - Sensor data ingestion API     │  │
│  │  - Kaggle datasets   │  │  - Physical measurement schemas  │  │
│  │  - Financial APIs    │  │  - Intervention logging          │  │
│  │  - Web scrapers      │  │  - Environment interaction logs  │  │
│  │  - News aggregators  │  │                                  │  │
│  │                      │  │  Pearl Level 2 (Intervention)    │  │
│  │  Pearl Level 1       │  │                                  │  │
│  │  (Association)       │  │                                  │  │
│  └──────────────────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    WARRIORS (Implementation Agents)               │
│  - Hypothesis testing pipelines                                   │
│  - A/B testing frameworks                                         │
│  - Deployment automation                                          │
│  - Real-world feedback collection                                 │
│  - Anomaly detection (Kuhnian crisis detection)                   │
│  - Results reporting back to philosopher-kings                    │
├─────────────────────────────────────────────────────────────────┤
│                    KNOWLEDGE GRAPH (Core Infrastructure)          │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ Causal DAGs │  │ Entity Store │  │ Temporal Event Store    │ │
│  │ (Pearl)     │  │ (Embeddings) │  │ (Historical tracking)  │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ Hypothesis  │  │ Validation   │  │ Governance Layer        │ │
│  │ Registry    │  │ Evidence DB  │  │ (Smart Contracts)       │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components to Build

**1. Causal DAG Engine**
```
backend/causal/
├── dag.py              # DAG data structure and operations
├── pearl.py            # Pearl's do-calculus implementation
├── identifiability.py  # Causal effect identifiability checks
├── estimation.py       # Statistical estimation of causal effects
└── counterfactual.py   # Counterfactual reasoning engine
```

- Use `networkx` for graph operations
- Implement basic do-calculus (backdoor criterion, front-door criterion)
- Support interventional queries: P(Y | do(X))
- Store DAGs as JSON with metadata (author, timestamp, hypothesis link)

**2. Knowledge Graph Store**
```
backend/graph/
├── store.py            # Graph database interface
├── entities.py         # Entity types and schemas
├── relations.py        # Relationship types
├── embeddings.py       # Entity embedding computation
├── temporal.py         # Temporal versioning of graph state
└── query.py            # Graph query interface
```

- Use Neo4j or NetworkX for graph storage (start with NetworkX for simplicity, migrate later)
- Entity types: Hypotheses, Variables, DataSources, CausalModels, Experiments, Results
- Embedding computation using sentence-transformers for semantic similarity
- Temporal tracking: every graph mutation is timestamped and attributable

**3. Merchant Agent Framework**
```
backend/merchants/
├── base.py             # Abstract merchant agent
├── registry.py         # Agent registration and lifecycle
├── online/
│   ├── huggingface.py  # HuggingFace dataset/model discovery
│   ├── polymarket.py   # Polymarket price/trade data
│   ├── kaggle.py       # Kaggle dataset integration
│   ├── financial.py    # Financial market data (Yahoo Finance, Alpha Vantage)
│   ├── news.py         # News aggregation (RSS, APIs)
│   └── web.py          # General web scraping
├── offline/
│   ├── sensor_api.py   # API for ingesting sensor/robot data
│   ├── schemas.py      # Physical measurement data schemas
│   └── intervention.py # Intervention logging
└── scheduler.py        # Agent scheduling and orchestration
```

- Each merchant agent: discovery → collection → validation → ingestion pipeline
- Standardized data format for all merchants (schema with provenance metadata)
- Scheduling: configurable polling intervals per source
- Rate limiting and error handling

**4. Warrior Agent Framework (Boyd's OODA Loop)**
```
backend/warriors/
├── base.py             # Abstract warrior agent
├── ooda.py             # Boyd's OODA loop implementation (Observe-Orient-Decide-Act)
├── hypothesis_test.py  # Statistical hypothesis testing
├── ab_testing.py       # A/B test framework
├── anomaly.py          # Anomaly detection (Kuhnian crisis detection)
├── destruction.py      # Boyd's destructive deduction — pattern shattering when anomalies accumulate
├── creation.py         # Boyd's creative induction — new concept synthesis from shattered constituents
├── deployment.py       # Hypothesis deployment pipeline
└── feedback.py         # Results collection and reporting
```

The warrior framework explicitly implements Boyd's Dialectic Engine:
- **Observe**: Collect data from merchant agents and real-world deployment
- **Orient**: Apply existing causal models to interpret observations. Detect anomalies (mismatch between concept and reality)
- **Decide**: When anomalies accumulate past threshold (Kuhnian crisis), trigger destructive deduction — shatter existing hypothesis into constituents. Then creative induction — synthesize new hypothesis from shattered parts plus new data
- **Act**: Deploy new hypothesis, feed results back to observation
- The OODA loop speed is the competitive advantage — faster cycle = better adaptation = higher capacity for independent action (Boyd's fundamental goal)

**5. Philosopher-King Interface (Frontend)**
```
frontend/src/
├── components/
│   ├── DAGEditor/          # Visual causal DAG editor
│   ├── HypothesisForm/     # Hypothesis creation and management
│   ├── MerchantDashboard/  # Monitor data collection agents
│   ├── WarriorDashboard/   # Monitor testing and deployment
│   ├── KnowledgeGraph/     # Interactive graph visualization
│   └── EvidencePanel/      # Review evidence for/against hypotheses
├── pages/
│   ├── Dashboard.tsx       # Main philosopher-king dashboard
│   ├── Hypotheses.tsx      # Hypothesis management
│   ├── CausalModels.tsx    # Causal model editor
│   ├── Merchants.tsx       # Data source management
│   └── Warriors.tsx        # Testing and deployment
└── lib/
    ├── api.ts              # Backend API client
    ├── dag.ts              # Client-side DAG operations
    └── types.ts            # TypeScript type definitions
```

- React + TypeScript frontend
- D3.js or vis-network for DAG visualization
- Real-time updates via WebSocket for agent status

**6. Governance Layer (Smart Contracts)**
```
contracts/
├── HypothesisRegistry.sol  # On-chain hypothesis registration
├── ValidationBounty.sol    # Bounties for hypothesis validation/falsification
├── ReputationToken.sol     # Reputation tracking for agents and humans
├── GovernanceDAO.sol       # Voting on paradigm shifts / major decisions
└── DataProvenance.sol      # Immutable data provenance tracking
```

- Solidity smart contracts for governance
- Hypothesis registration with stake (skin in the game — Popperian falsifiability as economic mechanism)
- Validation bounties: reward for providing evidence that falsifies a hypothesis
- Reputation tokens: earned through successful predictions, lost through failed ones
- DAO governance for major paradigm decisions

### Data Schemas

**Hypothesis Schema:**
```json
{
  "id": "uuid",
  "author": "philosopher_king_id",
  "title": "string",
  "description": "string",
  "causal_model_id": "dag_id",
  "variables": ["var_id"],
  "predictions": [
    {
      "if": "intervention_description",
      "then": "expected_outcome",
      "falsification_criteria": "what_would_disprove_this"
    }
  ],
  "status": "proposed | testing | validated | falsified | paradigm",
  "evidence": ["evidence_id"],
  "stake": "amount",
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

**Causal DAG Schema:**
```json
{
  "id": "uuid",
  "nodes": [
    {
      "id": "var_id",
      "name": "string",
      "type": "observable | latent | intervention",
      "embedding": [float],
      "data_sources": ["merchant_id"]
    }
  ],
  "edges": [
    {
      "source": "var_id",
      "target": "var_id",
      "type": "causal | confounding | mediating",
      "strength": float,
      "evidence": ["evidence_id"]
    }
  ],
  "hypothesis_id": "uuid",
  "version": int,
  "created_at": "timestamp"
}
```

### Implementation Priority
1. Core graph store + entity schemas + basic CRUD API
2. Causal DAG engine with do-calculus
3. Online merchant agents (start with Polymarket + financial data)
4. Hypothesis registry + evidence tracking
5. Frontend: DAG editor + dashboard
6. Warrior framework: hypothesis testing pipeline
7. Smart contract governance layer
8. Offline merchant API schemas (for future robot integration)

### Tech Stack
- Backend: Python (FastAPI)
- Frontend: React + TypeScript + D3.js
- Graph: NetworkX (initial) → Neo4j (scale)
- Database: PostgreSQL for relational data, Redis for caching
- ML: sentence-transformers for embeddings, DoWhy/EconML for causal inference
- Blockchain: Solidity + Hardhat (Ethereum/Polygon)
- Deployment: Docker + docker-compose

---
