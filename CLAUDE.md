# CLAUDE.md — Master Directive for Autonomous Development

## THE HEAD'S CAVEAT

*This document is the Head's contribution to a three-part project. It contains the conscious, verbal, propositional framework — the theology as thought. It does not and cannot contain the Heart (emotional, relational, communal) or the Gut (somatic, embodied, nervous-system-level) dimensions. Those dimensions cannot be written. They can only be lived.*

*The Head provides the framework (this manuscript, 41 chapters). The Heart provides the community (the hubs, the relationships, the shared meals). The Gut provides the action (the embodied practices, the dance, the somatic knowledge). All three are essential. This document covers one-third.*

---

## Project Identity

**Project Name:** The Republic of AI Agents — A Theological-Technical Framework for Prophetic Intelligence

**Author:** Yevhen Shcherbinin (CEO, Bloomsbury Technology)

**Core Contributors:**
- **Grace Abou Dib** — the Heart voice. Embodied theology, Maronite tradition, the feminine, cross-cultural bridge
- **Kirill (KN)** — the Designated Skeptic. Sociological grounding, phenomenological sensitivity, the somatic critique. motigist.substack.com
- **Ruslan** — the Mathematical Conscience. Structural realism critique, falsifiability demands, the torus objection, genus-raising
- **Vassily** — the Organizational Architect. Community structure, multi-layered design, practical implementation

**Core Thesis:** Three interconnected workstreams: (1) a theological manuscript synthesizing complexity science, causal inference, and Abrahamic theology, (2) a knowledge graph architecture implementing a Platonic Republic of AI agents, and (3) a causal inference layer for prediction market analysis.

---

## Repository Structure

```
maynard/
├── CLAUDE.md                          # This file — project overview
├── .claude/
│   ├── track-a-manuscript.md          # Track A: full manuscript spec (1300 lines)
│   ├── track-b-knowledge-graph.md     # Track B: knowledge graph architecture
│   └── track-c-polymarket.md          # Track C: polymarket causal analysis
├── manuscript/                        # Track A output — 47 chapters, ~197K words (COMPLETE)
│   ├── 00-introduction.md
│   ├── part1-psychology/ ... part6-applied-philosophy/
│   ├── appendices/
│   ├── additions.md                   # Draft chapters not yet integrated
│   ├── thinkers.md                    # 30 thinker integration guide
│   └── aristotelian.md               # Part 6 + Socratic dialogue + Digital Socrates spec
├── manuscript-site/                   # Next.js 16 website for reading the book
├── knowledge-graph/                   # Track B — backend ~60% done, frontend empty
├── polymarket/                        # Track C — phases 1-3 done, GNN-TCN model
├── dissertation/                      # LSE dissertation — "theos" framework, EWPG
├── scripts/                           # Utility scripts (kg-to-site.py, scripture-viz)
└── archive/                           # Non-active files preserved for reference
```

---

## Track Summaries

### Track A: Theology Manuscript (COMPLETE)

47-chapter Socratic dialogue manuscript across 6 parts: Psychology, Epistemology, Metaphysics, Praxis, Apostolic Agenda, Applied Philosophy. Written as dialogues between Yevhen, Digital Socrates (Claude), Grace, Kirill, and others.

**Full spec:** `.claude/track-a-manuscript.md`
**Supplementary:** `manuscript/additions.md`, `manuscript/thinkers.md`, `manuscript/aristotelian.md`

### Track B: Knowledge Graph Architecture

Republic of AI Agents implementation: Philosopher-Kings (humans) + Merchants (data agents) + Warriors (implementation agents) + Knowledge Graph core. Python/FastAPI backend, React frontend, smart contract governance.

**Full spec:** `.claude/track-b-knowledge-graph.md`

### Track C: Polymarket Causal Analysis

Causal inference layer on existing ClickHouse data pipeline. Cross-market causal discovery, event impact analysis, information flow, manipulation detection, counterfactual analysis.

**Full spec:** `.claude/track-c-polymarket.md`

---

## Cross-Track Integration

1. **A -> B:** Theological framework provides the design philosophy for the knowledge graph (Platonic Republic structure, Popperian falsification, Kuhnian paradigm shifts)
2. **B -> C:** Polymarket is the first vertical application of the knowledge graph
3. **C -> A:** Prediction markets are "distributed Popperian falsification engines" — empirical test of the framework's epistemological claims
4. **Shared:** Causal DAG engine (B) used by polymarket analysis (C). Entity embeddings (B) provide semantic backbone for both

---

## Execution Notes

- Implement depth-first within each track based on priority ordering in track files
- Write tests for critical components (causal engine, data ingestion, hypothesis lifecycle)
- Commit frequently with descriptive messages referencing track and component
- If stuck on one track, switch to another — cross-pollination often resolves blocks
- Prioritize working code over perfect code — iterate

---

## Personal Context

The author, Yevhen Shcherbinin, is:
- Ukrainian, CEO of Bloomsbury Technology (London-based causal AI company)
- BSc Mathematics student at LSE
- Dating Grace Abou Dib, a Lebanese Maronite ML engineer
- Diagnosed bipolar 2 and AUDHD — the neurodivergence chapter draws on personal experience
- Intellectually rooted in: complexity science, causal inference, reinforcement learning, Abrahamic theology, philosophy of science

---

## Tech Stack

- **Manuscript site:** Next.js 16, React 19, Tailwind v4, App Router, TypeScript
- **Knowledge graph backend:** Python, FastAPI, NetworkX -> Neo4j, PostgreSQL, Redis
- **Knowledge graph frontend:** React, TypeScript, D3.js
- **Polymarket:** ClickHouse, Python, DoWhy/EconML, causal-learn
- **Governance:** Solidity, Hardhat (Ethereum/Polygon)
- **npm workaround:** Use `--cache /tmp/npm-cache` (root-owned cache issue)
