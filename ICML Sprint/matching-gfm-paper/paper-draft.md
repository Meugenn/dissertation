# Graph Foundation Models for Two-Sided Market Matching

## Working Title

Graph Foundation Models for Two-Sided Market Matching: Recovering Preferences
 and Stability from Temporal Bipartite Multigraphs

## Positioning

This project sits between representation learning and market design. The core
claim is that temporal graph representations can recover latent preference
structure from interaction histories, and that those recovered preferences can
be composed with matching operators in ways that preserve useful notions of
stability.

The current repo milestone is not the full paper. It is the first executable
research scaffold that lets us test the proposal's core objects in a controlled
setting:

1. Generate a temporal bipartite multigraph with latent buyer and seller
   utilities.
2. Recover preferences from typed interaction history.
3. Produce a matching via learned scores and a stability-aware objective.
4. Evaluate ranking quality, welfare, and blocking-pair behavior.

## Current Draft Thesis

Classical matching theory assumes preference orderings are given. Real
platforms observe timestamped interaction traces instead. A temporal graph model
should therefore be judged not only by pointwise ranking quality, but by
whether the induced preference profiles yield matchings with better stability
and welfare properties than standard tabular baselines.

## Initial Experimental Scope

The first synthetic experiment in this folder targets the narrowest version of
the proposal:

- one-to-one matching
- equal numbers of buyers and sellers
- typed temporal edges: `view`, `message`, `order`
- latent buyer-side and seller-side utility matrices
- evaluation against both latent utilities and future held-out interactions

Implemented comparison points:

- `pointwise_gbm`: tabular pair scoring using boosted trees
- `compact_graph_matcher`: recency-weighted typed message aggregation plus
  Sinkhorn and blocking-pair regularization
- `oracle_gale_shapley`: stable matching under true latent preferences

## Research Questions for This Milestone

1. Does graph-aware preference recovery improve Kendall-tau agreement with the
   latent preferences over a pointwise baseline?
2. Does adding a stability loss improve the true blocking-pair ratio under the
   latent utilities?
3. How much welfare do we lose, if any, when optimizing for stability-aware
   matching rather than raw interaction reconstruction?

## Mapping to Proposal Sections

- Proposal Section 2:
  covered by the synthetic temporal bipartite market generator and preference
  recovery evaluation.
- Proposal Section 3:
  partially covered by the compact graph matcher, Sinkhorn layer, and blocking
  loss.
- Proposal Section 4:
  not proved here; only empirical diagnostics are implemented.
- Proposal Section 6:
  partially covered by ranking, welfare, and stability metrics.

## Immediate Next Steps

1. Add a static bipartite spectral baseline and a true temporal GNN baseline.
2. Add ablations for edge-type collapse, no-temporal-decay, and no-stability
   penalty.
3. Replace the synthetic market with a Polymarket-derived trader-market slice.
4. Lift the matching layer from one-to-one to capacity-constrained matching.
