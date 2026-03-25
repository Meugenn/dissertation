"""Bridge module: reconciles the Meta-MAPG framework (dissertation ch05-06)
with the West Africa network infrastructure.

The MARL theory provides the gradient decomposition (Terms 1-3).
The network provides the environment (cities, trade edges, cascades).
This bridge connects them: agents operate on the network, Meta-MAPG
governs how they learn from each other's actions on that network.

Components:
    - TradeNetworkGame: wraps WestAfricaGraph as a stochastic game
    - MetaMAPGTrainer: implements the three-term gradient from Theorem 6.1
    - CascadeDampingAnalyser: tests the spectral-radius damping prediction
"""

from .marl_network_env import TradeNetworkGame, TradeAgent, TradeAction
from .meta_mapg import MetaMAPGTrainer, GradientTerms
from .cascade_damping import CascadeDampingAnalyser

__all__ = [
    "TradeNetworkGame",
    "TradeAgent",
    "TradeAction",
    "MetaMAPGTrainer",
    "GradientTerms",
    "CascadeDampingAnalyser",
]
