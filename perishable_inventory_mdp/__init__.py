"""
Multi-Supplier Perishable Inventory MDP Implementation

This package implements a Markov Decision Process for managing perishable
pharmaceutical inventory with multiple suppliers, stochastic demand, 
lead times, and spoilage dynamics.

Based on: "Mathematical Formulation of a Multi-Supplier Perishable Inventory 
MDP with Stochastic Demand, Lead Times, and Spoilage Dynamics"
"""

from .state import InventoryState, SupplierPipeline
from .environment import PerishableInventoryMDP
from .demand import DemandProcess, PoissonDemand, NegativeBinomialDemand
from .policies import BasePolicy, BaseStockPolicy, TailoredBaseSurgePolicy
from .costs import CostParameters
from .solver import ValueIteration, PolicyIteration

__version__ = "1.0.0"
__all__ = [
    "InventoryState",
    "SupplierPipeline", 
    "PerishableInventoryMDP",
    "DemandProcess",
    "PoissonDemand",
    "NegativeBinomialDemand",
    "BasePolicy",
    "BaseStockPolicy",
    "TailoredBaseSurgePolicy",
    "CostParameters",
    "ValueIteration",
    "PolicyIteration",
]

