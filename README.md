# Multi-Supplier Perishable Inventory MDP

A Python implementation of a Markov Decision Process (MDP) for managing perishable pharmaceutical inventory with multiple suppliers, stochastic demand, lead times, and spoilage dynamics.

## Overview

This implementation is based on the mathematical formulation presented in *"Mathematical Formulation of a Multi-Supplier Perishable Inventory MDP with Stochastic Demand, Lead Times, and Spoilage Dynamics"*.

### Key Features

- **Multi-supplier support**: Order from multiple suppliers with different lead times and costs
- **Perishable inventory**: Track inventory by expiry buckets with FIFO consumption
- **Stochastic demand**: Support for Poisson, Negative Binomial, and seasonal demand processes
- **Stochastic lead times**: Bernoulli-based lead time uncertainty
- **Multiple cost components**: Purchase, holding, shortage, and spoilage costs
- **Various policies**: Base-stock, Tailored Base-Surge (TBS), and myopic policies
- **MDP solvers**: Value iteration, policy iteration, and approximate DP

## Installation

```bash
pip install -r requirements.txt
```

## Mathematical Model

### State Variables

The system state at time $t$ is:

$$X_t = (\mathbf{I}_t, \{P_t^{(s)}\}_{s \in \mathcal{S}}, B_t, z_t)$$

Where:
- $\mathbf{I}_t = (I_t^{(1)}, \ldots, I_t^{(N)})$ - On-hand inventory by expiry bucket
- $P_t^{(s)}$ - Supplier pipelines
- $B_t$ - Backorders
- $z_t$ - Exogenous state (seasonality, trends)

### Sequence of Events

Each period follows this sequence:

1. **Arrivals**: $A_t = \sum_{s \in \mathcal{S}} (P_t^{(s,1)} + \tilde{P}_t^{(s,1)})$
2. **Serve Demand (FIFO)**: Oldest inventory consumed first
3. **Calculate Costs**: Purchase, holding, shortage
4. **Aging and Spoilage**: Inventory ages, expired units spoil
5. **Pipeline Shifts**: Orders move through pipeline
6. **Backorder Update**: Track unfulfilled demand

### Cost Structure

$$c_t = C_t^{purchase} + C_t^{hold} + C_t^{short} + w \cdot Spoiled_t$$

### Bellman Equation

$$V(X) = \max_{a \in \mathcal{A}(X)} \left\{ -c(X,a) + \gamma \cdot \mathbb{E}[V(X') | X, a] \right\}$$

## Quick Start

```python
from perishable_inventory_mdp import (
    PerishableInventoryMDP, PoissonDemand, CostParameters,
    BaseStockPolicy, TailoredBaseSurgePolicy
)
import numpy as np

# Create a simple two-supplier MDP
from perishable_inventory_mdp.environment import create_simple_mdp

mdp = create_simple_mdp(
    shelf_life=5,
    num_suppliers=2,
    mean_demand=10.0,
    fast_lead_time=1,
    slow_lead_time=3,
    fast_cost=2.0,
    slow_cost=1.0
)

# Create initial state
state = mdp.create_initial_state(
    initial_inventory=np.array([10.0, 20.0, 30.0, 40.0, 50.0])
)

# Define a policy
policy = TailoredBaseSurgePolicy(
    slow_supplier_id=1,
    fast_supplier_id=0,
    base_stock_level=60.0,
    reorder_point=30.0
)

# Simulate
results, total_reward = mdp.simulate_episode(
    initial_state=state,
    policy=policy,
    num_periods=100,
    seed=42
)

# Compute metrics
metrics = mdp.compute_inventory_metrics(results)
print(f"Fill Rate: {metrics['fill_rate']:.2%}")
print(f"Service Level: {metrics['service_level']:.2%}")
print(f"Spoilage Rate: {metrics['spoilage_rate']:.2%}")
```

## Module Structure

```
perishable_inventory_mdp/
├── __init__.py          # Package exports
├── state.py             # InventoryState, SupplierPipeline classes
├── demand.py            # Demand processes (Poisson, NegBin, Seasonal)
├── costs.py             # Cost parameters and calculations
├── environment.py       # Main MDP environment
├── policies.py          # Ordering policies (BaseStock, TBS, etc.)
└── solver.py            # MDP solvers (Value/Policy iteration, ADP)

tests/
├── test_state.py        # State representation tests
├── test_demand.py       # Demand process tests
├── test_costs.py        # Cost calculation tests
├── test_environment.py  # MDP environment tests
├── test_policies.py     # Policy tests
└── test_solver.py       # Solver tests
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest tests/ --cov=perishable_inventory_mdp
```

## Key Classes

### `InventoryState`

Represents the complete MDP state including:
- Inventory by expiry buckets (FIFO)
- Supplier pipelines
- Backorders
- Exogenous state

```python
state = InventoryState(
    shelf_life=5,
    inventory=np.array([10, 20, 30, 40, 50]),
    backorders=0.0
)
```

### `PerishableInventoryMDP`

The main environment class implementing the MDP dynamics:

```python
mdp = PerishableInventoryMDP(
    shelf_life=5,
    suppliers=[
        {"id": 0, "lead_time": 1, "unit_cost": 2.0},
        {"id": 1, "lead_time": 3, "unit_cost": 1.0}
    ],
    demand_process=PoissonDemand(10.0),
    cost_params=CostParameters.uniform_holding(5)
)

# Execute one step
result = mdp.step(state, action={0: 10.0, 1: 20.0})
```

### Policies

#### BaseStockPolicy
Orders up to a target inventory position:

```python
policy = BaseStockPolicy(target_level=50.0, supplier_id=0)
```

#### TailoredBaseSurgePolicy
Two-supplier policy allocating base demand to slow (cheap) supplier and surge to fast (expensive) supplier:

```python
policy = TailoredBaseSurgePolicy(
    slow_supplier_id=1,
    fast_supplier_id=0,
    base_stock_level=60.0,
    reorder_point=30.0
)
```

### Solvers

#### Value Iteration

```python
from perishable_inventory_mdp.solver import ValueIteration

solver = ValueIteration(mdp, num_demand_samples=50)
result = solver.solve(initial_states, action_space)
```

#### Policy Iteration

```python
from perishable_inventory_mdp.solver import PolicyIteration

solver = PolicyIteration(mdp, num_demand_samples=50)
result = solver.solve(initial_states, initial_policy=my_policy)
```

## Customization

### Custom Demand Process

```python
from perishable_inventory_mdp.demand import DemandProcess

class CustomDemand(DemandProcess):
    def sample(self, exogenous_state=None):
        # Your sampling logic
        pass
    
    def mean(self, exogenous_state=None):
        # Return expected demand
        pass
    
    def variance(self, exogenous_state=None):
        # Return variance
        pass
    
    def pmf(self, d, exogenous_state=None):
        # Return P(D=d)
        pass
```

### Custom Policy

```python
from perishable_inventory_mdp.policies import BasePolicy

class MyPolicy(BasePolicy):
    def get_action(self, state, mdp):
        # Your policy logic
        return {supplier_id: order_qty for ...}
```

## References

Based on the mathematical formulation in:
- "Mathematical Formulation of a Multi-Supplier Perishable Inventory MDP with Stochastic Demand, Lead Times, and Spoilage Dynamics"

Key theoretical foundations include:
- Scarf (1960) - Optimality of (s, S) policies
- Zipkin (2000) - Foundations of inventory management
- Tailored Base-Surge policy literature

## License

MIT License

