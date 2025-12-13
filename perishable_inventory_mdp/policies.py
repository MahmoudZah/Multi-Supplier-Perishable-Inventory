"""
Policies for the Perishable Inventory MDP

Implements various ordering policies including:
- Base-stock (order-up-to) policies
- Tailored Base-Surge (TBS) policies for two suppliers
- Myopic policies
- Advanced policies: PIL, DIP, PEIP, VectorBaseStock
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import InventoryState
    from .environment import PerishableInventoryMDP
from .interfaces import InventoryAgent, InventoryEnvironment
from .exceptions import InvalidParameterError, SupplierNotFoundError


class BasePolicy(InventoryAgent):
    """
    Abstract base class for ordering policies.
    
    A policy π: X → A maps states to actions.
    """
    
    @abstractmethod
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        """
        Get the action to take in the given state.
        
        Args:
            state: Current inventory state X_t
            mdp: The MDP environment
        
        Returns:
            Action dictionary {supplier_id: order_quantity}
        """
        pass
    
    def act(
        self,
        state: 'InventoryState',
        env: 'InventoryEnvironment'
    ) -> Dict[int, float]:
        """Alias for get_action to satisfy InventoryAgent interface"""
        return self.get_action(state, env)  # type: ignore

    def __call__(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        return self.get_action(state, mdp)


class DoNothingPolicy(BasePolicy):
    """Policy that never orders anything"""
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        return {s: 0.0 for s in state.pipelines.keys()}


class ConstantOrderPolicy(BasePolicy):
    """
    Policy that orders a constant amount each period.
    
    Useful for baseline comparisons.
    """
    
    def __init__(self, order_quantities: Dict[int, float]):
        self.order_quantities = order_quantities
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        return self.order_quantities.copy()


class BaseStockPolicy(BasePolicy):
    """
    Base-stock (order-up-to) policy.
    
    Orders to bring inventory position up to target level S*.
    
    a_t = max(0, S* - IP_t)
    
    where IP_t is the inventory position (on-hand + pipeline - backorders).
    
    Attributes:
        target_level: Order-up-to level S*
        supplier_id: Supplier to order from (if multiple suppliers)
    """
    
    def __init__(
        self,
        target_level: float,
        supplier_id: int = 0,
        respect_moq: bool = True
    ):
        if target_level < 0:
            raise InvalidParameterError(f"Target level must be non-negative, got {target_level}")
        self.target_level = target_level
        self.supplier_id = supplier_id
        self.respect_moq = respect_moq
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        # Calculate order-up-to quantity
        inventory_position = state.inventory_position
        order_qty = max(0, self.target_level - inventory_position)
        
        # Respect capacity first
        if self.supplier_id in state.pipelines:
            capacity = state.pipelines[self.supplier_id].capacity
            order_qty = min(order_qty, capacity)
        
        # Then apply MOQ rounding
        if self.respect_moq and self.supplier_id in state.pipelines:
            moq = state.pipelines[self.supplier_id].moq
            if order_qty > 0 and order_qty < moq:
                order_qty = moq
                if self.supplier_id in state.pipelines:
                    capacity = state.pipelines[self.supplier_id].capacity
                    if order_qty > capacity:
                        order_qty = 0.0
            elif order_qty > 0:
                order_qty = float(np.ceil(order_qty / moq) * moq)
                if self.supplier_id in state.pipelines:
                    capacity = state.pipelines[self.supplier_id].capacity
                    if order_qty > capacity:
                        order_qty = float(np.floor(capacity / moq) * moq)
        
        if self.supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.supplier_id, available)
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        action[self.supplier_id] = order_qty
        return action


class MultiSupplierBaseStockPolicy(BasePolicy):
    """
    Base-stock policy that allocates orders across multiple suppliers.
    
    Uses "cheapest effective arrival" logic:
    Orders from suppliers in order of increasing effective cost.
    """
    
    def __init__(
        self,
        target_level: float,
        allocation_strategy: str = "cheapest_first"
    ):
        if target_level < 0:
            raise InvalidParameterError(f"Target level must be non-negative, got {target_level}")
        self.target_level = target_level
        self.allocation_strategy = allocation_strategy
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        inventory_position = state.inventory_position
        order_qty = max(0, self.target_level - inventory_position)
        
        if order_qty <= 0:
            return {s: 0.0 for s in state.pipelines.keys()}
        
        sorted_suppliers = sorted(
            state.pipelines.items(),
            key=lambda x: x[1].unit_cost
        )
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        remaining = order_qty
        
        for supplier_id, pipeline in sorted_suppliers:
            if remaining <= 0:
                break
            
            alloc = min(remaining, pipeline.capacity)
            moq = pipeline.moq
            if alloc > 0 and alloc < moq:
                alloc = moq
            elif alloc > 0:
                alloc = np.ceil(alloc / moq) * moq
            
            action[supplier_id] = alloc
            remaining -= alloc
        
        return action


@dataclass
class TailoredBaseSurgePolicy(BasePolicy):
    """
    Tailored Base-Surge (TBS) policy for two suppliers.
    
    Allocates base demand to slow (cheap) supplier and 
    surge demand to fast (expensive) supplier.
    """
    slow_supplier_id: int
    fast_supplier_id: int
    base_stock_level: float
    reorder_point: float
    max_surge: float = float('inf')
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        if self.slow_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.slow_supplier_id, available)
        if self.fast_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.fast_supplier_id, available)
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        
        on_hand = state.total_inventory
        pipeline_slow = state.pipelines[self.slow_supplier_id].total_in_pipeline()
        
        ip_slow = on_hand + pipeline_slow - state.backorders
        ip_total = state.inventory_position
        
        base_order = max(0, self.base_stock_level - ip_slow)
        slow_pipeline = state.pipelines[self.slow_supplier_id]
        if base_order > 0:
            moq = slow_pipeline.moq
            if base_order < moq:
                base_order = moq
            else:
                base_order = np.ceil(base_order / moq) * moq
            base_order = min(base_order, slow_pipeline.capacity)
        
        action[self.slow_supplier_id] = base_order
        
        if ip_total < self.reorder_point:
            surge_order = min(self.reorder_point - ip_total, self.max_surge)
            fast_pipeline = state.pipelines[self.fast_supplier_id]
            if surge_order > 0:
                moq = fast_pipeline.moq
                if surge_order < moq:
                    surge_order = moq
                else:
                    surge_order = np.ceil(surge_order / moq) * moq
                surge_order = min(surge_order, fast_pipeline.capacity)
            action[self.fast_supplier_id] = surge_order
        
        return action
    
    @classmethod
    def from_demand_forecast(
        cls,
        slow_supplier_id: int,
        fast_supplier_id: int,
        mean_demand: float,
        std_demand: float,
        slow_lead_time: int,
        fast_lead_time: int,
        service_level: float = 0.95
    ) -> 'TailoredBaseSurgePolicy':
        """Create TBS policy from demand forecast."""
        from scipy import stats
        z_alpha = stats.norm.ppf(service_level)
        
        slow_demand_mean = mean_demand * slow_lead_time
        slow_demand_std = std_demand * np.sqrt(slow_lead_time)
        base_stock = slow_demand_mean + z_alpha * slow_demand_std
        
        fast_demand_mean = mean_demand * fast_lead_time
        fast_demand_std = std_demand * np.sqrt(fast_lead_time)
        reorder_point = fast_demand_mean + z_alpha * fast_demand_std
        
        return cls(
            slow_supplier_id=slow_supplier_id,
            fast_supplier_id=fast_supplier_id,
            base_stock_level=base_stock,
            reorder_point=reorder_point
        )


class MyopicPolicy(BasePolicy):
    """
    Myopic (one-step lookahead) policy.
    
    Chooses the action that minimizes expected one-step cost.
    """
    
    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        feasible_actions = mdp.get_feasible_actions(state)
        
        best_action = feasible_actions[0]
        best_cost = float('inf')
        
        for action in feasible_actions:
            expected_cost = mdp.expected_cost(state, action, self.num_samples)
            if expected_cost < best_cost:
                best_cost = expected_cost
                best_action = action
        
        return best_action


class SurvivalAdjustedPolicy(BasePolicy):
    """
    Policy that uses survival-adjusted inventory position.
    
    Accounts for probability that inventory will be consumed
    before expiry.
    """
    
    def __init__(
        self,
        target_level: float,
        survival_probs: np.ndarray,
        supplier_id: int = 0
    ):
        self.target_level = target_level
        self.survival_probs = survival_probs
        self.supplier_id = supplier_id
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        ip_surv = state.survival_adjusted_inventory_position(self.survival_probs)
        ip_surv += state.total_pipeline - state.backorders
        
        order_qty = max(0, self.target_level - ip_surv)
        
        if self.supplier_id in state.pipelines:
            pipeline = state.pipelines[self.supplier_id]
            moq = pipeline.moq
            if order_qty > 0 and order_qty < moq:
                order_qty = moq
            elif order_qty > 0:
                order_qty = np.ceil(order_qty / moq) * moq
            order_qty = min(order_qty, pipeline.capacity)
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        action[self.supplier_id] = order_qty
        return action
    
    @classmethod
    def compute_survival_probs(
        cls,
        shelf_life: int,
        mean_demand: float,
        inventory_level: float
    ) -> np.ndarray:
        """Compute survival probabilities ρ_n."""
        if inventory_level <= 0:
            return np.ones(shelf_life)
        
        probs = np.array([
            min(1.0, n * mean_demand / inventory_level)
            for n in range(1, shelf_life + 1)
        ])
        return probs


# =============================================================================
# ADVANCED BASELINE POLICIES
# =============================================================================

class ProjectedInventoryLevelPolicy(BasePolicy):
    """
    Projected Inventory Level (PIL) policy for perishable systems.
    
    Orders to raise expected inventory level at next arrival to target U,
    accounting for spoilage probabilities and expected demand during lead time.
    
    PIL outperforms base-stock in perishable/non-stationary settings.
    """
    
    def __init__(
        self,
        target_level: float,
        lead_time: int = 2,
        mean_demand: float = 10.0,
        survival_probs: Optional[np.ndarray] = None,
        supplier_id: int = 0
    ):
        if target_level < 0:
            raise InvalidParameterError(f"Target level must be non-negative, got {target_level}")
        self.target_level = target_level
        self.lead_time = lead_time
        self.mean_demand = mean_demand
        self.survival_probs = survival_probs
        self.supplier_id = supplier_id
    
    def project_inventory(self, state: 'InventoryState', lead_time: int) -> float:
        """Project inventory level after lead_time periods."""
        shelf_life = len(state.inventory)
        
        if self.survival_probs is not None:
            survival = self.survival_probs
        else:
            survival = np.array([n / shelf_life for n in range(1, shelf_life + 1)])
        
        adjusted_inventory = 0.0
        for n, inv in enumerate(state.inventory):
            remaining_life = n + 1
            if remaining_life > lead_time:
                adjusted_inventory += inv * survival[n]
        
        pipeline_arrivals = 0.0
        for pipeline in state.pipelines.values():
            for i, qty in enumerate(pipeline.pipeline):
                if i < lead_time:  # Arrives within projection horizon
                    pipeline_arrivals += qty
        
        expected_demand = self.mean_demand * lead_time
        projected = adjusted_inventory + pipeline_arrivals - expected_demand - state.backorders
        return projected
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        projected_ip = self.project_inventory(state, self.lead_time)
        order_qty = max(0.0, self.target_level - projected_ip)
        
        if self.supplier_id in state.pipelines:
            pipeline = state.pipelines[self.supplier_id]
            moq = pipeline.moq
            if order_qty > 0 and order_qty < moq:
                order_qty = moq
            elif order_qty > 0:
                order_qty = np.ceil(order_qty / moq) * moq
            order_qty = min(order_qty, pipeline.capacity)
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        action[self.supplier_id] = order_qty
        return action


class DualIndexPolicy(BasePolicy):
    """
    Dual-Index Policy (DIP) for dual-sourcing systems.
    
    Tracks two separate inventory positions (slow/fast) and orders
    to separate base-stock levels. Outperforms TBS in finite lead-time scenarios.
    """
    
    def __init__(
        self,
        slow_base_stock: float,
        fast_base_stock: float,
        slow_supplier_id: int = 0,
        fast_supplier_id: int = 1
    ):
        if slow_base_stock < 0 or fast_base_stock < 0:
            raise InvalidParameterError("Base stock levels must be non-negative")
        self.slow_base_stock = slow_base_stock
        self.fast_base_stock = fast_base_stock
        self.slow_supplier_id = slow_supplier_id
        self.fast_supplier_id = fast_supplier_id
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        if self.slow_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.slow_supplier_id, available)
        if self.fast_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.fast_supplier_id, available)
        
        on_hand = state.total_inventory
        backorders = state.backorders
        
        slow_pipeline = state.pipelines[self.slow_supplier_id].total_in_pipeline()
        slow_index = on_hand + slow_pipeline - backorders
        
        fast_pipeline = state.pipelines[self.fast_supplier_id].total_in_pipeline()
        fast_index = on_hand + fast_pipeline - backorders
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        
        slow_order = max(0.0, self.slow_base_stock - slow_index)
        if slow_order > 0:
            slow_config = state.pipelines[self.slow_supplier_id]
            moq = slow_config.moq
            if slow_order < moq:
                slow_order = moq
            else:
                slow_order = np.ceil(slow_order / moq) * moq
            slow_order = min(slow_order, slow_config.capacity)
        action[self.slow_supplier_id] = slow_order
        
        fast_order = max(0.0, self.fast_base_stock - fast_index)
        if fast_order > 0:
            fast_config = state.pipelines[self.fast_supplier_id]
            moq = fast_config.moq
            if fast_order < moq:
                fast_order = moq
            else:
                fast_order = np.ceil(fast_order / moq) * moq
            fast_order = min(fast_order, fast_config.capacity)
        action[self.fast_supplier_id] = fast_order
        
        return action
    
    @classmethod
    def from_demand_forecast(
        cls,
        slow_supplier_id: int,
        fast_supplier_id: int,
        mean_demand: float,
        std_demand: float,
        slow_lead_time: int,
        fast_lead_time: int,
        service_level: float = 0.95
    ) -> 'DualIndexPolicy':
        """Create DIP from demand forecast."""
        from scipy import stats
        z = stats.norm.ppf(service_level)
        
        slow_base = mean_demand * slow_lead_time + z * std_demand * np.sqrt(slow_lead_time)
        fast_base = mean_demand * fast_lead_time + z * std_demand * np.sqrt(fast_lead_time)
        
        return cls(
            slow_base_stock=slow_base,
            fast_base_stock=fast_base,
            slow_supplier_id=slow_supplier_id,
            fast_supplier_id=fast_supplier_id
        )


class ProjectedEffectiveInventoryPolicy(BasePolicy):
    """
    Projected Effective Inventory Position (PEIP) policy.
    
    Projects expedited inventory position ahead by lead-time difference.
    Outperforms TBS in non-asymptotic regimes with finite lead times.
    """
    
    def __init__(
        self,
        target_U: float,
        slow_supplier_id: int = 0,
        fast_supplier_id: int = 1,
        lead_time_diff: int = 2,
        mean_demand: float = 10.0
    ):
        if target_U < 0:
            raise InvalidParameterError(f"Target must be non-negative, got {target_U}")
        self.target_U = target_U
        self.slow_supplier_id = slow_supplier_id
        self.fast_supplier_id = fast_supplier_id
        self.lead_time_diff = lead_time_diff
        self.mean_demand = mean_demand
    
    def project_effective_position(self, state: 'InventoryState') -> float:
        """Project effective inventory position l periods ahead."""
        on_hand = state.total_inventory
        total_pipeline = state.total_pipeline
        backorders = state.backorders
        expected_demand = self.mean_demand * self.lead_time_diff
        return on_hand + total_pipeline - backorders - expected_demand
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        if self.slow_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.slow_supplier_id, available)
        if self.fast_supplier_id not in state.pipelines:
            available = list(state.pipelines.keys())
            raise SupplierNotFoundError(self.fast_supplier_id, available)
        
        action = {s: 0.0 for s in state.pipelines.keys()}
        
        projected_ep = self.project_effective_position(state)
        
        slow_order = max(0.0, self.target_U - projected_ep)
        if slow_order > 0:
            slow_config = state.pipelines[self.slow_supplier_id]
            moq = slow_config.moq
            if slow_order < moq:
                slow_order = moq
            else:
                slow_order = np.ceil(slow_order / moq) * moq
            slow_order = min(slow_order, slow_config.capacity)
        action[self.slow_supplier_id] = slow_order
        
        total_ip = state.inventory_position
        reorder_point = self.mean_demand * 2
        if total_ip < reorder_point:
            fast_order = max(0.0, reorder_point - total_ip)
            fast_config = state.pipelines[self.fast_supplier_id]
            moq = fast_config.moq
            if fast_order > 0 and fast_order < moq:
                fast_order = moq
            elif fast_order > 0:
                fast_order = np.ceil(fast_order / moq) * moq
            fast_order = min(fast_order, fast_config.capacity)
            action[self.fast_supplier_id] = fast_order
        
        return action
    
    @classmethod
    def from_demand_and_leadtimes(
        cls,
        slow_supplier_id: int,
        fast_supplier_id: int,
        mean_demand: float,
        std_demand: float,
        slow_lead_time: int,
        fast_lead_time: int,
        service_level: float = 0.95
    ) -> 'ProjectedEffectiveInventoryPolicy':
        """Create PEIP from demand and lead time info."""
        from scipy import stats
        z = stats.norm.ppf(service_level)
        
        lead_time_diff = slow_lead_time - fast_lead_time
        target_U = mean_demand * slow_lead_time + z * std_demand * np.sqrt(slow_lead_time)
        
        return cls(
            target_U=target_U,
            slow_supplier_id=slow_supplier_id,
            fast_supplier_id=fast_supplier_id,
            lead_time_diff=lead_time_diff,
            mean_demand=mean_demand
        )


class VectorBaseStockPolicy(BasePolicy):
    """
    Vector Base-Stock policy for perishable inventory.
    
    Tracks multi-dimensional inventory state (by age bucket and supplier)
    and orders to vector targets. Closer to optimal than standard base-stock.
    """
    
    def __init__(
        self,
        supplier_targets: Optional[Dict[int, float]] = None,
        critical_fractile: float = 0.95,
        default_target: float = 50.0,
        use_age_weighted: bool = True
    ):
        if critical_fractile < 0 or critical_fractile > 1:
            raise InvalidParameterError(f"Critical fractile must be in [0,1], got {critical_fractile}")
        self.supplier_targets = supplier_targets or {}
        self.critical_fractile = critical_fractile
        self.default_target = default_target
        self.use_age_weighted = use_age_weighted
    
    def compute_age_weighted_inventory(self, state: 'InventoryState') -> float:
        """Compute age-weighted inventory position."""
        shelf_life = len(state.inventory)
        weighted_sum = 0.0
        
        for n, inv in enumerate(state.inventory):
            remaining_life = n + 1
            weight = remaining_life / shelf_life if self.use_age_weighted else 1.0
            weighted_sum += weight * inv
        
        return weighted_sum
    
    def get_action(
        self,
        state: 'InventoryState',
        mdp: 'PerishableInventoryMDP'
    ) -> Dict[int, float]:
        action = {s: 0.0 for s in state.pipelines.keys()}
        
        weighted_inventory = self.compute_age_weighted_inventory(state)
        
        sorted_suppliers = sorted(
            state.pipelines.items(),
            key=lambda x: x[1].unit_cost
        )
        
        for supplier_id, pipeline in sorted_suppliers:
            target = self.supplier_targets.get(supplier_id, self.default_target)
            supplier_pipeline = pipeline.total_in_pipeline()
            local_ip = weighted_inventory + supplier_pipeline - state.backorders
            deficit = max(0.0, target - local_ip)
            
            if deficit > 0:
                moq = pipeline.moq
                if deficit < moq:
                    deficit = moq
                else:
                    deficit = np.ceil(deficit / moq) * moq
                deficit = min(deficit, pipeline.capacity)
            
            action[supplier_id] = deficit
        
        return action
    
    @classmethod
    def from_demand_forecast(
        cls,
        suppliers: List[Dict],
        mean_demand: float,
        std_demand: float,
        shelf_life: int,
        service_level: float = 0.95
    ) -> 'VectorBaseStockPolicy':
        """Create VectorBaseStock from demand and supplier info."""
        from scipy import stats
        z = stats.norm.ppf(service_level)
        
        supplier_targets = {}
        for supplier in suppliers:
            sid = supplier['id']
            lead_time = supplier.get('lead_time', 2)
            base_target = mean_demand * lead_time + z * std_demand * np.sqrt(lead_time)
            spoilage_factor = 1.0 + (1.0 / shelf_life)
            supplier_targets[sid] = base_target * spoilage_factor
        
        return cls(
            supplier_targets=supplier_targets,
            critical_fractile=service_level,
            use_age_weighted=True
        )
