"""
Tests for the Environment module.

Tests the complete MDP environment including
state transitions and the sequence of events.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.environment import (
    PerishableInventoryMDP, TransitionResult, create_simple_mdp
)
from perishable_inventory_mdp.simulation import run_episode
from perishable_inventory_mdp.state import InventoryState, SupplierPipeline
from perishable_inventory_mdp.demand import PoissonDemand
from perishable_inventory_mdp.costs import CostParameters
from perishable_inventory_mdp.policies import (
    DoNothingPolicy, ConstantOrderPolicy, BaseStockPolicy
)


class TestPerishableInventoryMDP:
    """Tests for the main MDP environment"""
    
    @pytest.fixture
    def simple_mdp(self):
        """Create a simple MDP for testing"""
        return create_simple_mdp(
            shelf_life=4,
            num_suppliers=1,
            mean_demand=10.0,
            fast_lead_time=2,
            fast_cost=1.0
        )
    
    @pytest.fixture
    def two_supplier_mdp(self):
        """Create a two-supplier MDP for testing"""
        return create_simple_mdp(
            shelf_life=4,
            num_suppliers=2,
            mean_demand=10.0,
            fast_lead_time=1,
            slow_lead_time=3,
            fast_cost=2.0,
            slow_cost=1.0
        )
    
    def test_create_initial_state(self, simple_mdp):
        """Test initial state creation"""
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([5.0, 10.0, 15.0, 20.0]),
            initial_backorders=5.0
        )
        
        assert state.shelf_life == 4
        assert state.total_inventory == 50.0
        assert state.backorders == 5.0
        assert len(state.pipelines) == 1
    
    def test_step_deterministic_demand(self, simple_mdp):
        """Test step with deterministic demand"""
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0])
        )
        
        # No order action
        action = {0: 0.0}
        
        # Fixed demand of 15
        result = simple_mdp.step(state, action, demand=15.0)
        
        assert result.demand_realized == 15.0
        assert result.sales == 15.0
        assert result.new_backorders == 0.0
    
    def test_step_fifo_consumption(self, simple_mdp):
        """Test that demand is consumed FIFO"""
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([5.0, 10.0, 15.0, 20.0])
        )
        
        action = {0: 0.0}
        result = simple_mdp.step(state, action, demand=12.0)
        
        # After FIFO: depletes 5 from bucket 0, 7 from bucket 1
        # Then aging shifts: [3, 15, 20, 0]
        assert_array_almost_equal(
            result.next_state.inventory,
            [3.0, 15.0, 20.0, 0.0]
        )
    
    def test_step_spoilage(self, simple_mdp):
        """Test inventory spoilage"""
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([10.0, 0.0, 0.0, 0.0])
        )
        
        action = {0: 0.0}
        result = simple_mdp.step(state, action, demand=5.0)
        
        # 5 consumed from oldest bucket, 5 remain and spoil
        assert result.spoiled == 5.0
        assert result.costs.spoilage_cost > 0
    
    def test_step_backorders(self, simple_mdp):
        """Test backorder creation"""
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([5.0, 5.0, 5.0, 5.0])
        )
        
        action = {0: 0.0}
        result = simple_mdp.step(state, action, demand=30.0)
        
        # Total inventory is 20, demand is 30
        assert result.sales == 20.0
        assert result.new_backorders == 10.0
        assert result.costs.shortage_cost > 0
    
    def test_step_arrivals(self, simple_mdp):
        """Test inventory arrivals from pipeline"""
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([0.0, 0.0, 0.0, 0.0])
        )
        
        # Set pipeline with arrival
        state.pipelines[0].pipeline = np.array([20.0, 0.0])
        
        action = {0: 0.0}
        result = simple_mdp.step(state, action, demand=5.0)
        
        assert result.arrivals == 20.0
        # After arrival, consumption, and aging
        # Arrives in freshest (index 3), consumes 5, ages
        # [0, 0, 15, 0] after aging
        assert result.next_state.inventory[2] == 15.0
    
    def test_step_order_placement(self, simple_mdp):
        """Test that orders are placed in pipeline"""
        state = simple_mdp.create_initial_state()
        
        action = {0: 25.0}
        result = simple_mdp.step(state, action, demand=0.0)
        
        # Order should be at end of pipeline
        assert result.next_state.pipelines[0].pipeline[-1] == 25.0
    
    def test_step_pipeline_shift(self, simple_mdp):
        """Test pipeline shifts correctly"""
        state = simple_mdp.create_initial_state()
        state.pipelines[0].pipeline = np.array([10.0, 20.0])
        
        action = {0: 30.0}
        result = simple_mdp.step(state, action, demand=0.0)
        
        # After shift: [20, 30]
        assert_array_equal(
            result.next_state.pipelines[0].pipeline,
            [20.0, 30.0]
        )
    
    def test_step_purchase_cost(self, simple_mdp):
        """Test purchase cost calculation in step"""
        state = simple_mdp.create_initial_state()
        
        # Order 10 units at cost 1.0 per unit
        action = {0: 10.0}
        result = simple_mdp.step(state, action, demand=0.0)
        
        assert result.costs.purchase_cost == 10.0  # 10 * 1.0
    
    def test_step_time_advance(self, simple_mdp):
        """Test that time step advances"""
        state = simple_mdp.create_initial_state()
        assert state.time_step == 0
        
        result = simple_mdp.step(state, {0: 0.0}, demand=0.0)
        
        assert result.next_state.time_step == 1
    
    def test_two_supplier_orders(self, two_supplier_mdp):
        """Test ordering from two suppliers"""
        state = two_supplier_mdp.create_initial_state()
        
        action = {0: 15.0, 1: 25.0}
        result = two_supplier_mdp.step(state, action, demand=0.0)
        
        # Check orders placed correctly
        assert result.next_state.pipelines[0].pipeline[-1] == 15.0
        assert result.next_state.pipelines[1].pipeline[-1] == 25.0
        
        # Check costs include both suppliers
        # Fast: 15 * 2.0 = 30, Slow: 25 * 1.0 = 25
        assert result.costs.purchase_cost == 55.0


class TestMDPSimulation:
    """Tests for MDP simulation functionality"""
    
    @pytest.fixture
    def mdp(self):
        return create_simple_mdp(shelf_life=4, num_suppliers=1, mean_demand=10.0)
    
    def test_simulate_episode(self, mdp):
        """Test episode simulation"""
        initial_state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0])
        )
        
        policy = DoNothingPolicy()
        results, total_reward = run_episode(
            mdp, policy, num_periods=10, seed=42, initial_state=initial_state
        )
        
        assert len(results) == 10
        assert isinstance(total_reward, float)
    
    def test_simulate_with_constant_policy(self, mdp):
        """Test simulation with constant ordering"""
        initial_state = mdp.create_initial_state()
        
        # Order 10 units each period
        policy = ConstantOrderPolicy({0: 10.0})
        results, _ = run_episode(
            mdp, policy, num_periods=20, seed=42, initial_state=initial_state
        )
        
        # Inventory should build up over time (with lead time)
        final_inventory = results[-1].next_state.total_inventory
        assert final_inventory >= 0  # Should have some inventory
    
    def test_compute_metrics(self, mdp):
        """Test metric computation"""
        initial_state = mdp.create_initial_state(
            initial_inventory=np.array([20.0, 20.0, 20.0, 20.0])
        )
        
        policy = DoNothingPolicy()
        results, _ = run_episode(
            mdp, policy, num_periods=50, seed=42, initial_state=initial_state
        )
        
        metrics = mdp.compute_inventory_metrics(results)
        
        assert "fill_rate" in metrics
        assert "spoilage_rate" in metrics
        assert "service_level" in metrics
        assert "average_inventory" in metrics
        
        # Fill rate should be between 0 and 1
        assert 0 <= metrics["fill_rate"] <= 1
        assert 0 <= metrics["service_level"] <= 1


class TestMDPFeasibility:
    """Tests for action feasibility checks"""
    
    @pytest.fixture
    def mdp(self):
        suppliers = [
            {"id": 0, "lead_time": 2, "unit_cost": 1.0, "capacity": 50, "moq": 5}
        ]
        return PerishableInventoryMDP(
            shelf_life=4,
            suppliers=suppliers,
            demand_process=PoissonDemand(10.0),
            cost_params=CostParameters.uniform_holding(4)
        )
    
    def test_feasible_action(self, mdp):
        """Test that valid action is feasible"""
        state = mdp.create_initial_state()
        
        # Valid: multiple of MOQ, within capacity
        assert mdp.is_action_feasible(state, {0: 10.0})
        assert mdp.is_action_feasible(state, {0: 50.0})  # At capacity
        assert mdp.is_action_feasible(state, {0: 0.0})   # Zero order
    
    def test_infeasible_exceeds_capacity(self, mdp):
        """Test that action exceeding capacity is infeasible"""
        state = mdp.create_initial_state()
        
        assert not mdp.is_action_feasible(state, {0: 60.0})  # Over capacity
    
    def test_infeasible_moq_violation(self, mdp):
        """Test that MOQ violation is infeasible"""
        state = mdp.create_initial_state()
        
        # Not multiple of MOQ=5
        assert not mdp.is_action_feasible(state, {0: 7.0})
        assert not mdp.is_action_feasible(state, {0: 3.0})
    
    def test_get_feasible_actions(self, mdp):
        """Test enumeration of feasible actions"""
        state = mdp.create_initial_state()
        
        actions = mdp.get_feasible_actions(state)
        
        # Should include zero order
        assert {0: 0.0} in actions
        
        # All actions should be feasible
        for action in actions:
            assert mdp.is_action_feasible(state, action)


class TestLostSalesModel:
    """Tests for lost sales model variant"""
    
    def test_lost_sales_no_backorders(self):
        """Test that lost sales model has no backorders"""
        mdp = PerishableInventoryMDP(
            shelf_life=4,
            suppliers=[{"id": 0, "lead_time": 2, "unit_cost": 1.0}],
            demand_process=PoissonDemand(10.0),
            cost_params=CostParameters.uniform_holding(4),
            lost_sales=True
        )
        
        state = mdp.create_initial_state(
            initial_inventory=np.array([5.0, 5.0, 5.0, 5.0])
        )
        
        result = mdp.step(state, {0: 0.0}, demand=30.0)
        
        # With lost sales, backorders should be 0
        assert result.next_state.backorders == 0.0
        # But shortage cost should still be incurred
        assert result.costs.shortage_cost > 0


class TestTransitionResult:
    """Tests for TransitionResult dataclass"""
    
    def test_reward_property(self):
        """Test that reward equals negative cost"""
        costs = type('Costs', (), {'total_cost': 100.0, 'reward': -100.0})()
        
        result = TransitionResult(
            next_state=None,
            costs=costs,
            demand_realized=10.0,
            sales=8.0,
            new_backorders=2.0,
            spoiled=1.0,
            arrivals=5.0
        )
        
        assert result.reward == -100.0


class TestSequenceOfEvents:
    """
    Comprehensive tests verifying the exact sequence of events
    as specified in the paper.
    """
    
    def test_full_sequence(self):
        """Test complete sequence of events in one period"""
        mdp = create_simple_mdp(shelf_life=4, num_suppliers=1, mean_demand=10.0, fast_lead_time=2)
        
        # Set up specific state
        state = mdp.create_initial_state(
            initial_inventory=np.array([8.0, 12.0, 16.0, 20.0])
        )
        state.pipelines[0].pipeline = np.array([15.0, 10.0])
        
        # Execute step with known demand
        action = {0: 25.0}
        result = mdp.step(state, action, demand=20.0)
        
        # Verify sequence:
        # 1. Arrivals: 15 arrives, added to bucket 3 (index 3)
        #    Inventory: [8, 12, 16, 35]
        # 2. FIFO consumption of 20:
        #    Consume 8 from bucket 0, 12 from bucket 1
        #    Inventory: [0, 0, 16, 35]
        # 3. Aging:
        #    Bucket 0 spoils (0 units), shift
        #    Inventory: [0, 16, 35, 0]
        # 4. Pipeline shift with new order:
        #    Pipeline: [10, 25]
        
        assert result.arrivals == 15.0
        assert result.sales == 20.0
        assert result.spoiled == 0.0
        assert_array_almost_equal(
            result.next_state.inventory,
            [0.0, 16.0, 35.0, 0.0]
        )
        assert_array_almost_equal(
            result.next_state.pipelines[0].pipeline,
            [10.0, 25.0]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

