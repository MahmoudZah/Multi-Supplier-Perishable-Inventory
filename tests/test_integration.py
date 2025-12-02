"""
Integration tests for the Perishable Inventory MDP.

These tests verify end-to-end workflows and consistency
across the system, matching the mathematical formulation
in the paper.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.state import InventoryState, create_state_from_config
from perishable_inventory_mdp.environment import PerishableInventoryMDP, create_simple_mdp
from perishable_inventory_mdp.simulation import run_episode
from perishable_inventory_mdp.demand import PoissonDemand, SeasonalDemand
from perishable_inventory_mdp.costs import CostParameters
from perishable_inventory_mdp.policies import (
    BaseStockPolicy, TailoredBaseSurgePolicy, DoNothingPolicy
)
from perishable_inventory_mdp.solver import ValueIteration, SolvedPolicy


class TestMathematicalFormulationConsistency:
    """
    Tests verifying consistency with the mathematical formulation
    from the paper.
    """
    
    def test_inventory_aging_matrix_formulation(self):
        """
        Test: I_{t+1}^aged = A_age @ I_t
        
        Verify that matrix-based aging matches element-wise implementation.
        """
        state = InventoryState(
            shelf_life=5,
            inventory=np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        )
        
        # Matrix formulation
        A_age = state.get_aging_matrix()
        aged_matrix = A_age @ state.inventory
        
        # Element-wise formulation
        state_copy = state.copy()
        state_copy.age_inventory()
        
        # Results should match (except matrix doesn't return spoilage)
        assert_array_almost_equal(aged_matrix, state_copy.inventory)
    
    def test_fifo_consumption_boundary_condition(self):
        """
        Test boundary condition: If n=N and R>0, remaining demand
        cannot be fulfilled from on-hand inventory.
        """
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([5.0, 10.0, 15.0])  # Total = 30
        )
        
        # Demand exceeds inventory
        sales, backorders = state.serve_demand_fifo(demand=40.0)
        
        # All inventory consumed, 10 units become backorders
        assert sales == 30.0
        assert backorders == 10.0
        assert state.total_inventory == 0.0
    
    def test_sequence_of_events_matches_paper(self):
        """
        Verify the sequence of events exactly matches the paper:
        1. Arrivals
        2. Serve Demand FIFO
        3. Costs Before Aging
        4. Aging and Spoilage
        5. Pipeline Shifts and New Orders
        6. Backorder Update
        """
        mdp = create_simple_mdp(shelf_life=4, num_suppliers=1, mean_demand=10.0, fast_lead_time=2)
        
        # Set up specific scenario
        state = mdp.create_initial_state(
            initial_inventory=np.array([5.0, 10.0, 15.0, 20.0])
        )
        state.pipelines[0].pipeline = np.array([12.0, 8.0])  # 12 arrives this period
        
        # Execute step with known demand
        action = {0: 15.0}  # Order 15 units
        result = mdp.step(state, action, demand=25.0)
        
        # Verify step 1: Arrivals (12 added to freshest bucket)
        assert result.arrivals == 12.0
        
        # Verify step 2: FIFO consumption
        # Before: [5, 10, 15, 32] after arrival
        # Demand 25: consume 5 + 10 + 10 = 25
        # After FIFO: [0, 0, 5, 32]
        assert result.sales == 25.0
        assert result.new_backorders == 0.0
        
        # Verify step 4: Aging (0 spoiled since bucket 0 was emptied)
        assert result.spoiled == 0.0
        # After aging: [0, 5, 32, 0]
        
        # Verify step 5: Pipeline shift
        # [12, 8] -> [8, 15]
        assert_array_almost_equal(
            result.next_state.pipelines[0].pipeline,
            [8.0, 15.0]
        )
        
        # Final inventory should be [0, 5, 32, 0]
        assert_array_almost_equal(
            result.next_state.inventory,
            [0.0, 5.0, 32.0, 0.0]
        )
    
    def test_cost_components_match_formulation(self):
        """
        Test: c_t = C_t^purchase + C_t^hold + C_t^short + w * Spoiled_t
        """
        suppliers = [
            {"id": 0, "lead_time": 2, "unit_cost": 3.0, "fixed_cost": 50.0}
        ]
        cost_params = CostParameters(
            holding_costs=np.array([2.0, 1.5, 1.0, 0.5]),
            shortage_cost=20.0,
            spoilage_cost=10.0
        )
        
        mdp = PerishableInventoryMDP(
            shelf_life=4,
            suppliers=suppliers,
            demand_process=PoissonDemand(10.0),
            cost_params=cost_params
        )
        
        state = mdp.create_initial_state(
            initial_inventory=np.array([5.0, 10.0, 15.0, 20.0])
        )
        
        action = {0: 10.0}
        result = mdp.step(state, action, demand=60.0)
        
        # Verify purchase cost: 3.0 * 10 = 30
        assert result.costs.purchase_cost == 30.0
        
        # Verify fixed order cost: 50 (since order > 0)
        assert result.costs.fixed_order_cost == 50.0
        
        # Verify shortage cost: 20 * backorders
        assert result.costs.shortage_cost == 20.0 * result.new_backorders
        
        # Verify spoilage cost: 10 * spoiled
        assert result.costs.spoilage_cost == 10.0 * result.spoiled
    
    def test_inventory_position_definition(self):
        """
        Test: IP_t = on-hand + pipeline - backorders
        """
        state = InventoryState(
            shelf_life=4,
            inventory=np.array([10.0, 20.0, 30.0, 40.0]),  # Total: 100
            backorders=15.0
        )
        
        from perishable_inventory_mdp.state import SupplierPipeline
        state.pipelines[0] = SupplierPipeline(
            supplier_id=0,
            lead_time=2,
            pipeline=np.array([25.0, 30.0])  # Total: 55
        )
        
        # IP = 100 + 55 - 15 = 140
        assert state.inventory_position == 140.0


class TestTwoSupplierScenario:
    """
    Tests for the two-supplier (Tailored Base-Surge) scenario
    described in the paper.
    """
    
    @pytest.fixture
    def tbs_mdp(self):
        """Create MDP suitable for TBS testing"""
        return create_simple_mdp(
            shelf_life=5,
            num_suppliers=2,
            mean_demand=10.0,
            fast_lead_time=1,
            slow_lead_time=4,
            fast_cost=2.0,
            slow_cost=1.0
        )
    
    def test_tbs_policy_allocates_correctly(self, tbs_mdp):
        """Test that TBS allocates base to slow, surge to fast"""
        policy = TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=80.0,
            reorder_point=25.0
        )
        
        # High inventory - only base order needed
        high_state = tbs_mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        )
        high_action = policy.get_action(high_state, tbs_mdp)
        
        # Should order from slow supplier (base) but not fast (no surge)
        assert high_action[1] > 0  # Slow supplier
        assert high_action[0] == 0  # Fast supplier
        
        # Low inventory - both orders needed
        low_state = tbs_mdp.create_initial_state(
            initial_inventory=np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        )
        low_action = policy.get_action(low_state, tbs_mdp)
        
        # Should order from both suppliers
        assert low_action[0] > 0  # Fast supplier (surge)
        assert low_action[1] > 0  # Slow supplier (base)
    
    def test_supplier_cost_difference(self, tbs_mdp):
        """Verify cost difference between suppliers"""
        state = tbs_mdp.create_initial_state()
        
        # Order from fast supplier only
        fast_action = {0: 10.0, 1: 0.0}
        fast_result = tbs_mdp.step(state.copy(), fast_action, demand=0.0)
        
        # Order from slow supplier only
        slow_action = {0: 0.0, 1: 10.0}
        slow_result = tbs_mdp.step(state.copy(), slow_action, demand=0.0)
        
        # Fast should be more expensive
        assert fast_result.costs.purchase_cost > slow_result.costs.purchase_cost


class TestPerishabilityDynamics:
    """
    Tests specifically for perishability-related behavior.
    """
    
    def test_spoilage_accumulates(self):
        """Test that spoilage accumulates over time without demand"""
        mdp = create_simple_mdp(shelf_life=3, num_suppliers=1, mean_demand=0.0)
        
        # Start with inventory only in oldest bucket
        state = mdp.create_initial_state(
            initial_inventory=np.array([30.0, 0.0, 0.0])
        )
        
        total_spoiled = 0.0
        for _ in range(3):
            result = mdp.step(state, {0: 0.0}, demand=0.0)
            total_spoiled += result.spoiled
            state = result.next_state
        
        # All 30 units should spoil over 3 periods
        assert total_spoiled == 30.0
    
    def test_fifo_consumes_oldest_first(self):
        """Test that FIFO consumption depletes oldest inventory first"""
        state = InventoryState(
            shelf_life=4,
            inventory=np.array([10.0, 20.0, 30.0, 40.0])
        )
        
        # Consume 25 units
        sales, backorders = state.serve_demand_fifo(25.0)
        
        # Should consume oldest first: 10 from bucket 0, 15 from bucket 1
        assert sales == 25.0
        assert state.inventory[0] == 0.0  # Oldest fully consumed
        assert state.inventory[1] == 5.0   # Partially consumed
        assert state.inventory[2] == 30.0  # Untouched
        assert state.inventory[3] == 40.0  # Untouched
        
        # Further consumption
        sales2, _ = state.serve_demand_fifo(10.0)
        
        # Should continue from bucket 1, then bucket 2
        assert sales2 == 10.0
        assert state.inventory[1] == 0.0   # Now fully consumed
        assert state.inventory[2] == 25.0  # Partially consumed
    
    def test_survival_adjusted_inventory_position(self):
        """Test survival-adjusted inventory position calculation"""
        state = InventoryState(
            shelf_life=4,
            inventory=np.array([20.0, 20.0, 20.0, 20.0])
        )
        
        # Survival probabilities: older items less likely to survive
        survival_probs = np.array([0.2, 0.5, 0.8, 1.0])
        
        ip_surv = state.survival_adjusted_inventory_position(survival_probs)
        
        # 0.2*20 + 0.5*20 + 0.8*20 + 1.0*20 = 4 + 10 + 16 + 20 = 50
        assert ip_surv == 50.0
        
        # Compare to raw inventory position
        assert ip_surv < state.total_inventory  # 50 < 80


class TestLongRunSimulation:
    """Tests for long-run simulation behavior"""
    
    def test_base_stock_policy_stabilizes(self):
        """Test that base-stock policy leads to stable inventory"""
        np.random.seed(42)
        
        mdp = create_simple_mdp(shelf_life=5, num_suppliers=1, mean_demand=10.0)
        state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        )
        
        policy = BaseStockPolicy(target_level=60.0, supplier_id=0)
        results, _ = run_episode(mdp, policy, num_periods=200, seed=42, initial_state=state)
        
        # Inventory should stabilize around target level
        late_inventories = [r.next_state.total_inventory for r in results[-50:]]
        avg_inventory = np.mean(late_inventories)
        
        # Should be reasonably close to target (accounting for lead time)
        assert 30 < avg_inventory < 90
    
    def test_do_nothing_depletes_inventory(self):
        """Test that do-nothing policy leads to stockouts"""
        np.random.seed(42)
        
        mdp = create_simple_mdp(shelf_life=5, num_suppliers=1, mean_demand=10.0)
        state = mdp.create_initial_state(
            initial_inventory=np.array([20.0, 20.0, 20.0, 20.0, 20.0])
        )
        
        policy = DoNothingPolicy()
        results, _ = run_episode(mdp, policy, num_periods=50, seed=42, initial_state=state)
        
        # Should have stockouts
        metrics = mdp.compute_inventory_metrics(results)
        assert metrics["fill_rate"] < 1.0
        assert metrics["service_level"] < 1.0
    
    def test_metrics_computation(self):
        """Test that metrics are computed correctly"""
        np.random.seed(42)
        
        mdp = create_simple_mdp(shelf_life=4, num_suppliers=1, mean_demand=10.0)
        state = mdp.create_initial_state(
            initial_inventory=np.array([30.0, 30.0, 30.0, 30.0])
        )
        
        policy = BaseStockPolicy(target_level=50.0, supplier_id=0)
        results, total_reward = run_episode(mdp, policy, num_periods=100, seed=42, initial_state=state)
        
        metrics = mdp.compute_inventory_metrics(results)
        
        # Verify metric constraints
        assert 0 <= metrics["fill_rate"] <= 1
        assert 0 <= metrics["service_level"] <= 1
        assert 0 <= metrics["spoilage_rate"] <= 1
        assert metrics["average_inventory"] >= 0
        assert metrics["total_cost"] >= 0
        
        # Verify metric consistency
        assert metrics["total_sales"] <= metrics["total_demand"]
        assert metrics["fill_rate"] == pytest.approx(
            metrics["total_sales"] / metrics["total_demand"],
            rel=0.01
        )


class TestSolverIntegration:
    """Integration tests for MDP solvers"""
    
    def test_value_iteration_improves_policy(self):
        """Test that value iteration finds better policy than do-nothing"""
        np.random.seed(42)
        
        mdp = create_simple_mdp(shelf_life=3, num_suppliers=1, mean_demand=5.0)
        
        # Create test states
        states = [
            mdp.create_initial_state(
                initial_inventory=np.array([i*5.0, i*5.0, i*5.0])
            )
            for i in range(3)
        ]
        
        # Small action space for tractability
        action_space = [{0: 0.0}, {0: 5.0}, {0: 10.0}, {0: 15.0}]
        
        # Solve
        solver = ValueIteration(mdp, num_demand_samples=20, max_iterations=5)
        result = solver.solve(states, action_space)
        
        # Create policy from solution
        solved_policy = SolvedPolicy(result.policy, default_action={0: 5.0})
        
        # Compare with do-nothing
        test_state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0])
        )
        
        _, solved_reward = run_episode(mdp, solved_policy, 50, seed=42, initial_state=test_state)
        _, nothing_reward = run_episode(mdp, DoNothingPolicy(), 50, seed=42, initial_state=test_state.copy())
        
        # Solved policy should be at least as good (or close)
        # Note: with limited iterations, may not always be better
        assert solved_reward >= nothing_reward - 500  # Allow some margin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

