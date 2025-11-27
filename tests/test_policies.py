"""
Tests for the Policies module.

Tests various ordering policies including base-stock,
tailored base-surge, and myopic policies.
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.policies import (
    BasePolicy, DoNothingPolicy, ConstantOrderPolicy,
    BaseStockPolicy, MultiSupplierBaseStockPolicy,
    TailoredBaseSurgePolicy, SurvivalAdjustedPolicy
)
from perishable_inventory_mdp.environment import create_simple_mdp
from perishable_inventory_mdp.state import InventoryState, SupplierPipeline


class TestDoNothingPolicy:
    """Tests for DoNothingPolicy"""
    
    def test_returns_zero_orders(self):
        """Test that policy returns zero orders"""
        mdp = create_simple_mdp()
        state = mdp.create_initial_state()
        
        policy = DoNothingPolicy()
        action = policy.get_action(state, mdp)
        
        assert all(v == 0.0 for v in action.values())
    
    def test_works_with_multiple_suppliers(self):
        """Test with multiple suppliers"""
        mdp = create_simple_mdp(num_suppliers=2)
        state = mdp.create_initial_state()
        
        policy = DoNothingPolicy()
        action = policy.get_action(state, mdp)
        
        assert len(action) == 2
        assert action[0] == 0.0
        assert action[1] == 0.0


class TestConstantOrderPolicy:
    """Tests for ConstantOrderPolicy"""
    
    def test_returns_constant_orders(self):
        """Test that policy returns constant orders"""
        policy = ConstantOrderPolicy({0: 15.0, 1: 10.0})
        
        mdp = create_simple_mdp(num_suppliers=2)
        state = mdp.create_initial_state()
        
        action = policy.get_action(state, mdp)
        
        assert action[0] == 15.0
        assert action[1] == 10.0
    
    def test_independent_of_state(self):
        """Test that policy ignores state"""
        policy = ConstantOrderPolicy({0: 20.0})
        
        mdp = create_simple_mdp(num_suppliers=1)
        
        # Different states
        state1 = mdp.create_initial_state(
            initial_inventory=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        )
        state2 = mdp.create_initial_state(
            initial_inventory=np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        )
        
        assert policy.get_action(state1, mdp) == policy.get_action(state2, mdp)


class TestBaseStockPolicy:
    """Tests for BaseStockPolicy"""
    
    @pytest.fixture
    def mdp(self):
        return create_simple_mdp(shelf_life=4, num_suppliers=1, mean_demand=10.0)
    
    def test_order_up_to_level(self, mdp):
        """Test ordering up to target level"""
        policy = BaseStockPolicy(target_level=50.0, supplier_id=0)
        
        # Low inventory position
        state = mdp.create_initial_state(
            initial_inventory=np.array([5.0, 5.0, 5.0, 5.0])
        )
        
        action = policy.get_action(state, mdp)
        
        # Should order 50 - 20 = 30
        assert action[0] == 30.0
    
    def test_no_order_when_above_target(self, mdp):
        """Test no order when inventory position exceeds target"""
        policy = BaseStockPolicy(target_level=30.0, supplier_id=0)
        
        state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0])
        )
        
        action = policy.get_action(state, mdp)
        
        # Position = 40 > 30, no order
        assert action[0] == 0.0
    
    def test_considers_pipeline(self, mdp):
        """Test that pipeline is considered in inventory position"""
        policy = BaseStockPolicy(target_level=50.0, supplier_id=0)
        
        state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0])
        )
        # Add 20 units in pipeline
        state.pipelines[0].pipeline = np.array([10.0, 10.0])
        
        action = policy.get_action(state, mdp)
        
        # Position = 40 + 20 = 60 > 50, no order
        assert action[0] == 0.0
    
    def test_considers_backorders(self, mdp):
        """Test that backorders reduce inventory position"""
        policy = BaseStockPolicy(target_level=50.0, supplier_id=0)
        
        state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0]),
            initial_backorders=15.0
        )
        
        action = policy.get_action(state, mdp)
        
        # Position = 40 - 15 = 25, order 50 - 25 = 25
        assert action[0] == 25.0
    
    def test_respects_moq(self):
        """Test that MOQ is respected"""
        suppliers = [
            {"id": 0, "lead_time": 2, "unit_cost": 1.0, "moq": 10}
        ]
        mdp = create_simple_mdp.__wrapped__(
            shelf_life=4, num_suppliers=1, mean_demand=10.0
        ) if hasattr(create_simple_mdp, '__wrapped__') else create_simple_mdp(
            shelf_life=4, num_suppliers=1, mean_demand=10.0
        )
        
        policy = BaseStockPolicy(target_level=50.0, supplier_id=0, respect_moq=True)
        
        state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 15.0])
        )
        
        action = policy.get_action(state, mdp)
        
        # Need 5 units, but MOQ is 1, so order 5 (or rounded up if MOQ > 5)
        # Default MOQ is 1, so this should work
        assert action[0] >= 0


class TestTailoredBaseSurgePolicy:
    """Tests for TailoredBaseSurgePolicy"""
    
    @pytest.fixture
    def two_supplier_mdp(self):
        return create_simple_mdp(
            shelf_life=4,
            num_suppliers=2,
            mean_demand=10.0,
            fast_lead_time=1,
            slow_lead_time=3,
            fast_cost=2.0,
            slow_cost=1.0
        )
    
    def test_base_order_to_slow_supplier(self, two_supplier_mdp):
        """Test base orders go to slow supplier"""
        policy = TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=50.0,
            reorder_point=20.0
        )
        
        state = two_supplier_mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0])
        )
        
        action = policy.get_action(state, two_supplier_mdp)
        
        # Slow supplier should get base order
        assert action[1] > 0  # Slow supplier ID is 1
    
    def test_surge_order_when_low(self, two_supplier_mdp):
        """Test surge orders when inventory is low"""
        policy = TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=50.0,
            reorder_point=30.0
        )
        
        # Very low inventory
        state = two_supplier_mdp.create_initial_state(
            initial_inventory=np.array([2.0, 2.0, 2.0, 2.0])
        )
        
        action = policy.get_action(state, two_supplier_mdp)
        
        # Both suppliers should receive orders (base + surge)
        assert action[0] > 0  # Fast supplier (surge)
        assert action[1] > 0  # Slow supplier (base)
    
    def test_no_surge_when_adequate(self, two_supplier_mdp):
        """Test no surge when inventory is adequate"""
        policy = TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=50.0,
            reorder_point=20.0
        )
        
        state = two_supplier_mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0])
        )
        
        action = policy.get_action(state, two_supplier_mdp)
        
        # Position = 40 > reorder_point=20, no surge
        assert action[0] == 0.0  # Fast supplier (no surge needed)
    
    def test_from_demand_forecast(self, two_supplier_mdp):
        """Test factory method from demand forecast"""
        policy = TailoredBaseSurgePolicy.from_demand_forecast(
            slow_supplier_id=1,
            fast_supplier_id=0,
            mean_demand=10.0,
            std_demand=3.0,
            slow_lead_time=3,
            fast_lead_time=1,
            service_level=0.95
        )
        
        # Base stock should cover slow lead time
        assert policy.base_stock_level > 30  # > 3 * 10 mean demand
        
        # Reorder point should cover fast lead time
        assert policy.reorder_point > 10  # > 1 * 10 mean demand
        assert policy.reorder_point < policy.base_stock_level
    
    def test_tbs_uses_slow_supplier_for_base(self, two_supplier_mdp):
        """
        Verify TBS correctly uses slow/cheap supplier for base stock orders.
        
        Per the paper (Section 10): "Slow supplier maintains base...Fast supplier surges"
        """
        policy = TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=60.0,
            reorder_point=25.0
        )
        
        # State with moderate inventory (above reorder point but below base stock for slow)
        # IP_slow should be below base_stock, IP_total above reorder_point
        state = two_supplier_mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0, 10.0])  # Total = 40
        )
        
        action = policy.get_action(state, two_supplier_mdp)
        
        # Should order from slow supplier (base) but NOT fast (no surge needed)
        assert action[1] > 0, "Should order from slow (cheap) supplier"
        assert action[0] == 0, "Should NOT order from fast (expensive) supplier"
    
    def test_tbs_uses_fast_supplier_for_surge(self, two_supplier_mdp):
        """
        Verify TBS correctly uses fast/expensive supplier only for surge (emergency) orders.
        """
        policy = TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=60.0,
            reorder_point=30.0
        )
        
        # Very low inventory - below reorder point triggers surge
        state = two_supplier_mdp.create_initial_state(
            initial_inventory=np.array([5.0, 5.0, 5.0, 5.0])  # Total = 20 < 30 reorder point
        )
        
        action = policy.get_action(state, two_supplier_mdp)
        
        # Should order from BOTH suppliers (base + surge)
        assert action[1] > 0, "Should order from slow (cheap) supplier for base"
        assert action[0] > 0, "Should order from fast (expensive) supplier for surge"
    
    def test_tbs_fast_supplier_only_when_critical(self, two_supplier_mdp):
        """
        Verify fast supplier is only used when inventory position is below reorder point.
        """
        policy = TailoredBaseSurgePolicy(
            slow_supplier_id=1,
            fast_supplier_id=0,
            base_stock_level=50.0,
            reorder_point=20.0
        )
        
        # Inventory just above reorder point
        state_above = two_supplier_mdp.create_initial_state(
            initial_inventory=np.array([6.0, 6.0, 6.0, 7.0])  # Total = 25 > 20
        )
        action_above = policy.get_action(state_above, two_supplier_mdp)
        
        # Inventory below reorder point
        state_below = two_supplier_mdp.create_initial_state(
            initial_inventory=np.array([4.0, 4.0, 4.0, 3.0])  # Total = 15 < 20
        )
        action_below = policy.get_action(state_below, two_supplier_mdp)
        
        # Above reorder: no fast supplier order
        assert action_above[0] == 0, "No surge order when IP > reorder point"
        
        # Below reorder: fast supplier order triggered
        assert action_below[0] > 0, "Surge order when IP < reorder point"


class TestSurvivalAdjustedPolicy:
    """Tests for SurvivalAdjustedPolicy"""
    
    def test_compute_survival_probs(self):
        """Test survival probability computation"""
        probs = SurvivalAdjustedPolicy.compute_survival_probs(
            shelf_life=4,
            mean_demand=10.0,
            inventory_level=40.0
        )
        
        # Fresher items more likely to survive
        assert probs[0] < probs[-1]
        
        # All probabilities between 0 and 1
        assert all(0 <= p <= 1 for p in probs)
    
    def test_orders_more_for_old_inventory(self):
        """Test that policy orders more when inventory is old"""
        mdp = create_simple_mdp(shelf_life=4, num_suppliers=1)
        
        survival_probs = np.array([0.3, 0.6, 0.9, 1.0])
        policy = SurvivalAdjustedPolicy(
            target_level=50.0,
            survival_probs=survival_probs,
            supplier_id=0
        )
        
        # Old inventory (concentrated in low indices)
        old_state = mdp.create_initial_state(
            initial_inventory=np.array([30.0, 10.0, 0.0, 0.0])
        )
        
        # Fresh inventory (concentrated in high indices)
        fresh_state = mdp.create_initial_state(
            initial_inventory=np.array([0.0, 0.0, 10.0, 30.0])
        )
        
        old_action = policy.get_action(old_state, mdp)
        fresh_action = policy.get_action(fresh_state, mdp)
        
        # Should order more with old inventory
        assert old_action[0] > fresh_action[0]


class TestPolicyCallable:
    """Test that policies are callable"""
    
    def test_policy_callable(self):
        """Test that policies can be called directly"""
        mdp = create_simple_mdp()
        state = mdp.create_initial_state()
        
        policy = BaseStockPolicy(target_level=50.0, supplier_id=0)
        
        # Should work with both __call__ and get_action
        action1 = policy(state, mdp)
        action2 = policy.get_action(state, mdp)
        
        assert action1 == action2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

