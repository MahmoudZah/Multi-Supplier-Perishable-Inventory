"""
Tests for the Costs module.

Tests cost calculations including purchase, holding,
shortage, and spoilage costs.
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.costs import (
    CostParameters, PeriodCosts,
    calculate_purchase_costs, calculate_holding_cost,
    calculate_shortage_cost, calculate_spoilage_cost,
    calculate_safety_violation_cost, calculate_safe_threshold
)
from perishable_inventory_mdp.state import SupplierPipeline


class TestCostParameters:
    """Tests for CostParameters class"""
    
    def test_uniform_holding(self):
        """Test uniform holding cost creation"""
        params = CostParameters.uniform_holding(
            shelf_life=5,
            holding_cost=2.0,
            shortage_cost=15.0,
            spoilage_cost=8.0
        )
        
        assert len(params.holding_costs) == 5
        assert all(h == 2.0 for h in params.holding_costs)
        assert params.shortage_cost == 15.0
        assert params.spoilage_cost == 8.0
    
    def test_age_dependent_holding(self):
        """Test age-dependent holding cost creation"""
        params = CostParameters.age_dependent_holding(
            shelf_life=5,
            base_holding=1.0,
            age_premium=0.5
        )
        
        # Older items (lower index) should have higher cost
        assert params.holding_costs[0] > params.holding_costs[-1]
        
        # Check formula: h_n = base + premium * (N - n) / N
        # For n=1 (index 0): h = 1 + 0.5 * (5-1)/5 = 1 + 0.4 = 1.4
        assert params.holding_costs[0] == pytest.approx(1.4)
        
        # For n=5 (index 4): h = 1 + 0.5 * (5-5)/5 = 1.0
        assert params.holding_costs[4] == pytest.approx(1.0)


class TestPeriodCosts:
    """Tests for PeriodCosts class"""
    
    def test_total_cost(self):
        """Test total cost calculation"""
        costs = PeriodCosts(
            purchase_cost=100.0,
            fixed_order_cost=20.0,
            holding_cost=15.0,
            shortage_cost=50.0,
            spoilage_cost=10.0,
            safety_violation_cost=5.0
        )
        
        assert costs.total_cost == 200.0
    
    def test_reward(self):
        """Test reward (negative cost)"""
        costs = PeriodCosts(purchase_cost=50.0, holding_cost=10.0)
        
        assert costs.reward == -60.0
    
    def test_addition(self):
        """Test adding PeriodCosts"""
        costs1 = PeriodCosts(purchase_cost=100.0, holding_cost=20.0)
        costs2 = PeriodCosts(purchase_cost=50.0, shortage_cost=30.0)
        
        combined = costs1 + costs2
        
        assert combined.purchase_cost == 150.0
        assert combined.holding_cost == 20.0
        assert combined.shortage_cost == 30.0


class TestCostCalculations:
    """Tests for cost calculation functions"""
    
    def test_purchase_costs_single_supplier(self):
        """Test purchase cost calculation for single supplier"""
        pipelines = {
            0: SupplierPipeline(
                supplier_id=0,
                lead_time=2,
                unit_cost=5.0,
                fixed_cost=100.0
            )
        }
        
        actions = {0: 20.0}
        costs = calculate_purchase_costs(actions, pipelines)
        
        # Purchase: 5 * 20 = 100
        assert costs.purchase_cost == 100.0
        # Fixed cost incurred
        assert costs.fixed_order_cost == 100.0
    
    def test_purchase_costs_no_order(self):
        """Test that zero order incurs no fixed cost"""
        pipelines = {
            0: SupplierPipeline(
                supplier_id=0,
                lead_time=2,
                unit_cost=5.0,
                fixed_cost=100.0
            )
        }
        
        actions = {0: 0.0}
        costs = calculate_purchase_costs(actions, pipelines)
        
        assert costs.purchase_cost == 0.0
        assert costs.fixed_order_cost == 0.0
    
    def test_purchase_costs_multiple_suppliers(self):
        """Test purchase costs with multiple suppliers"""
        pipelines = {
            0: SupplierPipeline(
                supplier_id=0,
                lead_time=1,
                unit_cost=3.0,
                fixed_cost=50.0
            ),
            1: SupplierPipeline(
                supplier_id=1,
                lead_time=3,
                unit_cost=2.0,
                fixed_cost=30.0
            )
        }
        
        actions = {0: 10.0, 1: 20.0}
        costs = calculate_purchase_costs(actions, pipelines)
        
        # Purchase: 3*10 + 2*20 = 30 + 40 = 70
        assert costs.purchase_cost == 70.0
        # Fixed: 50 + 30 = 80
        assert costs.fixed_order_cost == 80.0
    
    def test_holding_cost(self):
        """Test holding cost calculation"""
        inventory = np.array([10.0, 20.0, 30.0])
        holding_costs = np.array([2.0, 1.5, 1.0])
        
        cost = calculate_holding_cost(inventory, holding_costs)
        
        # 10*2 + 20*1.5 + 30*1 = 20 + 30 + 30 = 80
        assert cost == 80.0
    
    def test_shortage_cost(self):
        """Test shortage cost calculation"""
        cost = calculate_shortage_cost(new_backorders=15.0, shortage_penalty=10.0)
        
        assert cost == 150.0
    
    def test_spoilage_cost(self):
        """Test spoilage cost calculation"""
        cost = calculate_spoilage_cost(spoiled_qty=8.0, spoilage_penalty=5.0)
        
        assert cost == 40.0
    
    def test_safety_violation_no_violation(self):
        """Test safety violation when threshold met"""
        cost = calculate_safety_violation_cost(
            inventory_position=100.0,
            safe_threshold=80.0,
            safety_penalty=10.0
        )
        
        assert cost == 0.0
    
    def test_safety_violation_with_violation(self):
        """Test safety violation when below threshold"""
        cost = calculate_safety_violation_cost(
            inventory_position=60.0,
            safe_threshold=80.0,
            safety_penalty=10.0
        )
        
        # Violation = 80 - 60 = 20
        assert cost == 200.0  # 10 * 20


class TestSafeThreshold:
    """Tests for safe threshold calculation"""
    
    def test_basic_threshold(self):
        """Test basic safe threshold calculation"""
        threshold = calculate_safe_threshold(
            mean_demand=100.0,
            std_demand=20.0,
            service_level=0.95
        )
        
        # S = μ + z_α * σ
        # z_0.95 ≈ 1.645
        from scipy import stats
        z = stats.norm.ppf(0.95)
        expected = 100.0 + z * 20.0
        
        assert threshold == pytest.approx(expected)
    
    def test_high_service_level(self):
        """Test that higher service level gives higher threshold"""
        threshold_95 = calculate_safe_threshold(100.0, 20.0, 0.95)
        threshold_99 = calculate_safe_threshold(100.0, 20.0, 0.99)
        
        assert threshold_99 > threshold_95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

