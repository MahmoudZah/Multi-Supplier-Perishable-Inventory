"""
Tests for the State module.

Tests the InventoryState and SupplierPipeline classes,
verifying correct state representation and transitions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.state import (
    InventoryState, SupplierPipeline, create_state_from_config
)


class TestSupplierPipeline:
    """Tests for SupplierPipeline class"""
    
    def test_initialization_default(self):
        """Test default pipeline initialization"""
        pipeline = SupplierPipeline(supplier_id=0, lead_time=3)
        
        assert pipeline.supplier_id == 0
        assert pipeline.lead_time == 3
        assert len(pipeline.pipeline) == 3
        assert_array_equal(pipeline.pipeline, [0, 0, 0])
        assert_array_equal(pipeline.scheduled, [0, 0, 0])
        assert pipeline.unit_cost == 1.0
        assert pipeline.fixed_cost == 0.0
    
    def test_initialization_with_values(self):
        """Test pipeline initialization with custom values"""
        pipeline = SupplierPipeline(
            supplier_id=1,
            lead_time=2,
            pipeline=np.array([10.0, 20.0]),
            unit_cost=2.5,
            fixed_cost=50.0,
            capacity=100,
            moq=5
        )
        
        assert_array_equal(pipeline.pipeline, [10.0, 20.0])
        assert pipeline.unit_cost == 2.5
        assert pipeline.fixed_cost == 50.0
        assert pipeline.capacity == 100
        assert pipeline.moq == 5
    
    def test_get_arriving(self):
        """Test getting arriving quantity"""
        pipeline = SupplierPipeline(
            supplier_id=0,
            lead_time=3,
            pipeline=np.array([5.0, 10.0, 15.0]),
            scheduled=np.array([2.0, 0.0, 0.0])
        )
        
        # P_t^(s,1) + P̃_t^(s,1) = 5 + 2 = 7
        assert pipeline.get_arriving() == 7.0
    
    def test_shift_and_add_order(self):
        """Test pipeline shift with new order"""
        pipeline = SupplierPipeline(
            supplier_id=0,
            lead_time=3,
            pipeline=np.array([5.0, 10.0, 15.0])
        )
        
        arrived = pipeline.shift_and_add_order(20.0)
        
        # First position (5) should be returned
        assert arrived == 5.0
        # Pipeline should shift: [10, 15, 20]
        assert_array_equal(pipeline.pipeline, [10.0, 15.0, 20.0])
    
    def test_shift_scheduled(self):
        """Test scheduled supply shift"""
        pipeline = SupplierPipeline(
            supplier_id=0,
            lead_time=3,
            scheduled=np.array([5.0, 10.0, 15.0])
        )
        
        arrived = pipeline.shift_scheduled()
        
        assert arrived == 5.0
        # Last position should be 0
        assert_array_equal(pipeline.scheduled, [10.0, 15.0, 0.0])
    
    def test_total_in_pipeline(self):
        """Test total pipeline calculation"""
        pipeline = SupplierPipeline(
            supplier_id=0,
            lead_time=3,
            pipeline=np.array([5.0, 10.0, 15.0]),
            scheduled=np.array([2.0, 3.0, 0.0])
        )
        
        # Total = 5 + 10 + 15 + 2 + 3 = 35
        assert pipeline.total_in_pipeline() == 35.0
    
    def test_copy(self):
        """Test deep copy of pipeline"""
        pipeline = SupplierPipeline(
            supplier_id=0,
            lead_time=3,
            pipeline=np.array([5.0, 10.0, 15.0])
        )
        
        copy = pipeline.copy()
        copy.pipeline[0] = 100.0
        
        # Original should be unchanged
        assert pipeline.pipeline[0] == 5.0


class TestInventoryState:
    """Tests for InventoryState class"""
    
    def test_initialization_default(self):
        """Test default state initialization"""
        state = InventoryState(shelf_life=5)
        
        assert state.shelf_life == 5
        assert len(state.inventory) == 5
        assert_array_equal(state.inventory, [0, 0, 0, 0, 0])
        assert state.backorders == 0.0
        assert len(state.pipelines) == 0
    
    def test_initialization_with_inventory(self):
        """Test state initialization with inventory"""
        inventory = np.array([10.0, 20.0, 30.0])
        state = InventoryState(shelf_life=3, inventory=inventory)
        
        assert_array_equal(state.inventory, [10.0, 20.0, 30.0])
    
    def test_invalid_inventory_length(self):
        """Test that mismatched inventory length raises error"""
        with pytest.raises(ValueError):
            InventoryState(shelf_life=5, inventory=np.array([1, 2, 3]))
    
    def test_total_inventory(self):
        """Test total inventory calculation"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        assert state.total_inventory == 60.0
    
    def test_inventory_position_no_pipeline(self):
        """Test inventory position without pipeline"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0]),
            backorders=5.0
        )
        # IP = 60 - 5 = 55
        assert state.inventory_position == 55.0
    
    def test_inventory_position_with_pipeline(self):
        """Test inventory position with pipeline"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0]),
            backorders=5.0
        )
        state.pipelines[0] = SupplierPipeline(
            supplier_id=0,
            lead_time=2,
            pipeline=np.array([15.0, 25.0])
        )
        # IP = 60 + 40 - 5 = 95
        assert state.inventory_position == 95.0
    
    def test_add_arrivals(self):
        """Test adding arrivals to freshest bucket"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        
        state.add_arrivals(15.0)
        
        # Only last bucket should change
        assert_array_equal(state.inventory, [10.0, 20.0, 45.0])
    
    def test_serve_demand_fifo_partial(self):
        """Test FIFO demand serving - partial fulfillment"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        
        sales, backorders = state.serve_demand_fifo(15.0)
        
        assert sales == 15.0
        assert backorders == 0.0
        # Should deplete oldest first: [0, 15, 30]
        assert_array_equal(state.inventory, [0.0, 15.0, 30.0])
    
    def test_serve_demand_fifo_full(self):
        """Test FIFO demand serving - full depletion"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        
        sales, backorders = state.serve_demand_fifo(60.0)
        
        assert sales == 60.0
        assert backorders == 0.0
        assert_array_equal(state.inventory, [0.0, 0.0, 0.0])
    
    def test_serve_demand_fifo_stockout(self):
        """Test FIFO demand serving - stockout with backorders"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        
        sales, backorders = state.serve_demand_fifo(80.0)
        
        assert sales == 60.0
        assert backorders == 20.0
        assert_array_equal(state.inventory, [0.0, 0.0, 0.0])
    
    def test_serve_demand_fifo_across_buckets(self):
        """Test FIFO serving across multiple buckets"""
        state = InventoryState(
            shelf_life=4,
            inventory=np.array([5.0, 10.0, 15.0, 20.0])
        )
        
        sales, backorders = state.serve_demand_fifo(25.0)
        
        assert sales == 25.0
        assert backorders == 0.0
        # Depletes 5, 10, then 10 from third bucket
        assert_array_equal(state.inventory, [0.0, 0.0, 5.0, 20.0])
    
    def test_age_inventory(self):
        """Test inventory aging"""
        state = InventoryState(
            shelf_life=4,
            inventory=np.array([5.0, 10.0, 15.0, 20.0])
        )
        
        spoiled = state.age_inventory()
        
        assert spoiled == 5.0
        # Should shift: [10, 15, 20, 0]
        assert_array_equal(state.inventory, [10.0, 15.0, 20.0, 0.0])
    
    def test_age_inventory_no_spoilage(self):
        """Test aging with empty oldest bucket"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([0.0, 10.0, 20.0])
        )
        
        spoiled = state.age_inventory()
        
        assert spoiled == 0.0
        assert_array_equal(state.inventory, [10.0, 20.0, 0.0])
    
    def test_aging_matrix(self):
        """Test aging matrix generation"""
        state = InventoryState(shelf_life=4)
        A_age = state.get_aging_matrix()
        
        expected = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=np.float64)
        
        assert_array_equal(A_age, expected)
    
    def test_aging_matrix_multiplication(self):
        """Test that aging matrix correctly ages inventory"""
        state = InventoryState(
            shelf_life=4,
            inventory=np.array([5.0, 10.0, 15.0, 20.0])
        )
        
        A_age = state.get_aging_matrix()
        aged = A_age @ state.inventory
        
        # Should shift: [10, 15, 20, 0]
        assert_array_equal(aged, [10.0, 15.0, 20.0, 0.0])
    
    def test_survival_adjusted_position(self):
        """Test survival-adjusted inventory position"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        
        # Survival probs: older items less likely to be consumed
        survival_probs = np.array([0.5, 0.8, 1.0])
        
        ip_surv = state.survival_adjusted_inventory_position(survival_probs)
        
        # 0.5*10 + 0.8*20 + 1.0*30 = 5 + 16 + 30 = 51
        assert ip_surv == 51.0
    
    def test_copy_independence(self):
        """Test that copy is truly independent"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0]),
            backorders=5.0
        )
        state.pipelines[0] = SupplierPipeline(
            supplier_id=0,
            lead_time=2,
            pipeline=np.array([5.0, 10.0])
        )
        
        copy = state.copy()
        
        # Modify copy
        copy.inventory[0] = 100.0
        copy.backorders = 50.0
        copy.pipelines[0].pipeline[0] = 200.0
        
        # Original should be unchanged
        assert state.inventory[0] == 10.0
        assert state.backorders == 5.0
        assert state.pipelines[0].pipeline[0] == 5.0
    
    def test_to_tuple_hashable(self):
        """Test that state can be converted to hashable tuple"""
        state = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        
        key = state.to_tuple()
        
        # Should be usable as dict key
        d = {key: "value"}
        assert d[key] == "value"
    
    def test_equality(self):
        """Test state equality"""
        state1 = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        state2 = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 30.0])
        )
        state3 = InventoryState(
            shelf_life=3,
            inventory=np.array([10.0, 20.0, 31.0])
        )
        
        assert state1 == state2
        assert state1 != state3


class TestCreateStateFromConfig:
    """Tests for state factory function"""
    
    def test_basic_creation(self):
        """Test basic state creation from config"""
        suppliers = [
            {"id": 0, "lead_time": 2, "unit_cost": 1.5},
            {"id": 1, "lead_time": 4, "unit_cost": 1.0}
        ]
        
        state = create_state_from_config(
            shelf_life=5,
            suppliers=suppliers,
            initial_inventory=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            initial_backorders=10.0
        )
        
        assert state.shelf_life == 5
        assert len(state.pipelines) == 2
        assert state.pipelines[0].lead_time == 2
        assert state.pipelines[0].unit_cost == 1.5
        assert state.pipelines[1].lead_time == 4
        assert state.backorders == 10.0
    
    def test_with_initial_pipeline(self):
        """Test creation with initial pipeline quantities"""
        suppliers = [
            {
                "id": 0,
                "lead_time": 3,
                "unit_cost": 2.0,
                "initial_pipeline": [5.0, 10.0, 15.0]
            }
        ]
        
        state = create_state_from_config(shelf_life=4, suppliers=suppliers)
        
        assert_array_equal(state.pipelines[0].pipeline, [5.0, 10.0, 15.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

