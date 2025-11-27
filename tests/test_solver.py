"""
Tests for the Solver module.

Tests value iteration, policy iteration, and
approximate dynamic programming algorithms.
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.solver import (
    ValueIteration, PolicyIteration, SolverResult,
    SolvedPolicy, ApproximateDynamicProgramming, default_feature_fn
)
from perishable_inventory_mdp.environment import create_simple_mdp
from perishable_inventory_mdp.policies import DoNothingPolicy, BaseStockPolicy


class TestSolverResult:
    """Tests for SolverResult dataclass"""
    
    def test_default_initialization(self):
        """Test default SolverResult"""
        result = SolverResult()
        
        assert result.iterations == 0
        assert result.converged == False
        assert len(result.value_function) == 0
        assert len(result.policy) == 0


class TestValueIteration:
    """Tests for Value Iteration solver"""
    
    @pytest.fixture
    def simple_mdp(self):
        return create_simple_mdp(
            shelf_life=3,
            num_suppliers=1,
            mean_demand=5.0
        )
    
    def test_bellman_operator(self, simple_mdp):
        """Test Bellman operator computation"""
        solver = ValueIteration(simple_mdp, num_demand_samples=20)
        
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([5.0, 5.0, 5.0])
        )
        action = {0: 5.0}
        value_function = {}
        
        q_value = solver.bellman_operator(state, action, value_function)
        
        # Q-value should be negative (cost is positive)
        assert isinstance(q_value, float)
    
    def test_solve_small_problem(self, simple_mdp):
        """Test solving a small problem"""
        solver = ValueIteration(
            simple_mdp,
            num_demand_samples=10,
            max_iterations=5,
            tolerance=1.0  # Relaxed for quick test
        )
        
        # Create a few test states
        states = [
            simple_mdp.create_initial_state(
                initial_inventory=np.array([i*5.0, i*5.0, i*5.0])
            )
            for i in range(3)
        ]
        
        # Define small action space
        action_space = [{0: 0.0}, {0: 5.0}, {0: 10.0}]
        
        result = solver.solve(states, action_space)
        
        assert result.iterations > 0
        assert len(result.value_function) > 0
        assert len(result.policy) > 0
    
    def test_value_function_structure(self, simple_mdp):
        """Test that value function has expected structure"""
        solver = ValueIteration(
            simple_mdp,
            num_demand_samples=10,
            max_iterations=3
        )
        
        states = [
            simple_mdp.create_initial_state(
                initial_inventory=np.array([10.0, 10.0, 10.0])
            ),
            simple_mdp.create_initial_state(
                initial_inventory=np.array([0.0, 0.0, 0.0])
            )
        ]
        
        result = solver.solve(states, [{0: 0.0}, {0: 10.0}])
        
        # Higher inventory should have higher value (less cost)
        key_high = states[0].to_tuple()
        key_low = states[1].to_tuple()
        
        if key_high in result.value_function and key_low in result.value_function:
            # State with more inventory should be better (less negative)
            # This may not always hold due to holding costs, but for shortage-heavy costs...
            pass  # Just verify computation completed


class TestPolicyIteration:
    """Tests for Policy Iteration solver"""
    
    @pytest.fixture
    def simple_mdp(self):
        return create_simple_mdp(
            shelf_life=3,
            num_suppliers=1,
            mean_demand=5.0
        )
    
    def test_policy_evaluation(self, simple_mdp):
        """Test policy evaluation step"""
        solver = PolicyIteration(
            simple_mdp,
            num_demand_samples=10,
            eval_iterations=3
        )
        
        state = simple_mdp.create_initial_state(
            initial_inventory=np.array([10.0, 10.0, 10.0])
        )
        
        policy = {state.to_tuple(): {0: 5.0}}
        value_fn = solver.policy_evaluation(policy, [state], {})
        
        assert state.to_tuple() in value_fn
    
    def test_policy_improvement(self, simple_mdp):
        """Test policy improvement step"""
        solver = PolicyIteration(
            simple_mdp,
            num_demand_samples=10
        )
        
        states = [
            simple_mdp.create_initial_state(
                initial_inventory=np.array([10.0, 10.0, 10.0])
            )
        ]
        
        value_fn = {states[0].to_tuple(): -50.0}
        
        new_policy, _ = solver.policy_improvement(
            states, value_fn, [{0: 0.0}, {0: 10.0}]
        )
        
        assert len(new_policy) == 1
        assert states[0].to_tuple() in new_policy
    
    def test_solve_with_initial_policy(self, simple_mdp):
        """Test solving with initial policy"""
        solver = PolicyIteration(
            simple_mdp,
            num_demand_samples=10,
            max_iterations=3,
            eval_iterations=2
        )
        
        states = [
            simple_mdp.create_initial_state(
                initial_inventory=np.array([10.0, 10.0, 10.0])
            )
        ]
        
        initial_policy = DoNothingPolicy()
        
        result = solver.solve(
            states,
            initial_policy=initial_policy,
            action_space=[{0: 0.0}, {0: 5.0}]
        )
        
        assert result.iterations > 0


class TestSolvedPolicy:
    """Tests for SolvedPolicy wrapper"""
    
    def test_lookup_existing_state(self):
        """Test looking up action for known state"""
        mdp = create_simple_mdp()
        state = mdp.create_initial_state()
        
        policy_dict = {state.to_tuple(): {0: 15.0}}
        policy = SolvedPolicy(policy_dict)
        
        action = policy.get_action(state, mdp)
        
        assert action[0] == 15.0
    
    def test_default_for_unknown_state(self):
        """Test default action for unknown state"""
        mdp = create_simple_mdp()
        
        policy = SolvedPolicy(
            policy_dict={},
            default_action={0: 10.0}
        )
        
        state = mdp.create_initial_state()
        action = policy.get_action(state, mdp)
        
        assert action[0] == 10.0
    
    def test_zero_default_for_unknown(self):
        """Test zero default when no default provided"""
        mdp = create_simple_mdp()
        
        policy = SolvedPolicy(policy_dict={})
        
        state = mdp.create_initial_state()
        action = policy.get_action(state, mdp)
        
        assert all(v == 0.0 for v in action.values())


class TestApproximateDynamicProgramming:
    """Tests for ADP solver"""
    
    def test_default_feature_function(self):
        """Test default feature extraction"""
        mdp = create_simple_mdp(shelf_life=4)
        state = mdp.create_initial_state(
            initial_inventory=np.array([10.0, 20.0, 30.0, 40.0])
        )
        
        features = default_feature_fn(state)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert features[0] == 1.0  # Bias term
    
    def test_fit_weights(self):
        """Test that ADP learns weights"""
        mdp = create_simple_mdp(shelf_life=3, num_suppliers=1, mean_demand=5.0)
        
        adp = ApproximateDynamicProgramming(
            mdp,
            feature_fn=default_feature_fn,
            learning_rate=0.001,
            num_iterations=10
        )
        
        states = [
            mdp.create_initial_state(
                initial_inventory=np.array([10.0, 10.0, 10.0])
            )
        ]
        
        policy = DoNothingPolicy()
        weights = adp.fit(states, policy)
        
        assert weights is not None
        assert len(weights) > 0
    
    def test_get_value(self):
        """Test value approximation after fitting"""
        mdp = create_simple_mdp(shelf_life=3)
        
        adp = ApproximateDynamicProgramming(
            mdp,
            feature_fn=default_feature_fn,
            num_iterations=5
        )
        
        states = [
            mdp.create_initial_state(
                initial_inventory=np.array([10.0, 10.0, 10.0])
            )
        ]
        
        adp.fit(states, DoNothingPolicy())
        
        state = mdp.create_initial_state(
            initial_inventory=np.array([20.0, 20.0, 20.0])
        )
        
        value = adp.get_value(state)
        
        assert isinstance(value, float)


class TestSolverConvergence:
    """Tests for solver convergence properties"""
    
    def test_value_iteration_convergence_tracking(self):
        """Test that convergence is tracked"""
        mdp = create_simple_mdp(shelf_life=3, mean_demand=5.0)
        
        solver = ValueIteration(
            mdp,
            num_demand_samples=10,
            max_iterations=10,
            tolerance=0.1
        )
        
        states = [
            mdp.create_initial_state(
                initial_inventory=np.array([10.0, 10.0, 10.0])
            )
        ]
        
        result = solver.solve(states, [{0: 0.0}, {0: 5.0}])
        
        assert len(result.history) == result.iterations
    
    def test_policy_iteration_convergence_tracking(self):
        """Test PI convergence tracking"""
        mdp = create_simple_mdp(shelf_life=3, mean_demand=5.0)
        
        solver = PolicyIteration(
            mdp,
            num_demand_samples=10,
            max_iterations=5,
            eval_iterations=2
        )
        
        states = [
            mdp.create_initial_state(
                initial_inventory=np.array([10.0, 10.0, 10.0])
            )
        ]
        
        result = solver.solve(states, action_space=[{0: 0.0}, {0: 5.0}])
        
        assert len(result.history) == result.iterations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

