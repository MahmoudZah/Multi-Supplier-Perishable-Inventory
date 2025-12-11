import numpy as np
import pytest
from colab_training.environment_suite import generate_ultra_environments, build_environment_from_config

def test_ultra_environment_properties():
    rng = np.random.RandomState(42)
    configs = generate_ultra_environments(rng, count=1)
    config = configs[0]
    
    print(f"Generated Ultra Config: {config.env_id}")
    print(f"Suppliers: {config.num_suppliers}")
    print(f"Demand: {config.mean_demand}")
    print(f"Crisis Prob: {config.crisis_probability}")
    
    assert config.num_suppliers >= 10, "Ultra environment should have at least 10 suppliers"
    assert config.complexity == "ultra", "Complexity should be 'ultra'"
    assert config.demand_type == "composite", "Demand type should be composite"
    
    # Build MDP
    mdp = build_environment_from_config(config)
    assert len(mdp.suppliers) == config.num_suppliers
    
    # Run simulation
    state = mdp.reset(seed=42)
    print("Environment reset successful")
    
    for t in range(10):
        # Random action
        action = {}
        for s in mdp.suppliers:
            # Order random amount up to capacity, rounded to integer for MOQ
            qty = round(np.random.uniform(0, s['capacity']))
            action[s['id']] = qty
            
        result = mdp.step(state, action)
        state = result.next_state
        print(f"Step {t+1}: Demand={result.demand_realized:.2f}, Sales={result.sales:.2f}, Cost={result.costs.total_cost:.2f}")
        
    print("Simulation successful")

if __name__ == "__main__":
    test_ultra_environment_properties()
