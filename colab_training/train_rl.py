"""
RL Training Script for Perishable Inventory MDP.

Loads configuration, sets up the environment and wrapper,
and trains a PPO agent using Stable Baselines3.
"""

import json
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perishable_inventory_mdp.environment import create_simple_mdp
from perishable_inventory_mdp.costs import CostParameters
from perishable_inventory_mdp.demand import PoissonDemand
from colab_training.gym_wrapper import PerishableInventoryGymWrapper

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return json.load(f)

def make_env(mdp_params):
    """Factory function to create wrapped environment."""
    def _init():
        # Reconstruct MDP from params
        # Note: This is a simplified reconstruction. 
        # For full flexibility, you might need a more robust factory.
        
        # Extract params
        shelf_life = mdp_params['shelf_life']
        suppliers = mdp_params['suppliers']
        
        demand_params = mdp_params['demand']
        if demand_params['type'] == 'poisson':
            demand_process = PoissonDemand(demand_params['mean'])
        else:
            raise ValueError(f"Unsupported demand type: {demand_params['type']}")
            
        cost_params_dict = mdp_params['costs']
        cost_params = CostParameters.uniform_holding(
            shelf_life=shelf_life,
            holding_cost=cost_params_dict['holding'],
            shortage_cost=cost_params_dict['shortage'],
            spoilage_cost=cost_params_dict['spoilage'],
            discount_factor=cost_params_dict['discount_factor']
        )
        
        # Create MDP
        # We use the class directly instead of create_simple_mdp to support custom params
        from perishable_inventory_mdp.environment import PerishableInventoryMDP
        mdp = PerishableInventoryMDP(
            shelf_life=shelf_life,
            suppliers=suppliers,
            demand_process=demand_process,
            cost_params=cost_params
        )
        
        # Wrap it
        env = PerishableInventoryGymWrapper(mdp)
        return env
    return _init

def train():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = load_config(config_path)
    
    mdp_params = config['mdp_params']
    train_params = config['training_params']
    eval_params = config['eval_params']
    
    # Create Vectorized Environment
    n_envs = train_params.get('n_envs', 1)
    env = make_vec_env(
        make_env(mdp_params),
        n_envs=n_envs,
        seed=train_params['seed'],
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None
    )
    
    # Create Evaluation Environment (separate seed)
    eval_env = make_vec_env(
        make_env(mdp_params),
        n_envs=1,
        seed=train_params['seed'] + 1000
    )
    
    # Setup Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/results',
        eval_freq=eval_params['eval_freq'],
        n_eval_episodes=eval_params['n_eval_episodes'],
        deterministic=eval_params['deterministic'],
        render=False
    )
    
    # Initialize Model
    algo = train_params['algorithm']
    if algo == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=train_params['learning_rate'],
            tensorboard_log="./logs/tensorboard/"
        )
    elif algo == 'SAC':
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=train_params['learning_rate'],
            tensorboard_log="./logs/tensorboard/"
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
        
    print(f"Starting training with {algo} for {train_params['total_timesteps']} timesteps...")
    
    # Train
    model.learn(
        total_timesteps=train_params['total_timesteps'],
        callback=eval_callback
    )
    
    # Save final model
    model.save("final_model")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
