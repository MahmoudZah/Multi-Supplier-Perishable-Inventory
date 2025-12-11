"""Verification that mixed-complexity environments can be vectorized."""
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from colab_training.environment_suite import create_environment_suite, build_environment_from_config
from colab_training.gym_env import PerishableInventoryGymWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

def create_env_from_config(config):
    mdp = build_environment_from_config(config)
    env = PerishableInventoryGymWrapper(mdp=mdp)
    env = TimeLimit(env, max_episode_steps=100)
    env = Monitor(env)
    return env

# Create suite
suite = create_environment_suite(seed=42)
print(f"Total environments: {len(suite)}")
print(f"Summary: {suite.get_summary()}")

# Get one environment from each complexity level
complexities = ["simple", "moderate", "complex", "extreme", "ultra"]
configs = []
for c in complexities:
    envs = suite.get_by_complexity(c)
    if envs:
        configs.append(envs[0])
        print(f"{c}: {envs[0].num_suppliers} suppliers, shelf_life={envs[0].shelf_life}")

# Create vectorized environment with all complexity levels
def make_env(config):
    def _init():
        return create_env_from_config(config)
    return _init

env_fns = [make_env(cfg) for cfg in configs]
vec_env = DummyVecEnv(env_fns)

print(f"\nVectorized environment created with {len(configs)} environments")
print(f"Observation space: {vec_env.observation_space}")
print(f"Action space: {vec_env.action_space}")

# Run a few steps
obs = vec_env.reset()
print(f"\nReset successful. Observation shape: {obs.shape}")

for step in range(10):
    actions = [vec_env.action_space.sample() for _ in range(len(configs))]
    actions = np.array(actions)
    obs, rewards, dones, infos = vec_env.step(actions)
    print(f"Step {step+1}: obs_shape={obs.shape}, rewards_shape={rewards.shape}")

print("\n✅ Vectorized training with mixed complexity levels WORKS!")
vec_env.close()
