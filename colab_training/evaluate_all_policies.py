"""
Comprehensive Policy Evaluation Script.

Evaluates RL model against ALL baseline policies:
- TBS (Tailored Base-Surge)
- BaseStock
- DoNothing
- PIL (Projected Inventory Level)
- DIP (Dual-Index Policy)
- PEIP (Projected Effective Inventory Position)
- VectorBS (Vector Base-Stock)

Usage:
    python evaluate_all_policies.py --model_path logs/best_model/best_model.zip

This script provides comprehensive comparison of trained RL against all baselines.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Project imports
from colab_training.gym_env import (
    PerishableInventoryGymWrapper,
    RewardConfig,
)
from colab_training.environment_suite import (
    get_canonical_suite,
    build_environment_from_config,
)
from colab_training.benchmark import (
    evaluate_policy,
    get_tbs_policy_for_env,
    get_basestock_policy_for_env,
    get_pil_policy_for_env,
    get_dip_policy_for_env,
    get_peip_policy_for_env,
    get_vector_bs_policy_for_env,
    get_all_policies_for_env,
    generate_performance_report,
    visualize_comparison,
    ComparisonReport,
    AVAILABLE_BASELINES
)
from perishable_inventory_mdp.policies import (
    DoNothingPolicy,
    ProjectedInventoryLevelPolicy,
    DualIndexPolicy,
    ProjectedEffectiveInventoryPolicy,
    VectorBaseStockPolicy
)


def create_env_from_config(config, reward_config, episode_length=500):
    """Create gym environment from config."""
    mdp = build_environment_from_config(config)
    env = PerishableInventoryGymWrapper(mdp=mdp, reward_config=reward_config)
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = Monitor(env)
    return env


def evaluate_all_policies(
    model_path: str,
    n_eval: int = 5,
    n_envs_per_complexity: int = 5,
    output_path: Optional[str] = None,
    include_policies: Optional[List[str]] = None
):
    """
    Evaluate RL model against all baseline policies.
    
    Args:
        model_path: Path to trained model zip file
        n_eval: Number of episodes per environment
        n_envs_per_complexity: Number of environments to evaluate per complexity level
        output_path: Optional path to save results
        include_policies: List of policies to include (default: all)
    """
    # Load model
    print(f"📦 Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Load suite
    suite = get_canonical_suite()
    print(f"📊 Loaded {len(suite)} environments")
    
    # Reward config
    reward_config = RewardConfig()
    
    # Initialize report
    report = ComparisonReport()
    
    # Policy selection
    if include_policies is None:
        include_policies = AVAILABLE_BASELINES
    
    # All complexity levels
    complexities = ["simple", "moderate", "complex", "extreme", "ultra", "multi_item"]
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE POLICY EVALUATION")
    print("=" * 70)
    print(f"Policies: RL + {', '.join(include_policies)}")
    print("=" * 70)
    
    for complexity in complexities:
        configs = suite.get_by_complexity(complexity)
        n_envs = min(n_envs_per_complexity, len(configs))
        
        if n_envs == 0:
            print(f"\n⏭️ Skipping {complexity.upper()} (no environments)")
            continue
        
        print(f"\n🔍 Evaluating {complexity.upper()} environments ({n_envs} samples)...")
        
        for i, config in enumerate(configs[:n_envs]):
            # Skip multi-item configs for now
            if hasattr(config, 'is_multi_item') and config.is_multi_item:
                print(f"   ⏭️ Skipping multi-item env (gym wrapper not yet supported)")
                continue
            
            try:
                env = create_env_from_config(config, reward_config, 500)
            except Exception as e:
                print(f"   ❌ Failed to create env {config.env_id}: {e}")
                continue
            
            env_id = config.env_id
            rl_cost = "N/A"
            
            # Evaluate RL
            try:
                rl_result = evaluate_policy(
                    policy=model,
                    env=env,
                    n_episodes=n_eval,
                    max_steps=500,
                    policy_name="RL",
                    env_id=env_id,
                    complexity=complexity
                )
                report.add_result(rl_result)
                rl_cost = f"{rl_result.mean_cost:.1f}"
            except Exception as e:
                print(f"   ❌ RL failed: {e}")
            
            # Evaluate TBS
            if "TBS" in include_policies:
                try:
                    tbs = get_tbs_policy_for_env(env)
                    tbs_result = evaluate_policy(
                        policy=tbs, env=env, n_episodes=n_eval, max_steps=500,
                        policy_name="TBS", env_id=env_id, complexity=complexity
                    )
                    report.add_result(tbs_result)
                except Exception as e:
                    pass  # Silent fail for policies that don't apply
            
            # Evaluate BaseStock
            if "BaseStock" in include_policies:
                try:
                    bs = get_basestock_policy_for_env(env)
                    bs_result = evaluate_policy(
                        policy=bs, env=env, n_episodes=n_eval, max_steps=500,
                        policy_name="BaseStock", env_id=env_id, complexity=complexity
                    )
                    report.add_result(bs_result)
                except Exception as e:
                    pass
            
            # Evaluate DoNothing
            if "DoNothing" in include_policies:
                try:
                    dn = DoNothingPolicy()
                    dn_result = evaluate_policy(
                        policy=dn, env=env, n_episodes=n_eval, max_steps=500,
                        policy_name="DoNothing", env_id=env_id, complexity=complexity
                    )
                    report.add_result(dn_result)
                except Exception as e:
                    pass
            
            # Evaluate PIL
            if "PIL" in include_policies:
                try:
                    pil = get_pil_policy_for_env(env)
                    pil_result = evaluate_policy(
                        policy=pil, env=env, n_episodes=n_eval, max_steps=500,
                        policy_name="PIL", env_id=env_id, complexity=complexity
                    )
                    report.add_result(pil_result)
                except Exception as e:
                    pass
            
            # Evaluate DIP
            if "DIP" in include_policies:
                try:
                    dip = get_dip_policy_for_env(env)
                    dip_result = evaluate_policy(
                        policy=dip, env=env, n_episodes=n_eval, max_steps=500,
                        policy_name="DIP", env_id=env_id, complexity=complexity
                    )
                    report.add_result(dip_result)
                except Exception as e:
                    pass
            
            # Evaluate PEIP
            if "PEIP" in include_policies:
                try:
                    peip = get_peip_policy_for_env(env)
                    peip_result = evaluate_policy(
                        policy=peip, env=env, n_episodes=n_eval, max_steps=500,
                        policy_name="PEIP", env_id=env_id, complexity=complexity
                    )
                    report.add_result(peip_result)
                except Exception as e:
                    pass
            
            # Evaluate VectorBS
            if "VectorBS" in include_policies:
                try:
                    vbs = get_vector_bs_policy_for_env(env)
                    vbs_result = evaluate_policy(
                        policy=vbs, env=env, n_episodes=n_eval, max_steps=500,
                        policy_name="VectorBS", env_id=env_id, complexity=complexity
                    )
                    report.add_result(vbs_result)
                except Exception as e:
                    pass
            
            env.close()
            print(f"   [{i+1}/{n_envs}] {env_id}: RL cost={rl_cost}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    
    # Generate report
    print("\n" + generate_performance_report(report))
    
    # Save results
    if output_path:
        report.save(output_path)
        print(f"\n💾 Results saved to {output_path}")
    
    # Generate visualization
    try:
        if output_path:
            viz_path = str(Path(output_path).parent / "comparison_all_policies.png")
        else:
            viz_path = None
        fig = visualize_comparison(report, save_path=viz_path)
        print("📈 Visualization generated")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("📋 SUMMARY BY POLICY")
    print("=" * 70)
    df = report.to_dataframe()
    if not df.empty:
        summary = df.groupby('policy_name').agg({
            'mean_cost': ['mean', 'std'],
            'mean_fill_rate': 'mean'
        }).round(2)
        print(summary)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL model against ALL policies")
    parser.add_argument("--model_path", type=str, default="logs/best_model/best_model.zip",
                        help="Path to trained model")
    parser.add_argument("--n_eval", type=int, default=5,
                        help="Episodes per environment")
    parser.add_argument("--n_envs", type=int, default=5,
                        help="Environments per complexity level")
    parser.add_argument("--output", type=str, default="logs/evaluation_all_policies.json",
                        help="Path to save results")
    parser.add_argument("--policies", type=str, nargs="+", 
                        default=["TBS", "BaseStock", "DoNothing", "PIL", "DIP", "PEIP", "VectorBS"],
                        help="Policies to include")
    
    args = parser.parse_args()
    
    evaluate_all_policies(
        model_path=args.model_path,
        n_eval=args.n_eval,
        n_envs_per_complexity=args.n_envs,
        output_path=args.output,
        include_policies=args.policies
    )
