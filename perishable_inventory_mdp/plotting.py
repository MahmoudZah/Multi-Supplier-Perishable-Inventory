
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional, List

def plot_simulation_trace(
    trace: pd.DataFrame,
    policy_name: str,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plots the simulation trace for inventory, orders, and spoilage.
    Inventory is plotted as a line.
    Orders and Spoilage are plotted as lines for clarity.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(trace.index, trace['inventory'], label='Inventory', color='blue', linewidth=2)
    ax.plot(trace.index, trace['orders'], label='Orders', color='green', linestyle='--', alpha=0.7)
    # Plot spoilage on the same axis but maybe with points or a thin line
    ax.plot(trace.index, trace['spoilage'], label='Spoilage', color='red', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Quantity')
    ax.set_title(f'{policy_name} Simulation Trace')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if ax is None:
        plt.tight_layout()

def plot_paid_cost_comparison(
    results_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plots a comparison of paid costs (purchase + holding) across policies.
    """
    # Calculate costs if not present
    if 'paid_cost' not in results_df.columns:
        results_df['paid_cost'] = results_df['purchase_cost'] + results_df['holding_cost']
    
    # Group by policy and calculate mean
    policy_costs = results_df.groupby('policy')['paid_cost'].mean().sort_values()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    policy_costs.plot(kind='bar', ax=ax, color=sns.color_palette('viridis', len(policy_costs)))
    ax.set_xlabel('Policy', fontsize=12)
    ax.set_ylabel('Mean Paid Cost', fontsize=12)
    ax.set_title('Paid Cost Comparison Across Policies', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    if ax is None:
        plt.tight_layout()

def plot_metric_comparison(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = False,
    complexity_order: List[str] = ['simple', 'moderate', 'complex', 'extreme']
) -> None:
    """
    Plots a grouped bar chart comparing a metric across policies and complexity levels.
    """
    # Pivot data: Index=Policy, Columns=Complexity
    # Aggregate by mean if there are multiple runs per scenario
    pivot_data = df.pivot_table(
        values=metric_col, 
        index='policy', 
        columns='complexity', 
        aggfunc='mean'
    )
    
    # Reorder columns if they exist
    cols = [c for c in complexity_order if c in pivot_data.columns]
    pivot_data = pivot_data[cols]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_data.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=1)
    
    if log_scale:
        ax.set_yscale('log')
        
    ax.set_xlabel('Policy', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Complexity', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, which='both')
    
    # Add labels on bars (optional, might get crowded)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)

    if ax is None:
        plt.tight_layout()
