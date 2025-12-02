"""
Gymnasium Wrapper for Perishable Inventory MDP.

Converts the custom InventoryEnvironment into a standard Gymnasium environment
compatible with RL libraries like Stable Baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple

from perishable_inventory_mdp.environment import PerishableInventoryMDP
from perishable_inventory_mdp.state import InventoryState

class PerishableInventoryGymWrapper(gym.Env):
    """
    Gymnasium wrapper for PerishableInventoryMDP.
    
    Observation Space:
        Box(low=0, high=inf, shape=(N + S*L_s + 1 + E,))
        - Inventory buckets (N)
        - Pipeline quantities (sum of lead times across suppliers)
        - Backorders (1)
        - Exogenous state (E)
        
    Action Space:
        Box(low=0, high=1, shape=(S,))
        - Normalized order quantity for each supplier.
        - Scaled by supplier capacity (or a default max order).
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, mdp: PerishableInventoryMDP, max_order_qty: float = 100.0):
        super().__init__()
        self.mdp = mdp
        self.max_order_qty = max_order_qty
        
        # --- Define Action Space ---
        # One continuous value [0, 1] per supplier
        self.num_suppliers = len(mdp.suppliers)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_suppliers,),
            dtype=np.float32
        )
        
        # --- Define Observation Space ---
        # 1. Inventory buckets (shelf_life)
        # 2. Pipeline queues (sum of lead times)
        # 3. Backorders (1)
        # 4. Exogenous state (depends on demand process)
        
        self.shelf_life = mdp.shelf_life
        
        # Calculate total pipeline size
        self.pipeline_size = sum(s['lead_time'] for s in mdp.suppliers)
        
        # Exogenous state size (assume 1 for simple cases, e.g. time/seasonality)
        # Ideally, we'd query the demand process, but for now we'll infer or fix it.
        # Let's assume size 1 (time index or demand factor)
        self.exog_size = 1 
        
        total_obs_size = self.shelf_life + self.pipeline_size + 1 + self.exog_size
        
        self.observation_space = spaces.Box(
            low=-float('inf'), # Exogenous could be negative? Usually inventory >= 0
            high=float('inf'),
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        # Internal state tracking
        self.current_state: Optional[InventoryState] = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_state = self.mdp.reset(seed=seed, options=options)
        
        return self._get_observation(self.current_state), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step."""
        if self.current_state is None:
            raise RuntimeError("Call reset() before step()")
        
        # Convert normalized action to dictionary {supplier_id: qty}
        mdp_action = self._decode_action(action)
        
        # Execute MDP step
        result = self.mdp.step(self.current_state, mdp_action)
        
        # Update internal state
        self.current_state = result.next_state
        
        # Get observation
        obs = self._get_observation(self.current_state)
        
        # Get reward (negative cost)
        # Normalize reward? For now, raw reward.
        reward = result.reward
        
        # Done condition (infinite horizon, so usually False unless time limit)
        # We'll let the TimeLimit wrapper handle truncation
        terminated = False
        truncated = False
        
        info = {
            "demand": result.demand_realized,
            "sales": result.sales,
            "spoilage": result.spoiled,
            "total_cost": result.costs.total_cost
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, state: InventoryState) -> np.ndarray:
        """Flatten state into observation vector."""
        obs_parts = []
        
        # 1. Inventory buckets
        obs_parts.append(state.inventory)
        
        # 2. Pipelines
        # Ensure consistent order of suppliers
        sorted_suppliers = sorted(state.pipelines.items())
        for _, pipeline in sorted_suppliers:
            obs_parts.append(pipeline.pipeline)
            # Note: We are ignoring 'scheduled' for now to keep it simple, 
            # or we should add it if used.
            
        # 3. Backorders
        obs_parts.append([state.backorders])
        
        # 4. Exogenous state
        if state.exogenous_state is not None:
            obs_parts.append(state.exogenous_state)
        else:
            obs_parts.append([0.0]) # Default placeholder
            
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _decode_action(self, action: np.ndarray) -> Dict[int, float]:
        """Convert normalized action array to MDP action dict."""
        mdp_action = {}
        
        # Ensure action is clipped to [0, 1]
        action = np.clip(action, 0.0, 1.0)
        
        for i, supplier in enumerate(self.mdp.suppliers):
            supplier_id = supplier['id']
            capacity = supplier.get('capacity', self.max_order_qty)
            moq = supplier.get('moq', 1)
            
            # Scale to capacity
            raw_qty = action[i] * capacity
            
            # Round to MOQ
            if moq > 0:
                if raw_qty < moq / 2: # Round down to 0 if small
                    qty = 0.0
                else:
                    qty = np.round(raw_qty / moq) * moq
            else:
                qty = raw_qty
                
            mdp_action[supplier_id] = qty
            
        return mdp_action
