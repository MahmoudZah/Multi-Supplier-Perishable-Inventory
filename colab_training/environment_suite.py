"""
Environment Suite for RL vs TBS Comparison.

Generates 100+ unique MDP environments with varying complexity levels
for comprehensive benchmarking of RL agents against TBS policies.

Based on recommendations from rl_performance_analysis.md:
- Random cost ratios (1.2x to 5x)
- Random lead time gaps (1 to 5)
- Random demand volatility
- Different shelf lives (3 to 10)
- Stochastic lead times
- Non-stationary demand (seasonality, trends, spikes)
- Capacity constraints
- Multiple suppliers (2-4)
- Crisis dynamics
"""

import numpy as np
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Iterator
from enum import Enum, auto
import json

from perishable_inventory_mdp.environment import (
    PerishableInventoryMDP, 
    EnhancedPerishableInventoryMDP,
    create_simple_mdp,
    create_enhanced_mdp
)
from perishable_inventory_mdp.demand import (
    PoissonDemand, 
    SeasonalDemand,
    TrendDemand,
    DemandSpikeProcess,
    CompositeDemand,
    StochasticLeadTime
)
from perishable_inventory_mdp.costs import CostParameters
from perishable_inventory_mdp.crisis import create_crisis_process


class ComplexityLevel(Enum):
    """Environment complexity categories."""
    SIMPLE = auto()       # Basic TBS-optimal scenario
    MODERATE = auto()     # Some complexity, TBS still competitive
    COMPLEX = auto()      # High complexity, RL should excel
    EXTREME = auto()      # Maximum complexity for stress testing
    ULTRA = auto()        # 10+ suppliers, extreme volatility (200x complex)


@dataclass
class EnvironmentConfig:
    """Configuration for a single environment instance.
    
    Captures all parameters needed to recreate an environment.
    Used for deduplication and serialization.
    """
    # Basic parameters
    shelf_life: int = 5
    mean_demand: float = 10.0
    num_suppliers: int = 2
    
    # Supplier parameters (per supplier)
    lead_times: Tuple[int, ...] = (1, 3)
    unit_costs: Tuple[float, ...] = (2.0, 1.0)
    capacities: Tuple[float, ...] = (100.0, 100.0)
    
    # Demand complexity
    demand_type: str = "stationary"  # stationary, seasonal, trend, spiky, composite
    seasonality_amplitude: float = 0.0
    trend_slope: float = 0.0
    spike_probability: float = 0.0
    spike_multiplier: float = 1.0
    
    # Supply complexity
    stochastic_lead_times: bool = False
    lead_time_variance: float = 0.0
    
    # Crisis dynamics
    enable_crisis: bool = False
    crisis_probability: float = 0.0
    
    # Capacity constraints
    total_capacity_limit: Optional[float] = None
    
    # Cost structure
    holding_cost_base: float = 0.5
    holding_cost_age_premium: float = 0.5
    shortage_cost: float = 10.0
    spoilage_cost: float = 5.0
    
    # Metadata
    complexity: str = "simple"
    env_id: str = ""
    
    def __post_init__(self):
        # Generate unique ID if not set
        if not self.env_id:
            self.env_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID from config hash."""
        # Create deterministic representation
        config_str = json.dumps({
            k: v for k, v in sorted(asdict(self).items()) 
            if k != 'env_id'
        }, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create from dictionary."""
        # Convert tuple fields
        for field_name in ['lead_times', 'unit_costs', 'capacities']:
            if field_name in data and isinstance(data[field_name], list):
                data[field_name] = tuple(data[field_name])
        return cls(**data)


@dataclass
class MultiItemEnvironmentConfig:
    """Configuration for a multi-item environment instance.
    
    Extends EnvironmentConfig with per-item parameters.
    """
    # Item parameters
    num_items: int = 2
    item_shelf_lives: Tuple[int, ...] = (5, 4)
    item_demand_multipliers: Tuple[float, ...] = (1.0, 0.8)
    item_names: Tuple[str, ...] = ("Item A", "Item B")
    
    # Base demand (scaled by multipliers)
    mean_demand: float = 10.0
    
    # Supplier parameters
    num_suppliers: int = 2
    lead_times: Tuple[int, ...] = (1, 3)
    unit_costs: Tuple[float, ...] = (2.0, 1.0)
    capacities: Tuple[float, ...] = (100.0, 100.0)
    supplier_item_coverage: Optional[Dict[int, List[int]]] = None  # None = all items
    
    # Demand complexity
    demand_type: str = "stationary"
    seasonality_amplitude: float = 0.0
    spike_probability: float = 0.0
    spike_multiplier: float = 1.0
    
    # Supply complexity
    stochastic_lead_times: bool = False
    lead_time_variance: float = 0.0
    
    # Crisis dynamics
    enable_crisis: bool = False
    crisis_probability: float = 0.0
    
    # Cost structure
    holding_cost_base: float = 0.5
    holding_cost_age_premium: float = 0.5
    shortage_cost: float = 10.0
    spoilage_cost: float = 5.0
    
    # Metadata
    complexity: str = "multi_item"
    env_id: str = ""
    is_multi_item: bool = True
    
    def __post_init__(self):
        if not self.env_id:
            self.env_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID from config hash."""
        config_dict = {
            'num_items': self.num_items,
            'item_shelf_lives': self.item_shelf_lives,
            'item_demand_multipliers': self.item_demand_multipliers,
            'mean_demand': self.mean_demand,
            'num_suppliers': self.num_suppliers,
            'lead_times': self.lead_times,
            'unit_costs': self.unit_costs,
            'demand_type': self.demand_type,
            'complexity': self.complexity
        }
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return "mi_" + hashlib.md5(config_str.encode()).hexdigest()[:10]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiItemEnvironmentConfig':
        """Create from dictionary."""
        for field_name in ['item_shelf_lives', 'item_demand_multipliers', 'item_names',
                           'lead_times', 'unit_costs', 'capacities']:
            if field_name in data and isinstance(data[field_name], list):
                data[field_name] = tuple(data[field_name])
        return cls(**data)


@dataclass
class EnvironmentSuite:
    """Suite of environments for RL benchmarking.
    
    Generates and manages 100+ unique environment configurations
    categorized by complexity level.
    """
    configs: List[EnvironmentConfig] = field(default_factory=list)
    seed: int = 42
    
    def __post_init__(self):
        self._rng = np.random.RandomState(self.seed)
        self._config_hashes: set = set()
    
    def __len__(self) -> int:
        return len(self.configs)
    
    def __iter__(self) -> Iterator[EnvironmentConfig]:
        return iter(self.configs)
    
    def __getitem__(self, idx: int) -> EnvironmentConfig:
        return self.configs[idx]
    
    def add_config(self, config: EnvironmentConfig) -> bool:
        """Add config if not duplicate. Returns True if added."""
        config_hash = config.env_id
        if config_hash in self._config_hashes:
            return False
        self._config_hashes.add(config_hash)
        self.configs.append(config)
        return True
    
    def get_by_complexity(self, complexity: str) -> List[EnvironmentConfig]:
        """Get all configs of a given complexity level."""
        return [c for c in self.configs if c.complexity == complexity]
    
    def get_summary(self) -> Dict[str, int]:
        """Get count of environments by complexity."""
        summary = {}
        for c in self.configs:
            summary[c.complexity] = summary.get(c.complexity, 0) + 1
        return summary
    
    def build_environment(self, config: EnvironmentConfig) -> PerishableInventoryMDP:
        """Build MDP environment from configuration."""
        return build_environment_from_config(config)
    
    def save(self, filepath: str):
        """Save suite to JSON file."""
        data = {
            'seed': self.seed,
            'configs': [c.to_dict() for c in self.configs]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'EnvironmentSuite':
        """Load suite from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        suite = cls(seed=data['seed'])
        for config_data in data['configs']:
            config = EnvironmentConfig.from_dict(config_data)
            suite.add_config(config)
        return suite


def build_environment_from_config(config):
    """Build MDP environment from configuration.
    
    Creates the appropriate environment type based on config complexity.
    Handles both EnvironmentConfig and MultiItemEnvironmentConfig.
    """
    # Dispatch to multi-item builder if applicable
    if isinstance(config, MultiItemEnvironmentConfig) or getattr(config, 'is_multi_item', False):
        return build_multi_item_environment_from_config(config)
    
    # Build suppliers list
    suppliers = []
    for i in range(config.num_suppliers):
        suppliers.append({
            'id': i,
            'lead_time': config.lead_times[i] if i < len(config.lead_times) else 1,
            'unit_cost': config.unit_costs[i] if i < len(config.unit_costs) else 1.0,
            'capacity': config.capacities[i] if i < len(config.capacities) else 100.0,
            'fixed_cost': 0.0,
            'moq': 1
        })
    
    # Build demand process
    demand_process = _build_demand_process(config)
    
    # Build cost parameters
    cost_params = CostParameters.age_dependent_holding(
        shelf_life=config.shelf_life,
        base_holding=config.holding_cost_base,
        age_premium=config.holding_cost_age_premium,
        shortage_cost=config.shortage_cost,
        spoilage_cost=config.spoilage_cost
    )
    
    # Build stochastic lead times if enabled
    stochastic_lead_times = None
    if config.stochastic_lead_times:
        stochastic_lead_times = {}
        for i, supplier in enumerate(suppliers):
            # Convert variance to advancement probability
            # Higher variance = lower advancement probability
            adv_prob = 1.0 - config.lead_time_variance
            stochastic_lead_times[i] = StochasticLeadTime(
                supplier_id=i,
                advancement_prob=max(0.5, min(1.0, adv_prob))
            )
    
    # Use enhanced MDP if crisis enabled
    if config.enable_crisis:
        crisis_process = create_crisis_process(
            crisis_probability=config.crisis_probability,
            recovery_rate=0.2,
            severity="moderate"
        )
        return EnhancedPerishableInventoryMDP(
            shelf_life=config.shelf_life,
            suppliers=suppliers,
            demand_process=demand_process,
            cost_params=cost_params,
            stochastic_lead_times=stochastic_lead_times,
            crisis_process=crisis_process,
            enable_rejection=True
        )
    else:
        return PerishableInventoryMDP(
            shelf_life=config.shelf_life,
            suppliers=suppliers,
            demand_process=demand_process,
            cost_params=cost_params,
            stochastic_lead_times=stochastic_lead_times
        )


def build_multi_item_environment_from_config(config: MultiItemEnvironmentConfig):
    """Build multi-item MDP from MultiItemEnvironmentConfig.
    
    Returns:
        MultiItemPerishableInventoryMDP
    """
    from perishable_inventory_mdp.environment import MultiItemPerishableInventoryMDP
    from perishable_inventory_mdp.costs import CostParameters
    
    # Build items list
    items = []
    for i in range(config.num_items):
        items.append({
            'id': i,
            'shelf_life': config.item_shelf_lives[i] if i < len(config.item_shelf_lives) else 5,
            'name': config.item_names[i] if i < len(config.item_names) else f"Item_{i}",
            'demand_multiplier': config.item_demand_multipliers[i] if i < len(config.item_demand_multipliers) else 1.0
        })
    
    # Build suppliers list
    suppliers = []
    for i in range(config.num_suppliers):
        supplier = {
            'id': i,
            'lead_time': config.lead_times[i] if i < len(config.lead_times) else 1,
            'unit_cost': config.unit_costs[i] if i < len(config.unit_costs) else 1.0,
            'capacity': config.capacities[i] if i < len(config.capacities) else 100.0,
            'fixed_cost': 0.0,
            'moq': 1
        }
        # Add item coverage if specified
        if config.supplier_item_coverage and i in config.supplier_item_coverage:
            supplier['items_supplied'] = config.supplier_item_coverage[i]
        suppliers.append(supplier)
    
    # Build demand process (base demand, will be scaled by item multipliers)
    demand_process = _build_demand_process_multi_item(config)
    
    # Build cost parameters using max shelf life
    max_shelf_life = max(config.item_shelf_lives)
    cost_params = CostParameters.age_dependent_holding(
        shelf_life=max_shelf_life,
        base_holding=config.holding_cost_base,
        age_premium=config.holding_cost_age_premium,
        shortage_cost=config.shortage_cost,
        spoilage_cost=config.spoilage_cost
    )
    
    return MultiItemPerishableInventoryMDP(
        items=items,
        suppliers=suppliers,
        demand_process=demand_process,
        cost_params=cost_params
    )


def _build_demand_process_multi_item(config: MultiItemEnvironmentConfig):
    """Build demand process for multi-item config."""
    if config.demand_type == "stationary":
        return PoissonDemand(config.mean_demand)
    elif config.demand_type == "seasonal":
        return SeasonalDemand(
            base_rate=config.mean_demand,
            amplitude=config.seasonality_amplitude,
            period=50
        )
    elif config.demand_type == "composite":
        return CompositeDemand(
            base_rate=config.mean_demand,
            seasonality_amplitude=config.seasonality_amplitude,
            spike_prob=config.spike_probability,
            spike_multiplier=config.spike_multiplier
        )
    else:
        return PoissonDemand(config.mean_demand)


def _build_demand_process(config: EnvironmentConfig):
    """Build demand process based on configuration."""
    if config.demand_type == "stationary":
        return PoissonDemand(config.mean_demand)
    
    elif config.demand_type == "seasonal":
        return SeasonalDemand(
            base_rate=config.mean_demand,
            amplitude=config.seasonality_amplitude,
            period=50
        )
    
    elif config.demand_type == "trend":
        return TrendDemand(
            base_rate=config.mean_demand,
            trend_rate=config.trend_slope,
            min_rate=1.0,
            max_rate=config.mean_demand * 3
        )
    
    elif config.demand_type == "spiky":
        return DemandSpikeProcess(
            base_process=PoissonDemand(config.mean_demand),
            spike_prob=config.spike_probability,
            spike_multiplier=config.spike_multiplier
        )
    
    elif config.demand_type == "composite":
        return CompositeDemand(
            base_rate=config.mean_demand,
            seasonality_amplitude=config.seasonality_amplitude,
            spike_prob=config.spike_probability,
            spike_multiplier=config.spike_multiplier
        )
    
    else:
        return PoissonDemand(config.mean_demand)


def generate_simple_environments(
    rng: np.random.RandomState,
    count: int = 20
) -> List[EnvironmentConfig]:
    """Generate simple environments where TBS is near-optimal.
    
    Characteristics:
    - Deterministic lead times
    - Stationary demand
    - 2 suppliers
    - Standard cost structure
    """
    configs = []
    
    for _ in range(count):
        # Random but simple parameters
        shelf_life = rng.choice([4, 5, 6])
        mean_demand = rng.uniform(8, 15)
        fast_lt = 1
        slow_lt = rng.choice([2, 3, 4])
        cost_ratio = rng.uniform(1.5, 2.5)
        slow_cost = rng.uniform(0.8, 1.2)
        fast_cost = slow_cost * cost_ratio
        
        config = EnvironmentConfig(
            shelf_life=shelf_life,
            mean_demand=mean_demand,
            num_suppliers=2,
            lead_times=(fast_lt, slow_lt),
            unit_costs=(fast_cost, slow_cost),
            capacities=(100.0, 100.0),
            demand_type="stationary",
            complexity="simple"
        )
        configs.append(config)
    
    return configs


def generate_moderate_environments(
    rng: np.random.RandomState,
    count: int = 30
) -> List[EnvironmentConfig]:
    """Generate moderate complexity environments.
    
    Characteristics:
    - Some seasonality or demand variation
    - Possibly stochastic lead times
    - 2 suppliers
    - Varied cost ratios
    """
    configs = []
    
    for _ in range(count):
        shelf_life = rng.choice([3, 4, 5, 6, 7])
        mean_demand = rng.uniform(5, 20)
        fast_lt = rng.choice([1, 2])
        slow_lt = fast_lt + rng.choice([1, 2, 3])
        cost_ratio = rng.uniform(1.2, 3.0)
        slow_cost = rng.uniform(0.5, 1.5)
        fast_cost = slow_cost * cost_ratio
        
        # Add some complexity
        demand_type = rng.choice(["stationary", "seasonal", "spiky"])
        seasonality = rng.uniform(0.1, 0.3) if demand_type == "seasonal" else 0.0
        spike_prob = rng.uniform(0.02, 0.08) if demand_type == "spiky" else 0.0
        spike_mult = rng.uniform(1.5, 2.5) if demand_type == "spiky" else 1.0
        
        stochastic_lt = rng.random() < 0.3
        lt_variance = rng.uniform(0.1, 0.2) if stochastic_lt else 0.0
        
        config = EnvironmentConfig(
            shelf_life=shelf_life,
            mean_demand=mean_demand,
            num_suppliers=2,
            lead_times=(fast_lt, slow_lt),
            unit_costs=(fast_cost, slow_cost),
            capacities=(100.0, 100.0),
            demand_type=demand_type,
            seasonality_amplitude=seasonality,
            spike_probability=spike_prob,
            spike_multiplier=spike_mult,
            stochastic_lead_times=stochastic_lt,
            lead_time_variance=lt_variance,
            complexity="moderate"
        )
        configs.append(config)
    
    return configs


def generate_complex_environments(
    rng: np.random.RandomState,
    count: int = 35
) -> List[EnvironmentConfig]:
    """Generate complex environments where RL should excel.
    
    Characteristics:
    - Composite demand (seasonality + spikes)
    - Stochastic lead times
    - 2-3 suppliers
    - Crisis dynamics (optional)
    - Varied parameters
    """
    configs = []
    
    for _ in range(count):
        shelf_life = rng.choice([3, 4, 5, 6, 7, 8])
        mean_demand = rng.uniform(5, 25)
        num_suppliers = rng.choice([2, 3])
        
        # Generate supplier parameters
        lead_times = []
        unit_costs = []
        capacities = []
        
        base_cost = rng.uniform(0.5, 1.5)
        for i in range(num_suppliers):
            lt = i + rng.choice([1, 2])
            lead_times.append(lt)
            # Cost inversely related to lead time
            cost_mult = 1.0 + (num_suppliers - 1 - i) * rng.uniform(0.3, 0.8)
            unit_costs.append(base_cost * cost_mult)
            capacities.append(rng.choice([50.0, 75.0, 100.0]))
        
        # Complex demand
        demand_type = rng.choice(["seasonal", "composite", "spiky"])
        seasonality = rng.uniform(0.15, 0.4)
        spike_prob = rng.uniform(0.03, 0.1)
        spike_mult = rng.uniform(1.5, 3.0)
        
        # Stochastic lead times
        stochastic_lt = rng.random() < 0.6
        lt_variance = rng.uniform(0.1, 0.3) if stochastic_lt else 0.0
        
        # Crisis dynamics
        enable_crisis = rng.random() < 0.4
        crisis_prob = rng.uniform(0.02, 0.08) if enable_crisis else 0.0
        
        config = EnvironmentConfig(
            shelf_life=shelf_life,
            mean_demand=mean_demand,
            num_suppliers=num_suppliers,
            lead_times=tuple(lead_times),
            unit_costs=tuple(unit_costs),
            capacities=tuple(capacities),
            demand_type=demand_type,
            seasonality_amplitude=seasonality,
            spike_probability=spike_prob,
            spike_multiplier=spike_mult,
            stochastic_lead_times=stochastic_lt,
            lead_time_variance=lt_variance,
            enable_crisis=enable_crisis,
            crisis_probability=crisis_prob,
            complexity="complex"
        )
        configs.append(config)
    
    return configs


def generate_extreme_environments(
    rng: np.random.RandomState,
    count: int = 20
) -> List[EnvironmentConfig]:
    """Generate extreme complexity environments for stress testing.
    
    Characteristics:
    - 3-4 suppliers
    - Composite demand with high variability
    - Crisis dynamics
    - Stochastic lead times
    - Tight capacity constraints
    - Extreme parameter values
    """
    configs = []
    
    for _ in range(count):
        shelf_life = rng.choice([3, 4, 5, 10])  # Short or long
        mean_demand = rng.uniform(10, 40)
        num_suppliers = rng.choice([3, 4])
        
        # Generate supplier parameters
        lead_times = []
        unit_costs = []
        capacities = []
        
        base_cost = rng.uniform(0.3, 2.0)
        for i in range(num_suppliers):
            lt = 1 + i + rng.choice([0, 1])
            lead_times.append(lt)
            cost_mult = 1.0 + (num_suppliers - 1 - i) * rng.uniform(0.5, 1.5)
            unit_costs.append(base_cost * cost_mult)
            # Tighter capacity
            capacities.append(rng.uniform(30.0, 70.0))
        
        # High demand variability
        seasonality = rng.uniform(0.25, 0.5)
        spike_prob = rng.uniform(0.05, 0.15)
        spike_mult = rng.uniform(2.0, 4.0)
        
        # Always stochastic lead times
        lt_variance = rng.uniform(0.2, 0.4)
        
        # Crisis often enabled
        enable_crisis = rng.random() < 0.7
        crisis_prob = rng.uniform(0.05, 0.15) if enable_crisis else 0.0
        
        config = EnvironmentConfig(
            shelf_life=shelf_life,
            mean_demand=mean_demand,
            num_suppliers=num_suppliers,
            lead_times=tuple(lead_times),
            unit_costs=tuple(unit_costs),
            capacities=tuple(capacities),
            demand_type="composite",
            seasonality_amplitude=seasonality,
            spike_probability=spike_prob,
            spike_multiplier=spike_mult,
            stochastic_lead_times=True,
            lead_time_variance=lt_variance,
            enable_crisis=enable_crisis,
            crisis_probability=crisis_prob,
            shortage_cost=rng.uniform(5.0, 20.0),
            spoilage_cost=rng.uniform(3.0, 10.0),
            complexity="extreme"
        )
        configs.append(config)
    
    return configs


def generate_ultra_environments(
    rng: np.random.RandomState,
    count: int = 10
) -> List[EnvironmentConfig]:
    """Generate ULTRA complexity environments.
    
    Characteristics:
    - 10-15 suppliers
    - Extreme demand volatility (composite + trends + spikes)
    - High probability of crisis
    - Tight constraints and MOQs
    - "200x more complex" than extreme
    """
    configs = []
    
    for _ in range(count):
        shelf_life = rng.choice([5, 10, 15])
        mean_demand = rng.uniform(50, 200)  # Much higher demand
        num_suppliers = rng.choice([10, 11, 12, 13, 14, 15])
        
        # Generate supplier parameters
        lead_times = []
        unit_costs = []
        capacities = []
        
        base_cost = rng.uniform(0.5, 5.0)
        
        for i in range(num_suppliers):
            # Highly varied lead times (1 to 10)
            lt = rng.randint(1, 11)
            lead_times.append(lt)
            
            # Cost structure: some cheap/slow, some fast/expensive, some random
            if i < 3: # Cheap base suppliers
                cost_mult = rng.uniform(0.5, 0.8)
            elif i > num_suppliers - 3: # Expensive emergency suppliers
                cost_mult = rng.uniform(2.0, 5.0)
            else:
                cost_mult = rng.uniform(0.8, 2.0)
                
            unit_costs.append(base_cost * cost_mult)
            
            # Tight capacities relative to demand
            # Total capacity should be just enough to cover demand + safety
            # Individual capacities small
            cap = rng.uniform(mean_demand * 0.1, mean_demand * 0.3)
            capacities.append(cap)
        
        # Extreme demand variability
        seasonality = rng.uniform(0.4, 0.8)
        spike_prob = rng.uniform(0.1, 0.25)
        spike_mult = rng.uniform(5.0, 10.0) # Massive spikes
        trend_slope = rng.uniform(-0.05, 0.05) # Significant trends
        
        # Always stochastic lead times with high variance
        lt_variance = rng.uniform(0.3, 0.6)
        
        # Crisis almost always enabled
        enable_crisis = rng.random() < 0.9
        crisis_prob = rng.uniform(0.1, 0.3) if enable_crisis else 0.0
        
        config = EnvironmentConfig(
            shelf_life=shelf_life,
            mean_demand=mean_demand,
            num_suppliers=num_suppliers,
            lead_times=tuple(lead_times),
            unit_costs=tuple(unit_costs),
            capacities=tuple(capacities),
            demand_type="composite",
            seasonality_amplitude=seasonality,
            trend_slope=trend_slope,
            spike_probability=spike_prob,
            spike_multiplier=spike_mult,
            stochastic_lead_times=True,
            lead_time_variance=lt_variance,
            enable_crisis=enable_crisis,
            crisis_probability=crisis_prob,
            shortage_cost=rng.uniform(20.0, 100.0), # High penalties
            spoilage_cost=rng.uniform(10.0, 50.0),
            complexity="ultra"
        )
        configs.append(config)
    
    return configs


def generate_multi_item_environments(
    rng: np.random.RandomState,
    count: int = 15
) -> List[MultiItemEnvironmentConfig]:
    """Generate multi-item environments with varying complexity.
    
    Characteristics:
    - 2-5 items with different shelf lives
    - 2-4 suppliers
    - Varying demand multipliers per item
    - Optional supplier-item coverage restrictions
    """
    configs = []
    
    for _ in range(count):
        num_items = rng.choice([2, 3, 4, 5])
        num_suppliers = rng.choice([2, 3, 4])
        
        # Generate item parameters
        item_shelf_lives = tuple(rng.choice([3, 4, 5, 6, 7]) for _ in range(num_items))
        item_demand_multipliers = tuple(rng.uniform(0.5, 1.5) for _ in range(num_items))
        item_names = tuple(f"Item_{chr(65+i)}" for i in range(num_items))  # Item_A, Item_B, ...
        
        # Generate supplier parameters
        base_cost = rng.uniform(0.5, 2.0)
        lead_times = []
        unit_costs = []
        capacities = []
        
        for i in range(num_suppliers):
            lt = 1 + i + rng.choice([0, 1])
            lead_times.append(lt)
            cost_mult = 1.0 + (num_suppliers - 1 - i) * rng.uniform(0.3, 0.8)
            unit_costs.append(base_cost * cost_mult)
            capacities.append(rng.choice([50.0, 75.0, 100.0]))
        
        # Random demand type
        demand_type = rng.choice(["stationary", "seasonal", "composite"])
        seasonality = rng.uniform(0.1, 0.3) if demand_type in ["seasonal", "composite"] else 0.0
        spike_prob = rng.uniform(0.02, 0.08) if demand_type == "composite" else 0.0
        spike_mult = rng.uniform(1.5, 2.5) if demand_type == "composite" else 1.0
        
        # Optional supplier-item coverage (30% chance of restriction)
        supplier_item_coverage = None
        if rng.random() < 0.3 and num_suppliers >= 2 and num_items >= 2:
            supplier_item_coverage = {}
            for sid in range(num_suppliers):
                # Each supplier covers 50-100% of items
                num_covered = rng.randint(num_items // 2 + 1, num_items + 1)
                covered_items = list(rng.choice(num_items, size=num_covered, replace=False))
                supplier_item_coverage[sid] = covered_items
        
        config = MultiItemEnvironmentConfig(
            num_items=num_items,
            item_shelf_lives=item_shelf_lives,
            item_demand_multipliers=item_demand_multipliers,
            item_names=item_names,
            mean_demand=rng.uniform(8, 20),
            num_suppliers=num_suppliers,
            lead_times=tuple(lead_times),
            unit_costs=tuple(unit_costs),
            capacities=tuple(capacities),
            supplier_item_coverage=supplier_item_coverage,
            demand_type=demand_type,
            seasonality_amplitude=seasonality,
            spike_probability=spike_prob,
            spike_multiplier=spike_mult,
            complexity="multi_item"
        )
        configs.append(config)
    
    return configs


def create_environment_suite(
    seed: int = 42,
    simple_count: int = 20,
    moderate_count: int = 30,
    complex_count: int = 35,
    extreme_count: int = 20,
    ultra_count: int = 10,
    multi_item_count: int = 15
) -> EnvironmentSuite:
    """Create a complete environment suite with 100+ unique environments.
    
    Args:
        seed: Random seed for reproducibility
        simple_count: Number of simple environments
        moderate_count: Number of moderate environments
        complex_count: Number of complex environments
        extreme_count: Number of extreme environments
        ultra_count: Number of ultra environments
        multi_item_count: Number of multi-item environments
    
    Returns:
        EnvironmentSuite with unique configurations
    """
    suite = EnvironmentSuite(seed=seed)
    rng = np.random.RandomState(seed)
    
    # Generate each complexity level
    simple_configs = generate_simple_environments(rng, simple_count)
    moderate_configs = generate_moderate_environments(rng, moderate_count)
    complex_configs = generate_complex_environments(rng, complex_count)
    extreme_configs = generate_extreme_environments(rng, extreme_count)
    ultra_configs = generate_ultra_environments(rng, ultra_count)
    multi_item_configs = generate_multi_item_environments(rng, multi_item_count)
    
    # Add to suite (duplicates are automatically filtered)
    for config in simple_configs:
        suite.add_config(config)
    for config in moderate_configs:
        suite.add_config(config)
    for config in complex_configs:
        suite.add_config(config)
    for config in extreme_configs:
        suite.add_config(config)
    for config in ultra_configs:
        suite.add_config(config)
    for config in multi_item_configs:
        suite.add_config(config)
    
    return suite


def get_canonical_suite() -> EnvironmentSuite:
    """Get the canonical 120-environment suite for benchmarking.
    
    Returns a reproducible suite with:
    - 20 simple environments
    - 30 moderate environments
    - 35 complex environments
    - 20 extreme environments
    - 10 ultra environments
    - 15 multi-item environments
    """
    return create_environment_suite(seed=42)


# Convenience function for quick iteration
def iter_environments(
    complexity: Optional[str] = None,
    suite: Optional[EnvironmentSuite] = None
) -> Iterator[Tuple[EnvironmentConfig, PerishableInventoryMDP]]:
    """Iterate over environment configurations and their built MDPs.
    
    Args:
        complexity: Optional filter by complexity level
        suite: Optional custom suite (uses canonical if None)
    
    Yields:
        Tuple of (EnvironmentConfig, PerishableInventoryMDP)
    """
    if suite is None:
        suite = get_canonical_suite()
    
    configs = suite.get_by_complexity(complexity) if complexity else suite.configs
    
    for config in configs:
        mdp = build_environment_from_config(config)
        yield config, mdp
