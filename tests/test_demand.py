"""
Tests for the Demand Process module.

Tests stochastic demand models including Poisson,
Negative Binomial, and seasonal demand.
"""

import pytest
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, '.')

from perishable_inventory_mdp.demand import (
    PoissonDemand, NegativeBinomialDemand, SeasonalDemand,
    StochasticLeadTime, create_demand_scenario
)


class TestPoissonDemand:
    """Tests for Poisson demand process"""
    
    def test_basic_properties(self):
        """Test mean and variance for Poisson"""
        demand = PoissonDemand(base_rate=10.0)
        
        assert demand.mean() == 10.0
        assert demand.variance() == 10.0  # Var = mean for Poisson
        assert demand.std() == pytest.approx(np.sqrt(10.0))
    
    def test_sampling_distribution(self):
        """Test that samples follow Poisson distribution"""
        np.random.seed(42)
        demand = PoissonDemand(base_rate=15.0)
        
        samples = [demand.sample() for _ in range(5000)]
        
        # Sample mean should be close to rate
        assert np.mean(samples) == pytest.approx(15.0, rel=0.1)
        # Sample variance should be close to rate
        assert np.var(samples) == pytest.approx(15.0, rel=0.15)
    
    def test_pmf(self):
        """Test probability mass function"""
        demand = PoissonDemand(base_rate=5.0)
        
        # Compare to scipy
        for d in [0, 1, 5, 10]:
            expected = stats.poisson.pmf(d, 5.0)
            assert demand.pmf(d) == pytest.approx(expected)
    
    def test_seasonality_function(self):
        """Test demand with seasonality"""
        def seasonal_fn(z):
            return 1.5 if z[0] > 0.5 else 0.5
        
        demand = PoissonDemand(base_rate=10.0, seasonality_fn=seasonal_fn)
        
        # High season
        high_state = np.array([0.8])
        assert demand.mean(high_state) == 15.0  # 10 * 1.5
        
        # Low season
        low_state = np.array([0.3])
        assert demand.mean(low_state) == 5.0  # 10 * 0.5


class TestNegativeBinomialDemand:
    """Tests for Negative Binomial demand process"""
    
    def test_basic_properties(self):
        """Test mean and variance for NegBin"""
        # r=5, p=0.5 => mean = 5*(1-0.5)/0.5 = 5
        demand = NegativeBinomialDemand(n_successes=5, prob_success=0.5)
        
        assert demand.mean() == pytest.approx(5.0)
        # Var = r(1-p)/p^2 = 5*0.5/0.25 = 10
        assert demand.variance() == pytest.approx(10.0)
    
    def test_overdispersion(self):
        """Test that NegBin has variance > mean"""
        demand = NegativeBinomialDemand(n_successes=5, prob_success=0.5)
        
        # Characteristic of overdispersed demand
        assert demand.variance() > demand.mean()
    
    def test_sampling_distribution(self):
        """Test that samples follow NegBin distribution"""
        np.random.seed(42)
        demand = NegativeBinomialDemand(n_successes=10, prob_success=0.4)
        
        samples = [demand.sample() for _ in range(5000)]
        
        expected_mean = 10 * (1 - 0.4) / 0.4
        assert np.mean(samples) == pytest.approx(expected_mean, rel=0.1)
    
    def test_pmf(self):
        """Test probability mass function"""
        r, p = 5, 0.5
        demand = NegativeBinomialDemand(n_successes=r, prob_success=p)
        
        for d in [0, 1, 5, 10]:
            expected = stats.nbinom.pmf(d, r, p)
            assert demand.pmf(d) == pytest.approx(expected)


class TestSeasonalDemand:
    """Tests for seasonal demand process"""
    
    def test_seasonal_pattern(self):
        """Test that demand varies with time"""
        demand = SeasonalDemand(
            base_rate=10.0,
            amplitude=0.5,
            period=12,
            phase=0.0
        )
        
        # At different time points
        rates = []
        for t in range(12):
            state = np.array([float(t)])
            rates.append(demand.mean(state))
        
        # Should have variation
        assert max(rates) > min(rates)
        # Peak should be around amplitude above base
        assert max(rates) == pytest.approx(15.0, rel=0.1)
        # Trough should be around amplitude below base
        assert min(rates) == pytest.approx(5.0, rel=0.1)
    
    def test_period_length(self):
        """Test that pattern repeats after period"""
        demand = SeasonalDemand(base_rate=10.0, period=12)
        
        for t in range(24):
            state_t = np.array([float(t)])
            state_t_plus_period = np.array([float(t + 12)])
            
            assert demand.mean(state_t) == pytest.approx(
                demand.mean(state_t_plus_period)
            )
    
    def test_exogenous_state_update(self):
        """Test that time advances in exogenous state"""
        demand = SeasonalDemand(base_rate=10.0)
        
        state = np.array([5.0])
        new_state = demand.update_exogenous_state(state)
        
        assert new_state[0] == 6.0
    
    def test_null_state_handling(self):
        """Test handling of null exogenous state"""
        demand = SeasonalDemand(base_rate=10.0)
        
        # Should use base rate when no state
        assert demand.mean(None) == 10.0
        
        # Should initialize state
        new_state = demand.update_exogenous_state(None)
        assert new_state is not None


class TestStochasticLeadTime:
    """Tests for stochastic lead time model"""
    
    def test_deterministic_advancement(self):
        """Test deterministic (p=1) lead time"""
        slt = StochasticLeadTime(supplier_id=0, advancement_prob=1.0)
        
        # Should always advance
        for _ in range(100):
            assert slt.sample_advancement() == True
    
    def test_no_advancement(self):
        """Test no advancement (p=0)"""
        slt = StochasticLeadTime(supplier_id=0, advancement_prob=0.0)
        
        # Should never advance
        for _ in range(100):
            assert slt.sample_advancement() == False
    
    def test_probabilistic_advancement(self):
        """Test probabilistic advancement"""
        np.random.seed(42)
        slt = StochasticLeadTime(supplier_id=0, advancement_prob=0.7)
        
        advancements = [slt.sample_advancement() for _ in range(1000)]
        
        # Should be roughly 70% True
        assert np.mean(advancements) == pytest.approx(0.7, abs=0.05)


class TestDemandScenarioFactory:
    """Tests for demand scenario factory"""
    
    def test_stationary_scenario(self):
        """Test creation of stationary demand"""
        demand = create_demand_scenario(10.0, "stationary")
        
        assert isinstance(demand, PoissonDemand)
        assert demand.mean() == 10.0
    
    def test_seasonal_scenario(self):
        """Test creation of seasonal demand"""
        demand = create_demand_scenario(
            10.0, "seasonal",
            amplitude=0.3,
            period=12
        )
        
        assert isinstance(demand, SeasonalDemand)
    
    def test_overdispersed_scenario(self):
        """Test creation of overdispersed demand"""
        demand = create_demand_scenario(10.0, "overdispersed", dispersion=5.0)
        
        assert isinstance(demand, NegativeBinomialDemand)
        assert demand.mean() == pytest.approx(10.0, rel=0.01)
    
    def test_unknown_scenario_raises(self):
        """Test that unknown scenario raises error"""
        with pytest.raises(ValueError):
            create_demand_scenario(10.0, "unknown_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

