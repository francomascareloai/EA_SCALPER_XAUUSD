"""
Unit Tests for Oracle Validation Scripts
=========================================

Tests for:
- Walk-Forward Analysis
- Monte Carlo Block Bootstrap
- Deflated Sharpe (PSR/DSR)
- Unified Confidence Scoring
- Sample Data Generation
- GO/NO-GO Validator Integration

For: EA_SCALPER_XAUUSD - ORACLE v2.2.1

Usage:
    pytest tests/oracle/test_oracle.py -v
    pytest tests/oracle/test_oracle.py -v -k "test_wfa"
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.oracle.sample_data import (
    generate_sample_trades,
    generate_realistic_xauusd_trades,
    generate_edge_case_trades,
    calculate_metrics_summary
)
from scripts.oracle.walk_forward import WalkForwardAnalyzer, WFAResult
from scripts.oracle.monte_carlo import BlockBootstrapMC, MCResult
from scripts.oracle.deflated_sharpe import SharpeAnalyzer, SharpeAnalysisResult
from scripts.oracle.confidence import (
    UnifiedConfidenceCalculator,
    ConfidenceComponents,
    calculate_confidence_score
)
from scripts.oracle.metrics import calculate_sqn, calculate_sharpe, calculate_sortino


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_trades():
    """Generate sample trades for testing"""
    return generate_sample_trades(n_trades=200, seed=42)


@pytest.fixture
def realistic_trades():
    """Generate realistic XAUUSD trades"""
    return generate_realistic_xauusd_trades(n_trades=300, seed=42)


@pytest.fixture
def small_sample():
    """Small sample for edge case testing"""
    return generate_sample_trades(n_trades=50, seed=42)


@pytest.fixture
def high_sharpe_trades():
    """Suspiciously good trades (for overfitting detection)"""
    return generate_edge_case_trades(scenario='high_sharpe', n_trades=200, seed=42)


# =============================================================================
# SAMPLE DATA GENERATION TESTS
# =============================================================================

class TestSampleDataGeneration:
    """Tests for sample data generators"""
    
    def test_generate_sample_trades_count(self):
        """Test that correct number of trades are generated"""
        df = generate_sample_trades(n_trades=100, seed=42)
        assert len(df) == 100
    
    def test_generate_sample_trades_columns(self):
        """Test that required columns exist"""
        df = generate_sample_trades(n_trades=50, seed=42)
        required_cols = ['datetime', 'direction', 'entry_price', 'profit', 'is_win']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_generate_sample_trades_win_rate(self):
        """Test that win rate is approximately as specified"""
        df = generate_sample_trades(n_trades=1000, win_rate=0.55, seed=42)
        actual_wr = df['is_win'].mean()
        assert 0.50 <= actual_wr <= 0.60, f"Win rate {actual_wr} not in expected range"
    
    def test_realistic_trades_has_sessions(self, realistic_trades):
        """Test realistic trades include session info"""
        assert 'session' in realistic_trades.columns
        sessions = realistic_trades['session'].unique()
        assert len(sessions) >= 2  # Should have multiple sessions
    
    def test_realistic_trades_has_regimes(self, realistic_trades):
        """Test realistic trades include regime info"""
        assert 'regime' in realistic_trades.columns
    
    def test_edge_case_high_drawdown(self):
        """Test high drawdown scenario generates data"""
        df = generate_edge_case_trades(scenario='high_drawdown', n_trades=200)
        assert len(df) >= 190  # Allow some margin
    
    def test_metrics_summary(self, sample_trades):
        """Test metrics summary calculation"""
        summary = calculate_metrics_summary(sample_trades)
        assert 'n_trades' in summary
        assert 'win_rate' in summary
        assert 'sharpe' in summary
        assert 'max_dd' in summary


# =============================================================================
# WALK-FORWARD ANALYSIS TESTS
# =============================================================================

class TestWalkForwardAnalysis:
    """Tests for WFA implementation"""
    
    def test_wfa_runs_without_error(self, sample_trades):
        """Test WFA executes successfully"""
        wfa = WalkForwardAnalyzer(n_windows=5, is_ratio=0.7, min_trades_per_window=10)
        result = wfa.run(sample_trades, mode='rolling', return_col='profit')
        assert isinstance(result, WFAResult)
    
    def test_wfa_returns_correct_structure(self, sample_trades):
        """Test WFA result has expected attributes"""
        wfa = WalkForwardAnalyzer(n_windows=5, min_trades_per_window=10)
        result = wfa.run(sample_trades, return_col='profit')
        
        assert hasattr(result, 'wfe')
        assert hasattr(result, 'windows')
        assert hasattr(result, 'status')
        assert hasattr(result, 'oos_positive_pct')
    
    def test_wfa_wfe_in_valid_range(self, sample_trades):
        """Test WFE is in reasonable range"""
        wfa = WalkForwardAnalyzer(n_windows=5, min_trades_per_window=10)
        result = wfa.run(sample_trades, return_col='profit')
        
        # WFE should be between -2 and 2 for normal strategies
        assert -2 <= result.wfe <= 2, f"WFE {result.wfe} out of expected range"
    
    def test_wfa_anchored_mode(self, sample_trades):
        """Test anchored WFA mode"""
        wfa = WalkForwardAnalyzer(n_windows=5, min_trades_per_window=10)
        result = wfa.run(sample_trades, mode='anchored', return_col='profit')
        assert result.mode == 'anchored'
    
    def test_wfa_rolling_mode(self, sample_trades):
        """Test rolling WFA mode"""
        wfa = WalkForwardAnalyzer(n_windows=5, min_trades_per_window=10)
        result = wfa.run(sample_trades, mode='rolling', return_col='profit')
        assert result.mode == 'rolling'
    
    def test_wfa_insufficient_data_raises(self, small_sample):
        """Test WFA raises error for insufficient data"""
        wfa = WalkForwardAnalyzer(n_windows=10, min_trades_per_window=20)
        with pytest.raises(ValueError):
            wfa.run(small_sample, return_col='profit')
    
    def test_wfa_report_generation(self, sample_trades):
        """Test WFA report can be generated"""
        wfa = WalkForwardAnalyzer(n_windows=5, min_trades_per_window=10)
        result = wfa.run(sample_trades, return_col='profit')
        report = wfa.generate_report(result)
        assert isinstance(report, str)
        assert 'WFE' in report


# =============================================================================
# MONTE CARLO TESTS
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo simulation"""
    
    def test_mc_runs_without_error(self, sample_trades):
        """Test MC executes successfully"""
        mc = BlockBootstrapMC(n_simulations=100)
        result = mc.run(sample_trades, use_block=True)
        assert isinstance(result, MCResult)
    
    def test_mc_returns_correct_structure(self, sample_trades):
        """Test MC result has expected attributes"""
        mc = BlockBootstrapMC(n_simulations=100)
        result = mc.run(sample_trades, use_block=True)
        
        assert hasattr(result, 'dd_95th')
        assert hasattr(result, 'var_95')
        assert hasattr(result, 'cvar_95')
        assert hasattr(result, 'prob_profit')
        assert hasattr(result, 'confidence_score')
    
    def test_mc_dd_percentiles_ordered(self, sample_trades):
        """Test DD percentiles are in correct order"""
        mc = BlockBootstrapMC(n_simulations=500)
        result = mc.run(sample_trades, use_block=True)
        
        assert result.dd_5th <= result.dd_25th <= result.dd_50th
        assert result.dd_50th <= result.dd_75th <= result.dd_95th
    
    def test_mc_traditional_vs_block(self, sample_trades):
        """Test both traditional and block bootstrap work"""
        mc = BlockBootstrapMC(n_simulations=100)
        
        result_block = mc.run(sample_trades, use_block=True)
        result_trad = mc.run(sample_trades, use_block=False)
        
        assert result_block.method == 'block_bootstrap'
        assert result_trad.method == 'traditional'
    
    def test_mc_optimal_block_size(self):
        """Test optimal block size calculation"""
        mc = BlockBootstrapMC()
        
        # n^(1/3) rule
        assert mc.optimal_block_size(1000) >= 5
        assert mc.optimal_block_size(1000) <= 15
    
    def test_mc_trades_per_day_estimation(self, realistic_trades):
        """Test trades per day estimation from timestamps"""
        mc = BlockBootstrapMC()
        tpd = mc._estimate_trades_per_day(realistic_trades)
        assert tpd >= 1
        assert tpd <= 50  # Reasonable range for scalping
    
    def test_mc_report_generation(self, sample_trades):
        """Test MC report can be generated"""
        mc = BlockBootstrapMC(n_simulations=100)
        result = mc.run(sample_trades, use_block=True)
        report = mc.generate_report(result)
        assert isinstance(report, str)
        assert 'DRAWDOWN' in report


# =============================================================================
# SHARPE ANALYSIS TESTS
# =============================================================================

class TestSharpeAnalysis:
    """Tests for PSR/DSR analysis"""
    
    def test_sharpe_analyzer_runs(self, sample_trades):
        """Test Sharpe analyzer executes"""
        returns = sample_trades['profit'].values / 100000
        analyzer = SharpeAnalyzer()
        result = analyzer.analyze(returns, n_trials=1)
        assert isinstance(result, SharpeAnalysisResult)
    
    def test_sharpe_result_attributes(self, sample_trades):
        """Test result has all attributes"""
        returns = sample_trades['profit'].values / 100000
        analyzer = SharpeAnalyzer()
        result = analyzer.analyze(returns, n_trials=1)
        
        assert hasattr(result, 'observed_sharpe')
        assert hasattr(result, 'probabilistic_sharpe')
        assert hasattr(result, 'deflated_sharpe')
        assert hasattr(result, 'verdict')
    
    def test_psr_in_valid_range(self, sample_trades):
        """Test PSR is between 0 and 1"""
        returns = sample_trades['profit'].values / 100000
        analyzer = SharpeAnalyzer()
        result = analyzer.analyze(returns, n_trials=1)
        
        assert 0 <= result.probabilistic_sharpe <= 1
    
    def test_dsr_decreases_with_trials(self, sample_trades):
        """Test DSR decreases with more trials (multiple testing penalty)"""
        returns = sample_trades['profit'].values / 100000
        analyzer = SharpeAnalyzer()
        
        result_1 = analyzer.analyze(returns, n_trials=1)
        result_100 = analyzer.analyze(returns, n_trials=100)
        
        # More trials should decrease DSR (more penalty)
        assert result_100.deflated_sharpe <= result_1.deflated_sharpe
    
    def test_high_sharpe_flagged(self, high_sharpe_trades):
        """Test suspiciously high Sharpe is detected"""
        returns = high_sharpe_trades['profit'].values / 100000
        analyzer = SharpeAnalyzer()
        result = analyzer.analyze(returns, n_trials=100)
        
        # High sharpe with many trials should have concerns
        if result.observed_sharpe > 3:
            # Either DSR should be low or verdict should indicate concern
            assert result.deflated_sharpe < result.observed_sharpe or 'OVERFIT' in result.verdict


# =============================================================================
# CONFIDENCE SCORE TESTS
# =============================================================================

class TestConfidenceScore:
    """Tests for unified confidence scoring"""
    
    def test_confidence_calculator_creates(self):
        """Test calculator instantiates"""
        calc = UnifiedConfidenceCalculator()
        assert calc is not None
    
    def test_wfa_component_scoring(self):
        """Test WFA component scoring logic"""
        score_good, _ = UnifiedConfidenceCalculator.calculate_wfa_component(0.7, 0.85)
        score_bad, _ = UnifiedConfidenceCalculator.calculate_wfa_component(0.3, 0.4)
        
        assert score_good > score_bad
        assert 0 <= score_good <= 25
        assert 0 <= score_bad <= 25
    
    def test_mc_component_scoring(self):
        """Test Monte Carlo component scoring"""
        score_good, _ = UnifiedConfidenceCalculator.calculate_mc_component(5.0, 2.0)
        score_bad, _ = UnifiedConfidenceCalculator.calculate_mc_component(12.0, 15.0)
        
        assert score_good > score_bad
        assert 0 <= score_good <= 25
        assert 0 <= score_bad <= 25
    
    def test_sharpe_component_scoring(self):
        """Test Sharpe component scoring"""
        score_good, _ = UnifiedConfidenceCalculator.calculate_sharpe_component(0.95, 1.5)
        score_bad, _ = UnifiedConfidenceCalculator.calculate_sharpe_component(0.7, -0.5)
        
        assert score_good > score_bad
        assert 0 <= score_good <= 20
        assert 0 <= score_bad <= 20
    
    def test_propfirm_component_scoring(self):
        """Test Prop Firm component scoring"""
        score_good, _ = UnifiedConfidenceCalculator.calculate_propfirm_component(2.0, 1.0)
        score_bad, _ = UnifiedConfidenceCalculator.calculate_propfirm_component(15.0, 10.0)
        
        assert score_good > score_bad
        assert 0 <= score_good <= 20
        assert 0 <= score_bad <= 20
    
    def test_total_score_calculation(self):
        """Test complete score calculation"""
        calc = UnifiedConfidenceCalculator()
        result = calc.calculate(
            wfe=0.65,
            oos_positive_pct=0.8,
            dd_95=7.0,
            prob_ruin=3.0,
            psr=0.92,
            dsr=0.8,
            p_daily=4.0,
            p_total=1.5
        )
        
        assert isinstance(result, ConfidenceComponents)
        assert 0 <= result.total <= 100
    
    def test_decision_thresholds(self):
        """Test decision threshold logic"""
        calc = UnifiedConfidenceCalculator()
        
        # Good strategy
        good = calc.calculate(0.7, 0.85, 5.0, 2.0, 0.95, 1.0, 2.0, 1.0)
        # Bad strategy
        bad = calc.calculate(0.3, 0.4, 15.0, 20.0, 0.6, -1.0, 20.0, 15.0)
        
        assert good.decision.value in ['STRONG_GO', 'GO']
        assert bad.decision.value in ['NO_GO', 'INVESTIGATE']
    
    def test_warnings_generated(self):
        """Test warning generation for concerning metrics"""
        warnings = UnifiedConfidenceCalculator.generate_warnings(
            wfe=0.3,
            dd_95=12.0,
            psr=0.7,
            dsr=-0.5,
            p_daily=15.0,
            p_total=10.0,
            sharpe_observed=5.0
        )
        
        assert len(warnings) > 0
        assert any('DSR' in w for w in warnings)
    
    def test_confidence_report_generation(self):
        """Test report string generation"""
        calc = UnifiedConfidenceCalculator()
        result = calc.calculate(0.6, 0.75, 8.0, 5.0, 0.90, 0.5, 5.0, 2.0)
        report = result.generate_report()
        
        assert isinstance(report, str)
        assert 'CONFIDENCE' in report
        assert 'DECISION' in report


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestMetrics:
    """Tests for metrics calculations"""
    
    def test_calculate_sqn(self, sample_trades):
        """Test SQN calculation"""
        sqn, interp = calculate_sqn(sample_trades)
        assert isinstance(sqn, (int, float))
        assert isinstance(interp, str)
    
    def test_sharpe_calculation(self, sample_trades):
        """Test Sharpe ratio calculation"""
        returns = sample_trades['profit'].values / 100000
        sharpe = calculate_sharpe(returns)
        assert isinstance(sharpe, (int, float))
    
    def test_sortino_calculation(self, sample_trades):
        """Test Sortino ratio calculation"""
        returns = sample_trades['profit'].values / 100000
        sortino = calculate_sortino(returns)
        assert isinstance(sortino, (int, float))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_full_validation_pipeline(self, realistic_trades):
        """Test complete validation flow"""
        # 1. WFA
        wfa = WalkForwardAnalyzer(n_windows=5, min_trades_per_window=15)
        wfa_result = wfa.run(realistic_trades, return_col='profit')
        
        # 2. Monte Carlo
        mc = BlockBootstrapMC(n_simulations=100)
        mc_result = mc.run(realistic_trades, use_block=True)
        
        # 3. Sharpe Analysis
        returns = realistic_trades['profit'].values / 100000
        sharpe_analyzer = SharpeAnalyzer()
        sharpe_result = sharpe_analyzer.analyze(returns, n_trials=10)
        
        # 4. Confidence Score
        calc = UnifiedConfidenceCalculator()
        confidence = calc.calculate(
            wfe=wfa_result.wfe,
            oos_positive_pct=wfa_result.oos_positive_pct / 100,
            dd_95=mc_result.dd_95th,
            prob_ruin=mc_result.risk_of_ruin_10pct,
            psr=sharpe_result.probabilistic_sharpe,
            dsr=sharpe_result.deflated_sharpe,
            p_daily=mc_result.ftmo_daily_violation_prob,
            p_total=mc_result.ftmo_total_violation_prob
        )
        
        # Verify all pieces work together
        assert 0 <= confidence.total <= 100
        assert confidence.decision is not None
    
    def test_edge_cases_handled(self):
        """Test various edge cases are handled gracefully"""
        # Small sample
        small_df = generate_sample_trades(n_trades=20, seed=42)
        
        # Should not crash
        mc = BlockBootstrapMC(n_simulations=50)
        mc_result = mc.run(small_df, use_block=False)
        assert mc_result is not None
        
        # Single profit column
        returns = small_df['profit'].values / 100000
        analyzer = SharpeAnalyzer()
        result = analyzer.analyze(returns, n_trials=1)
        assert result is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
