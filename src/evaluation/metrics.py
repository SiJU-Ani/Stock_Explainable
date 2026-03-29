"""
Evaluation Module - Financial and ML metrics for backtesting and performance analysis.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class MLMetrics:
    """Standard machine learning metrics."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (0 or 1)
            y_pred_proba: Predicted probabilities (0-1)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # ROC AUC (requires probabilities)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics


class FinancialMetrics:
    """Financial trading metrics."""
    
    @staticmethod
    def compute_returns(prices: np.ndarray) -> np.ndarray:
        """
        Compute log returns.
        
        Args:
            prices: Closing prices
            
        Returns:
            Log returns
        """
        return np.log(prices[1:] / prices[:-1])
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Compute Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.05,
        target_return: float = 0.0
    ) -> float:
        """
        Compute Sortino ratio (considers only downside volatility).
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            target_return: Target return
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = np.minimum(excess_returns - target_return, 0)
        downside_std = np.sqrt(np.mean(downside_returns**2))
        
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def maximum_drawdown(returns: np.ndarray) -> float:
        """
        Compute maximum drawdown.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown (as negative number)
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def cumulative_return(returns: np.ndarray) -> float:
        """
        Compute cumulative return.
        
        Args:
            returns: Array of returns
            
        Returns:
            Total return
        """
        return np.prod(1 + returns) - 1
    
    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Compute Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        annual_return = (np.prod(1 + returns) ** (periods_per_year / len(returns))) - 1
        max_dd = FinancialMetrics.maximum_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def win_rate(
        predicted_direction: np.ndarray,
        actual_returns: np.ndarray
    ) -> float:
        """
        Compute win rate (% of correct direction predictions).
        
        Args:
            predicted_direction: Array of predicted directions (0 or 1)
            actual_returns: Array of actual returns
            
        Returns:
            Win rate (0-1)
        """
        actual_direction = (actual_returns > 0).astype(int)
        correct = (predicted_direction == actual_direction).sum()
        return correct / len(actual_returns)


class Backtester:
    """Backtest trading strategy based on model predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtester.
        
        Args:
            config: Configuration dictionary
        """
        backtest_config = config.get('evaluation', {}).get('backtesting', {})
        self.initial_capital = backtest_config.get('initial_capital', 100000)
        self.transaction_costs = backtest_config.get('transaction_costs', 0.001)
        self.slippage = backtest_config.get('slippage', 0.0005)
        self.risk_free_rate = backtest_config.get('risk_free_rate', 0.05)
    
    def backtest(
        self,
        predictions: np.ndarray,  # 0 or 1
        prices: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Backtest predictions on price data.
        
        Args:
            predictions: Binary predictions (0=DOWN, 1=UP)
            prices: Closing prices
            returns: Log returns
            
        Returns:
            Dictionary with backtest results
        """
        if len(predictions) != len(returns):
            raise ValueError("predictions and returns must have same length")
        
        # Trading logic: 1 means long position
        positions = predictions  # 1 = long, 0 = cash
        
        # Calculate P&L
        trade_returns = positions[:-1] * returns[1:]  # Align with next day return
        strategy_returns = trade_returns.copy()
        
        # Apply transaction costs
        position_changes = np.abs(np.diff(positions.astype(float)))
        transaction_costs_returns = -position_changes * self.transaction_costs
        strategy_returns += transaction_costs_returns
        
        # Apply slippage
        slippage_returns = -np.abs(positions[:-1]) * self.slippage
        strategy_returns += slippage_returns
        
        # Calculate metrics
        cumulative_return = FinancialMetrics.cumulative_return(strategy_returns)
        sharpe = FinancialMetrics.sharpe_ratio(strategy_returns, self.risk_free_rate)
        sortino = FinancialMetrics.sortino_ratio(strategy_returns, self.risk_free_rate)
        max_dd = FinancialMetrics.maximum_drawdown(strategy_returns)
        calmar = FinancialMetrics.calmar_ratio(strategy_returns)
        win_rate = FinancialMetrics.win_rate(predictions[:-1], returns[1:])
        
        # Benchmark returns (buy and hold)
        benchmark_returns = returns[1:]
        benchmark_cumulative = FinancialMetrics.cumulative_return(benchmark_returns)
        benchmark_sharpe = FinancialMetrics.sharpe_ratio(benchmark_returns, self.risk_free_rate)
        
        results = {
            'strategy': {
                'cumulative_return': cumulative_return,
                'annual_return': (1 + cumulative_return) ** (252 / len(strategy_returns)) - 1,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'calmar_ratio': calmar,
                'win_rate': win_rate,
                'num_trades': int(position_changes.sum()),
                'avg_trade_return': strategy_returns.mean() if len(strategy_returns) > 0 else 0,
            },
            'benchmark': {
                'cumulative_return': benchmark_cumulative,
                'annual_return': (1 + benchmark_cumulative) ** (252 / len(benchmark_returns)) - 1,
                'sharpe_ratio': benchmark_sharpe,
                'max_drawdown': FinancialMetrics.maximum_drawdown(benchmark_returns),
            },
            'outperformance': {
                'excess_return': cumulative_return - benchmark_cumulative,
                'excess_sharpe': sharpe - benchmark_sharpe,
            }
        }
        
        return results


class PerformanceAnalyzer:
    """
    Complete performance analysis combining ML and financial metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analyzer."""
        self.config = config
        self.backtester = Backtester(config)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        prices: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Complete evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            prices: Closing prices
            returns: Log returns
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive performance evaluation")
        
        # ML metrics
        ml_metrics = MLMetrics.compute_metrics(y_true, y_pred, y_pred_proba)
        logger.info(f"ML Metrics: {ml_metrics}")
        
        # Backtest
        backtest_results = self.backtester.backtest(y_pred, prices, returns)
        logger.info(f"Backtest Results: Strategy Sharpe={backtest_results['strategy']['sharpe_ratio']:.3f}")
        
        results = {
            'ml_metrics': ml_metrics,
            'backtest_results': backtest_results,
            'combined_score': ml_metrics['f1'] * backtest_results['strategy']['sharpe_ratio']
        }
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """Print evaluation report."""
        report = "\n" + "="*60 + "\n"
        report += "PERFORMANCE EVALUATION REPORT\n"
        report += "="*60 + "\n\n"
        
        # ML Metrics
        report += "ML METRICS\n"
        report += "-"*40 + "\n"
        for key, value in results['ml_metrics'].items():
            report += f"  {key:20s}: {value:.4f}\n"
        
        # Strategy Results
        report += "\nSTRATEGY PERFORMANCE\n"
        report += "-"*40 + "\n"
        for key, value in results['backtest_results']['strategy'].items():
            if isinstance(value, float):
                report += f"  {key:20s}: {value:.4f}\n"
            else:
                report += f"  {key:20s}: {value}\n"
        
        # Benchmark
        report += "\nBENCHMARK (Buy & Hold)\n"
        report += "-"*40 + "\n"
        for key, value in results['backtest_results']['benchmark'].items():
            report += f"  {key:20s}: {value:.4f}\n"
        
        # Outperformance
        report += "\nOUTPERFORMANCE\n"
        report += "-"*40 + "\n"
        for key, value in results['backtest_results']['outperformance'].items():
            report += f"  {key:20s}: {value:.4f}\n"
        
        report += "\n" + "="*60 + "\n"
        logger.info(report)
