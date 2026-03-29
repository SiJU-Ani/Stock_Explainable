"""
Unit tests for all modules.
"""

import unittest
import numpy as np
import torch
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data generation utilities
def create_dummy_ohlcv_data(n_days: int = 100, n_tickers: int = 1) -> dict:
    """Create dummy OHLCV data for testing."""
    data = {}
    base_price = 100
    
    for i in range(n_tickers):
        ticker = f"TEST{i}"
        dates = pd.date_range(start='2023-01-01', periods=n_days)
        
        prices = base_price + np.cumsum(np.random.randn(n_days) * 0.5)
        
        df = pd.DataFrame({
            'Open': prices + np.random.randn(n_days) * 0.5,
            'High': prices + np.abs(np.random.randn(n_days) * 0.7),
            'Low': prices - np.abs(np.random.randn(n_days) * 0.7),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        data[ticker] = df
    
    return data


class TestDataAcquisition(unittest.TestCase):
    """Test data acquisition module."""
    
    def test_ohlcv_validation(self):
        """Test OHLCV data validation."""
        from src.data_acquisition.fetcher import StockDataFetcher
        
        dummy_data = create_dummy_ohlcv_data(100)
        df = dummy_data['TEST0']
        
        fetcher = StockDataFetcher({})
        is_valid = fetcher.validate_ohlcv(df)
        self.assertTrue(is_valid)
    
    def test_invalid_ohlcv(self):
        """Test invalid OHLCV detection."""
        from src.data_acquisition.fetcher import StockDataFetcher
        
        # Create invalid data (negative prices)
        df = pd.DataFrame({
            'Open': [-1, 2, 3],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000000, 1000000, 1000000]
        })
        
        fetcher = StockDataFetcher({})
        is_valid = fetcher.validate_ohlcv(df)
        self.assertFalse(is_valid)


class TestPreprocessing(unittest.TestCase):
    """Test text preprocessing module."""
    
    def test_text_cleaning(self):
        """Test text preprocessing."""
        from src.preprocessing.text_processor import TextPreprocessor
        
        preprocessor = TextPreprocessor({})
        
        text = "APPLE Inc. reported STRONG earnings   today!!!"
        cleaned = preprocessor.clean_text(text)
        
        # Should be lowercase
        self.assertTrue(cleaned.islower() or ' ' in cleaned)
        # Should have no extra spaces
        self.assertNotIn('   ', cleaned)
    
    def test_text_truncation(self):
        """Test text truncation."""
        from src.preprocessing.text_processor import TextPreprocessor
        
        preprocessor = TextPreprocessor({'preprocessing': {'max_sequence_length': 100}})
        
        long_text = "a" * 1000
        truncated = preprocessor.truncate_text(long_text)
        
        self.assertLessEqual(len(truncated), 400)  # 100 * 4


class TestKnowledgeGraph(unittest.TestCase):
    """Test knowledge graph module."""
    
    def test_graph_creation(self):
        """Test graph creation."""
        from src.graph_construction.knowledge_graph import FinancialKnowledgeGraph
        
        graph = FinancialKnowledgeGraph({})
        
        # Add nodes
        graph.add_node('AAPL', 'Apple Inc.')
        graph.add_node('MSFT', 'Microsoft')
        
        # Add edge
        graph.add_edge('AAPL', 'MSFT', 'competitor', weight=0.8)
        
        self.assertEqual(graph.graph.number_of_nodes(), 2)
        self.assertEqual(graph.graph.number_of_edges(), 1)
    
    def test_neighbors(self):
        """Test neighbor retrieval."""
        from src.graph_construction.knowledge_graph import FinancialKnowledgeGraph
        
        graph = FinancialKnowledgeGraph({})
        
        graph.add_node('A', 'Company A')
        graph.add_node('B', 'Company B')
        graph.add_node('C', 'Company C')
        
        graph.add_edge('A', 'B', 'supplier')
        graph.add_edge('A', 'C', 'customer')
        
        neighbors = graph.get_neighbors('A', direction='out')
        self.assertEqual(len(neighbors), 2)
        self.assertIn('B', neighbors)
        self.assertIn('C', neighbors)


class TestAgenticDebate(unittest.TestCase):
    """Test agentic debate module."""
    
    def test_bull_agent(self):
        """Test Bull agent."""
        from src.agentic_debate.debate_module import BullAgent
        
        bull = BullAgent({'agentic_debate': {
            'bull_model': 'distilbert-base-uncased'
        }})
        
        text = "Apple beat expectations and announced record growth."
        result = bull.analyze(text)
        
        self.assertIn('reasoning_vector', result)
        self.assertIn('positive_catalysts', result)
        self.assertIn('bullish_score', result)
        self.assertTrue(len(result['positive_catalysts']) > 0)
    
    def test_bear_agent(self):
        """Test Bear agent."""
        from src.agentic_debate.debate_module import BearAgent
        
        bear = BearAgent({'agentic_debate': {
            'bear_model': 'distilbert-base-uncased'
        }})
        
        text = "Company faces regulatory risks and market disruption."
        result = bear.analyze(text)
        
        self.assertIn('reasoning_vector', result)
        self.assertIn('risk_factors', result)
        self.assertIn('bearish_score', result)
        self.assertTrue(len(result['risk_factors']) > 0)


class TestTemporalModels(unittest.TestCase):
    """Test temporal models."""
    
    def test_lstm_model(self):
        """Test LSTM model."""
        from src.temporal_model.temporal import LSTMModel
        
        model = LSTMModel(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            dropout=0.3
        )
        
        # Test forward pass
        x = torch.randn(4, 20, 10)  # batch_size=4, seq_length=20, input_size=10
        output, (h, c) = model(x)
        
        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(h.shape, (2, 4, 32))
    
    def test_transformer_model(self):
        """Test Transformer model."""
        from src.temporal_model.temporal import TemporalTransformer
        
        model = TemporalTransformer(
            input_size=10,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128
        )
        
        x = torch.randn(4, 20, 10)
        output = model(x)
        
        self.assertEqual(output.shape, (4, 1))
        self.assertTrue((output >= 0).all() and (output <= 1).all())  # Sigmoid output
    
    def test_hybrid_model(self):
        """Test Hybrid temporal model."""
        from src.temporal_model.temporal import HybridTemporalModel
        
        model = HybridTemporalModel(
            historical_dim=10,
            sentiment_dim=5,
            gnn_dim=8,
            model_type='lstm',
            config={'temporal_model': {'lstm': {'hidden_size': 32, 'num_layers': 2}}}
        )
        
        historical = torch.randn(4, 20, 10)
        sentiment = torch.randn(4, 20, 5)
        gnn = torch.randn(4, 20, 8)
        
        output = model(historical, sentiment, gnn)
        self.assertEqual(output.shape, (4, 1))


class TestGNN(unittest.TestCase):
    """Test GNN module."""
    
    def test_gcn_forward(self):
        """Test GCN forward pass."""
        from src.gnn_module.gnn import GCNModel
        
        model = GCNModel(
            input_dim=10,
            hidden_dim=32,
            output_dim=16,
            num_layers=2
        )
        
        # Create sample data
        x = torch.randn(5, 10)  # 5 nodes, 10 features
        adj = torch.eye(5)  # Simple identity adjacency
        
        output = model(x, adj)
        self.assertEqual(output.shape, (5, 16))
    
    def test_event_propagation(self):
        """Test event propagation."""
        from src.gnn_module.gnn import EventPropagationGNN
        
        gnn = EventPropagationGNN({
            'gnn': {
                'input_dim': 10,
                'hidden_dim': 32,
                'output_dim': 16,
                'num_layers': 2
            }
        })
        
        node_features = torch.randn(5, 10)
        edge_indices = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
        
        result = gnn.propagate(node_features, edge_indices, num_hops=2)
        
        self.assertIn('final', result)
        self.assertEqual(result['final'].shape, (5, 16))


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def test_ml_metrics(self):
        """Test ML metrics."""
        from src.evaluation.metrics import MLMetrics
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        
        metrics = MLMetrics.compute_metrics(y_true, y_pred, y_proba)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('roc_auc', metrics)
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        from src.evaluation.metrics import FinancialMetrics
        
        returns = np.array([0.01, -0.005, 0.015, 0.002, -0.001])
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        
        self.assertIsInstance(sharpe, float)
    
    def test_maximum_drawdown(self):
        """Test maximum drawdown."""
        from src.evaluation.metrics import FinancialMetrics
        
        returns = np.array([-0.1, 0.05, -0.15, 0.10, 0.05])
        mdd = FinancialMetrics.maximum_drawdown(returns)
        
        self.assertLess(mdd, 0)


class TestTrainer(unittest.TestCase):
    """Test training module."""
    
    def test_loss_computation(self):
        """Test loss computation."""
        from src.training.trainer import HybridLoss
        
        loss_fn = HybridLoss(
            classification_weight=0.7,
            regression_weight=0.3
        )
        
        logits = torch.randn(4, 1)
        target = torch.randint(0, 2, (4, 1)).float()
        target_return = torch.randn(4, 1)
        
        loss, loss_dict = loss_fn(logits, target, target_return)
        
        self.assertGreater(loss.item(), 0)
        self.assertIn('total', loss_dict)
        self.assertIn('classification', loss_dict)
        self.assertIn('regression', loss_dict)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataAcquisition))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestAgenticDebate))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalModels))
    suite.addTests(loader.loadTestsFromTestCase(TestGNN))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainer))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
