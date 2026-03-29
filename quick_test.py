#!/usr/bin/env python
"""
Quick test script to verify all modules are working.
Run: python quick_test.py
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger_config import setup_logging, load_config, create_directories


def test_imports():
    """Test that all modules can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("Testing imports...")
    
    try:
        from src.data_acquisition import DataPipeline
        logger.info("✓ Data acquisition module")
    except Exception as e:
        logger.error(f"✗ Data acquisition: {e}")
        return False
    
    try:
        from src.preprocessing import TextPreprocessor, TextEmbedder
        logger.info("✓ Preprocessing module")
    except Exception as e:
        logger.error(f"✗ Preprocessing: {e}")
        return False
    
    try:
        from src.graph_construction import FinancialKnowledgeGraph
        logger.info("✓ Graph construction module")
    except Exception as e:
        logger.error(f"✗ Graph construction: {e}")
        return False
    
    try:
        from src.agentic_debate import BullAgent, BearAgent
        logger.info("✓ Agentic debate module")
    except Exception as e:
        logger.error(f"✗ Agentic debate: {e}")
        return False
    
    try:
        from src.gnn_module import EventPropagationGNN
        logger.info("✓ GNN module")
    except Exception as e:
        logger.error(f"✗ GNN module: {e}")
        return False
    
    try:
        from src.temporal_model import LSTMModel, HybridTemporalModel
        logger.info("✓ Temporal model module")
    except Exception as e:
        logger.error(f"✗ Temporal model: {e}")
        return False
    
    try:
        from src.training import Trainer, HybridLoss
        logger.info("✓ Training module")
    except Exception as e:
        logger.error(f"✗ Training: {e}")
        return False
    
    try:
        from src.evaluation import MLMetrics, FinancialMetrics, Backtester
        logger.info("✓ Evaluation module")
    except Exception as e:
        logger.error(f"✗ Evaluation: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting configuration loading...")
    
    try:
        config = load_config('configs/config.yaml')
        assert 'data' in config
        assert 'preprocessing' in config
        assert 'gnn' in config
        assert 'training' in config
        logger.info("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False


def test_knowledge_graph():
    """Test knowledge graph functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting knowledge graph...")
    
    try:
        import networkx as nx
        from src.graph_construction import FinancialKnowledgeGraph
        
        graph = FinancialKnowledgeGraph({})
        
        # Add nodes
        graph.add_node('AAPL', 'Apple Inc.')
        graph.add_node('MSFT', 'Microsoft')
        graph.add_node('GOOGL', 'Google')
        
        # Add edges
        graph.add_edge('AAPL', 'MSFT', 'competitor', weight=0.8)
        graph.add_edge('MSFT', 'GOOGL', 'competitor', weight=0.7)
        
        # Test queries
        neighbors = graph.get_neighbors('AAPL')
        assert 'MSFT' in neighbors
        
        stats = graph.get_graph_stats()
        assert stats['num_nodes'] == 3
        assert stats['num_edges'] == 2
        
        logger.info("✓ Knowledge graph works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Knowledge graph test failed: {e}")
        return False


def test_torch_models():
    """Test PyTorch model instantiation."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting PyTorch models...")
    
    try:
        import torch
        from src.temporal_model import LSTMModel, TemporalTransformer
        
        # Test LSTM
        lstm = LSTMModel(input_size=10, hidden_size=32, num_layers=2)
        x = torch.randn(4, 20, 10)
        output, (h, c) = lstm(x)
        assert output.shape == (4, 1)
        logger.info("✓ LSTM model works")
        
        # Test Transformer
        transformer = TemporalTransformer(
            input_size=10,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128
        )
        output = transformer(x)
        assert output.shape == (4, 1)
        logger.info("✓ Transformer model works")
        
        return True
    except Exception as e:
        logger.error(f"✗ PyTorch models test failed: {e}")
        return False


def test_metrics():
    """Test evaluation metrics."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting metrics...")
    
    try:
        import numpy as np
        from src.evaluation import MLMetrics, FinancialMetrics
        
        # Test ML metrics
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = MLMetrics.compute_metrics(y_true, y_pred)
        assert 'accuracy' in metrics
        logger.info("✓ ML Metrics work")
        
        # Test financial metrics
        returns = np.array([0.01, -0.005, 0.015, 0.002, -0.001])
        sharpe = FinancialMetrics.sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        logger.info("✓ Financial Metrics work")
        
        return True
    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}")
        return False


def main():
    """Run all quick tests."""
    # Setup logging
    logger = setup_logging(level='INFO')
    
    print("\n" + "="*60)
    print("STOCK EXPLAINABLE AI - QUICK TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_config_loading()
    all_passed &= test_knowledge_graph()
    all_passed &= test_torch_models()
    all_passed &= test_metrics()
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - System is ready!")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
