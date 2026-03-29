"""
Main Pipeline - Orchestrate the complete Stock Explainable AI system.
"""

import logging
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.utils.logger_config import setup_logging, load_config, create_directories
from src.data_acquisition import DataPipeline
from src.preprocessing import TextPreprocessor, TextEmbedder, SentimentFeatureExtractor
from src.graph_construction import FinancialKnowledgeGraph, GraphDataBuilder
from src.agentic_debate import AgenticDebateModule
from src.gnn_module import EventPropagationGNN
from src.temporal_model import HybridTemporalModel
from src.training import Trainer
from src.evaluation import PerformanceAnalyzer


logger = logging.getLogger(__name__)


class StockExplainableAIPipeline:
    """
    Main orchestrator for the complete Stock Explainable AI trading system.
    
    Orchestrates:
    1. Data acquisition (stock prices, macro indicators, news)
    2. Text preprocessing and embeddings
    3. Knowledge graph construction
    4. Agentic debate (Bull/Bear sentiment analysis)
    5. GNN-based event propagation
    6. Hybrid temporal prediction
    7. Training and evaluation
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logging(
            log_dir=log_config.get('log_dir', './logs'),
            level=log_config.get('level', 'INFO')
        )
        
        # Create directories
        create_directories(self.config)
        self.logger.info("Pipeline initialized")
        
        # Device
        self.device = self.config.get('training', {}).get('device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            self.logger.warning("CUDA not available, using CPU")
        
        # Module holders (lazy initialization)
        self.data_pipeline = None
        self.text_preprocessor = None
        self.text_embedder = None
        self.knowledge_graph = None
        self.debate_module = None
        self.gnn = None
        self.temporal_model = None
        self.trainer = None
        self.analyzer = None
    
    def init_data_pipeline(self):
        """Initialize data acquisition pipeline."""
        self.logger.info("Initializing data pipeline...")
        self.data_pipeline = DataPipeline(self.config)
        self.logger.info("Data pipeline ready")
        return self
    
    def init_text_processing(self):
        """Initialize text processing modules."""
        self.logger.info("Initializing text processing...")
        self.text_preprocessor = TextPreprocessor(self.config)
        self.text_embedder = TextEmbedder(self.config)
        self.logger.info("Text processing ready")
        return self
    
    def init_graph(self, edges_df: pd.DataFrame = None):
        """Initialize knowledge graph."""
        self.logger.info("Initializing knowledge graph...")
        
        if edges_df is not None:
            self.knowledge_graph = GraphDataBuilder.build_from_manual_data(
                self.config, edges_df
            )
        else:
            self.knowledge_graph = FinancialKnowledgeGraph(self.config)
        
        self.logger.info("Knowledge graph ready")
        return self
    
    def init_agentic_debate(self):
        """Initialize agentic debate module."""
        self.logger.info("Initializing agentic debate module...")
        self.debate_module = AgenticDebateModule(self.config)
        self.logger.info("Agentic debate ready")
        return self
    
    def init_gnn(self):
        """Initialize GNN module."""
        self.logger.info("Initializing GNN module...")
        self.gnn = EventPropagationGNN(self.config)
        self.gnn.to(self.device)
        self.logger.info("GNN ready")
        return self
    
    def init_temporal_model(self):
        """Initialize temporal prediction model."""
        self.logger.info("Initializing temporal model...")
        
        model_type = self.config.get('temporal_model', {}).get('model_type', 'lstm')
        
        self.temporal_model = HybridTemporalModel(
            historical_dim=10,  # OHLCV + technicals
            sentiment_dim=1,    # Judge sentiment score
            gnn_dim=self.config.get('gnn', {}).get('output_dim', 128),
            model_type=model_type,
            config=self.config
        )
        
        self.temporal_model.to(self.device)
        self.logger.info(f"Temporal model ({model_type}) ready")
        return self
    
    def init_trainer(self):
        """Initialize training module."""
        self.logger.info("Initializing trainer...")
        
        if self.temporal_model is None:
            self.init_temporal_model()
        
        self.trainer = Trainer(self.temporal_model, self.config, self.device)
        self.logger.info("Trainer ready")
        return self
    
    def init_analyzer(self):
        """Initialize performance analyzer."""
        self.logger.info("Initializing performance analyzer...")
        self.analyzer = PerformanceAnalyzer(self.config)
        self.logger.info("Analyzer ready")
        return self
    
    def fetch_data(self, tickers: list, start_date: str = None, end_date: str = None) -> Dict:
        """
        Step 1: Fetch all data (stock prices, macro indicators, news).
        
        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with stock_data, macro_data, news
        """
        if self.data_pipeline is None:
            self.init_data_pipeline()
        
        self.logger.info(f"Fetching data for {len(tickers)} tickers...")
        data = self.data_pipeline.fetch_all(tickers, start_date, end_date)
        
        self.logger.info("Data fetching completed")
        return data
    
    def preprocess_and_embed(self, news_texts: list) -> np.ndarray:
        """
        Step 2: Preprocess news texts and generate embeddings.
        
        Args:
            news_texts: List of news article texts
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if self.text_embedder is None:
            self.init_text_processing()
        
        self.logger.info(f"Preprocessing {len(news_texts)} news articles...")
        
        # Clean text
        cleaned_texts = [self.text_preprocessor.preprocess(text) for text in news_texts]
        
        # Embed
        embeddings = self.text_embedder.embed(cleaned_texts)
        
        self.logger.info(f"Generated embeddings: {embeddings.shape}")
        return embeddings
    
    def build_knowledge_graph(self, edges_df: pd.DataFrame = None):
        """
        Step 1B: Build financial knowledge graph.
        
        Args:
            edges_df: DataFrame with columns: source, target, relation_type, weight
        """
        if self.knowledge_graph is None:
            self.init_graph(edges_df)
        
        stats = self.knowledge_graph.get_graph_stats()
        self.logger.info(f"Graph stats: {stats}")
        return self.knowledge_graph
    
    def run_agentic_debate(self, news_texts: list) -> list:
        """
        Step 2: Run agentic debate on news articles.
        
        Args:
            news_texts: List of financial news texts
            
        Returns:
            List of debate results with sentiment scores
        """
        if self.debate_module is None:
            self.init_agentic_debate()
        
        self.logger.info(f"Running agentic debate on {len(news_texts)} texts...")
        results = self.debate_module.debate_batch(news_texts)
        
        self.logger.info("Agentic debate completed")
        return results
    
    def propagate_through_graph(
        self,
        node_embeddings: np.ndarray,
        edge_indices: np.ndarray
    ) -> torch.Tensor:
        """
        Step 3: Propagate sentiment/events through knowledge graph using GNN.
        
        Args:
            node_embeddings: Node feature embeddings
            edge_indices: Graph edge indices
            
        Returns:
            Propagated node embeddings
        """
        if self.gnn is None:
            self.init_gnn()
        
        self.logger.info("Propagating through knowledge graph...")
        
        node_features = torch.FloatTensor(node_embeddings).to(self.device)
        result = self.gnn.propagate(node_features, edge_indices)
        
        self.logger.info("GNN propagation completed")
        return result['final']
    
    def predict(
        self,
        historical_features: torch.Tensor,
        sentiment_scores: torch.Tensor,
        gnn_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Step 4: Generate predictions using hybrid temporal model.
        
        Args:
            historical_features: Historical price/technical data
            sentiment_scores: Sentiment scores from debate module
            gnn_embeddings: Propagated GNN embeddings
            
        Returns:
            Direction predictions (0=DOWN, 1=UP)
        """
        if self.temporal_model is None:
            self.init_temporal_model()
        
        self.temporal_model.eval()
        with torch.no_grad():
            predictions = self.temporal_model(
                historical_features.to(self.device),
                sentiment_scores.to(self.device),
                gnn_embeddings.to(self.device)
            )
        
        return predictions
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Step 5: Train the complete system.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.trainer is None:
            self.init_trainer()
        
        self.logger.info("Starting training...")
        history = self.trainer.train(
            train_loader,
            val_loader,
            num_epochs,
            early_stopping_patience
        )
        
        self.logger.info("Training completed")
        return history
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        prices: np.ndarray,
        returns: np.ndarray
    ) -> Dict:
        """
        Step 6: Comprehensive evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            prices: Closing prices
            returns: Log returns
            
        Returns:
            Complete evaluation results
        """
        if self.analyzer is None:
            self.init_analyzer()
        
        self.logger.info("Running comprehensive evaluation...")
        results = self.analyzer.evaluate(y_true, y_pred, y_pred_proba, prices, returns)
        
        self.analyzer.print_report(results)
        
        return results
    
    def save(self, save_dir: str):
        """
        Save all models and configurations.
        
        Args:
            save_dir: Directory to save models
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'temporal_model': self.temporal_model.state_dict() if self.temporal_model else None,
            'gnn': self.gnn.state_dict() if self.gnn else None,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pt'))
        self.logger.info(f"Checkpoint saved to {save_dir}")
    
    def load(self, checkpoint_path: str):
        """
        Load saved models.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.temporal_model:
            self.temporal_model.load_state_dict(checkpoint['temporal_model'])
        
        if self.gnn:
            self.gnn.load_state_dict(checkpoint['gnn'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """Example usage of the pipeline."""
    # Load configuration
    config_path = 'configs/config.yaml'
    
    # Initialize pipeline
    pipeline = StockExplainableAIPipeline(config_path)
    
    # Initialize all modules
    (pipeline
     .init_data_pipeline()
     .init_text_processing()
     .init_agentic_debate()
     .init_gnn()
     .init_temporal_model()
     .init_trainer()
     .init_analyzer())
    
    logger.info("Stock Explainable AI Pipeline initialized successfully!")
    
    # Example: Fetch data
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = pipeline.fetch_data(tickers, start_date='2023-01-01', end_date='2024-01-01')
    
    logger.info(f"Fetched data for: {list(data['stock_data'].keys())}")


if __name__ == '__main__':
    main()
