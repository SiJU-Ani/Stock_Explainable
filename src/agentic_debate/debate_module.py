"""
Agentic Debate Module - Bull/Bear agents with Judge model for sentiment scoring.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    raise ImportError("transformers required")

logger = logging.getLogger(__name__)


class BaseFinancialAgent:
    """Base class for Bull/Bear agents."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize agent.
        
        Args:
            name: Agent name ('Bull' or 'Bear')
            config: Configuration dictionary
        """
        self.name = name
        self.config = config.get('agentic_debate', {})
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_name = self.config.get('bull_model' if name == 'Bull' else 'bear_model',
                                     'distilbert-base-uncased')
        
        logger.info(f"Initializing {name} agent with {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.embedding_dim = self.model.config.hidden_size
    
    def extract_sentiment_vector(self, text: str) -> np.ndarray:
        """
        Extract sentiment vector from text using transformer.
        
        Args:
            text: Input financial text
            
        Returns:
            Sentiment vector (mean pooled embeddings)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = (embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1)
            mean_embeddings = sum_embeddings / (sum_mask + 1e-9)
        
        return mean_embeddings.cpu().numpy()[0]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and extract reasoning vector.
        Override in subclasses.
        
        Args:
            text: Input text
            
        Returns:
            Analysis results with 'reasoning_vector'
        """
        reasoning_vector = self.extract_sentiment_vector(text)
        return {
            'agent': self.name,
            'reasoning_vector': reasoning_vector,
            'vector_dim': reasoning_vector.shape[0],
        }


class BullAgent(BaseFinancialAgent):
    """Bull agent - identifies positive catalysts and growth opportunities."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('Bull', config)
        self.positive_keywords = [
            'growth', 'gain', 'profit', 'positive', 'upgrade', 'beat',
            'expansion', 'partnership', 'acquisition', 'innovation',
            'record', 'strong', 'outperform'
        ]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Extract positive catalysts and bullish factors.
        
        Args:
            text: Financial news text
            
        Returns:
            Bull analysis with reasoning vector
        """
        result = super().analyze(text)
        
        # Identify positive keywords
        text_lower = text.lower()
        found_keywords = [kw for kw in self.positive_keywords if kw in text_lower]
        
        result['positive_catalysts'] = found_keywords
        result['bullish_score'] = min(len(found_keywords) / len(self.positive_keywords), 1.0)
        
        logger.debug(f"Bull analysis: found {len(found_keywords)} positive catalysts")
        return result


class BearAgent(BaseFinancialAgent):
    """Bear agent - identifies risks and negative factors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('Bear', config)
        self.risk_keywords = [
            'risk', 'loss', 'decline', 'challenge', 'downgrade', 'miss',
            'regulatory', 'lawsuit', 'disruption', 'warning', 'recession',
            'decline', 'weakness', 'underperform', 'sell-off'
        ]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Extract risk factors and bearish signals.
        
        Args:
            text: Financial news text
            
        Returns:
            Bear analysis with reasoning vector
        """
        result = super().analyze(text)
        
        # Identify risk keywords
        text_lower = text.lower()
        found_keywords = [kw for kw in self.risk_keywords if kw in text_lower]
        
        result['risk_factors'] = found_keywords
        result['bearish_score'] = min(len(found_keywords) / len(self.risk_keywords), 1.0)
        
        logger.debug(f"Bear analysis: found {len(found_keywords)} risk factors")
        return result


class JudgeModel(nn.Module):
    """
    Judge model - evaluates Bull/Bear reasoning and assigns sentiment scores.
    """
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        """
        Initialize judge model.
        
        Args:
            input_dim: Input feature dimension (2 * embedding_dim)
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config.get('agentic_debate', {})
        self.input_dim = input_dim
        
        model_type = self.config.get('judge_model_type', 'logistic_regression')
        
        if model_type == 'logistic_regression':
            # Will use sklearn
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.is_nn = False
        else:  # feedforward network
            hidden_dim = self.config.get('judge_hidden_dim', 128)
            dropout = self.config.get('judge_dropout', 0.3)
            
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output: 0-1 sentiment score
            )
            self.is_nn = True
    
    def forward(self, bull_vector: torch.Tensor, bear_vector: torch.Tensor) -> torch.Tensor:
        """
        Judge the bull/bear debate.
        
        Args:
            bull_vector: Bull agent reasoning vector
            bear_vector: Bear agent reasoning vector
            
        Returns:
            Sentiment score (0 = bearish, 1 = bullish)
        """
        # Concatenate vectors
        combined = torch.cat([bull_vector, bear_vector], dim=-1)
        
        if self.is_nn:
            return self.model(combined)
        else:
            # For sklearn, need numpy
            return combined
    
    def judge_debate(
        self,
        bull_analysis: Dict[str, Any],
        bear_analysis: Dict[str, Any],
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Judge the entire debate between Bull and Bear agents.
        
        Args:
            bull_analysis: Bull agent analysis results
            bear_analysis: Bear agent analysis results
            device: Device to use
            
        Returns:
            Judge verdict with sentiment score
        """
        bull_vector = bull_analysis['reasoning_vector']
        bear_vector = bear_analysis['reasoning_vector']
        
        if self.is_nn:
            bull_tensor = torch.FloatTensor(bull_vector).unsqueeze(0).to(device)
            bear_tensor = torch.FloatTensor(bear_vector).unsqueeze(0).to(device)
            
            with torch.no_grad():
                sentiment_score = self.forward(bull_tensor, bear_tensor).item()
        else:
            # For sklearn
            combined = np.concatenate([bull_vector, bear_vector]).reshape(1, -1)
            # Fit on single sample (simplified)
            sentiment_score = 0.5 + 0.4 * (
                bull_analysis.get('bullish_score', 0.5) - 
                bear_analysis.get('bearish_score', 0.5)
            )
        
        verdict = {
            'sentiment_score': sentiment_score,
            'sentiment_label': 'BULLISH' if sentiment_score > 0.6 else 'BEARISH' if sentiment_score < 0.4 else 'NEUTRAL',
            'bull_strength': bull_analysis.get('bullish_score', 0),
            'bear_strength': bear_analysis.get('bearish_score', 0),
            'confidence': abs(sentiment_score - 0.5) * 2,  # 0-1
        }
        
        logger.info(f"Judge verdict: {verdict['sentiment_label']} (score: {sentiment_score:.3f})")
        return verdict


class AgenticDebateModule:
    """
    Full agentic debate system: Bull agent, Bear agent, and Judge.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the debate module."""
        self.config = config
        self.bull = BullAgent(config)
        self.bear = BearAgent(config)
        
        # Create judge model
        input_dim = 2 * self.bull.embedding_dim  # Both agents' embeddings
        self.judge = JudgeModel(input_dim, config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def debate(self, text: str) -> Dict[str, Any]:
        """
        Run full debate on a news article.
        
        Args:
            text: Financial news text
            
        Returns:
            Debate results with sentiment scores
        """
        logger.info(f"Starting debate on text ({len(text)} chars)")
        
        # Bull analysis
        bull_analysis = self.bull.analyze(text)
        logger.info(f"Bull found {len(bull_analysis.get('positive_catalysts', []))} catalysts")
        
        # Bear analysis
        bear_analysis = self.bear.analyze(text)
        logger.info(f"Bear found {len(bear_analysis.get('risk_factors', []))} risks")
        
        # Judge verdict
        verdict = self.judge.judge_debate(bull_analysis, bear_analysis, self.device)
        
        return {
            'bull_analysis': bull_analysis,
            'bear_analysis': bear_analysis,
            'judge_verdict': verdict,
            'text_length': len(text),
        }
    
    def debate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run debate on multiple texts.
        
        Args:
            texts: List of financial news texts
            
        Returns:
            List of debate results
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            results.append(self.debate(text))
        
        return results
