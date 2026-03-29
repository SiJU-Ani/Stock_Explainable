"""
Text Preprocessing Module - Tokenization and embeddings using Transformers.
"""

import logging
from typing import List, Dict, Any, Tuple, Union
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError:
    raise ImportError("transformers and torch required. Install with: pip install transformers torch")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")


logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess financial text: tokenization, cleaning, normalization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config.get('preprocessing', {})
        self.max_length = self.config.get('max_sequence_length', 512)
        self.lowercase = self.config.get('lowercase', True)
        self.remove_special = self.config.get('remove_special_chars', True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean financial text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters (optional)
        if self.remove_special:
            import re
            # Keep alphanumeric, spaces, and financial symbols
            text = re.sub(r'[^a-z0-9\s\-\$\%\(\)]', '', text)
        
        return text
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """
        Truncate text to max length (before tokenization).
        
        Args:
            text: Text to truncate
            max_length: Max characters (approximate)
            
        Returns:
            Truncated text
        """
        max_length = max_length or self.max_length * 4  # Rough estimate
        return text[:max_length]
    
    def preprocess(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Preprocess single text or batch.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Cleaned text(s)
        """
        if isinstance(texts, str):
            return self.clean_text(self.truncate_text(texts))
        
        return [self.clean_text(self.truncate_text(t)) for t in texts]


class TextEmbedder:
    """Generate embeddings for financial text using pretrained Transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text embedder.
        
        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config.get('preprocessing', {})
        self.model_name = self.config.get(
            'embedding_model',
            'sentence-transformers/distilbert-base-uncased-finetuned-sst-2-english'
        )
        self.max_length = self.config.get('max_sequence_length', 512)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            Embeddings as numpy array shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("texts cannot be empty")
        
        # Remove empty texts
        texts = [t for t in texts if t and len(str(t).strip()) > 0]
        if not texts:
            raise ValueError("All texts are empty after filtering")
        
        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_batch(
        self,
        texts_list: List[List[str]],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        Embed multiple batches of texts.
        
        Args:
            texts_list: List of lists of texts
            batch_size: Batch size
            
        Returns:
            List of embedding arrays
        """
        return [self.embed(texts, batch_size) for texts in texts_list]


class TokenizedTextProcessor:
    """Tokenize texts using BERT tokenizer (for fine-tuning models)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tokenizer."""
        self.config = config.get('preprocessing', {})
        self.tokenizer_name = self.config.get('tokenizer', 'distilbert-base-uncased')
        self.max_length = self.config.get('max_sequence_length', 512)
        
        logger.info(f"Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    
    def tokenize(
        self,
        texts: List[str],
        padding: str = 'max_length',
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts for model input.
        
        Args:
            texts: List of texts
            padding: Padding strategy
            truncation: Whether to truncate long texts
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', 'token_type_ids'
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt'
        )
        
        logger.info(f"Tokenized {len(texts)} texts")
        return encoded


class SentimentFeatureExtractor:
    """
    Extract sentiment-related features from text.
    Designed for Bull/Bear agent analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature extractor."""
        self.config = config
        self.embedder = TextEmbedder(config)
        self.tokenizer = TokenizedTextProcessor(config)
    
    def extract_features(
        self,
        text: str,
        feature_type: str = 'both'
    ) -> Dict[str, np.ndarray]:
        """
        Extract sentiment features from text.
        
        Args:
            text: Input text
            feature_type: 'embedding', 'tokenized', or 'both'
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        if feature_type in ['embedding', 'both']:
            embedding = self.embedder.embed(text, batch_size=1)
            features['embedding'] = embedding[0]  # shape: (embedding_dim,)
        
        if feature_type in ['tokenized', 'both']:
            tokenized = self.tokenizer.tokenize([text])
            features['input_ids'] = tokenized['input_ids']
            features['attention_mask'] = tokenized['attention_mask']
        
        return features
