# Development Guide

## Project Architecture

### Modular Design Philosophy

This project follows a **strict modular design**:

- **Single Responsibility**: Each module handles one distinct task
- **Loose Coupling**: Modules communicate via standard interfaces
- **High Cohesion**: Related functionality grouped together
- **Testability**: Every module independently testable

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                              │
│  (yfinance, FRED, News APIs) → Raw Data                         │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│           PREPROCESSING & FEATURE EXTRACTION                     │
│  Text → Embeddings (768-dim) + OHLCV + Macro + Graphs          │
└───────────────┬─────────────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
    ┌────────┐     ┌──────────┐
    │ GRAPH  │     │ SENTIMENT│
    │CONSTRUCTION│  │ANALYSIS  │
    └────┬───┘     └──────┬───┘
         │                │
         ▼                ▼
    ┌──────────────────────────┐
    │   GNN PROPAGATION        │
    │   (2-hop diffusion)      │
    └────┬────────────────────┘
         │
         ▼
    ┌──────────────────────────┐
    │  FEATURE FUSION          │
    │ (Historical + Sentiment  │
    │  + GNN embeddings)       │
    └────┬────────────────────┘
         │
         ▼
    ┌──────────────────────────┐
    │  TEMPORAL MODEL          │
    │  (LSTM / Transformer)    │
    └────┬────────────────────┘
         │
         ▼
    ┌──────────────────────────┐
    │  PREDICTIONS             │
    │  (Direction 0/1)         │
    └────┬────────────────────┘
         │
         ▼
    ┌──────────────────────────┐
    │  EVALUATION              │
    │  (ML + Financial Metrics)│
    └──────────────────────────┘
```

## Module Deep Dive

### 1. Data Acquisition Module

**File**: `src/data_acquisition/fetcher.py`

**Responsibilities**:
- Fetch historical stock prices (OHLCV)
- Fetch macro indicators (VIX, inflation, unemployment)
- Load financial news articles
- Validate data integrity

**Key Classes**:
```python
StockDataFetcher         # yfinance wrapper
MacroeconomicDataFetcher # FRED integration
NewsDataLoader          # News data handling
DataPipeline            # Unified interface
```

**Design Pattern**: Factory + Adapter

### 2. Preprocessing Module

**File**: `src/preprocessing/text_processor.py`

**Responsibilities**:
- Clean and normalize text
- Tokenize with BERT tokenizer
- Generate embeddings with sentence-transformers
- Extract sentiment features

**Key Classes**:
```python
TextPreprocessor            # Text cleaning
TextEmbedder               # Embedding generation
TokenizedTextProcessor     # Tokenization
SentimentFeatureExtractor  # Feature engineering
```

**Design Pattern**: Pipeline

### 3. Graph Construction Module

**File**: `src/graph_construction/knowledge_graph.py`

**Responsibilities**:
- Build company relationship graphs
- Store node embeddings
- Query neighbors and multi-hop paths
- Convert to tensor format for GNN

**Key Classes**:
```python
FinancialKnowledgeGraph  # Main graph class
GraphDataBuilder         # Builder pattern
```

**Design Pattern**: Builder + Graph Pattern

### 4. Agentic Debate Module

**File**: `src/agentic_debate/debate_module.py`

**Responsibilities**:
- Extract positive catalysts (Bull)
- Identify risk factors (Bear)
- Judge and score sentiment
- Explain reasoning

**Key Classes**:
```python
BaseFinancialAgent    # Agent base class
BullAgent            # Bullish sentiment extraction
BearAgent            # Bearish sentiment extraction
JudgeModel           # Sentiment scoring
AgenticDebateModule  # Orchestrator
```

**Design Pattern**: Strategy + Chain of Responsibility

### 5. GNN Module

**File**: `src/gnn_module/gnn.py`

**Responsibilities**:
- Multi-layer graph convolution
- Propagate embeddings through graph
- Capture second-order effects
- Compute node impact scores

**Key Classes**:
```python
GraphConvolutionLayer    # Single GC layer
GCNModel                # Multi-layer GCN
GATLayer               # Attention-based layer
EventPropagationGNN    # Main propagation model
GNNTrainer            # Training loop
```

**Design Pattern**: Progressive Enhancement (layers)

### 6. Temporal Model Module

**File**: `src/temporal_model/temporal.py`

**Responsibilities**:
- LSTM sequence modeling
- Transformer attention-based processing
- Feature fusion from multiple modalities
- Direction prediction

**Key Classes**:
```python
LSTMModel                # LSTM variant
TransformerBlock        # Self-attention block
TemporalTransformer     # Transformer model
HybridTemporalModel     # Feature fusion + temporal
```

**Design Pattern**: Composite + Adapter

### 7. Training Module

**File**: `src/training/trainer.py`

**Responsibilities**:
- Define hybrid loss (BCE + MSE)
- Manage optimization (Adam, AdamW)
- Learning rate scheduling
- Model checkpointing

**Key Classes**:
```python
HybridLoss  # Weighted loss combination
Trainer     # Training loop orchestrator
```

**Design Pattern**: Strategy for loss/optimizer

### 8. Evaluation Module

**File**: `src/evaluation/metrics.py`

**Responsibilities**:
- Compute ML metrics (accuracy, F1, etc.)
- Calculate financial metrics (Sharpe, Sortino, etc.)
- Backtest trading strategy
- Generate performance reports

**Key Classes**:
```python
MLMetrics            # Classification metrics
FinancialMetrics     # Trading metrics
Backtester          # Strategy evaluation
PerformanceAnalyzer # Unified evaluation
```

**Design Pattern**: Strategy + Report Generation

## Adding New Features

### Add a New Data Source

1. **Create fetcher class** in `src/data_acquisition/fetcher.py`:
```python
class MyDataFetcher:
    def __init__(self, config):
        self.config = config
    
    def fetch(self, ...):
        # Implement fetching logic
        pass
```

2. **Register in DataPipeline**:
```python
class DataPipeline:
    def __init__(self, config):
        self.my_fetcher = MyDataFetcher(config)
    
    def fetch_all(self, ...):
        result['my_data'] = self.my_fetcher.fetch(...)
```

3. **Add tests** in `tests/test_modules.py`

### Add a New Model

1. **Create model class** in appropriate module:
```python
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Build model
    
    def forward(self, x):
        # Forward pass
        pass
```

2. **Add to pipeline initialization**:
```python
def init_my_model(self):
    self.logger.info("Initializing my model...")
    self.my_model = MyModel(self.config)
    return self
```

3. **Test with sample data**

## Testing Strategy

### Unit Tests
- Test each class in isolation
- Use mock objects for dependencies
- Test edge cases and error handling

### Integration Tests
- Test module interactions
- Use real data (but small samples)
- End-to-end workflow tests

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_modules.py::TestDataAcquisition -v

# With coverage
pytest tests/ --cov=src

# Generate HTML report
pytest tests/ --cov=src --cov-report=html
```

## Performance Optimization

### Model Inference
```python
model.eval()
with torch.no_grad():
    output = model(input)
```

### Batch Processing
```python
embedder.embed(texts, batch_size=64)  # Process in batches
```

### GPU Memory
```python
# Reduce batch size
config['training']['batch_size'] = 8

# Use gradient checkpointing (advanced)
model = torch.nn.utils.checkpoint(model, x)
```

## Debugging

### Enable Verbose Logging
```python
logger = setup_logging(level='DEBUG')
```

### Monitor Training
```bash
tensorboard --logdir=./logs
# or
wandb login && wandb sync
```

### Inspect Data
```python
# Check shapes
print(f"Features shape: {features.shape}")
print(f"Graph nodes: {graph.graph.number_of_nodes()}")
print(f"Embeddings: {embeddings.shape}")

# Validate values
assert (embeddings >= -1).all() and (embeddings <= 1).all()
assert not np.isnan(predictions).any()
```

## Code Style

### Naming Conventions
- Classes: `PascalCase` (e.g., `StockDataFetcher`)
- Functions/methods: `snake_case` (e.g., `fetch_data`)
- Constants: `UPPER_CASE` (e.g., `MAX_SEQUENCE_LENGTH`)
- Private: `_leading_underscore` (e.g., `_validate`)

### Documentation
```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Clear description of what function does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When validation fails
    """
```

### Imports
```python
# Standard library
import logging
from typing import Dict, List, Tuple

# Third party
import numpy as np
import torch

# Local
from src.utils import load_config
```

## Common Issues

### ImportError: No module named 'src'
**Solution**: Ensure you're running from project root and `src/` is importable:
```bash
export PYTHONPATH="/path/to/Stock_Explainable:$PYTHONPATH"
python script.py
```

### CUDA Out of Memory
**Solution**: Reduce batch size or use CPU:
```python
config['training']['batch_size'] = 8
config['training']['device'] = 'cpu'
```

### Transformer model download fails
**Solution**: Pre-download models:
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"
```

## Contributing Checklist

- [ ] Code follows style guide
- [ ] Added docstrings to all functions
- [ ] Written unit tests
- [ ] Tests pass locally
- [ ] No breaking changes to APIs
- [ ] Updated README if needed
- [ ] Committed with clear message

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Graph Neural Networks](https://arxiv.org/abs/1812.04202)
- [LSTM & RNNs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Financial Machine Learning](https://mlfinlab.readthedocs.io/)

---

**Last Updated**: 2024
**Maintained**: Active Development
