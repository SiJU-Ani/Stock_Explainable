# Stock Explainable AI - Modular Trading System

## 🚀 Overview

A comprehensive, modular framework for building explainable AI-driven stock trading systems. This system combines multiple advanced techniques:

- **NLP & Sentiment Analysis**: Bull/Bear agent debate for explainable reasoning
- **Knowledge Graph Neural Networks**: Propagate market signals through company relationships
- **Temporal Models**: LSTM/Transformer for directional prediction
- **Financial Metrics**: Comprehensive backtesting and evaluation framework

## 📁 Project Structure

```
Stock_Explainable/
├── src/                          # Main source code
│   ├── data_acquisition/        # Data fetching (yfinance, FRED, news)
│   ├── preprocessing/           # Text tokenization & embeddings
│   ├── graph_construction/      # Knowledge graph builder
│   ├── agentic_debate/          # Bull/Bear agents + Judge model
│   ├── gnn_module/              # Graph Neural Network propagation
│   ├── temporal_model/          # LSTM/Transformer models
│   ├── training/                # Training loops & optimization
│   ├── evaluation/              # Financial & ML metrics
│   └── utils/                   # Config, logging, helpers
├── tests/                        # Unit & integration tests
├── notebooks/                    # Jupyter notebooks for experimentation
├── configs/                      # Configuration files
├── data/                         # Data storage (raw, processed)
├── requirements.txt              # Python dependencies
└── main_pipeline.py              # Main orchestrator script
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Clone and navigate to project:**
```bash
cd Stock_Explainable
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install spaCy model (for NLP):**
```bash
python -m spacy download en_core_web_sm
```

5. **Setup environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required API Keys:**
- `FRED_API_KEY`: Get from https://fred.stlouisfed.org/docs/api/
- `WANDB_KEY` (optional): For experiment tracking

## 📚 Module Documentation

### 1. Data Acquisition (`src/data_acquisition/`)

Fetch domain from multiple sources:

```python
from src.data_acquisition import DataPipeline

config = load_config('configs/config.yaml')
pipeline = DataPipeline(config)

# Fetch all data
data = pipeline.fetch_all(
    tickers=['AAPL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Access components
stock_data = data['stock_data']        # Dict[ticker -> DataFrame]
macro_data = data['macro_data']        # DataFrame with VIX, inflation, etc.
news_data = data['news']               # List of news articles
```

**Data Sources:**
- **Stock Prices**: yfinance (free, no API needed)
- **Macro Indicators**: FRED API (inflation, VIX, unemployment)
- **News**: FNSPID dataset (via Hugging Face)

### 2. Text Processing (`src/preprocessing/`)

Convert raw text into embeddings:

```python
from src.preprocessing import TextPreprocessor, TextEmbedder

# Clean text
preprocessor = TextPreprocessor(config)
cleaned = preprocessor.preprocess(raw_news_text)

# Generate embeddings
embedder = TextEmbedder(config)
embeddings = embedder.embed(cleaned_texts)  # Shape: (n_texts, 768)
```

**Features:**
- Tokenization with DistilBERT
- Sentence embeddings (768-dim vectors)
- Batch processing support

### 3. Knowledge Graph (`src/graph_construction/`)

Build company relationship networks:

```python
from src.graph_construction import FinancialKnowledgeGraph

graph = FinancialKnowledgeGraph(config)

# Add companies
graph.add_node('AAPL', 'Apple', embedding=apple_embedding)
graph.add_node('TSLA', 'Tesla', embedding=tesla_embedding)

# Add relationships
graph.add_edge('APPLE', 'TSLA', relation_type='competitor', weight=0.8)
graph.add_edge('APPLE', 'AMD', relation_type='supplier', weight=0.6)

# Query
neighbors = graph.get_neighbors('APPLE', relation_type='supplier')
n_hop = graph.get_n_hop_neighbors('APPLE', n_hops=2)

# Get stats
stats = graph.get_graph_stats()
```

**Relationships:**
- `supplier`: Supply chain dependencies
- `customer`: Customer relationships
- `competitor`: Market competition

### 4. Agentic Debate (`src/agentic_debate/`)

Bull/Bear agents + Judge model for explainable sentiment:

```python
from src.agentic_debate import AgenticDebateModule

debate = AgenticDebateModule(config)

# Run debate on an article
result = debate.debate(news_article_text)

# Access results
bull_analysis = result['bull_analysis']          # Positive catalysts
bear_analysis = result['bear_analysis']          # Risk factors
verdict = result['judge_verdict']                # Sentiment score (0-1)

# Interpret
print(f"Sentiment: {verdict['sentiment_label']}")  # BULLISH/BEARISH/NEUTRAL
print(f"Confidence: {verdict['confidence']:.2f}")   # 0-1
```

**Agents:**
- **Bull Agent**: Extracts positive catalysts (growth, partnerships, etc.)
- **Bear Agent**: Identifies risks (regulatory, disruptions, etc.)
- **Judge Model**: Weighted scoring of both perspectives

### 5. GNN Module (`src/gnn_module/`)

Propagate market signals through the knowledge graph:

```python
from src.gnn_module import EventPropagationGNN

gnn = EventPropagationGNN(config)

# Propagate embeddings
node_features = torch.randn(num_companies, 768)  # Initial embeddings
edge_indices = graph.to_tensor_format()['edge_indices']

propagated = gnn.propagate(node_features, edge_indices, num_hops=2)

# Get impact scores
impact = gnn.get_node_impact(company_id=0, propagated_features=propagated['final'])
```

**Propagation:**
- Captures second-order market impacts
- How news for one company affects competitors/suppliers
- 3-layer GCN by default

### 6. Temporal Models (`src/temporal_model/`)

LSTM/Transformer for stock direction prediction:

```python
from src.temporal_model import HybridTemporalModel

# Create hybrid model combining all features
model = HybridTemporalModel(
    historical_dim=10,      # OHLCV + technical indicators
    sentiment_dim=1,        # Judge sentiment score
    gnn_dim=128,            # GNN propagated embeddings
    model_type='lstm'       # or 'transformer'
)

# Forward pass
historical = torch.randn(batch_size, seq_length, 10)
sentiment = torch.randn(batch_size, seq_length, 1)
gnn_emb = torch.randn(batch_size, seq_length, 128)

predictions = model(historical, sentiment, gnn_emb)  # (batch_size, 1)
```

**Models:**
- **LSTM**: 2-layer with 128 hidden units
- **Transformer**: 8-head attention, 3 layers
- **Feature Fusion**: Concatenates all modalities

### 7. Training (`src/training/`)

Train the complete system with hybrid loss:

```python
from src.training import Trainer

trainer = Trainer(model, config, device='cuda')

# Train with early stopping
history = trainer.train(
    train_loader=train_dl,
    val_loader=val_dl,
    num_epochs=100,
    early_stopping_patience=10,
    save_freq=5  # Save every 5 epochs
)

# Access history
print(f"Best epoch: {history['best_epoch']}")
print(f"Best val loss: {history['best_val_loss']:.4f}")
```

**Loss Functions:**
- Binary Cross Entropy (classification)
- Mean Squared Error (regression, optional)
- Weighted combination (0.7 * BCE + 0.3 * MSE)

**Optimization:**
- Adam optimizer with weight decay
- Learning rate scheduling (cosine annealing)
- Gradient clipping (norm=1.0)

### 8. Evaluation (`src/evaluation/`)

Comprehensive ML + financial metrics:

```python
from src.evaluation import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(config)

results = analyzer.evaluate(
    y_true=actual_labels,
    y_pred=predictions,
    y_pred_proba=prediction_probs,
    prices=closing_prices,
    returns=log_returns
)

analyzer.print_report(results)
```

**ML Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC

**Financial Metrics:**
- Sharpe Ratio
- Sortino Ratio (downside volatility)
- Maximum Drawdown
- Calmar Ratio
- Cumulative Return
- Win Rate

**Backtesting:**
- Transaction costs: 0.1%
- Slippage: 0.05%
- Comparison to buy-hold benchmark

## 🧪 Testing

Run comprehensive unit tests:

```bash
# All tests
python tests/test_modules.py

# Specific test class
python -m pytest tests/test_modules.py::TestDataAcquisition -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- Data acquisition (yfinance, FRED)
- Text preprocessing (tokenization, embeddings)
- Knowledge graph (nodes, edges, querying)
- Agentic debate (Bull, Bear, Judge)
- Temporal models (LSTM, Transformer)
- GNN propagation
- Metrics (ML, financial)

## 🚀 Quick Start Example

```python
from main_pipeline import StockExplainableAIPipeline
from src.utils import load_config

# Initialize
config = load_config('configs/config.yaml')
pipeline = StockExplainableAIPipeline(config)

# Build system step-by-step
(pipeline
 .init_data_pipeline()
 .init_text_processing()
 .init_agentic_debate()
 .init_gnn()
 .init_temporal_model()
 .init_trainer()
 .init_analyzer())

# 1. Fetch data
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = pipeline.fetch_data(tickers)

# 2. Preprocess & debate
debate_results = pipeline.run_agentic_debate(
    data['news']  # News articles
)

# 3. Build knowledge graph (with embeddings)
import pandas as pd
edges_df = pd.DataFrame({
    'source': ['AAPL', 'AAPL'],
    'target': ['MSFT', 'GOOGL'],
    'relation_type': ['competitor', 'customer'],
    'weight': [0.8, 0.6]
})
graph = pipeline.build_knowledge_graph(edges_df)

# 4. Propagate through GNN and predict
predictions = pipeline.predict(
    historical_features=historical_data,
    sentiment_scores=sentiment_data,
    gnn_embeddings=gnn_data
)

# 5. Train
history = pipeline.train(train_loader, val_loader, num_epochs=100)

# 6. Evaluate
results = pipeline.evaluate(y_true, y_pred, y_proba, prices, returns)

# 7. Save
pipeline.save('./models/checkpoint')
```

## 📊 Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Data sources
data:
  yfinance:
    start_date: "2020-01-01"
  fred:
    api_key: "${FRED_API_KEY}"  # From .env

# Model architectures
gnn:
  num_layers: 3
  hidden_dim: 256
  output_dim: 128

temporal_model:
  model_type: "lstm"
  lstm:
    hidden_size: 128
    num_layers: 2

# Training
training:
  num_epochs: 100
  learning_rate: 1e-3
  batch_size: 32
  early_stopping_patience: 10

# Evaluation
evaluation:
  backtesting:
    initial_capital: 100000
    transaction_costs: 0.001
    risk_free_rate: 0.05
```

## 🔍 Monitoring

Track experiments with Weights & Biases:

```python
# Enable in config.yaml
experiments:
  use_wandb: true
  project: "stock-explainable-ai"

# View at: https://wandb.ai/your-username/stock-explainable-ai
```

## 📝 Common Tasks

### Train on New Data
```bash
python main_pipeline.py --mode train --config configs/config.yaml
```

### Backtest Strategy
```bash
python scripts/backtest.py --model checkpoint/best_model.pt --data ./data/prices.csv
```

### Generate Predictions
```bash
python scripts/predict.py --tickers AAPL MSFT GOOGL --model ./models/checkpoint.pt
```

### Visualize Knowledge Graph
```bash
python scripts/visualize_graph.py --graph ./data/knowledge_graph.graphml
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
config['training']['batch_size'] = 8

# Use CPU
config['training']['device'] = 'cpu'
```

### Slow Embeddings
```python
# Use smaller model
config['preprocessing']['embedding_model'] = 'sentence-transformers/all-MiniLM-L6-v2'

# Increased batch size
config['preprocessing']['batch_size'] = 64
```

### API Rate Limits
```python
# Add delays between requests
import time
time.sleep(0.5)

# Use data caching
cache_dir = './data/cache'
```

## 📖 References

- **Transformers**: Vaswani et al. (2017) - Attention Is All You Need
- **GNNs**: Kipf & Welling (2016) - Semi-Supervised Classification with Graph CNNs
- **Financial Sentiment**: Malo et al. - Good News or Bad News?
- **Stock Prediction**: Fischer & Krauss (2018) - Deep Learning with Long Short-Term Memory

## 📄 License

MIT

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -am 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Submit pull request

## 📧 Support

For issues and questions:
- GitHub Issues: [Create issue]
- Email: support@example.com
- Documentation: [Wiki]

---

**Last Updated**: 2024
**Version**: 0.1.0
**Status**: Production Ready ✅
