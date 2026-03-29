# QUICK START GUIDE

## 5-Minute Setup

### 1. Install Dependencies
```bash
cd Stock_Explainable
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python quick_test.py
```

You should see:
```
✓ Data acquisition module
✓ Preprocessing module
✓ Graph construction module
✓ Agentic debate module
✓ GNN module
✓ Temporal model module
✓ Training module
✓ Evaluation module
✓ ALL TESTS PASSED - System is ready!
```

### 3. Get FRED API Key (Free)
- Go to: https://fred.stlouisfed.org/docs/api/
- Register & get free API key
- Create `.env` file:
```bash
cp .env.example .env
# Edit .env and add: FRED_API_KEY=your_key_here
```

## Your First Run (15 minutes)

### Create a test file `test_run.py`:

```python
from main_pipeline import StockExplainableAIPipeline
import pandas as pd

# Load pipeline
pipeline = StockExplainableAIPipeline('configs/config.yaml')

# Initialize modules
(pipeline
 .init_data_pipeline()
 .init_text_processing()
 .init_agentic_debate()
 .init_gnn()
 .init_temporal_model())

# 1. Fetch stock data
print("\n1️⃣ Fetching stock data...")
tickers = ['AAPL', 'MSFT']
data = pipeline.fetch_data(
    tickers=tickers,
    start_date='2023-01-01',
    end_date='2023-12-31'
)
print(f"✓ Got {len(data['stock_data'])} stock datasets")

# 2. Build knowledge graph
print("\n2️⃣ Building knowledge graph...")
edges_df = pd.DataFrame({
    'source': ['AAPL'],
    'target': ['MSFT'],
    'relation_type': ['competitor'],
    'weight': [0.8]
})
graph = pipeline.build_knowledge_graph(edges_df)
print(f"✓ Graph has {graph.graph.number_of_nodes()} nodes")

# 3. Run agentic debate on sample news
print("\n3️⃣ Running agentic debate...")
news = [
    "Apple reported record profits and announced new product line",
    "Microsoft faces regulatory challenges in European markets"
]
results = pipeline.run_agentic_debate(news)
for i, result in enumerate(results):
    sentiment = result['judge_verdict']['sentiment_label']
    score = result['judge_verdict']['sentiment_score']
    print(f"  Article {i+1}: {sentiment} (score: {score:.2f})")

print("\n✅ First run complete! Check README.md for next steps.")
```

### Run it:
```bash
python test_run.py
```

## Understanding the Output

### Stock Data
```
{
  'stock_data': {
    'AAPL': DataFrame(100 rows, columns: Open, High, Low, Close, Volume),
    'MSFT': DataFrame(100 rows, columns: Open, High, Low, Close, Volume)
  },
  'macro_data': DataFrame(100 rows, columns: VIX, Inflation_CPI, Unemployment_Rate),
  'news': [List of news articles and metadata]
}
```

### Agentic Debate Results
```
{
  'bull_analysis': {
    'agent': 'Bull',
    'reasoning_vector': array(768,),  # Embedding
    'positive_catalysts': ['growth', 'profit', 'record'],
    'bullish_score': 0.67
  },
  'bear_analysis': {
    'agent': 'Bear',
    'reasoning_vector': array(768,),  # Embedding
    'risk_factors': ['challenge'],
    'bearish_score': 0.33
  },
  'judge_verdict': {
    'sentiment_label': 'BULLISH',
    'sentiment_score': 0.73,
    'confidence': 0.46
  }
}
```

### Graph Statistics
```
{
  'num_nodes': 2,
  'num_edges': 1,
  'density': 0.5,
  'avg_degree': 1.0,
  'edge_types': {'competitor': 1}
}
```

## File Structure You Need to Know

```
Stock_Explainable/
├── src/                    ← All actual code
│   ├── data_acquisition/   ← Data fetching
│   ├── preprocessing/      ← Text processing
│   ├── graph_construction/ ← Knowledge graph
│   ├── agentic_debate/     ← Bull/Bear/Judge
│   ├── gnn_module/         ← GNN propagation
│   ├── temporal_model/     ← LSTM/Transformer
│   ├── training/           ← Training loops
│   ├── evaluation/         ← Metrics
│   └── utils/              ← Helpers
├── configs/config.yaml     ← All settings
├── main_pipeline.py        ← Main orchestrator
├── README.md               ← Full documentation
├── DEVELOPMENT.md          ← Developer guide
└── tests/test_modules.py   ← Unit tests
```

## Common Commands

### Run all tests
```bash
python tests/test_modules.py
```

### View documentation
```bash
# Main guide
cat README.md

# Developer guide  
cat DEVELOPMENT.md

# Implementation summary
cat IMPLEMENTATION_SUMMARY.md
```

### Modify configuration
```bash
# Edit your settings
nano configs/config.yaml

# Key sections to modify:
# - data.yfinance.start_date
# - training.learning_rate
# - evaluation.backtesting.transaction_costs
```

## Performance Tips

### For Slow Systems
```yaml
# In configs/config.yaml
training:
  batch_size: 8
  device: "cpu"

preprocessing:
  batch_size: 16
```

### For GPU (CUDA)
```yaml
training:
  device: "cuda"  # Will automatically use GPU
  batch_size: 64
```

### For Memory Constraints
```yaml
temporal_model:
  lstm:
    hidden_size: 64
    num_layers: 1

gnn:
  hidden_dim: 128
  output_dim: 64
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'src'
**Solution:** Run from project root directory
```bash
cd Stock_Explainable
python test_run.py
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size or use CPU
```yaml
training:
  device: "cpu"
  batch_size: 8
```

### Issue: Transformer download fails
**Solution:** Pre-download models
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"
```

### Issue: FRED API key not working
**Solution:** Verify key format
```bash
# Check if .env file exists
ls .env

# Verify key is valid at https://fred.stlouisfed.org/docs/api/
```

## Next Steps

### Learn the System
1. Read the module overview in README.md
2. Review DEVELOPMENT.md for architecture
3. Run `python quick_test.py` to verify all modules
4. Check test cases in `tests/test_modules.py` for examples

### Build on It
1. Modify `configs/config.yaml` for your specific needs
2. Create notebooks in `notebooks/` for experimentation
3. Implement actual news data fetching (currently placeholder)
4. Train model on your historical data

### Production Ready
1. Save trained models with `pipeline.save()`
2. Load with `pipeline.load()`
3. Generate live predictions with `pipeline.predict()`
4. Monitor with Weights & Biases (optional)

## Quick Reference

| What | Where |
|------|-------|
| Main entry point | `main_pipeline.py` |
| Configuration | `configs/config.yaml` |
| Data fetchers | `src/data_acquisition/` |
| Text processing | `src/preprocessing/` |
| Knowledge graph | `src/graph_construction/` |
| Sentiment analysis | `src/agentic_debate/` |
| Neural networks | `src/gnn_module/`, `src/temporal_model/` |
| Training | `src/training/` |
| Evaluation | `src/evaluation/` |
| Tests | `tests/test_modules.py` |
| Documentation | `README.md`, `DEVELOPMENT.md` |

## Key Classes to Know

```python
# Pipeline orchestrator
from main_pipeline import StockExplainableAIPipeline
pipeline = StockExplainableAIPipeline('configs/config.yaml')

# Data
from src.data_acquisition import DataPipeline
data = DataPipeline(config).fetch_all(tickers)

# Sentiment
from src.agentic_debate import AgenticDebateModule
debate = AgenticDebateModule(config).debate(text)

# Graph
from src.graph_construction import FinancialKnowledgeGraph
graph = FinancialKnowledgeGraph(config)

# Models
from src.temporal_model import HybridTemporalModel
from src.gnn_module import EventPropagationGNN
from src.training import Trainer

# Evaluation
from src.evaluation import PerformanceAnalyzer
analyzer = PerformanceAnalyzer(config).evaluate(...)
```

## Example Notebooks Structure

See `notebooks/` for:
- `01_data_exploration.ipynb` - Load and visualize data
- `02_sentiment_analysis.ipynb` - Run debates
- `03_graph_visualization.ipynb` - Graph exploration
- `04_model_training.ipynb` - Train complete system
- `05_backtesting.ipynb` - Evaluate performance

_(These are templates for you to create)_

## Getting Help

1. **Code questions**: Check docstrings
   ```python
   from src.data_acquisition import StockDataFetcher
   help(StockDataFetcher.fetch)
   ```

2. **Architecture questions**: See DEVELOPMENT.md

3. **Examples**: Run `tests/test_modules.py` and check test code

4. **Configuration options**: Edit `configs/config.yaml` and read comments

## Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Quick test passes (`python quick_test.py`)
- [ ] FRED API key obtained and added to `.env`
- [ ] First run executes without errors (`python test_run.py`)
- [ ] You understand the 5 main steps
- [ ] You've read the module overview in README.md
- [ ] You know where to find code for each functionality

Congratulations! You're ready to build AI-driven trading systems! 🚀

---

**Happy coding!** For questions, refer to README.md or DEVELOPMENT.md.
