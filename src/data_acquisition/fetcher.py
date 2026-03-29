"""
Data Acquisition Module - Fetch historical stock data, news, and macroeconomic indicators.
"""

import logging
import os
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import fredapi
except ImportError:
    fredapi = None


logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch historical OHLCV data from Yahoo Finance."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stock data fetcher.
        
        Args:
            config: Configuration dictionary with yfinance settings
        """
        if yf is None:
            raise ImportError("yfinance required. Install with: pip install yfinance")
        
        self.config = config.get('yfinance', {})
        self.interval = self.config.get('interval', '1d')
        
    def fetch(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.
        
        Args:
            tickers: List of stock tickers (e.g., ['AAPL', 'GOOGL'])
            start_date: Start date (YYYY-MM-DD). Uses config if None
            end_date: End date (YYYY-MM-DD). Uses today if None
            
        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data
            
        Raises:
            ValueError: If ticker list is empty
        """
        if not tickers:
            raise ValueError("tickers list cannot be empty")
        
        start_date = start_date or self.config.get('start_date', '2020-01-01')
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        data = {}
        for ticker in tickers:
            try:
                logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=self.interval,
                    progress=False
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue
                
                # Ensure required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {ticker}")
                    continue
                
                data[ticker] = df
                logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {str(e)}")
        
        if not data:
            raise RuntimeError(f"Failed to fetch data for any tickers in {tickers}")
        
        return data
    
    def validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data integrity.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Having: {df.columns.tolist()}")
            return False
        
        # Check High >= Open, Close, Low
        invalid_high = (df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)).sum()
        if invalid_high > 0:
            logger.warning(f"Found {invalid_high} rows with High < max(Open, Close, Low)")
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        negative = (df[price_cols] < 0).sum().sum()
        if negative > 0:
            logger.warning(f"Found {negative} negative prices")
            return False
        
        # Check for missing values
        missing = df[required_cols].isna().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values")
        
        return True


class MacroeconomicDataFetcher:
    """Fetch macroeconomic indicators from Federal Reserve Economic Data (FRED)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize macro data fetcher.
        
        Args:
            config: Configuration dictionary with FRED settings
        """
        if fredapi is None:
            raise ImportError("fredapi required. Install with: pip install fredapi")
        
        fred_config = config.get('fred', {})
        api_key = fred_config.get('api_key')
        
        if not api_key:
            raise ValueError("FRED_API_KEY not set in config or environment")
        
        self.fred = fredapi.Fred(api_key=api_key)
        self.indicators = fred_config.get('indicators', ['VIXCLS', 'CPIAUCSL', 'UNRATE'])
    
    def fetch(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, pd.Series]:
        """
        Fetch macroeconomic indicators.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping indicator series IDs to data
        """
        start_date = start_date or '2020-01-01'
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        data = {}
        indicator_names = {
            'VIXCLS': 'VIX',
            'CPIAUCSL': 'Inflation_CPI',
            'UNRATE': 'Unemployment_Rate',
        }
        
        for indicator in self.indicators:
            try:
                logger.info(f"Fetching {indicator} ({indicator_names.get(indicator, indicator)})")
                series = self.fred.get_series(
                    indicator,
                    observation_start=start_date,
                    observation_end=end_date
                )
                data[indicator_names.get(indicator, indicator)] = series
                logger.info(f"Successfully fetched {len(series)} rows for {indicator}")
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {str(e)}")
        
        return data
    
    def align_with_market_dates(
        self,
        macro_data: Dict[str, pd.Series],
        market_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Align macroeconomic data with market dates using forward fill.
        
        Args:
            macro_data: Dictionary of macro series
            market_dates: Market trading dates (from stock data)
            
        Returns:
            DataFrame aligned with market dates
        """
        # Create DataFrame from dict
        df = pd.DataFrame(macro_data)
        df.index.name = 'Date'
        
        # Reindex to market dates and forward fill
        df = df.reindex(market_dates, method='ffill')
        
        # Forward fill at the beginning if needed
        df = df.fillna(method='bfill')
        
        logger.info(f"Aligned macro data to {len(df)} market dates")
        return df


class NewsDataLoader:
    """Load financial news data (placeholder for actual news sources)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize news data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('news', {})
        self.source = self.config.get('news_source', 'fnspid')
        self.cache_dir = self.config.get('cache_dir', './data/news')
    
    def fetch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news data from configured source (FNSPID, NewsAPI, etc).
        
        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of news articles with text and timestamps
        """
        logger.info(f"Loading news for {len(tickers)} tickers from {self.source} ({start_date} to {end_date})")
        
        news_articles = []
        
        try:
            if self.source == 'fnspid':
                # FNSPID API implementation
                # To use: Register at https://www.fnspid.com/
                # Uncomment and add your API key to .env
                
                fnspid_api_key = os.environ.get('FNSPID_API_KEY')
                if not fnspid_api_key:
                    logger.warning("FNSPID_API_KEY not set. Skipping FNSPID data. Add to .env to enable.")
                    return self._fetch_fallback_news(tickers, start_date, end_date)
                
                # Example FNSPID API call (adjust endpoint based on actual API)
                for ticker in tickers:
                    try:
                        articles = self._fetch_fnspid_news(
                            ticker, fnspid_api_key, start_date, end_date
                        )
                        news_articles.extend(articles)
                    except Exception as e:
                        logger.debug(f"FNSPID fetch error for {ticker}: {str(e)[:100]}")
                        
            elif self.source == 'newsapi':
                # NewsAPI fallback (free tier available)
                # To use: Register at https://newsapi.org/
                newsapi_key = os.environ.get('NEWSAPI_KEY')
                if newsapi_key:
                    news_articles = self._fetch_newsapi_news(
                        tickers, newsapi_key, start_date, end_date
                    )
                else:
                    logger.warning("NEWSAPI_KEY not set. Using fallback.")
                    news_articles = self._fetch_fallback_news(tickers, start_date, end_date)
            else:
                news_articles = self._fetch_fallback_news(tickers, start_date, end_date)
                
        except Exception as e:
            logger.error(f"News fetch failed: {str(e)[:100]}")
            return []
        
        logger.info(f"Fetched {len(news_articles)} news articles")
        return news_articles
    
    def _fetch_fnspid_news(
        self,
        ticker: str,
        api_key: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Fetch from FNSPID API."""
        try:
            import requests
            
            url = f"https://api.fnspid.com/v1/news"
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {
                "ticker": ticker,
                "startDate": start_date,
                "endDate": end_date,
                "limit": 50
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                for item in data.get('articles', []):
                    articles.append({
                        'ticker': ticker,
                        'headline': item.get('title', ''),
                        'text': item.get('summary', ''),
                        'source': item.get('source', 'FNSPID'),
                        'timestamp': item.get('publishedDate', ''),
                        'url': item.get('url', '')
                    })
                return articles
            else:
                logger.warning(f"FNSPID API error {response.status_code} for {ticker}")
                return []
        except Exception as e:
            logger.debug(f"FNSPID API exception: {str(e)[:80]}")
            return []
    
    def _fetch_newsapi_news(
        self,
        tickers: List[str],
        api_key: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Fetch from NewsAPI (free alternative)."""
        try:
            import requests
            
            articles = []
            for ticker in tickers:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": f"{ticker} stock",
                    "from": start_date,
                    "to": end_date,
                    "sortBy": "publishedAt",
                    "apiKey": api_key,
                    "language": "en"
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', []):
                        articles.append({
                            'ticker': ticker,
                            'headline': article.get('title', ''),
                            'text': article.get('description', ''),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'timestamp': article.get('publishedAt', ''),
                            'url': article.get('url', '')
                        })
                else:
                    logger.debug(f"NewsAPI error {response.status_code}")
            
            return articles
        except Exception as e:
            logger.debug(f"NewsAPI exception: {str(e)[:80]}")
            return []
    
    def _fetch_fallback_news(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Fallback: Return synthetic news for demonstration."""
        logger.info("Using fallback news data (synthetic for demonstration)")
        
        sample_headlines = {
            'AAPL': [
                'Apple Q4 earnings beat estimates by 15%',
                'Apple announces new AI features in iOS',
                'Apple stock reaches all-time high'
            ],
            'MSFT': [
                'Microsoft expands Azure cloud services',
                'Microsoft partners with OpenAI for enterprise solutions',
                'Microsoft beats quarterly revenue targets'
            ],
            'GOOGL': [
                'Google announces breakthrough in quantum computing',
                'Google faces regulatory scrutiny in EU',
                'Google search updates improve AI integration'
            ],
            'AMZN': [
                'Amazon Web Services grows 30% YoY',
                'Amazon enters healthcare market',
                'Amazon announces new sustainability goals'
            ],
            'NVDA': [
                'NVIDIA demand for AI chips remains strong',
                'NVIDIA stock doubles in response to AI boom',
                'NVIDIA builds next-gen GPU architecture'
            ]
        }
        
        articles = []
        for ticker in tickers:
            if ticker in sample_headlines:
                for headline in sample_headlines[ticker]:
                    articles.append({
                        'ticker': ticker,
                        'headline': headline,
                        'text': f'Market update on {ticker}: {headline}',
                        'source': 'Demo',
                        'timestamp': start_date,
                        'url': ''
                    })
        
        return articles


class DataPipeline:
    """
    Unified data acquisition pipeline combining stock, macro, and news data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data pipeline."""
        self.config = config
        self.stock_fetcher = StockDataFetcher(config)
        
        if config.get('fred', {}).get('api_key'):
            self.macro_fetcher = MacroeconomicDataFetcher(config)
        else:
            self.macro_fetcher = None
            logger.warning("Macro data fetcher not initialized (no FRED API key)")
        
        self.news_loader = NewsDataLoader(config)
    
    def fetch_all(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        Fetch all data types (stock, macro, news).
        
        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with keys: 'stock_data', 'macro_data', 'news'
        """
        logger.info(f"Starting data pipeline for {len(tickers)} tickers")
        
        result = {
            'stock_data': {},
            'macro_data': None,
            'news': []
        }
        
        # Fetch stock data
        stock_data = self.stock_fetcher.fetch(tickers, start_date, end_date)
        result['stock_data'] = stock_data
        
        # Fetch macro data
        if self.macro_fetcher:
            try:
                macro_data_dict = self.macro_fetcher.fetch(start_date, end_date)
                # Get union of all stock dates
                all_dates = pd.DatetimeIndex([])
                for df in stock_data.values():
                    all_dates = all_dates.union(df.index)
                all_dates = all_dates.sort_values()
                
                result['macro_data'] = self.macro_fetcher.align_with_market_dates(
                    macro_data_dict, all_dates
                )
            except Exception as e:
                logger.error(f"Failed to fetch macro data: {e}")
        
        # Fetch news data
        if start_date and end_date:
            try:
                result['news'] = self.news_loader.fetch(tickers, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to fetch news data: {e}")
        
        logger.info("Data pipeline completed successfully")
        return result
