"""Data acquisition module."""
from . fetcher import StockDataFetcher, MacroeconomicDataFetcher, NewsDataLoader, DataPipeline

__all__ = ['StockDataFetcher', 'MacroeconomicDataFetcher', 'NewsDataLoader', 'DataPipeline']
