"""Stock data fetching and processing utilities."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class StockDataFetcher:
    """Utility class for fetching and processing stock market data."""
    
    def __init__(self):
        self.cache = {}  # Simple cache for data
        
    def get_stock_data(self, 
                      ticker: str, 
                      period: str = "1y",
                      include_volume: bool = True) -> Dict[str, Any]:
        """
        Fetch stock price and volume data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            include_volume: Whether to include volume data
            
        Returns:
            Dictionary containing price data, volume data, and metadata
        """
        try:
            cache_key = f"{ticker}_{period}"
            
            # Check cache (simple time-based cache for 1 hour)
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(hours=1):
                    logger.info(f"Using cached data for {ticker}")
                    return cached_data
            
            logger.info(f"Fetching stock data for {ticker} (period: {period})")
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period=period)
            
            if hist.empty:
                raise ValueError(f"No data available for ticker {ticker}")
            
            # Get basic info
            info = stock.info
            
            # Prepare price data
            price_data = {
                'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                'open': hist['Open'].tolist(),
                'high': hist['High'].tolist(),
                'low': hist['Low'].tolist(),
                'close': hist['Close'].tolist(),
                'adj_close': hist['Close'].tolist(),  # yfinance automatically adjusts
            }
            
            # Calculate price statistics
            current_price = hist['Close'].iloc[-1]
            price_change_1d = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
            price_change_pct_1d = (price_change_1d / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            
            # Calculate price statistics over period
            period_high = hist['High'].max()
            period_low = hist['Low'].min()
            period_start_price = hist['Close'].iloc[0]
            period_return = (current_price - period_start_price) / period_start_price * 100
            
            # Volume data
            volume_data = {}
            if include_volume and 'Volume' in hist.columns:
                volume_data = {
                    'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                    'volume': hist['Volume'].tolist(),
                    'avg_volume': hist['Volume'].mean(),
                    'volume_trend': self._calculate_volume_trend(hist['Volume'])
                }
            
            # Volatility calculation (using close prices)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Prepare result
            result = {
                'ticker': ticker,
                'period': period,
                'data_points': len(hist),
                'current_price': round(current_price, 2),
                'price_change_1d': round(price_change_1d, 2),
                'price_change_pct_1d': round(price_change_pct_1d, 2),
                'period_high': round(period_high, 2),
                'period_low': round(period_low, 2),
                'period_return_pct': round(period_return, 2),
                'volatility': round(volatility, 4),
                'price_data': price_data,
                'volume_data': volume_data,
                'company_info': {
                    'name': info.get('longName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', None),
                    'dividend_yield': info.get('dividendYield', None)
                },
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = (result, datetime.now())
            
            logger.info(f"Successfully fetched data for {ticker}: {len(hist)} data points")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            raise
    
    def _calculate_volume_trend(self, volume_series: pd.Series) -> str:
        """Calculate volume trend over the period."""
        if len(volume_series) < 10:
            return "insufficient_data"
        
        # Compare recent average vs earlier average
        recent_avg = volume_series.tail(10).mean()
        earlier_avg = volume_series.head(10).mean()
        
        change_pct = (recent_avg - earlier_avg) / earlier_avg * 100
        
        if change_pct > 20:
            return "increasing"
        elif change_pct < -20:
            return "decreasing"
        else:
            return "stable"
    
    def detect_price_anomalies(self, price_data: Dict[str, Any], 
                             volume_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect unusual price and volume patterns.
        
        Args:
            price_data: Price data dictionary
            volume_data: Volume data dictionary (optional)
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            closes = np.array(price_data['close'])
            dates = price_data['dates']
            
            # Calculate daily returns
            returns = np.diff(closes) / closes[:-1]
            
            # Detect price anomalies using statistical methods
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            threshold = 2.5  # 2.5 standard deviations
            
            price_anomalies = []
            for i, ret in enumerate(returns):
                if abs(ret - mean_return) > threshold * std_return:
                    price_anomalies.append({
                        'date': dates[i+1],  # i+1 because returns array is shorter
                        'return': round(ret * 100, 2),
                        'price': round(closes[i+1], 2),
                        'type': 'spike' if ret > mean_return else 'drop',
                        'severity': abs(ret - mean_return) / std_return
                    })
            
            # Volume anomalies (if volume data available)
            volume_anomalies = []
            if volume_data and volume_data.get('volume'):
                volumes = np.array(volume_data['volume'])
                mean_volume = np.mean(volumes)
                std_volume = np.std(volumes)
                
                for i, vol in enumerate(volumes):
                    if vol > mean_volume + 2 * std_volume:
                        volume_anomalies.append({
                            'date': dates[i],
                            'volume': int(vol),
                            'vs_average': round(vol / mean_volume, 2),
                            'type': 'high_volume'
                        })
            
            return {
                'price_anomalies': price_anomalies[-10:],  # Last 10 anomalies
                'volume_anomalies': volume_anomalies[-10:],  # Last 10 anomalies
                'total_price_anomalies': len(price_anomalies),
                'total_volume_anomalies': len(volume_anomalies),
                'analysis_period': f"{len(closes)} trading days",
                'volatility_assessment': self._assess_volatility(returns)
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {'error': str(e)}
    
    def _assess_volatility(self, returns: np.ndarray) -> str:
        """Assess volatility level based on returns."""
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        if volatility > 0.5:
            return "very_high"
        elif volatility > 0.3:
            return "high"
        elif volatility > 0.15:
            return "moderate"
        else:
            return "low"
