"""
Module for implementing trading strategies for the WINFUT trading robot.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import datetime
import math

from config import STRATEGY_PARAMS, TRADING_HOURS, TRADING_DAYS, TECHNICAL_INDICATORS

# Configure logger
logger = logging.getLogger("winfut_robot")
from models import ModelManager
from data_processor import DataProcessor

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Class for implementing and executing trading strategies"""
    
    def __init__(self, model_manager: ModelManager, data_processor: DataProcessor):
        self.model_manager = model_manager
        self.data_processor = data_processor
        self.confidence_threshold = STRATEGY_PARAMS["confidence_threshold"]
        self.min_volume = STRATEGY_PARAMS["min_volume"]
        self.use_market_regime = STRATEGY_PARAMS["use_market_regime"]
        self.current_signal = None
        self.last_signal_time = None
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.current_stop_loss = None
        self.current_take_profit = None
        
    def _is_trading_hours(self) -> bool:
        """
        Checks if current time is within trading hours, with additional time filters.
        
        Returns:
            Boolean indicating if current time is within trading hours
        """
        now = datetime.datetime.now()
        
        # Check if today is a trading day
        if now.weekday() not in TRADING_DAYS:
            return False
        
        start_time_str = TRADING_HOURS["start"]
        end_time_str = TRADING_HOURS["end"]
        
        start_hour, start_min = map(int, start_time_str.split(':'))
        end_hour, end_min = map(int, end_time_str.split(':'))
        
        start_time = now.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
        end_time = now.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)
        
        # Basic trading hours check
        if not (start_time <= now <= end_time):
            return False
        
        # Check if we're in the first X minutes of trading (avoid opening volatility)
        avoid_first_minutes = STRATEGY_PARAMS["time_filters"]["avoid_first_minutes"]
        if (now - start_time).total_seconds() < avoid_first_minutes * 60:
            logger.info(f"Not trading during first {avoid_first_minutes} minutes of session")
            return False
        
        # Check if we're in lunch hour (typically lower volume)
        if STRATEGY_PARAMS["time_filters"]["avoid_lunch_hour"]:
            lunch_start_str, lunch_end_str = STRATEGY_PARAMS["time_filters"]["lunch_period"]
            lunch_start_hour, lunch_start_min = map(int, lunch_start_str.split(':'))
            lunch_end_hour, lunch_end_min = map(int, lunch_end_str.split(':'))
            
            lunch_start = now.replace(hour=lunch_start_hour, minute=lunch_start_min, second=0, microsecond=0)
            lunch_end = now.replace(hour=lunch_end_hour, minute=lunch_end_min, second=0, microsecond=0)
            
            if lunch_start <= now <= lunch_end:
                logger.info("Not trading during lunch hour")
                return False
        
        # Check if current time is in preferred trading hours
        prefer_hours = STRATEGY_PARAMS["time_filters"]["prefer_hours"]
        if prefer_hours:
            # Convert preferred hours to datetime objects for today
            preferred_periods = []
            for i in range(0, len(prefer_hours), 2):
                if i + 1 < len(prefer_hours):
                    start_h, start_m = map(int, prefer_hours[i].split(':'))
                    end_h, end_m = map(int, prefer_hours[i+1].split(':'))
                    
                    period_start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
                    period_end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
                    preferred_periods.append((period_start, period_end))
            
            # Check if current time falls within any preferred period
            in_preferred_period = any(period_start <= now <= period_end for period_start, period_end in preferred_periods)
            
            # If we're enforcing preferred hours, return result
            if not in_preferred_period:
                logger.debug("Current time is not in preferred trading hours")
                # Still allow trading but at reduced confidence
                # We don't return False here because we'll adjust confidence instead
        
        return True
        
    def _calculate_time_multiplier(self) -> float:
        """
        Calculates a multiplier for signal confidence based on time of day.
        Higher values during preferred trading hours.
        
        Returns:
            Float multiplier value between 0.8 and 1.2
        """
        now = datetime.datetime.now()
        
        # Get preferred hours from config
        prefer_hours = STRATEGY_PARAMS["time_filters"]["prefer_hours"]
        
        # Default multiplier
        multiplier = 1.0
        
        if prefer_hours:
            # Convert preferred hours to datetime objects for today
            preferred_periods = []
            for i in range(0, len(prefer_hours), 2):
                if i + 1 < len(prefer_hours):
                    start_h, start_m = map(int, prefer_hours[i].split(':'))
                    end_h, end_m = map(int, prefer_hours[i+1].split(':'))
                    
                    period_start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
                    period_end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
                    preferred_periods.append((period_start, period_end))
            
            # Check if current time falls within any preferred period
            in_preferred_period = any(period_start <= now <= period_end for period_start, period_end in preferred_periods)
            
            if in_preferred_period:
                # Boost confidence during preferred hours
                multiplier = 1.2
            else:
                # Reduce confidence outside preferred hours
                multiplier = 0.9
                
        # Also adjust for early or late in the trading day
        start_time_str = TRADING_HOURS["start"]
        end_time_str = TRADING_HOURS["end"]
        
        start_hour, start_min = map(int, start_time_str.split(':'))
        end_hour, end_min = map(int, end_time_str.split(':'))
        
        start_time = now.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
        end_time = now.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)
        
        # Avoid trading in the last 10 minutes
        minutes_to_close = (end_time - now).total_seconds() / 60
        if minutes_to_close < 10:
            multiplier *= 0.8  # Further reduce confidence near close
            
        return multiplier
    
    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes candlestick patterns for potential trade signals.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            results = {
                "bullish_patterns": [],
                "bearish_patterns": [],
                "bullish_strength": 0,
                "bearish_strength": 0
            }
            
            # Get the last few candles for pattern detection
            candles = df.iloc[-5:].copy()
            
            # Calculate candle properties
            candles['body_size'] = abs(candles['close'] - candles['open'])
            candles['upper_shadow'] = candles.apply(
                lambda x: x['high'] - max(x['open'], x['close']), axis=1
            )
            candles['lower_shadow'] = candles.apply(
                lambda x: min(x['open'], x['close']) - x['low'], axis=1
            )
            candles['range'] = candles['high'] - candles['low']
            candles['is_bullish'] = candles['close'] > candles['open']
            
            # Average values for comparison
            avg_body = candles['body_size'].mean()
            avg_range = candles['range'].mean()
            
            # Latest candle
            latest = candles.iloc[-1]
            prev = candles.iloc[-2] if len(candles) > 1 else None
            prev2 = candles.iloc[-3] if len(candles) > 2 else None
            
            # Pattern detection - Bullish patterns
            
            # Hammer (bullish reversal)
            if (latest['is_bullish'] and 
                latest['lower_shadow'] > 2 * latest['body_size'] and
                latest['upper_shadow'] < 0.2 * latest['body_size'] and
                latest['body_size'] < 0.3 * latest['range']):
                results["bullish_patterns"].append("hammer")
                results["bullish_strength"] += 1
            
            # Bullish Engulfing
            if (prev is not None and
                not prev['is_bullish'] and latest['is_bullish'] and
                latest['open'] < prev['close'] and
                latest['close'] > prev['open'] and
                latest['body_size'] > 1.5 * prev['body_size']):
                results["bullish_patterns"].append("bullish_engulfing")
                results["bullish_strength"] += 2
            
            # Morning Star (bullish reversal)
            if (prev2 is not None and prev is not None and
                not prev2['is_bullish'] and prev['body_size'] < 0.5 * avg_body and
                latest['is_bullish'] and latest['body_size'] > avg_body):
                results["bullish_patterns"].append("morning_star")
                results["bullish_strength"] += 2
            
            # Piercing Line
            if (prev is not None and
                not prev['is_bullish'] and latest['is_bullish'] and
                latest['open'] < prev['close'] and
                latest['close'] > (prev['open'] + prev['close']) / 2):
                results["bullish_patterns"].append("piercing_line")
                results["bullish_strength"] += 1
            
            # Pattern detection - Bearish patterns
            
            # Shooting Star (bearish reversal)
            if (latest['is_bullish'] and 
                latest['upper_shadow'] > 2 * latest['body_size'] and
                latest['lower_shadow'] < 0.2 * latest['body_size'] and
                latest['body_size'] < 0.3 * latest['range']):
                results["bearish_patterns"].append("shooting_star")
                results["bearish_strength"] += 1
            
            # Bearish Engulfing
            if (prev is not None and
                prev['is_bullish'] and not latest['is_bullish'] and
                latest['open'] > prev['close'] and
                latest['close'] < prev['open'] and
                latest['body_size'] > 1.5 * prev['body_size']):
                results["bearish_patterns"].append("bearish_engulfing")
                results["bearish_strength"] += 2
            
            # Evening Star (bearish reversal)
            if (prev2 is not None and prev is not None and
                prev2['is_bullish'] and prev['body_size'] < 0.5 * avg_body and
                not latest['is_bullish'] and latest['body_size'] > avg_body):
                results["bearish_patterns"].append("evening_star")
                results["bearish_strength"] += 2
            
            # Dark Cloud Cover
            if (prev is not None and
                prev['is_bullish'] and not latest['is_bullish'] and
                latest['open'] > prev['close'] and
                latest['close'] < (prev['open'] + prev['close']) / 2):
                results["bearish_patterns"].append("dark_cloud_cover")
                results["bearish_strength"] += 1
            
            # Doji (indecision)
            if latest['body_size'] < 0.1 * latest['range']:
                if latest['upper_shadow'] > 2 * latest['body_size'] and latest['lower_shadow'] > 2 * latest['body_size']:
                    results["bullish_patterns"].append("doji")
                    results["bearish_patterns"].append("doji")
                    # Doji at support or resistance has more significance
                    if latest['close'] < df['bb_lower'].iloc[-1]:
                        results["bullish_strength"] += 1
                    elif latest['close'] > df['bb_upper'].iloc[-1]:
                        results["bearish_strength"] += 1
            
            # Normalize strength scores to range 0-3
            results["bullish_strength"] = min(3, results["bullish_strength"])
            results["bearish_strength"] = min(3, results["bearish_strength"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing candlestick patterns: {str(e)}")
            return {"bullish_patterns": [], "bearish_patterns": [], "bullish_strength": 0, "bearish_strength": 0}
    
    def _check_divergences(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Checks for divergences between price and indicators (RSI, MACD).
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Dictionary with divergence analysis results
        """
        try:
            results = {
                "bullish_divergence": False,
                "bearish_divergence": False,
                "rsi_bullish_divergence": False,
                "rsi_bearish_divergence": False,
                "macd_bullish_divergence": False,
                "macd_bearish_divergence": False
            }
            
            # Need sufficient data for divergence analysis
            lookback = TECHNICAL_INDICATORS["divergence_lookback"]
            if len(df) < lookback + 5:
                return results
                
            # Function to find local extrema
            def find_extrema(series, lookback=5):
                # Prepare highs and lows arrays
                highs, lows = [], []
                
                # Look for local minima and maxima
                for i in range(lookback, len(series) - lookback):
                    # Check if this point is a local maximum
                    if all(series.iloc[i] > series.iloc[i-j] for j in range(1, lookback+1)) and \
                       all(series.iloc[i] > series.iloc[i+j] for j in range(1, lookback+1)):
                        highs.append((i, series.iloc[i]))
                        
                    # Check if this point is a local minimum
                    if all(series.iloc[i] < series.iloc[i-j] for j in range(1, lookback+1)) and \
                       all(series.iloc[i] < series.iloc[i+j] for j in range(1, lookback+1)):
                        lows.append((i, series.iloc[i]))
                
                return highs, lows
            
            # Get recent data for analysis
            data = df.iloc[-lookback*2:].copy()
            
            # Find price extrema
            price_highs, price_lows = find_extrema(data['close'], lookback=3)
            
            # Find RSI extrema
            rsi_highs, rsi_lows = find_extrema(data['rsi'], lookback=3)
            
            # Find MACD extrema
            macd_highs, macd_lows = find_extrema(data['macd'], lookback=3)
            
            # Check for RSI divergences
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                # Check for bullish divergence (price lower lows, RSI higher lows)
                latest_price_low_idx = price_lows[-1][0]
                prev_price_low_idx = price_lows[-2][0]
                
                # Find closest RSI lows to the price lows
                latest_rsi_low = next((low for low in rsi_lows if abs(low[0] - latest_price_low_idx) <= 2), None)
                prev_rsi_low = next((low for low in rsi_lows if abs(low[0] - prev_price_low_idx) <= 2), None)
                
                if latest_rsi_low and prev_rsi_low:
                    # Check if price made lower low but RSI made higher low
                    if (data['close'].iloc[latest_price_low_idx] < data['close'].iloc[prev_price_low_idx] and
                        latest_rsi_low[1] > prev_rsi_low[1]):
                        results["rsi_bullish_divergence"] = True
                        results["bullish_divergence"] = True
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                # Check for bearish divergence (price higher highs, RSI lower highs)
                latest_price_high_idx = price_highs[-1][0]
                prev_price_high_idx = price_highs[-2][0]
                
                # Find closest RSI highs to the price highs
                latest_rsi_high = next((high for high in rsi_highs if abs(high[0] - latest_price_high_idx) <= 2), None)
                prev_rsi_high = next((high for high in rsi_highs if abs(high[0] - prev_price_high_idx) <= 2), None)
                
                if latest_rsi_high and prev_rsi_high:
                    # Check if price made higher high but RSI made lower high
                    if (data['close'].iloc[latest_price_high_idx] > data['close'].iloc[prev_price_high_idx] and
                        latest_rsi_high[1] < prev_rsi_high[1]):
                        results["rsi_bearish_divergence"] = True
                        results["bearish_divergence"] = True
            
            # Check for MACD divergences (similar logic to RSI divergences)
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                latest_price_low_idx = price_lows[-1][0]
                prev_price_low_idx = price_lows[-2][0]
                
                latest_macd_low = next((low for low in macd_lows if abs(low[0] - latest_price_low_idx) <= 2), None)
                prev_macd_low = next((low for low in macd_lows if abs(low[0] - prev_price_low_idx) <= 2), None)
                
                if latest_macd_low and prev_macd_low:
                    if (data['close'].iloc[latest_price_low_idx] < data['close'].iloc[prev_price_low_idx] and
                        latest_macd_low[1] > prev_macd_low[1]):
                        results["macd_bullish_divergence"] = True
                        results["bullish_divergence"] = True
            
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                latest_price_high_idx = price_highs[-1][0]
                prev_price_high_idx = price_highs[-2][0]
                
                latest_macd_high = next((high for high in macd_highs if abs(high[0] - latest_price_high_idx) <= 2), None)
                prev_macd_high = next((high for high in macd_highs if abs(high[0] - prev_price_high_idx) <= 2), None)
                
                if latest_macd_high and prev_macd_high:
                    if (data['close'].iloc[latest_price_high_idx] > data['close'].iloc[prev_price_high_idx] and
                        latest_macd_high[1] < prev_macd_high[1]):
                        results["macd_bearish_divergence"] = True
                        results["bearish_divergence"] = True
                        
            return results
            
        except Exception as e:
            logger.error(f"Error checking divergences: {str(e)}")
            return {"bullish_divergence": False, "bearish_divergence": False}
    
    def _analyze_vwap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes VWAP (Volume Weighted Average Price) for intraday signals.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Dictionary with VWAP analysis results
        """
        try:
            results = {
                "price_above_vwap": False,
                "price_below_vwap": False,
                "crossing_vwap_upward": False,
                "crossing_vwap_downward": False,
                "distance_from_vwap": 0.0,
                "vwap_trend": "neutral"
            }
            
            # Check if VWAP column exists
            if 'vwap' not in df.columns:
                return results
                
            # Get latest values
            latest_close = df['close'].iloc[-1]
            latest_vwap = df['vwap'].iloc[-1]
            
            # Previous values for crossing detection
            prev_close = df['close'].iloc[-2] if len(df) > 1 else latest_close
            prev_vwap = df['vwap'].iloc[-2] if len(df) > 1 else latest_vwap
            
            # Basic VWAP relationship
            results["price_above_vwap"] = latest_close > latest_vwap
            results["price_below_vwap"] = latest_close < latest_vwap
            
            # Calculate relative distance
            if latest_vwap != 0:
                results["distance_from_vwap"] = abs((latest_close - latest_vwap) / latest_vwap) * 100  # percentage
            
            # Detect crossing
            results["crossing_vwap_upward"] = prev_close < prev_vwap and latest_close > latest_vwap
            results["crossing_vwap_downward"] = prev_close > prev_vwap and latest_close < latest_vwap
            
            # Determine VWAP trend
            vwap_samples = df['vwap'].iloc[-5:]
            if vwap_samples.is_monotonic_increasing:
                results["vwap_trend"] = "up"
            elif vwap_samples.is_monotonic_decreasing:
                results["vwap_trend"] = "down"
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing VWAP: {str(e)}")
            return {"price_above_vwap": False, "price_below_vwap": False}
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detects market regime (trending, ranging, volatile).
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            String indicating market regime
        """
        try:
            # Use ADX to determine trending vs ranging
            last_adx = df['adx'].iloc[-1]
            
            # Use ATR and Bollinger Band width for volatility
            last_atr = df['atr'].iloc[-1]
            last_bb_width = df['bb_width'].iloc[-1]
            
            # Average ATR and BB width for comparison
            avg_atr = df['atr'].iloc[-20:].mean()
            avg_bb_width = df['bb_width'].iloc[-20:].mean()
            
            # Determine market regime
            if last_adx > 25:  # Strong trend
                if last_atr > 1.5 * avg_atr:
                    return "volatile_trending"
                return "trending"
            else:  # Ranging market
                if last_bb_width > 1.5 * avg_bb_width:
                    return "volatile_ranging"
                return "ranging"
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
    
    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes technical indicators for additional trading signals.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Dictionary with technical analysis results
        """
        try:
            results = {}
            
            # Get the latest values
            latest = df.iloc[-1]
            
            # Trend indicators
            results["trend"] = {
                "sma_fast_above_slow": latest['sma_fast'] > latest['sma_slow'],
                "ema_fast_above_slow": latest['ema_fast'] > latest['ema_slow'],
                "price_above_sma_fast": latest['close'] > latest['sma_fast'],
                "price_above_sma_slow": latest['close'] > latest['sma_slow'],
                "adx_strength": latest['adx']
            }
            
            # Momentum indicators
            results["momentum"] = {
                "rsi": latest['rsi'],
                "rsi_oversold": latest['rsi'] < 30,
                "rsi_overbought": latest['rsi'] > 70,
                "macd_above_signal": latest['macd'] > latest['macd_signal'],
                "macd_positive": latest['macd'] > 0,
                "stoch_k_above_d": latest['slowk'] > latest['slowd'],
                "stoch_oversold": latest['slowk'] < 20 and latest['slowd'] < 20,
                "stoch_overbought": latest['slowk'] > 80 and latest['slowd'] > 80
            }
            
            # Volatility indicators
            results["volatility"] = {
                "bb_width": latest['bb_width'],
                "price_above_upper_band": latest['close'] > latest['bb_upper'],
                "price_below_lower_band": latest['close'] < latest['bb_lower'],
                "atr": latest['atr']
            }
            
            # Volume indicators
            results["volume"] = {
                "obv_rising": df['obv'].iloc[-1] > df['obv'].iloc[-2],
                "volume_rising": df['volume'].iloc[-1] > df['volume'].iloc[-2],
                "volume_above_average": df['volume'].iloc[-1] > df['volume'].iloc[-10:].mean()
            }
            
            # Calculate overall buy/sell scores based on indicators
            buy_signals = sum([
                results["trend"]["sma_fast_above_slow"],
                results["trend"]["ema_fast_above_slow"],
                results["trend"]["price_above_sma_fast"],
                results["momentum"]["macd_above_signal"],
                results["momentum"]["macd_positive"],
                results["momentum"]["stoch_k_above_d"],
                results["momentum"]["rsi_oversold"],
                results["volatility"]["price_below_lower_band"],
                results["volume"]["obv_rising"],
                results["volume"]["volume_above_average"]
            ])
            
            sell_signals = sum([
                not results["trend"]["sma_fast_above_slow"],
                not results["trend"]["ema_fast_above_slow"],
                not results["trend"]["price_above_sma_fast"],
                not results["momentum"]["macd_above_signal"],
                not results["momentum"]["macd_positive"],
                not results["momentum"]["stoch_k_above_d"],
                results["momentum"]["rsi_overbought"],
                results["volatility"]["price_above_upper_band"],
                not results["volume"]["obv_rising"]
            ])
            
            results["buy_score"] = buy_signals / 10
            results["sell_score"] = sell_signals / 9
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {str(e)}")
            return {"buy_score": 0, "sell_score": 0}
    
    def should_filter_signal(self, df: pd.DataFrame, signal: int, confidence: float) -> bool:
        """
        Applies additional filters to determine if a signal should be taken.
        
        Args:
            df: DataFrame with price data and indicators
            signal: Trading signal (1 for buy, -1 for sell, 0 for neutral)
            confidence: Confidence level of the signal
            
        Returns:
            Boolean indicating if the signal should be filtered out
        """
        try:
            # Check if confidence is above threshold
            if confidence < self.confidence_threshold:
                return True
                
            # Check if volume is sufficient
            if df['volume'].iloc[-1] < self.min_volume:
                return True
            
            # Check if within trading hours
            if not self._is_trading_hours():
                return True
                
            # Check market regime if enabled
            if self.use_market_regime:
                regime = self.detect_market_regime(df)
                
                # Avoid trading in volatile_ranging markets
                if regime == "volatile_ranging" and confidence < 0.8:
                    return True
                    
                # Require higher confidence in volatile markets
                if "volatile" in regime and confidence < 0.7:
                    return True
            
            # Check for potential reversal patterns
            if signal == 1:  # Buy signal
                # Don't buy in strong downtrends
                if df['close'].iloc[-5:].is_monotonic_decreasing and df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean():
                    return True
                    
                # Don't buy at resistance levels (multiple failed attempts to break upper BB)
                upper_bb_touches = sum(df['close'].iloc[-5:] > df['bb_upper'].iloc[-5:])
                if upper_bb_touches >= 3:
                    return True
                    
            elif signal == -1:  # Sell signal
                # Don't sell in strong uptrends
                if df['close'].iloc[-5:].is_monotonic_increasing and df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean():
                    return True
                    
                # Don't sell at support levels (multiple failed attempts to break lower BB)
                lower_bb_touches = sum(df['close'].iloc[-5:] < df['bb_lower'].iloc[-5:])
                if lower_bb_touches >= 3:
                    return True
            
            # Signal passed all filters
            return False
            
        except Exception as e:
            logger.error(f"Error in signal filtering: {str(e)}")
            return True  # Filter out if there's an error
    
    def calculate_entry_signal(self, df: pd.DataFrame) -> Tuple[int, float, Dict[str, Any]]:
        """
        Calculates trading signal based on ML predictions and technical analysis.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Tuple of (signal, confidence, metadata)
        """
        try:
            # Process data for ML models
            features, _ = self.data_processor.process_data(df)
            
            if features.empty:
                logger.error("Failed to process features for signal generation")
                return 0, 0.0, {}
            
            # Get the last row for prediction
            latest_features = features.iloc[[-1]]
            
            # Get ensemble prediction
            predictions, confidences = self.model_manager.ensemble_predict(latest_features)
            
            if len(predictions) == 0:
                logger.warning("No predictions generated")
                return 0, 0.0, {}
            
            # Get prediction and confidence
            prediction = predictions[0]
            confidence = confidences[0]
            
            # Convert to trading signal (-1, 0, 1)
            if prediction == 1:  # Predicting price increase
                raw_signal = 1  # Buy signal
            else:  # Predicting price decrease
                raw_signal = -1  # Sell signal
            
            # Get additional technical analysis
            tech_analysis = self.analyze_technical_indicators(df)
            
            # Analyze candlestick patterns if enabled
            candle_patterns = {}
            if STRATEGY_PARAMS.get("use_candlestick_patterns", False):
                candle_patterns = self._analyze_candlestick_patterns(df)
                
            # Check for divergences if enabled
            divergences = {}
            if STRATEGY_PARAMS.get("use_divergence_analysis", False):
                divergences = self._check_divergences(df)
                
            # Check VWAP position if enabled
            vwap_analysis = {}
            if STRATEGY_PARAMS.get("use_vwap", False):
                vwap_analysis = self._analyze_vwap(df)
            
            # Apply time multiplier to adjust confidence based on time of day
            time_multiplier = self._calculate_time_multiplier()
            
            # Combine all signals
            if raw_signal == 1:  # Buy signal from ML
                # Base confidence from ML and technical indicators
                base_confidence = 0.6 * confidence + 0.3 * tech_analysis["buy_score"]
                
                # Add contribution from candlestick patterns
                if candle_patterns.get("bullish_strength", 0) > 0:
                    base_confidence += 0.05 * candle_patterns.get("bullish_strength", 0)
                    
                # Add contribution from divergences
                if divergences.get("bullish_divergence", False):
                    base_confidence += 0.1
                    
                # Add contribution from VWAP
                if vwap_analysis.get("price_above_vwap", False):
                    base_confidence += 0.05
                
                # Apply time multiplier
                adjusted_confidence = base_confidence * time_multiplier
                
            else:  # Sell signal from ML
                # Base confidence from ML and technical indicators
                base_confidence = 0.6 * confidence + 0.3 * tech_analysis["sell_score"]
                
                # Add contribution from candlestick patterns
                if candle_patterns.get("bearish_strength", 0) > 0:
                    base_confidence += 0.05 * candle_patterns.get("bearish_strength", 0)
                    
                # Add contribution from divergences
                if divergences.get("bearish_divergence", False):
                    base_confidence += 0.1
                    
                # Add contribution from VWAP
                if vwap_analysis.get("price_below_vwap", False):
                    base_confidence += 0.05
                
                # Apply time multiplier
                adjusted_confidence = base_confidence * time_multiplier
            
            # Cap confidence at 1.0
            adjusted_confidence = min(adjusted_confidence, 1.0)
            
            # Determine final signal
            if self.should_filter_signal(df, raw_signal, adjusted_confidence):
                final_signal = 0
            else:
                final_signal = raw_signal
            
            # Check reward/risk ratio if enabled
            if final_signal != 0 and STRATEGY_PARAMS.get("risk_reward_settings", {}).get("min_reward_risk_ratio"):
                # Calculate potential stop loss and take profit
                current_price = df['close'].iloc[-1]
                stop_loss, take_profit = self.calculate_risk_params(df, current_price, final_signal)
                
                # Calculate reward/risk ratio
                if final_signal == 1:  # Long position
                    reward = take_profit - current_price
                    risk = current_price - stop_loss
                else:  # Short position
                    reward = current_price - take_profit
                    risk = stop_loss - current_price
                
                # Only take trades with sufficient reward/risk ratio
                min_ratio = STRATEGY_PARAMS["risk_reward_settings"]["min_reward_risk_ratio"]
                if risk > 0 and (reward / risk) < min_ratio:
                    logger.info(f"Signal filtered due to insufficient reward/risk ratio: {reward/risk:.2f} < {min_ratio}")
                    final_signal = 0
            
            # Create metadata for the signal
            metadata = {
                "ml_prediction": prediction,
                "ml_confidence": confidence,
                "technical_analysis": tech_analysis,
                "candlestick_patterns": candle_patterns,
                "divergences": divergences,
                "vwap_analysis": vwap_analysis,
                "time_multiplier": time_multiplier,
                "raw_confidence": base_confidence,
                "adjusted_confidence": adjusted_confidence,
                "market_regime": self.detect_market_regime(df) if self.use_market_regime else "not_used",
                "timestamp": pd.Timestamp.now()
            }
            
            return final_signal, adjusted_confidence, metadata
            
        except Exception as e:
            logger.error(f"Error calculating entry signal: {str(e)}")
            return 0, 0.0, {}
    
    def should_exit_position(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determines if current position should be exited.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.position == 0:
            return False, "no_position"
            
        try:
            current_price = df['close'].iloc[-1]
            
            # Check stop loss
            if self.position == 1 and self.current_stop_loss is not None and current_price <= self.current_stop_loss:
                return True, "stop_loss"
                
            if self.position == -1 and self.current_stop_loss is not None and current_price >= self.current_stop_loss:
                return True, "stop_loss"
                
            # Check take profit
            if self.position == 1 and self.current_take_profit is not None and current_price >= self.current_take_profit:
                return True, "take_profit"
                
            if self.position == -1 and self.current_take_profit is not None and current_price <= self.current_take_profit:
                return True, "take_profit"
                
            # Technical reversal signals
            if self.position == 1:  # Long position
                # Check for reversal signals
                if df['rsi'].iloc[-1] > 75:  # Overbought
                    return True, "technical_reversal"
                    
                # MACD crossed below signal line
                if df['macd'].iloc[-2] > df['macd_signal'].iloc[-2] and df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
                    return True, "technical_reversal"
                    
                # Price crossed below EMA fast
                if df['close'].iloc[-2] > df['ema_fast'].iloc[-2] and df['close'].iloc[-1] < df['ema_fast'].iloc[-1]:
                    return True, "technical_reversal"
                    
            elif self.position == -1:  # Short position
                # Check for reversal signals
                if df['rsi'].iloc[-1] < 25:  # Oversold
                    return True, "technical_reversal"
                    
                # MACD crossed above signal line
                if df['macd'].iloc[-2] < df['macd_signal'].iloc[-2] and df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                    return True, "technical_reversal"
                    
                # Price crossed above EMA fast
                if df['close'].iloc[-2] < df['ema_fast'].iloc[-2] and df['close'].iloc[-1] > df['ema_fast'].iloc[-1]:
                    return True, "technical_reversal"
            
            # ML model predicts reversal
            features, _ = self.data_processor.process_data(df)
            if not features.empty:
                latest_features = features.iloc[[-1]]
                predictions, confidences = self.model_manager.ensemble_predict(latest_features)
                
                if len(predictions) > 0:
                    prediction = predictions[0]
                    confidence = confidences[0]
                    
                    # Exit if ML strongly predicts opposite direction
                    if self.position == 1 and prediction == 0 and confidence > 0.75:
                        return True, "ml_reversal"
                        
                    if self.position == -1 and prediction == 1 and confidence > 0.75:
                        return True, "ml_reversal"
            
            # Check end of trading day
            now = datetime.datetime.now()
            end_time_str = TRADING_HOURS["end"]
            end_hour, end_min = map(int, end_time_str.split(':'))
            end_time = now.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)
            
            # Close positions 5 minutes before market close
            if now >= end_time - datetime.timedelta(minutes=5):
                return True, "end_of_day"
            
            return False, "keep_position"
            
        except Exception as e:
            logger.error(f"Error in exit position evaluation: {str(e)}")
            return False, "error"
    
    def calculate_risk_params(self, df: pd.DataFrame, entry_price: float, position_type: int) -> Tuple[float, float]:
        """
        Calculates dynamic stop loss and take profit levels.
        
        Args:
            df: DataFrame with price data and indicators
            entry_price: Entry price for the position
            position_type: Position type (1 for long, -1 for short)
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            # Get ATR for volatility-based stop loss
            atr = df['atr'].iloc[-1]
            
            if position_type == 1:  # Long position
                # Stop loss: entry price - 2 * ATR or recent swing low
                stop_distances = [
                    2 * atr,  # ATR-based
                    entry_price - df['low'].iloc[-20:].min()  # Recent swing low
                ]
                
                # Choose the smaller distance for tighter stop
                stop_distance = min(filter(lambda x: x > 0, stop_distances))
                stop_loss = entry_price - stop_distance
                
                # Take profit: entry price + 3 * ATR or recent swing high
                take_distances = [
                    3 * atr,  # ATR-based (reward:risk ratio of 1.5)
                    df['high'].iloc[-20:].max() - entry_price  # Recent swing high
                ]
                
                # Choose reasonable profit target
                take_distance = min(filter(lambda x: x > 0, take_distances))
                take_profit = entry_price + take_distance
                
            else:  # Short position
                # Stop loss: entry price + 2 * ATR or recent swing high
                stop_distances = [
                    2 * atr,  # ATR-based
                    df['high'].iloc[-20:].max() - entry_price  # Recent swing high
                ]
                
                # Choose the smaller distance for tighter stop
                stop_distance = min(filter(lambda x: x > 0, stop_distances))
                stop_loss = entry_price + stop_distance
                
                # Take profit: entry price - 3 * ATR or recent swing low
                take_distances = [
                    3 * atr,  # ATR-based (reward:risk ratio of 1.5)
                    entry_price - df['low'].iloc[-20:].min()  # Recent swing low
                ]
                
                # Choose reasonable profit target
                take_distance = min(filter(lambda x: x > 0, take_distances))
                take_profit = entry_price - take_distance
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating risk parameters: {str(e)}")
            # Return simple fixed stop loss and take profit
            if position_type == 1:  # Long
                return entry_price * 0.99, entry_price * 1.015
            else:  # Short
                return entry_price * 1.01, entry_price * 0.985
    
    def update_trailing_stop(self, current_price: float) -> None:
        """
        Updates trailing stop loss.
        
        Args:
            current_price: Current market price
        """
        if self.position == 0 or self.current_stop_loss is None:
            return
            
        try:
            # For long positions
            if self.position == 1:
                # Calculate potential new stop loss
                new_stop = current_price - (current_price - self.entry_price) * 0.4
                
                # Only move stop loss up, never down
                if new_stop > self.current_stop_loss:
                    self.current_stop_loss = new_stop
                    logger.info(f"Updated trailing stop loss to {self.current_stop_loss}")
                    
            # For short positions
            elif self.position == -1:
                # Calculate potential new stop loss
                new_stop = current_price + (self.entry_price - current_price) * 0.4
                
                # Only move stop loss down, never up
                if new_stop < self.current_stop_loss:
                    self.current_stop_loss = new_stop
                    logger.info(f"Updated trailing stop loss to {self.current_stop_loss}")
                    
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
    
    def generate_trading_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates trading signals based on current market data.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            Dict with trading signals and metadata
        """
        result = {
            "signal": 0,  # 0: no action, 1: buy, -1: sell, 2: exit long, -2: exit short
            "confidence": 0.0,
            "current_position": self.position,
            "stop_loss": None,
            "take_profit": None,
            "metadata": {}
        }
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Update trailing stop if in a position
            if self.position != 0:
                self.update_trailing_stop(current_price)
            
            # First, check if we should exit existing position
            if self.position != 0:
                should_exit, exit_reason = self.should_exit_position(df)
                
                if should_exit:
                    if self.position == 1:
                        result["signal"] = 2  # Exit long
                        logger.info(f"Exit LONG signal generated: {exit_reason}")
                    else:
                        result["signal"] = -2  # Exit short
                        logger.info(f"Exit SHORT signal generated: {exit_reason}")
                    
                    result["metadata"]["exit_reason"] = exit_reason
                    return result
            
            # If not in a position, check for new entry signals
            if self.position == 0:
                signal, confidence, metadata = self.calculate_entry_signal(df)
                
                if signal != 0:
                    # Calculate stop loss and take profit
                    stop_loss, take_profit = self.calculate_risk_params(df, current_price, signal)
                    
                    result["signal"] = signal
                    result["confidence"] = confidence
                    result["stop_loss"] = stop_loss
                    result["take_profit"] = take_profit
                    result["metadata"] = metadata
                    
                    if signal == 1:
                        logger.info(f"BUY signal generated with {confidence:.2f} confidence")
                    else:
                        logger.info(f"SELL signal generated with {confidence:.2f} confidence")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return result
