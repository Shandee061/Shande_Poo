"""
Configuration settings for the WINFUT trading robot.

Este arquivo contém todas as configurações necessárias para o robô de trading WINFUT,
incluindo parâmetros de API, estratégias, risco, análise técnica e machine learning.
"""
import os

# Profit Pro API Configuration
PROFIT_API_URL = os.getenv("PROFIT_API_URL", "http://localhost:8080")
PROFIT_API_KEY = os.getenv("PROFIT_API_KEY", "")
PROFIT_API_SECRET = os.getenv("PROFIT_API_SECRET", "")

# Profit Pro DLL/Socket Configuration
PROFIT_PRO_USE_DLL = os.getenv("PROFIT_PRO_USE_DLL", "true").lower() == "true"  # Now true by default
PROFIT_PRO_HOST = os.getenv("PROFIT_PRO_HOST", "localhost")
PROFIT_PRO_PORT = int(os.getenv("PROFIT_PRO_PORT", "8080"))

# Profit Pro Simulation/Demo Mode
PROFIT_PRO_SIMULATION_MODE = os.getenv("PROFIT_PRO_SIMULATION_MODE", "true").lower() == "true"

# Profit Pro DLL Configuration
# Para instalação local, ajuste o caminho abaixo para a localização da sua DLL
# Típico: "C:/Program Files/Nelogica/Profit/DLL/ProfitDLL.dll"
PROFIT_PRO_DLL_PATH = os.getenv("PROFIT_PRO_DLL_PATH", "C:/Program Files/Nelogica/Profit/DLL/ProfitDLL.dll")  
PROFIT_PRO_DLL_VERSION = os.getenv("PROFIT_PRO_DLL_VERSION", "5.0.3")
PROFIT_PRO_LOG_PATH = os.getenv("PROFIT_PRO_LOG_PATH", "./profit_dll.log")  # Path to save DLL logs

# MetaTrader 5 Configuration
USE_METATRADER = os.getenv("USE_METATRADER", "false").lower() == "true"
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")

# News Analysis Configuration
NEWS_ANALYSIS_ENABLED = os.getenv("NEWS_ANALYSIS_ENABLED", "false").lower() == "true"
NEWS_UPDATE_INTERVAL = int(os.getenv("NEWS_UPDATE_INTERVAL", "600"))  # 10 minutes
NEWS_MAX_ARTICLES = int(os.getenv("NEWS_MAX_ARTICLES", "50"))
NEWS_SOURCES = os.getenv("NEWS_SOURCES", "").split(",") if os.getenv("NEWS_SOURCES") else []
NEWS_KEYWORDS = os.getenv("NEWS_KEYWORDS", "").split(",") if os.getenv("NEWS_KEYWORDS") else []

# Trading Parameters
SYMBOL = "WINFUT"
TIMEFRAME = "1m"  # 1 minute candlesticks
MAX_POSITION = 1  # Maximum number of contracts
TRADING_HOURS = {
    "start": "09:00",
    "end": "17:55"
}
TRADING_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday

# Technical Indicators Parameters
TECHNICAL_INDICATORS = {
    # Moving Averages
    "sma_fast": 9,
    "sma_slow": 21,
    "ema_fast": 12,
    "ema_slow": 26,
    "ema_200": 200,  # Long-term trend
    
    # Oscillators
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_overbought": 80,
    "stoch_oversold": 20,
    
    # Trend indicators
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_period": 14,
    "adx_threshold": 25,  # ADX above this indicates strong trend
    "cci_period": 14,  # Commodity Channel Index period
    
    # Volatility indicators
    "bbands_period": 20,
    "bbands_std_dev": 2,
    "atr_period": 14,
    
    # Volume indicators
    "obv_signal_period": 5,  # OBV signal line period
    "vwap_period": "day",  # Calculate VWAP based on daily data
    
    # Ichimoku cloud parameters (advanced trend analysis)
    "ichimoku_tenkan": 9,
    "ichimoku_kijun": 26,
    "ichimoku_senkou_b": 52,
    
    # Pivot points
    "pivot_timeframe": "D",  # Calculate pivot points based on daily data
    
    # Divergence detection
    "divergence_lookback": 10  # Number of candles to look back for divergence
}

# Risk Management Parameters
RISK_MANAGEMENT = {
    "stop_loss_ticks": 100,  # Fixed stop loss in ticks
    "take_profit_ticks": 150,  # Fixed take profit in ticks
    "trailing_stop_ticks": 80,  # Trailing stop in ticks
    "max_daily_loss": 1000,  # Maximum daily loss in points
    "max_daily_trades": 10,  # Maximum number of trades per day
    "risk_per_trade": 0.02  # Percentage of account per trade
}

# Machine Learning Model Parameters
ML_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "prediction_horizon": 5,  # Predict price movement for next 5 candles
    "train_test_split": 0.8,  # 80% training, 20% testing
    "lookback_period": 30  # Use 30 periods of historical data for features
}

# Backtesting Parameters
BACKTEST_PARAMS = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "winfut_robot.log"

# Strategy Parameters
STRATEGY_PARAMS = {
    "confidence_threshold": 0.70,  # ML model confidence threshold for trades (increased for better precision)
    "ensemble_weights": {  # Weights for ensemble prediction
        "random_forest": 0.4,
        "xgboost": 0.6  # XGBoost tends to perform better on time series data
    },
    "min_volume": 1000,  # Minimum volume for valid signals
    "use_market_regime": True,  # Whether to consider market regime
    "use_vwap": True,  # Whether to use VWAP for intraday trading
    "use_candlestick_patterns": True,  # Whether to use candlestick pattern analysis
    "use_divergence_analysis": True,  # Whether to check for divergences
    "time_filters": {
        "avoid_first_minutes": 15,  # Avoid trading in first X minutes of session (market open volatility)
        "avoid_lunch_hour": True,   # Avoid trading during lunch hour (typically lower volume)
        "lunch_period": ["12:00", "13:30"],
        "prefer_hours": ["10:00", "11:30", "14:00", "15:30"]  # Prefer trading during these hours
    },
    "risk_reward_settings": {
        "min_reward_risk_ratio": 1.5,  # Minimum reward to risk ratio for valid trades
        "dynamic_position_sizing": True,  # Whether to adjust position size based on volatility
        "trailing_activation_threshold": 0.5,  # % of take profit to hit before activating trailing stop
        "breakeven_threshold": 0.3  # % of take profit to hit before moving stop to breakeven
    }
}
