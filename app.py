"""
Main application file for WINFUT trading robot.
Provides interface for configuration, monitoring, and backtesting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import os

from config import (
    SYMBOL, TIMEFRAME, TECHNICAL_INDICATORS, RISK_MANAGEMENT, ML_PARAMS, 
    BACKTEST_PARAMS, STRATEGY_PARAMS, TRADING_HOURS, PROFIT_PRO_USE_DLL,
    PROFIT_PRO_HOST, PROFIT_PRO_PORT, PROFIT_PRO_DLL_PATH, PROFIT_PRO_DLL_VERSION,
    PROFIT_PRO_LOG_PATH, PROFIT_API_URL, PROFIT_API_KEY, PROFIT_API_SECRET,
    PROFIT_PRO_SIMULATION_MODE, NEWS_ANALYSIS_ENABLED, NEWS_UPDATE_INTERVAL, 
    NEWS_MAX_ARTICLES, NEWS_SOURCES, NEWS_KEYWORDS, USE_METATRADER, MT5_LOGIN, 
    MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH
)
from profit_api import ProfitProAPI
# Removida a importação do MetaTraderAPI, pois vamos usar apenas o Profit Pro
# from metatrader_api import MetaTraderAPI
from data_processor import DataProcessor
from models import ModelManager
from strategy import TradingStrategy
from risk_manager import RiskManager
from backtester import Backtester
from order_manager import OrderManager
from logger import setup_logger
from news_analyzer import NewsAnalyzer
from investing_collector import InvestingCollector, get_market_snapshot
from web_scraper import WinfutWebScraper
from market_regime import MarketRegimeDetector
from auto_optimizer import AutoOptimizer, StrategyParameter
from performance_monitor import PerformanceMonitor, TradingStatus
from utils import (
    plot_price_chart, plot_equity_curve, format_number, 
    display_trade_list, render_technical_indicators,
    create_position_summary, create_backtest_summary,
    get_current_winfut_contract, get_available_winfut_contracts, select_winfut_contract_for_trading
)
from technical_indicators import TechnicalIndicators
from correlation_analyzer import CorrelationAnalyzer
from pattern_detector import PatternDetector
from deep_learning_integration import DeepLearningIntegration
from user_analytics import initialize_analytics, UserPerformanceTracker, UsageStatisticsCollector

# Setup logger
logger = setup_logger()

# Page configuration
st.set_page_config(
    page_title="WINFUT Trading Robot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for persistent data
if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "trader_running" not in st.session_state:
    st.session_state.trader_running = False

if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

if "notifications" not in st.session_state:
    st.session_state.notifications = []

if "price_data" not in st.session_state:
    st.session_state.price_data = None

if "trading_signals" not in st.session_state:
    st.session_state.trading_signals = []

if "open_positions" not in st.session_state:
    st.session_state.open_positions = []

if "order_history" not in st.session_state:
    st.session_state.order_history = []
    
if "news_analyzer" not in st.session_state:
    st.session_state.news_analyzer = None
    
if "news_analysis_enabled" not in st.session_state:
    st.session_state.news_analysis_enabled = NEWS_ANALYSIS_ENABLED
    
if "investing_collector" not in st.session_state:
    st.session_state.investing_collector = None
    
if "web_scraper" not in st.session_state:
    st.session_state.web_scraper = None
    
if "profit_api_initialized" not in st.session_state:
    st.session_state.profit_api_initialized = False
    
if "trading_active" not in st.session_state:
    st.session_state.trading_active = False
if "market_data" not in st.session_state:
    st.session_state.market_data = None
    
if "current_contract" not in st.session_state:
    st.session_state.current_contract = get_current_winfut_contract()
    
if "available_contracts" not in st.session_state:
    st.session_state.available_contracts = get_available_winfut_contracts()

if "market_regime_detector" not in st.session_state:
    st.session_state.market_regime_detector = None
    
if "auto_optimizer" not in st.session_state:
    st.session_state.auto_optimizer = None
    
if "performance_monitor" not in st.session_state:
    st.session_state.performance_monitor = None
    
if "current_regime" not in st.session_state:
    st.session_state.current_regime = None
    
if "optimization_enabled" not in st.session_state:
    st.session_state.optimization_enabled = True
    
if "circuit_breaker_enabled" not in st.session_state:
    st.session_state.circuit_breaker_enabled = True
    
if "technical_indicators" not in st.session_state:
    st.session_state.technical_indicators = None
    
if "correlation_analyzer" not in st.session_state:
    st.session_state.correlation_analyzer = None
    
if "pattern_detector" not in st.session_state:
    st.session_state.pattern_detector = None
    
if "deep_learning" not in st.session_state:
    st.session_state.deep_learning = None

def initialize_components():
    """Initialize all trading components"""
    try:
        # Initialize Profit Pro API
        # Adicionar controle de modo de simulação na barra lateral
        st.sidebar.markdown("## Modo de Operação")
        simulation_mode = st.sidebar.checkbox(
            "Modo de Simulação (sem envio de ordens reais)",
            value=True,  # Por padrão, começamos em modo de simulação por segurança
            help="Ative esta opção para operar em modo de simulação sem enviar ordens reais ao mercado"
        )
        
        # Armazenar o modo de simulação no estado da sessão
        st.session_state.simulation_mode = simulation_mode
        
        # Check if we should use DLL mode
        if PROFIT_PRO_USE_DLL:
            # DLL mode
            # Verificar ambiente Windows para a DLL
            import platform
            is_windows = platform.system() == 'Windows'
            dll_exists = False
            
            if not is_windows:
                st.warning("A DLL do Profit Pro só pode ser usada no ambiente Windows. Usando modo de simulação.")
                st.session_state.simulation_mode = True  # Forçar modo de simulação
            else:
                # Verificar se a DLL existe no caminho especificado
                import os
                dll_exists = os.path.exists(PROFIT_PRO_DLL_PATH)
                if not dll_exists:
                    st.warning(f"DLL não encontrada no caminho: {PROFIT_PRO_DLL_PATH}. Usando modo de simulação.")
                    st.session_state.simulation_mode = True  # Forçar modo de simulação
            
            st.session_state.profit_api = ProfitProAPI(
                use_dll=True,
                dll_path=PROFIT_PRO_DLL_PATH,
                dll_version=PROFIT_PRO_DLL_VERSION,
                symbol=SYMBOL,
                host=PROFIT_PRO_HOST,
                port=PROFIT_PRO_PORT,
                simulation_mode=st.session_state.simulation_mode  # Usar o valor do checkbox ou forçado
            )
            st.info("Inicializando API do Profit Pro no modo DLL...")
        else:
            # REST API mode
            st.session_state.profit_api = ProfitProAPI(
                use_dll=False,
                api_url=PROFIT_API_URL,
                api_key=PROFIT_API_KEY,
                api_secret=PROFIT_API_SECRET,
                host=PROFIT_PRO_HOST,
                port=PROFIT_PRO_PORT,
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                simulation_mode=st.session_state.simulation_mode,  # Usar o valor do checkbox
                dll_path=PROFIT_PRO_DLL_PATH,  # Incluir para caso seja necessário mudar para DLL mais tarde
                dll_version=PROFIT_PRO_DLL_VERSION
            )
            st.info("Inicializando API do Profit Pro no modo REST...")
            
        # Attempt to connect
        connected = st.session_state.profit_api.connect()
        
        # Atualiza o estado da conexão
        st.session_state.profit_api_initialized = connected
        
        if not connected:
            if PROFIT_PRO_USE_DLL:
                if is_windows and dll_exists:
                    st.warning("Falha ao conectar à DLL do Profit Pro. Verifique se o Profit Pro está aberto e a DLL está disponível.")
                    st.info("Certifique-se de que o Profit Pro está aberto e logado em sua conta antes de iniciar o robô.")
                else:
                    st.warning("Continuando com funcionalidade limitada devido à indisponibilidade da DLL.")
            else:
                st.warning("Falha ao conectar à API do Profit Pro. Continuando com funcionalidade limitada. Você pode usar Investing.com para dados.")
            # Não interrompemos a inicialização, permitindo continuar com outras funcionalidades
        else:
            st.success("Conexão com o Profit Pro estabelecida com sucesso!")
            if st.session_state.simulation_mode:
                st.info("Operando em modo de simulação - nenhuma ordem real será enviada.")
            else:
                st.warning("Operando em modo REAL - ordens serão enviadas ao mercado!")
        
        # Use Profit Pro as the trading platform for order manager
        trading_platform = st.session_state.profit_api
        
        # Store platform type in session state for reference
        st.session_state.trading_platform_type = "profit_pro"
        
        # Initialize other components
        st.session_state.data_processor = DataProcessor()
        st.session_state.model_manager = ModelManager()
        st.session_state.risk_manager = RiskManager()
        st.session_state.strategy = TradingStrategy(
            st.session_state.model_manager,
            st.session_state.data_processor
        )
        st.session_state.backtester = Backtester(
            st.session_state.data_processor,
            st.session_state.model_manager,
            st.session_state.strategy,
            st.session_state.risk_manager
        )
        st.session_state.order_manager = OrderManager(
            trading_platform,
            st.session_state.risk_manager
        )
        
        # Initialize News Analyzer if enabled
        if st.session_state.news_analysis_enabled:
            try:
                st.session_state.news_analyzer = NewsAnalyzer(
                    news_sources=NEWS_SOURCES if NEWS_SOURCES else None,
                    keywords=NEWS_KEYWORDS if NEWS_KEYWORDS else None,
                    update_interval=NEWS_UPDATE_INTERVAL,
                    max_articles=NEWS_MAX_ARTICLES
                )
                st.session_state.add_notification("News analyzer initialized successfully", "success")
                
                # Start news analysis if enabled
                st.session_state.news_analyzer.start()
                logger.info("News analyzer started successfully")
            except Exception as e:
                logger.error(f"Error initializing news analyzer: {str(e)}")
                st.session_state.add_notification(f"Error initializing news analyzer: {str(e)}", "error")
        
        # Initialize Investing.com data collector
        try:
            st.session_state.investing_collector = InvestingCollector()
            # Fetch initial market data snapshot
            st.session_state.market_data = get_market_snapshot()
            st.session_state.add_notification("Investing.com data collector initialized successfully", "success")
            logger.info("Investing.com data collector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Investing.com data collector: {str(e)}")
            st.session_state.add_notification(f"Error initializing Investing.com data collector: {str(e)}", "error")
            
        # Initialize B3 Web Scraper
        try:
            st.session_state.web_scraper = WinfutWebScraper()
            st.session_state.add_notification("B3 Web Scraper initialized successfully", "success")
            logger.info("B3 Web Scraper initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing B3 Web Scraper: {str(e)}")
            st.session_state.add_notification(f"Error initializing B3 Web Scraper: {str(e)}", "error")
            
        # Initialize User Analytics and Telemetry
        try:
            # Import user analytics module
            from user_analytics import initialize_analytics
            
            # Get info from environment variables or use defaults
            user_id = os.environ.get("USER_ID", None)
            license_key = os.environ.get("LICENSE_KEY", None)
            api_key = os.environ.get("ANALYTICS_API_KEY", None)
            
            # Initialize analytics
            performance_tracker, usage_collector = initialize_analytics(
                user_id=user_id,
                license_key=license_key,
                api_key=api_key
            )
            
            # Store in session state
            st.session_state.performance_tracker = performance_tracker
            st.session_state.usage_collector = usage_collector
            
            st.session_state.add_notification("Telemetria inicializada com sucesso", "success")
            logger.info("User Analytics initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing User Analytics: {str(e)}")
            st.session_state.add_notification(f"Erro ao inicializar telemetria: {str(e)}", "warning")
        
        # Load models if available
        try:
            models_loaded = st.session_state.model_manager.load_models()
            if models_loaded:
                st.session_state.add_notification("Models loaded successfully", "success")
            else:
                st.session_state.add_notification("No pre-trained models found", "warning")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            st.session_state.add_notification(f"Error loading models: {str(e)}", "error")
            
        # Initialize market regime detector
        try:
            st.session_state.market_regime_detector = MarketRegimeDetector(
                lookback_period=60,
                volatility_window=20,
                trend_threshold=0.3,
                volatility_threshold=0.8,
                range_threshold=0.2,
                use_kmeans=True,
                n_clusters=5
            )
            st.session_state.add_notification("Market regime detector initialized successfully", "success")
            logger.info("Market regime detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing market regime detector: {str(e)}")
            st.session_state.add_notification(f"Error initializing market regime detector: {str(e)}", "error")
            
        # Initialize advanced analysis tools
        try:
            # Technical indicators
            st.session_state.technical_indicators = TechnicalIndicators()
            
            # Correlation analyzer
            st.session_state.correlation_analyzer = CorrelationAnalyzer()
            
            # Pattern detector
            st.session_state.pattern_detector = PatternDetector()
            
            # Initialize Deep Learning models
            st.session_state.deep_learning = DeepLearningIntegration(
                model_dir="models",
                use_gpu=False,
                default_lookback=60,
                default_horizon=5,
                confidence_threshold=0.7
            )
            
            st.session_state.add_notification("Advanced analysis tools initialized successfully", "success")
            logger.info("Advanced analysis tools initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing advanced analysis tools: {str(e)}")
            st.session_state.add_notification(f"Error initializing advanced analysis tools: {str(e)}", "error")
            
        # Initialize performance monitor
        try:
            st.session_state.performance_monitor = PerformanceMonitor(
                monitoring_interval=60,
                warning_threshold=-2.0,
                circuit_breaker_threshold=-5.0,
                daily_loss_limit=-3.0,
                metrics_window=20,
                auto_restart_time=60,
                enable_circuit_breaker=st.session_state.circuit_breaker_enabled,
                save_metrics=True,
                metrics_dir="metrics"
            )
            
            # Initialize with default equity value (to be updated with actual value)
            initial_equity = 10000.0  # Default starting equity
            st.session_state.performance_monitor.initialize(initial_equity)
            
            # Register callbacks for important events
            st.session_state.performance_monitor.register_callbacks(
                status_change=lambda old, new: st.session_state.add_notification(
                    f"Trading status changed: {old.value} -> {new.value}", 
                    "warning"
                ),
                circuit_breaker=lambda reason: st.session_state.add_notification(
                    f"Circuit breaker activated: {reason}", 
                    "error"
                ),
                warning=lambda msg: st.session_state.add_notification(
                    msg, 
                    "warning"
                )
            )
            
            st.session_state.add_notification("Performance monitor initialized successfully", "success")
            logger.info("Performance monitor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing performance monitor: {str(e)}")
            st.session_state.add_notification(f"Error initializing performance monitor: {str(e)}", "error")
            
        # Initialize auto optimizer with strategy parameters
        try:
            # Create parameter definitions
            strategy_parameters = {
                "confidence_threshold": StrategyParameter(
                    name="confidence_threshold",
                    default_value=0.7,
                    min_value=0.5,
                    max_value=0.95,
                    step=0.05,
                    param_type="float",
                    description="Limiar de confiança para executar operações"
                ),
                "stop_loss_ticks": StrategyParameter(
                    name="stop_loss_ticks",
                    default_value=150,
                    min_value=50,
                    max_value=500,
                    step=10,
                    param_type="int",
                    description="Stop loss em ticks"
                ),
                "take_profit_ticks": StrategyParameter(
                    name="take_profit_ticks",
                    default_value=300,
                    min_value=100,
                    max_value=1000,
                    step=50,
                    param_type="int",
                    description="Take profit em ticks"
                ),
                "ema_fast_period": StrategyParameter(
                    name="ema_fast_period",
                    default_value=9,
                    min_value=5,
                    max_value=20,
                    step=1,
                    param_type="int",
                    description="Período da EMA rápida"
                ),
                "ema_slow_period": StrategyParameter(
                    name="ema_slow_period",
                    default_value=21,
                    min_value=15,
                    max_value=50,
                    step=1,
                    param_type="int",
                    description="Período da EMA lenta"
                ),
                "rsi_period": StrategyParameter(
                    name="rsi_period",
                    default_value=14,
                    min_value=7,
                    max_value=21,
                    step=1,
                    param_type="int",
                    description="Período do RSI"
                ),
                "rsi_overbought": StrategyParameter(
                    name="rsi_overbought",
                    default_value=70,
                    min_value=65,
                    max_value=85,
                    step=1,
                    param_type="int",
                    description="Nível de sobrevenda do RSI"
                ),
                "rsi_oversold": StrategyParameter(
                    name="rsi_oversold",
                    default_value=30,
                    min_value=15,
                    max_value=35,
                    step=1,
                    param_type="int",
                    description="Nível de sobrecompra do RSI"
                ),
                "macd_fast": StrategyParameter(
                    name="macd_fast",
                    default_value=12,
                    min_value=8,
                    max_value=20,
                    step=1,
                    param_type="int",
                    description="Período rápido do MACD"
                ),
                "macd_slow": StrategyParameter(
                    name="macd_slow",
                    default_value=26,
                    min_value=20,
                    max_value=40,
                    step=1,
                    param_type="int",
                    description="Período lento do MACD"
                ),
                "macd_signal": StrategyParameter(
                    name="macd_signal",
                    default_value=9,
                    min_value=5,
                    max_value=15,
                    step=1,
                    param_type="int",
                    description="Período do sinal do MACD"
                ),
                "volume_filter": StrategyParameter(
                    name="volume_filter",
                    default_value=True,
                    param_type="bool",
                    description="Filtrar sinais por volume"
                )
            }
            
            # Configure the backtesting function for optimizer
            def backtest_func(data, params):
                # Convert parameters dict to format expected by backtester
                backtest_params = {
                    "strategy_params": params,
                    "technical_indicators": TECHNICAL_INDICATORS,
                    "risk_management": RISK_MANAGEMENT
                }
                
                # Run backtest with the given parameters
                result = st.session_state.backtester.run_backtest(
                    data, 
                    custom_params=backtest_params
                )
                
                return result
            
            # Initialize auto optimizer
            st.session_state.auto_optimizer = AutoOptimizer(
                backtest_func=backtest_func,
                parameters=strategy_parameters,
                evaluation_metric="profit_factor",
                optimization_interval=50,
                exploration_rate=0.2,
                use_regime_detection=True,
                models_dir="models"
            )
            
            # Try to load any saved models
            models_loaded = st.session_state.auto_optimizer.load_models()
            if models_loaded:
                st.session_state.add_notification("Optimization models loaded successfully", "success")
            
            st.session_state.add_notification("Auto optimizer initialized successfully", "success")
            logger.info("Auto optimizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing auto optimizer: {str(e)}")
            st.session_state.add_notification(f"Error initializing auto optimizer: {str(e)}", "error")
            
        st.session_state.initialized = True
        logger.info("Trading components initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        return False

def add_notification(message, type="info"):
    """Add a notification to the app"""
    st.session_state.notifications.append({
        "message": message,
        "type": type,
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    })
    if len(st.session_state.notifications) > 100:
        st.session_state.notifications.pop(0)

# Add the method to session_state
st.session_state.add_notification = add_notification

def update_investing_data():
    """Update market data from Investing.com"""
    try:
        if st.session_state.investing_collector is None:
            logger.warning("Investing collector not initialized")
            return False
        
        # Tentar obter dados do mercado
        market_data = get_market_snapshot()
        
        # Verificar se os dados estão completos
        has_indices = bool(market_data and "indices" in market_data and market_data["indices"])
        has_currencies = bool(market_data and "currencies" in market_data and market_data["currencies"])
        
        if market_data and has_indices and has_currencies:
            st.session_state.market_data = market_data
            logger.info("Market data from Investing.com updated successfully")
            return True
        else:
            logger.warning("Incomplete or missing market data from Investing.com, using synthetic data")
            
            # Verificar quais dados estão faltando e gerar dados sintéticos quando necessário
            if not market_data:
                market_data = {
                    "timestamp": datetime.datetime.now(),
                    "indices": {},
                    "currencies": {},
                    "sectors": {}
                }
            
            # Se não tiver dados de índices, gerar sintéticos para IBOVESPA e WINFUT
            if not has_indices:
                logger.warning("Generating synthetic data for indices (Investing.com API failure)")
                now = datetime.datetime.now()
                
                # Gerar para IBOVESPA
                if "IBOVESPA" not in market_data["indices"]:
                    df = st.session_state.investing_collector._generate_synthetic_data(
                        "IBOVESPA", "1d", now - datetime.timedelta(days=2), now
                    )
                    if not df.empty:
                        # Calcular a variação do último dia
                        last_close = df['close'].iloc[-1]
                        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close * 0.99
                        change = last_close - prev_close
                        change_pct = (change / prev_close) * 100
                        
                        market_data["indices"]["IBOVESPA"] = {
                            "timestamp": now,
                            "open": df['open'].iloc[-1],
                            "high": df['high'].iloc[-1],
                            "low": df['low'].iloc[-1],
                            "close": last_close,
                            "volume": df['volume'].iloc[-1],
                            "change": change,
                            "change_pct": change_pct,
                            "symbol": "IBOVESPA",
                            "synthetic": True
                        }
                
                # Gerar para WINFUT
                if "WINFUT" not in market_data["indices"]:
                    df = st.session_state.investing_collector._generate_synthetic_data(
                        "WINFUT", "15m", now - datetime.timedelta(days=1), now
                    )
                    if not df.empty:
                        # Calcular a variação do último período
                        last_close = df['close'].iloc[-1]
                        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close * 0.99
                        change = last_close - prev_close
                        change_pct = (change / prev_close) * 100
                        
                        market_data["indices"]["WINFUT"] = {
                            "timestamp": now,
                            "open": df['open'].iloc[-1],
                            "high": df['high'].iloc[-1],
                            "low": df['low'].iloc[-1],
                            "close": last_close,
                            "volume": df['volume'].iloc[-1],
                            "change": change,
                            "change_pct": change_pct,
                            "symbol": "WINFUT",
                            "synthetic": True
                        }
            
            # Se não tiver dados de moedas, gerar sintéticos para USD_BRL
            if not has_currencies:
                logger.warning("Generating synthetic data for currencies (Investing.com API failure)")
                now = datetime.datetime.now()
                
                if "USD_BRL" not in market_data["currencies"]:
                    df = st.session_state.investing_collector._generate_synthetic_data(
                        "USD_BRL", "1d", now - datetime.timedelta(days=2), now
                    )
                    if not df.empty:
                        # Calcular a variação do último dia
                        last_close = df['close'].iloc[-1]
                        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close * 1.01
                        change = last_close - prev_close
                        change_pct = (change / prev_close) * 100
                        
                        market_data["currencies"]["USD_BRL"] = {
                            "timestamp": now,
                            "open": df['open'].iloc[-1],
                            "high": df['high'].iloc[-1],
                            "low": df['low'].iloc[-1],
                            "close": last_close,
                            "volume": df['volume'].iloc[-1],
                            "change": change,
                            "change_pct": change_pct,
                            "symbol": "USD_BRL",
                            "synthetic": True
                        }
            
            # Atualizar dados no session_state
            st.session_state.market_data = market_data
            logger.info("Market data updated with synthetic data where needed")
            
            return True
            
    except Exception as e:
        logger.error(f"Error updating Investing.com data: {str(e)}")
        return False


def fetch_market_data():
    """Fetch and update market data"""
    try:
        # Get OHLCV data from Profit Pro
        price_data = None
        
        # Get data from Profit Pro
        price_data = st.session_state.profit_api.get_price_data(period="1d", limit=1000)
        
        if price_data.empty:
            logger.warning("Received empty price data from API")
            
            # Try to get data from Investing.com as fallback
            try:
                if st.session_state.investing_collector is not None:
                    # Usa o contrato atual de Mini Índice
                    current_contract = st.session_state.current_contract
                    investing_data = st.session_state.investing_collector.get_historical_data(
                        symbol=current_contract,
                        interval="15m",
                        start_date=datetime.datetime.now() - datetime.timedelta(days=5)
                    )
                    logger.info(f"Buscando dados para contrato: {current_contract}")
                    
                    if not investing_data.empty:
                        logger.info(f"Using Investing.com data as fallback: {len(investing_data)} bars")
                        st.session_state.add_notification(
                            "Using Investing.com data as fallback for price data", 
                            "info"
                        )
                        
                        # Process Investing.com data
                        processed_data = st.session_state.data_processor.add_technical_indicators(investing_data)
                        st.session_state.price_data = processed_data
                    else:
                        logger.warning("No data available from Investing.com either")
            except Exception as investing_error:
                logger.error(f"Error getting Investing.com fallback data: {str(investing_error)}")
            
            # Update market data from Investing.com
            update_investing_data()
            return
        
        # Process data with technical indicators
        processed_data = st.session_state.data_processor.add_technical_indicators(price_data)
        
        # Update session state
        st.session_state.price_data = processed_data
        
        # Detect market regime if enabled
        if st.session_state.market_regime_detector is not None and not processed_data.empty:
            try:
                # Detect current market regime
                current_regime = st.session_state.market_regime_detector.detect_regime(processed_data)
                
                # If regime changed, notify
                if st.session_state.current_regime != current_regime:
                    old_regime = st.session_state.current_regime
                    old_regime_name = old_regime.value if old_regime else "undefined"
                    st.session_state.add_notification(
                        f"Regime de mercado alterado: {old_regime_name} -> {current_regime.value}",
                        "warning"
                    )
                    logger.info(f"Market regime changed: {old_regime_name} -> {current_regime.value}")
                
                # Update current regime in session state
                st.session_state.current_regime = current_regime
                
                # Get regime summary
                regime_summary = st.session_state.market_regime_detector.get_regime_summary()
                logger.info(f"Current market regime: {regime_summary['current_regime']} (stability: {regime_summary['stability']:.2f})")
                
                # Adapt strategy parameters based on regime if auto-optimization is enabled
                if st.session_state.optimization_enabled and st.session_state.auto_optimizer is not None:
                    # Check if it's time to optimize
                    time_to_optimize = st.session_state.auto_optimizer.update_trade_count()
                    
                    if time_to_optimize:
                        logger.info("Running parameter optimization...")
                        
                        # Optimizer will use ML or other methods to find optimal parameters
                        optimized_params = st.session_state.auto_optimizer.optimize(processed_data)
                        
                        if optimized_params:
                            st.session_state.add_notification(
                                "Parameters optimized for current market conditions",
                                "success"
                            )
                            logger.info(f"Parameters optimized: {optimized_params}")
                            
                            # Apply optimized parameters to strategy
                            st.session_state.strategy.update_parameters(optimized_params)
            except Exception as e:
                logger.error(f"Error in market regime detection: {str(e)}")
        
        # Generate trading signals
        if processed_data is not None and not processed_data.empty:
            # Get current trading status from performance monitor
            trading_status = TradingStatus.ACTIVE
            if st.session_state.performance_monitor is not None:
                trading_status = st.session_state.performance_monitor.check_status()
            
            # Only generate signals if we're allowed to trade
            if trading_status in [TradingStatus.ACTIVE, TradingStatus.LIMITED]:
                # If in LIMITED status, adjust confidence threshold
                if trading_status == TradingStatus.LIMITED and hasattr(st.session_state.strategy, 'confidence_threshold'):
                    # Store original threshold
                    original_threshold = st.session_state.strategy.confidence_threshold
                    # Increase threshold to be more selective in LIMITED mode
                    st.session_state.strategy.confidence_threshold = min(0.9, original_threshold * 1.2)
                    
                # Generate signals
                signals = st.session_state.strategy.generate_trading_signals(processed_data)
                
                # Restore original threshold if it was modified
                if trading_status == TradingStatus.LIMITED and hasattr(st.session_state.strategy, 'confidence_threshold'):
                    st.session_state.strategy.confidence_threshold = original_threshold
                
                if signals["signal"] != 0:
                    signal_type = "BUY" if signals["signal"] == 1 else "SELL"
                    st.session_state.add_notification(
                        f"New {signal_type} signal generated with {signals['confidence']:.2f} confidence", 
                        "info"
                    )
                    
                    # Add to signals list
                    signals["timestamp"] = datetime.datetime.now()
                    signals["regime"] = st.session_state.current_regime.value if st.session_state.current_regime else "undefined"
                    st.session_state.trading_signals.append(signals)
                    
                    # Keep only last 100 signals
                    if len(st.session_state.trading_signals) > 100:
                        st.session_state.trading_signals.pop(0)
        
        # Update open positions
        open_positions = st.session_state.order_manager.get_open_positions()
        st.session_state.open_positions = open_positions
        
        # Update order history
        order_history = st.session_state.order_manager.get_order_history()
        st.session_state.order_history = order_history
        
        # Update performance monitor with latest data
        if st.session_state.performance_monitor is not None:
            # Update current equity
            if open_positions or order_history:
                try:
                    # Calculate current equity based on open positions and order history
                    initial_equity = 10000.0  # Default value
                    realized_pnl = sum([order.get("profit", 0) for order in order_history])
                    unrealized_pnl = sum([pos.get("current_profit", 0) for pos in open_positions])
                    current_equity = initial_equity + realized_pnl + unrealized_pnl
                    
                    # Update monitor with current equity
                    st.session_state.performance_monitor.update_equity(current_equity)
                    
                    # Record any new closed trades
                    for order in order_history:
                        # Only process orders that haven't been recorded yet
                        if order.get("processed_by_monitor", False) == False:
                            st.session_state.performance_monitor.record_trade(order)
                            # Mark as processed
                            order["processed_by_monitor"] = True
                except Exception as e:
                    logger.error(f"Error updating performance monitor: {str(e)}")
        
        # Also update Investing.com market data
        update_investing_data()
        
        logger.info(f"Market data updated successfully: {len(processed_data)} bars")
    
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        st.session_state.add_notification(f"Error fetching market data: {str(e)}", "error")
        
        # Try to update at least the market summary from Investing.com
        update_investing_data()


def start_trading():
    """Start automated trading"""
    if not st.session_state.initialized:
        st.error("Trading components not initialized")
        return
    
    # Verificar se a API do Profit está inicializada
    if not st.session_state.profit_api_initialized:
        st.error("A API do Profit Pro não está conectada. Verifique a conexão antes de iniciar o trading.")
        return
    
    st.session_state.trader_running = True
    st.session_state.trading_active = True
    st.session_state.add_notification("Trading automatizado iniciado", "success")
    logger.info("Automated trading started")


def stop_trading():
    """Stop automated trading"""
    if not st.session_state.initialized:
        return
    
    st.session_state.trader_running = False
    st.session_state.trading_active = False
    st.session_state.add_notification("Trading automatizado interrompido", "warning")
    logger.info("Automated trading stopped")


def run_backtest():
    """Run backtest with current parameters"""
    if not st.session_state.initialized:
        st.error("Trading components not initialized")
        return
    
    try:
        # Get data source selection
        data_source = st.session_state.get("backtest_data_source", "profit_pro")
        
        # Try to get data based on selected source
        price_data = None
        
        if data_source == "profit_pro":
            # Get data from Profit Pro
            price_data = st.session_state.profit_api.get_price_data(period="1y", limit=5000)
            
            # If no data from Profit Pro, try Investing.com
            if price_data is None or price_data.empty:
                logger.warning("No data available from Profit Pro for backtesting, trying Investing.com...")
                data_source = "investing"
        
        if data_source == "investing":
            # Get data from Investing.com
            try:
                # Use custom days if available, otherwise use 365 days (1 year)
                days = st.session_state.get("backtest_days", 30)
                start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                
                if st.session_state.investing_collector is not None:
                    # Usa o contrato atual de Mini Índice
                    current_contract = st.session_state.get("backtest_contract", st.session_state.current_contract)
                    # Get historical data with selected timeframe
                    price_data = st.session_state.investing_collector.get_historical_data(
                        symbol=current_contract,
                        interval=st.session_state.get("backtest_timeframe", "15m"),
                        start_date=start_date
                    )
                    logger.info(f"Backtesting com contrato: {current_contract}")
                    
                    if not price_data.empty:
                        st.session_state.add_notification(
                            f"Using Investing.com data for backtesting: {len(price_data)} periods", 
                            "info"
                        )
                    else:
                        # Gerar dados sintéticos para o contrato atual
                        st.warning("Usando dados sintéticos para backtesting. Dados reais não disponíveis.")
                        logger.warning(f"Gerando dados sintéticos para backtesting com contrato: {current_contract}")
                        
                        # Determinar período para geração de dados sintéticos
                        days = st.session_state.get("backtest_days", 30)
                        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                        end_date = datetime.datetime.now()
                        
                        # Gerar dados sintéticos usando o método do InvestingCollector
                        price_data = st.session_state.investing_collector._generate_synthetic_data(
                            symbol=current_contract,
                            interval=st.session_state.get("backtest_timeframe", "15m"),
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if price_data.empty:
                            st.error("Falha ao gerar dados sintéticos para backtesting")
                            return
                        
                        # Adicionar notificação sobre dados sintéticos
                        st.session_state.add_notification(
                            f"Usando dados sintéticos para backtesting: {len(price_data)} períodos", 
                            "warning"
                        )
                else:
                    st.error("Investing.com collector not initialized")
                    return
            except Exception as investing_error:
                logger.error(f"Error getting Investing.com data for backtesting: {str(investing_error)}")
                st.error(f"Error getting data from Investing.com: {str(investing_error)}")
                return
        
        if price_data is None or price_data.empty:
            st.error("Failed to fetch any price data for backtesting")
            return
        
        # Run backtest
        with st.spinner("Running backtest..."):
            results = st.session_state.backtester.run_backtest(price_data)
            
            if "success" in results and results["success"] == False:
                st.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
                return
            
            st.session_state.backtest_results = results
            st.session_state.backtest_data_source = data_source
            
            # Create detailed backtest analysis
            analysis = st.session_state.backtester.analyze_results()
            st.session_state.backtest_analysis = analysis
            
            st.session_state.add_notification(
                f"Backtest completed successfully using {data_source.replace('_', ' ')} data", 
                "success"
            )
            logger.info(f"Backtest completed successfully using {data_source} data")
            
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        st.error(f"Error running backtest: {str(e)}")


def place_manual_order():
    """Place a manual order based on form inputs"""
    if not st.session_state.initialized:
        st.error("Trading components not initialized")
        return
    
    try:
        order_type = st.session_state.manual_order_type
        side = st.session_state.manual_order_side
        quantity = st.session_state.manual_order_quantity
        
        # Get current price for default stop loss and take profit
        current_price = None
        if st.session_state.price_data is not None and not st.session_state.price_data.empty:
            current_price = st.session_state.price_data['close'].iloc[-1]
        
        # Calculate stop loss and take profit if enabled
        stop_loss = None
        take_profit = None
        
        if st.session_state.manual_sl_enabled and current_price:
            if side == "BUY":
                stop_loss = current_price * (1 - st.session_state.manual_sl_pct / 100)
            else:
                stop_loss = current_price * (1 + st.session_state.manual_sl_pct / 100)
        
        if st.session_state.manual_tp_enabled and current_price:
            if side == "BUY":
                take_profit = current_price * (1 + st.session_state.manual_tp_pct / 100)
            else:
                take_profit = current_price * (1 - st.session_state.manual_tp_pct / 100)
        
        # Place order based on type
        if order_type == "MARKET":
            result = st.session_state.order_manager.place_market_order(
                side=side,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        else:  # LIMIT order
            limit_price = st.session_state.manual_limit_price
            result = st.session_state.order_manager.place_limit_order(
                side=side,
                quantity=quantity,
                price=limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        if result.get("success", False):
            order_id = result.get('order_id', 'Unknown')
            st.session_state.add_notification(
                f"Order placed successfully: {order_id}", 
                "success"
            )
            logger.info(f"Manual order placed successfully: {result}")
            
            # Record trade in telemetry if available
            if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                try:
                    # Create trade data for telemetry
                    trade_data = {
                        "id": order_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "contract": st.session_state.current_contract,
                        "direction": side,
                        "quantity": quantity,
                        "price": current_price,
                        "pnl": 0.0  # Initial PnL is zero, will be updated on close
                    }
                    
                    # Record in telemetry
                    st.session_state.performance_tracker.record_trade(trade_data)
                    logger.info(f"Trade recorded in telemetry: {order_id}")
                    
                    # Record usage if available
                    if 'usage_collector' in st.session_state and st.session_state.usage_collector:
                        st.session_state.usage_collector.record_feature_usage("place_order")
                except Exception as e:
                    logger.error(f"Error recording trade in telemetry: {str(e)}")
        else:
            st.session_state.add_notification(
                f"Failed to place order: {result.get('error', 'Unknown error')}", 
                "error"
            )
            logger.error(f"Failed to place manual order: {result}")
        
    except Exception as e:
        logger.error(f"Error placing manual order: {str(e)}")
        st.error(f"Error placing manual order: {str(e)}")


def train_models():
    """Train ML models with current data"""
    if not st.session_state.initialized:
        st.error("Trading components not initialized")
        return
    
    try:
        # Get data source selection
        data_source = st.session_state.get("model_data_source", "profit_pro")
        
        # Try to get data based on selected source
        price_data = None
        
        if data_source == "profit_pro":
            # Get data from Profit Pro
            price_data = st.session_state.profit_api.get_price_data(period="1y", limit=5000)
            
            # If no data from Profit Pro, try Investing.com
            if price_data is None or price_data.empty:
                logger.warning("No data available from Profit Pro for model training, trying Investing.com...")
                data_source = "investing"
        
        if data_source == "investing":
            # Get data from Investing.com
            try:
                # Get selected timeframe and days if available in session state
                timeframe = st.session_state.get("model_timeframe", "15m")
                days = st.session_state.get("model_days", 30)
                start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                
                if st.session_state.investing_collector is not None:
                    # Usa o contrato atual de Mini Índice
                    current_contract = st.session_state.get("model_contract", st.session_state.current_contract)
                    # Get historical data with selected params
                    price_data = st.session_state.investing_collector.get_historical_data(
                        symbol=current_contract,
                        interval=timeframe,
                        start_date=start_date
                    )
                    logger.info(f"Treinando modelos com contrato: {current_contract}")
                    
                    if not price_data.empty:
                        st.session_state.add_notification(
                            f"Using Investing.com data for model training: {len(price_data)} periods", 
                            "info"
                        )
                    else:
                        # Gerar dados sintéticos para o contrato atual
                        st.warning("Usando dados sintéticos para treinamento. Dados reais não disponíveis.")
                        logger.warning(f"Gerando dados sintéticos para treinamento com contrato: {current_contract}")
                        
                        # Determinar período para geração de dados sintéticos
                        days = st.session_state.get("model_days", 30)
                        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                        end_date = datetime.datetime.now()
                        
                        # Gerar dados sintéticos usando o método do InvestingCollector
                        price_data = st.session_state.investing_collector._generate_synthetic_data(
                            symbol=current_contract,
                            interval=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if price_data.empty:
                            st.error("Falha ao gerar dados sintéticos para treinamento")
                            return
                        
                        # Adicionar notificação sobre dados sintéticos
                        st.session_state.add_notification(
                            f"Usando dados sintéticos para treinamento: {len(price_data)} períodos", 
                            "warning"
                        )
                else:
                    st.error("Investing.com collector not initialized")
                    return
            except Exception as investing_error:
                logger.error(f"Error getting Investing.com data for model training: {str(investing_error)}")
                st.error(f"Error getting data from Investing.com: {str(investing_error)}")
                return
        
        if price_data is None or price_data.empty:
            st.error("Failed to fetch any price data for model training")
            return
        
        # Process data
        with st.spinner("Processing data for training..."):
            processed_data = st.session_state.data_processor.add_technical_indicators(price_data)
            features, target = st.session_state.data_processor.process_data(processed_data)
            
            if features.empty or target.empty:
                st.error("Failed to process features for training")
                return
        
        # Train models
        with st.spinner("Training models..."):
            models = st.session_state.model_manager.train_models(features, target)
            
            if not models:
                st.error("Failed to train models")
                return
            
            # Evaluate models on test data
            evaluation = st.session_state.model_manager.evaluate_models(features, target)
            
            # Save models
            saved = st.session_state.model_manager.save_models()
            
            if saved:
                st.session_state.add_notification(
                    f"Models trained and saved successfully using {data_source.replace('_', ' ')} data",
                    "success"
                )
                logger.info(f"Models trained and saved successfully using {data_source} data")
                
                # Save data source for reference
                st.session_state.model_data_source = data_source
                
                # Display evaluation metrics
                st.session_state.model_evaluation = evaluation
            else:
                st.session_state.add_notification("Models trained but failed to save", "warning")
                logger.warning("Models trained but failed to save")
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        st.error(f"Error training models: {str(e)}")


def close_all_positions():
    """Close all open positions"""
    if not st.session_state.initialized:
        st.error("Trading components not initialized")
        return
    
    try:
        result = st.session_state.order_manager.close_all_positions("manual_closure")
        
        closed_count = result.get("closed_positions", 0)
        if closed_count > 0:
            st.session_state.add_notification(f"Closed {closed_count} positions", "success")
            logger.info(f"Manually closed {closed_count} positions")
        else:
            st.session_state.add_notification("No positions to close", "info")
            logger.info("No positions to close")
        
    except Exception as e:
        logger.error(f"Error closing positions: {str(e)}")
        st.error(f"Error closing positions: {str(e)}")


# Main application layout
def main():
    # Inicializa a API e verifica o modo de simulação
    api = ProfitProAPI()
    
    # Mostrar título com indicação de modo simulação quando ativo
    if api.simulation_mode:
        st.title("WINFUT Automated Trading Robot 🧪 [MODO DE SIMULAÇÃO]")
        # Adiciona uma mensagem informativa sobre o modo de simulação
        st.warning("""
        **MODO DE SIMULAÇÃO ATIVO** - Nenhuma ordem real será enviada à corretora. 
        Perfeito para testar estratégias sem risco financeiro.
        """)
    else:
        st.title("WINFUT Automated Trading Robot")
    
    # Initialize components if not done already
    if not st.session_state.initialized:
        # Auto-initialize on page load for testing
        if True:  # st.button("Initialize Trading System", type="primary"):
            with st.spinner("Initializing trading components..."):
                initialized = initialize_components()
                if initialized:
                    st.success("Trading system initialized successfully!")
                    # Fetch initial data
                    fetch_market_data()
    
    # Only show main content if initialized
    if st.session_state.initialized:
        # Create tabs for different sections
        # Page title and description
        st.title("WINFUT Trading Robot")
        
        # Status indicator and system state
        col_status, col_version = st.columns([3, 1])
        
        with col_status:
            # Connection status indicators
            status_cols = st.columns(3)
            
            with status_cols[0]:
                connection_status = "🟢 Conectado" if st.session_state.profit_api_initialized else "🔴 Desconectado"
                connection_color = "green" if st.session_state.profit_api_initialized else "red"
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'><b>Profit Pro:</b> <span style='color:{connection_color};'>{connection_status}</span></div>", unsafe_allow_html=True)
            
            with status_cols[1]:
                investing_status = "🟢 Online" if st.session_state.investing_collector is not None else "🔴 Offline"
                investing_color = "green" if st.session_state.investing_collector is not None else "red"
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'><b>Investing.com:</b> <span style='color:{investing_color};'>{investing_status}</span></div>", unsafe_allow_html=True)
            
            with status_cols[2]:
                trading_status = "🟢 Ativo" if st.session_state.trading_active else "🔴 Inativo"
                trading_color = "green" if st.session_state.trading_active else "red"
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'><b>Trading Automático:</b> <span style='color:{trading_color};'>{trading_status}</span></div>", unsafe_allow_html=True)
        
        with col_version:
            st.info("Versão 1.0.0")
        
        # Quick action buttons
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("🚀 Iniciar Trading", use_container_width=True):
                start_trading()
        
        with action_cols[1]:
            if st.button("⏹️ Parar Trading", use_container_width=True):
                stop_trading()
        
        with action_cols[2]:
            if st.button("🔄 Atualizar Dados", use_container_width=True):
                with st.spinner("Atualizando dados de mercado..."):
                    fetch_market_data()
                st.success("Dados atualizados com sucesso!")
        
        with action_cols[3]:
            if st.button("❌ Fechar Posições", use_container_width=True, disabled=not st.session_state.open_positions):
                close_all_positions()
                
        # Separator
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "Dashboard", "Trading", "Backtesting", "Machine Learning", "Deep Learning", "Configuration", "Contratos", "Licenciamento", "Logs", "Análise Avançada"
        ])
        
        # Dashboard Tab
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Dados de Mercado")
                
                # Price chart
                if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                    fig = plot_price_chart(st.session_state.price_data)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No price data available. Initialize the system and fetch data.")
                
                # Indicadores técnicos
                if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                    with st.expander("Indicadores Técnicos", expanded=False):
                        st.write(render_technical_indicators(st.session_state.price_data))
                
                # Dados do Investing.com
                if st.session_state.investing_collector is not None:
                    st.subheader("Dados de Mercado (Investing.com)")
                    
                    # Botão para atualizar dados do Investing.com
                    if st.button("Atualizar Dados de Mercado"):
                        with st.spinner("Buscando dados atualizados do Investing.com..."):
                            updated = update_investing_data()
                            if updated:
                                st.success("Dados de mercado do Investing.com atualizados com sucesso")
                            else:
                                st.error("Falha ao atualizar dados do Investing.com")
                    
                    # Display market summary
                    if st.session_state.market_data:
                        indices = st.session_state.market_data.get("indices", {})
                        currencies = st.session_state.market_data.get("currencies", {})
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("#### Índices")
                            if "IBOVESPA" in indices:
                                ibov = indices["IBOVESPA"]
                                # Calcula a variação percentual
                                change = ibov.get("change_pct", 0)
                                change_value = ibov.get("change", 0)
                                st.metric(
                                    "IBOVESPA", 
                                    format_number(ibov.get("close", 0)),
                                    delta=f"{change:.2f}% ({format_number(change_value)})",
                                    delta_color="normal"
                                )
                            
                            if "WINFUT" in indices:
                                winfut = indices["WINFUT"]
                                # Calcula a variação percentual
                                change = winfut.get("change_pct", 0)
                                change_value = winfut.get("change", 0)
                                st.metric(
                                    "WINFUT", 
                                    format_number(winfut.get("close", 0)),
                                    delta=f"{change:.2f}% ({format_number(change_value)})",
                                    delta_color="normal"
                                )
                                
                        with col_b:
                            st.markdown("#### Câmbio")
                            if "USD_BRL" in currencies:
                                usd_brl = currencies["USD_BRL"]
                                # Calcula a variação percentual
                                change = usd_brl.get("change_pct", 0)
                                change_value = usd_brl.get("change", 0)
                                st.metric(
                                    "USD/BRL", 
                                    format_number(usd_brl.get("close", 0), decimals=4),
                                    delta=f"{change:.2f}% ({format_number(change_value, decimals=4)})",
                                    delta_color="inverse"  # Dólar caindo (negativo) é bom para o Brasil
                                )
                        
                        # Exibir gráfico histórico
                        with st.expander("Dados Históricos do WINFUT", expanded=False):
                            try:
                                timeframe = st.selectbox(
                                    "Período",
                                    options=["1m", "5m", "15m", "30m", "1h", "1d"],
                                    index=2  # Default to 15m
                                )
                                
                                days = st.slider(
                                    "Dias de Histórico",
                                    min_value=1,
                                    max_value=30,
                                    value=5
                                )
                                
                                if st.button("Buscar Dados Históricos"):
                                    with st.spinner(f"Buscando {days} dias de dados {timeframe} do WINFUT..."):
                                        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                                        
                                        # Usa o contrato atual de Mini Índice
                                        current_contract = st.session_state.current_contract
                                        historical_data = st.session_state.investing_collector.get_historical_data(
                                            symbol=current_contract,
                                            interval=timeframe,
                                            start_date=start_date,
                                            use_cache=False  # Don't use cache for this view
                                        )
                                        st.markdown(f"**Contrato:** {current_contract}")
                                        
                                        if historical_data.empty:
                                            # Gerar dados sintéticos quando não conseguir obter do Investing.com
                                            st.warning("Usando dados sintéticos. Dados reais não disponíveis.")
                                            logger.warning(f"Gerando dados sintéticos para {current_contract} com timeframe {timeframe}")
                                            
                                            # Gerar dados sintéticos usando o método do InvestingCollector
                                            end_date = datetime.datetime.now()
                                            historical_data = st.session_state.investing_collector._generate_synthetic_data(
                                                symbol=current_contract,
                                                interval=timeframe,
                                                start_date=start_date,
                                                end_date=end_date
                                            )
                                            
                                            if historical_data.empty:
                                                st.error("Falha ao gerar dados sintéticos")
                                                return  # Retorna do with st.spinner, interrompendo a exibição
                                        
                                        if not historical_data.empty:
                                            # Process with technical indicators
                                            processed_data = st.session_state.data_processor.add_technical_indicators(historical_data)
                                            
                                            # Show chart
                                            fig = plot_price_chart(processed_data)
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Show data statistics
                                            st.markdown(f"**Resumo:** {len(processed_data)} períodos")
                                            st.markdown(f"**Intervalo de Datas:** {processed_data.index[0]} a {processed_data.index[-1]}")
                                            
                                            # Show recent data
                                            st.markdown("**Dados Mais Recentes:**")
                                            st.dataframe(processed_data.tail(5))
                                        else:
                                            st.error("Falha ao buscar dados históricos para os parâmetros selecionados.")
                            except Exception as e:
                                st.error(f"Erro ao buscar dados históricos: {str(e)}")
                    else:
                        st.info("Nenhum dado de mercado disponível do Investing.com ainda. Clique em 'Atualizar Dados de Mercado' para buscar os dados mais recentes.")
            
            with col2:
                st.subheader("Status do Trading")
                
                # Trading control buttons (removidos porque já estão no topo da página)
                
                # Account information
                if st.session_state.initialized:
                    account_info = st.session_state.profit_api.get_account_info()
                    
                    if account_info:
                        daily_pnl = account_info.get("daily_pnl", 0)
                        pnl_delta_color = "normal" if daily_pnl >= 0 else "inverse"
                        
                        st.metric(
                            "Saldo da Conta", 
                            format_number(account_info.get("balance", 0)), 
                            format_number(daily_pnl),
                            delta_color=pnl_delta_color
                        )
                    
                    # Trading mode indicator
                    modo = "Ativo" if st.session_state.trader_running else "Parado"
                    mode_color = "green" if st.session_state.trader_running else "red"
                    
                    # Indicador de status com ícone
                    status_icon = "🟢" if st.session_state.trader_running else "🔴"
                    
                    # Adicionar indicador de modo simulação
                    if st.session_state.profit_api.simulation_mode:
                        st.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                        <strong>Modo de Trading:</strong> {status_icon} <span style='color:{mode_color}'>{modo}</span> 
                        <span style='background-color:yellow; color:black; padding:2px 5px; border-radius:3px;'>SIMULAÇÃO</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                        <strong>Modo de Trading:</strong> {status_icon} <span style='color:{mode_color}'>{modo}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Current positions
                    st.subheader("Posições Abertas")
                    if st.session_state.open_positions:
                        for i, pos in enumerate(st.session_state.open_positions):
                            with st.container():
                                st.markdown(create_position_summary(pos))
                    else:
                        st.info("Sem posições abertas")
                        
                # Seção de Análise de Notícias - Mostra apenas se habilitado e news_analyzer existe
                if st.session_state.news_analysis_enabled and st.session_state.news_analyzer:
                    st.subheader("Análise de Notícias do Mercado")
                    
                    try:
                        # Busca notícias recentes e pontuações de impacto
                        latest_news = st.session_state.news_analyzer.get_latest_news(limit=3)
                        impact_scores = st.session_state.news_analyzer.get_impact_scores()
                        
                        if latest_news:
                            for news in latest_news[:1]:  # Mostra apenas a notícia mais recente
                                sentiment = news.get('sentiment', {}).get('compound', 0)
                                sentiment_color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
                                sentiment_label = "Positivo" if sentiment > 0.2 else "Negativo" if sentiment < -0.2 else "Neutro"
                                
                                st.markdown(f"""
                                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                                <strong>Notícia Recente:</strong> {news.get('title', 'Sem título')}  
                                <em>Fonte: {news.get('source', 'Desconhecida')} | {news.get('date', 'Data desconhecida')}</em>  
                                <strong>Sentimento:</strong> <span style='color:{sentiment_color}'>{sentiment_label} ({sentiment:.2f})</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if impact_scores:
                            # Pega as 3 palavras-chave mais impactantes
                            top_keywords = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                            
                            st.markdown("**Palavras-chave com Maior Impacto no Mercado:**")
                            for kw, score in top_keywords:
                                impact_color = "green" if score > 0.5 else "red" if score < -0.5 else "orange"
                                st.markdown(f"- {kw}: <span style='color:{impact_color}'>{score:.2f}</span>", 
                                          unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.info(f"Análise de notícias indisponível: {e}")
                        
                    with st.expander("Ver Mais Notícias", expanded=False):
                        try:
                            more_news = st.session_state.news_analyzer.get_latest_news(limit=10)
                            
                            if more_news:
                                for news in more_news[1:4]:  # Pula a primeira já exibida acima
                                    sentiment = news.get('sentiment', {}).get('compound', 0)
                                    sentiment_color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
                                    sentiment_label = "Positivo" if sentiment > 0.2 else "Negativo" if sentiment < -0.2 else "Neutro"
                                    
                                    st.markdown(f"""
                                    **{news.get('title', 'Sem título')}**  
                                    *Fonte: {news.get('source', 'Desconhecida')} | {news.get('date', 'Data desconhecida')}*  
                                    **Sentimento:** <span style='color:{sentiment_color}'>{sentiment_label} ({sentiment:.2f})</span>
                                    """, unsafe_allow_html=True)
                                    st.markdown("---")
                        except Exception as e:
                            st.info(f"Não foi possível carregar notícias adicionais: {str(e)}")
                    
                    # Botão para fechar todas as posições
                    if st.session_state.open_positions:
                        if st.button("Fechar Todas as Posições"):
                            close_all_positions()
                
                # Sinais recentes de trading
                st.subheader("Sinais Recentes")
                if st.session_state.trading_signals:
                    for signal in reversed(st.session_state.trading_signals[-5:]):
                        signal_type = "COMPRA" if signal["signal"] == 1 else "VENDA" if signal["signal"] == -1 else "NEUTRO"
                        signal_color = "green" if signal["signal"] == 1 else "red" if signal["signal"] == -1 else "gray"
                        
                        st.markdown(
                            f"<div style='background-color: #f0f2f6; padding: 8px; border-radius: 5px; margin-bottom: 5px;'>" +
                            f"<span style='color:{signal_color}; font-weight:bold;'>{signal_type}</span> - " + 
                            f"Confiança: {signal['confidence']:.2f} - " +
                            f"Hora: {signal['timestamp'].strftime('%H:%M:%S')}" +
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Nenhum sinal de trading recente")
                
                # Notificações do sistema
                st.subheader("Notificações")
                if st.session_state.notifications:
                    for notification in reversed(st.session_state.notifications[-5:]):
                        notification_type = notification["type"]
                        if notification_type == "error":
                            st.error(f"{notification['time']} - {notification['message']}")
                        elif notification_type == "warning":
                            st.warning(f"{notification['time']} - {notification['message']}")
                        elif notification_type == "success":
                            st.success(f"{notification['time']} - {notification['message']}")
                        else:
                            st.info(f"{notification['time']} - {notification['message']}")
                else:
                    st.info("Nenhuma notificação")
                
                # Botão de atualização
                if st.button("Atualizar Dados"):
                    with st.spinner("Buscando dados atualizados do mercado..."):
                        fetch_market_data()
                    st.session_state.add_notification("Dados de mercado atualizados", "info")
        
        # Trading Tab
        with tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Manual Order Entry")
                
                # Create order form
                with st.form("order_form"):
                    # Order type
                    st.session_state.manual_order_type = st.selectbox(
                        "Order Type",
                        options=["MARKET", "LIMIT"],
                        key="order_type"
                    )
                    
                    # Order side
                    st.session_state.manual_order_side = st.selectbox(
                        "Side",
                        options=["BUY", "SELL"],
                        key="order_side"
                    )
                    
                    # Order quantity
                    st.session_state.manual_order_quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=10,
                        value=1,
                        step=1,
                        key="order_quantity"
                    )
                    
                    # Limit price (only for LIMIT orders)
                    if st.session_state.manual_order_type == "LIMIT":
                        current_price = 0
                        if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                            current_price = st.session_state.price_data['close'].iloc[-1]
                        
                        st.session_state.manual_limit_price = st.number_input(
                            "Limit Price",
                            min_value=0.0,
                            value=float(current_price),
                            step=0.1,
                            format="%.1f",
                            key="limit_price"
                        )
                    
                    # Stop Loss
                    st.session_state.manual_sl_enabled = st.checkbox("Enable Stop Loss", value=True)
                    if st.session_state.manual_sl_enabled:
                        st.session_state.manual_sl_pct = st.slider(
                            "Stop Loss (%)", 
                            min_value=0.1, 
                            max_value=5.0, 
                            value=1.0, 
                            step=0.1,
                            key="sl_pct"
                        )
                    
                    # Take Profit
                    st.session_state.manual_tp_enabled = st.checkbox("Enable Take Profit", value=True)
                    if st.session_state.manual_tp_enabled:
                        st.session_state.manual_tp_pct = st.slider(
                            "Take Profit (%)", 
                            min_value=0.1, 
                            max_value=10.0, 
                            value=2.0, 
                            step=0.1,
                            key="tp_pct"
                        )
                    
                    # Submit button
                    submit_order = st.form_submit_button("Place Order")
                
                if submit_order:
                    place_manual_order()
            
            with col2:
                st.subheader("Order History")
                
                if st.session_state.order_history:
                    display_trade_list(st.session_state.order_history)
                else:
                    st.info("No order history available")
                
                # Open positions
                st.subheader("Open Positions")
                if st.session_state.open_positions:
                    for i, pos in enumerate(st.session_state.open_positions):
                        with st.expander(f"Position {i+1} - {'Long' if pos['position_type'] == 1 else 'Short'}", expanded=True):
                            st.markdown(create_position_summary(pos))
                            
                            # Position management buttons
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                if st.button(f"Close Position #{i+1}", key=f"close_pos_{i}"):
                                    # Get current price
                                    current_price = 0
                                    if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                                        current_price = st.session_state.price_data['close'].iloc[-1]
                                    
                                    # Close position
                                    result = st.session_state.order_manager.close_position(i, current_price, "manual_closure")
                                    if result.get("success", False):
                                        st.session_state.add_notification(f"Position {i+1} closed successfully", "success")
                                    else:
                                        st.session_state.add_notification(f"Failed to close position: {result.get('error')}", "error")
                            
                            with col_b:
                                if st.button(f"Update SL #{i+1}", key=f"update_sl_{i}"):
                                    # Get current price
                                    current_price = 0
                                    if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                                        current_price = st.session_state.price_data['close'].iloc[-1]
                                    
                                    # Calculate new stop loss
                                    if pos["position_type"] == 1:  # Long
                                        new_stop = current_price * 0.99
                                    else:  # Short
                                        new_stop = current_price * 1.01
                                    
                                    # Update stop loss
                                    result = st.session_state.order_manager.modify_stop_loss(i, new_stop)
                                    if result.get("success", False):
                                        st.session_state.add_notification(f"Stop loss updated for position {i+1}", "success")
                                    else:
                                        st.session_state.add_notification(f"Failed to update stop loss: {result.get('error')}", "error")
                            
                            with col_c:
                                if st.button(f"Update TP #{i+1}", key=f"update_tp_{i}"):
                                    # Get current price
                                    current_price = 0
                                    if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                                        current_price = st.session_state.price_data['close'].iloc[-1]
                                    
                                    # Calculate new take profit
                                    if pos["position_type"] == 1:  # Long
                                        new_tp = current_price * 1.02
                                    else:  # Short
                                        new_tp = current_price * 0.98
                                    
                                    # Update take profit
                                    result = st.session_state.order_manager.modify_take_profit(i, new_tp)
                                    if result.get("success", False):
                                        st.session_state.add_notification(f"Take profit updated for position {i+1}", "success")
                                    else:
                                        st.session_state.add_notification(f"Failed to update take profit: {result.get('error')}", "error")
                else:
                    st.info("No open positions")
                
                # Risk management
                st.subheader("Risk Management")
                risk_report = st.session_state.risk_manager.get_risk_report()
                
                col_risk1, col_risk2 = st.columns(2)
                with col_risk1:
                    st.metric("Daily P&L", format_number(risk_report.get("daily_pnl", 0)))
                    st.metric("Success Rate", f"{risk_report.get('success_rate', 0):.1%}")
                
                with col_risk2:
                    st.metric("Daily Trades", f"{risk_report.get('daily_trades', 0)}/{risk_report.get('max_daily_trades', 0)}")
                    st.metric("Max Drawdown", f"{risk_report.get('max_drawdown', 0):.2f}")
        
        # Backtesting Tab
        with tab3:
            st.subheader("Backtest Strategy")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Backtest parameters
                st.write("Backtest Parameters")
                
                # Data source selection
                st.session_state["backtest_data_source"] = st.radio(
                    "Data Source",
                    options=["profit_pro", "investing"],
                    format_func=lambda x: "Profit Pro" if x == "profit_pro" else "Investing.com",
                    horizontal=True,
                    key="backtest_source"
                )
                
                # Timeframe selection for Investing.com
                if st.session_state["backtest_data_source"] == "investing":
                    st.session_state["backtest_timeframe"] = st.selectbox(
                        "Timeframe",
                        options=["1m", "5m", "15m", "30m", "1h", "1d"],
                        index=2,  # Default to 15m
                        key="bt_timeframe"
                    )
                    
                    st.session_state["backtest_days"] = st.slider(
                        "Days of History",
                        min_value=5,
                        max_value=365,
                        value=30,
                        key="bt_days"
                    )
                
                # Date range
                backtest_start = st.date_input(
                    "Start Date",
                    value=datetime.datetime.strptime(BACKTEST_PARAMS["start_date"], "%Y-%m-%d").date()
                )
                backtest_end = st.date_input(
                    "End Date",
                    value=datetime.datetime.strptime(BACKTEST_PARAMS["end_date"], "%Y-%m-%d").date()
                )
                
                # Initial capital
                backtest_capital = st.number_input(
                    "Initial Capital",
                    min_value=1000,
                    max_value=1000000,
                    value=BACKTEST_PARAMS["initial_capital"],
                    step=1000
                )
                
                # Update settings in backtester
                st.session_state.backtester.start_date = backtest_start.strftime("%Y-%m-%d")
                st.session_state.backtester.end_date = backtest_end.strftime("%Y-%m-%d")
                st.session_state.backtester.initial_capital = backtest_capital
                
                # Run backtest button
                if st.button("Run Backtest", type="primary"):
                    run_backtest()
            
            with col2:
                # Backtest results
                if st.session_state.backtest_results is not None:
                    st.write("Resultado do Backtest")
                    st.markdown(create_backtest_summary(st.session_state.backtest_results))
                else:
                    st.info("Nenhum resultado de backtest disponível. Execute um backtest primeiro.")
            
            # Detailed backtest analysis
            if st.session_state.backtest_results is not None:
                st.subheader("Análise Detalhada")
                
                # Create tabs for different analysis views
                performance_tab, trades_tab, metrics_tab = st.tabs(["Performance", "Trades", "Métricas Detalhadas"])
                
                with performance_tab:
                    # Equity curve
                    equity_curve = st.session_state.backtest_results.get("equity_curve")
                    if equity_curve is not None:
                        fig = plot_equity_curve(equity_curve)
                        st.plotly_chart(fig, use_container_width=True)
                
                with trades_tab:
                    trades = st.session_state.backtester.trade_history
                    
                    # Add trade distribution chart
                    trade_dist_fig = plot_trade_distribution(trades)
                    if trade_dist_fig:
                        st.plotly_chart(trade_dist_fig, use_container_width=True)
                    
                    # Trade list with details
                    st.subheader("Lista de Trades")
                    show_detailed = st.checkbox("Mostrar detalhes adicionais", value=False)
                    display_trade_list(trades, detailed=show_detailed)
                
                with metrics_tab:
                    if hasattr(st.session_state, 'backtest_analysis'):
                        analysis = st.session_state.backtest_analysis
                        
                        # Performance Metrics
                        st.subheader("Métricas de Performance")
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Total de Trades", analysis.get("total_trades", 0))
                            st.metric("Trades Vencedores", analysis.get("winning_trades", 0))
                            st.metric("Trades Perdedores", analysis.get("losing_trades", 0))
                            st.metric("Taxa de Acerto", f"{analysis.get('win_rate', 0):.1%}")
                        
                        with col_b:
                            st.metric("Profit Factor", f"{analysis.get('profit_factor', 0):.2f}")
                            st.metric("Ganho Médio", format_number(analysis.get("avg_winner", 0)))
                            st.metric("Perda Média", format_number(analysis.get("avg_loser", 0)))
                            st.metric("Razão G/P", f"{abs(analysis.get('avg_winner', 0) / analysis.get('avg_loser', 1)):.2f}")
                        
                        with col_c:
                            st.metric("Duração Média", f"{analysis.get('avg_holding_period', 0):.1f} min")
                            st.metric("Sequência de Ganhos", analysis.get("max_consecutive_wins", 0))
                            st.metric("Sequência de Perdas", analysis.get("max_consecutive_losses", 0))
                            st.metric("Maior Ganho", format_number(analysis.get("largest_winner", 0)))
                        
                        # Results by trade type
                        st.subheader("Resultados por Tipo de Operação")
                        col_long, col_short = st.columns(2)
                        
                        with col_long:
                            st.write("**Compras (Long):**")
                            st.metric("Quantidade", analysis.get("long_trades", 0))
                            st.metric("Taxa de Acerto", f"{analysis.get('long_win_rate', 0):.1%}")
                        
                        with col_short:
                            st.write("**Vendas (Short):**")
                            st.metric("Quantidade", analysis.get("short_trades", 0))
                            st.metric("Taxa de Acerto", f"{analysis.get('short_win_rate', 0):.1%}")
                        
                        # Results by exit reason
                        if "exit_reason_stats" in analysis:
                            st.subheader("Resultados por Motivo de Saída")
                            
                            exit_reasons = analysis["exit_reason_stats"]
                            
                            # If the field exists but is complex, handle it
                            if isinstance(exit_reasons, dict) and ("pnl" in exit_reasons) and ("result" in exit_reasons):
                                exit_stats = []
                                for reason, stats in exit_reasons.items():
                                    if isinstance(stats, dict) and "pnl" in stats and "count" in stats.get("pnl", {}):
                                        count = stats["pnl"].get("count", 0)
                                        mean = stats["pnl"].get("mean", 0)
                                        total = stats["pnl"].get("sum", 0)
                                        win_rate = stats.get("result", 0)
                                        
                                        exit_stats.append({
                                            "Motivo": reason.replace("_", " ").title(),
                                            "Quantidade": count,
                                            "Média": format_number(mean),
                                            "Total": format_number(total),
                                            "Taxa de Acerto": f"{win_rate:.1%}"
                                        })
                                
                                if exit_stats:
                                    # Display as a dataframe
                                    st.dataframe(pd.DataFrame(exit_stats).set_index("Motivo"), use_container_width=True)
        
        # Machine Learning Tab
        with tab4:
            st.title("Machine Learning Components")
            
            # Create tabs for each ML component
            ml_tab1, ml_tab2, ml_tab3 = st.tabs([
                "Market Regime Detection", "Parameter Auto-Optimization", "Performance Monitoring"
            ])
            
        # Deep Learning Tab
        with tab5:
            st.title("Deep Learning Models")
            
            # Create tabs for Deep Learning sections
            dl_tab1, dl_tab2, dl_tab3 = st.tabs([
                "Modelos", "Treinamento", "Previsões"
            ])
            
            # Models Tab
            with dl_tab1:
                st.markdown("### Modelos de Deep Learning")
                
                # Check models availability
                if st.session_state.deep_learning:
                    models_status = st.session_state.deep_learning.check_models_available()
                    
                    # Display model status
                    st.markdown("#### Status dos Modelos")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Framework availability
                        st.markdown("**Frameworks:**")
                        st.markdown(f"- TensorFlow: {'✅ Disponível' if models_status.get('tensorflow', False) else '❌ Não disponível'}")
                        st.markdown(f"- PyTorch: {'✅ Disponível' if models_status.get('torch', False) else '❌ Não disponível'}")
                        
                        # Add note about installation if not available
                        if not models_status.get('tensorflow', False) or not models_status.get('torch', False):
                            st.info("Para usar modelos Deep Learning localmente, é necessário instalar TensorFlow ou PyTorch.")
                            
                    with col2:
                        # Models availability
                        st.markdown("**Modelos:**")
                        st.markdown(f"- LSTM: {'✅ Carregado' if models_status.get('lstm', False) else '❌ Não disponível'}")
                        st.markdown(f"- Transformer: {'✅ Carregado' if models_status.get('transformer', False) else '❌ Não disponível'}")
                    
                    # Model parameters
                    st.markdown("#### Parâmetros dos Modelos")
                    
                    model_params = st.session_state.deep_learning.get_model_parameters()
                    
                    # Display as a form to allow updates
                    with st.form("dl_parameters_form"):
                        lookback = st.slider(
                            "Janela de lookback (períodos)", 
                            min_value=10, 
                            max_value=120, 
                            value=model_params.get("lookback_window", 60),
                            help="Número de períodos históricos usados como entrada para o modelo"
                        )
                        
                        horizon = st.slider(
                            "Horizonte de previsão (períodos)", 
                            min_value=1, 
                            max_value=10, 
                            value=model_params.get("forecast_horizon", 5),
                            help="Número de períodos futuros para prever"
                        )
                        
                        confidence = st.slider(
                            "Limiar de confiança", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=model_params.get("confidence_threshold", 0.7),
                            step=0.05,
                            help="Nível mínimo de confiança para gerar sinais de trading"
                        )
                        
                        submitted = st.form_submit_button("Atualizar Parâmetros")
                        
                        if submitted:
                            # Update parameters
                            update_params = {
                                "lookback_window": lookback,
                                "forecast_horizon": horizon,
                                "confidence_threshold": confidence
                            }
                            
                            result = st.session_state.deep_learning.update_model_parameters(update_params)
                            
                            if result.get("status") == "success":
                                st.success(result.get("message", "Parâmetros atualizados com sucesso"))
                            else:
                                st.error(result.get("message", "Erro ao atualizar parâmetros"))
                else:
                    st.warning("Modelos Deep Learning não foram inicializados. Reinicie a aplicação.")
            
            # Training Tab
            with dl_tab2:
                st.markdown("### Treinamento de Modelos")
                
                if st.session_state.deep_learning:
                    # Training form
                    with st.form("dl_training_form"):
                        st.markdown("#### Dados de Treinamento")
                        
                        # Data selection
                        data_source = st.selectbox(
                            "Fonte de dados",
                            options=["Dados atuais", "Dados históricos"],
                            help="Selecione a fonte de dados para treinamento"
                        )
                        
                        # Training parameters
                        st.markdown("#### Parâmetros de Treinamento")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            models_status = st.session_state.deep_learning.check_models_available()
                            models_to_train = st.multiselect(
                                "Modelos para treinar",
                                options=["lstm", "transformer"],
                                default=["lstm", "transformer"] if models_status.get('tensorflow', False) and models_status.get('torch', False) else 
                                        ["lstm"] if models_status.get('tensorflow', False) else 
                                        ["transformer"] if models_status.get('torch', False) else []
                            )
                            
                            epochs = st.slider(
                                "Épocas de treinamento", 
                                min_value=10, 
                                max_value=100, 
                                value=50
                            )
                            
                            batch_size = st.slider(
                                "Tamanho do batch", 
                                min_value=16, 
                                max_value=128, 
                                value=64,
                                step=16
                            )
                        
                        with col2:
                            patience = st.slider(
                                "Paciência (early stopping)", 
                                min_value=5, 
                                max_value=20, 
                                value=10
                            )
                            
                            train_ratio = st.slider(
                                "Proporção de treinamento", 
                                min_value=0.6, 
                                max_value=0.9, 
                                value=0.8,
                                step=0.05,
                                help="Proporção dos dados para treinamento (restante para validação)"
                            )
                            
                            # Target column
                            target_column = st.selectbox(
                                "Coluna alvo",
                                options=["close", "high", "low", "open"],
                                index=0,
                                help="Coluna a ser prevista pelo modelo"
                            )
                        
                        # Submit button
                        submitted = st.form_submit_button("Iniciar Treinamento")
                        
                        if submitted:
                            if len(models_to_train) == 0:
                                st.error("Selecione pelo menos um modelo para treinar")
                            else:
                                with st.spinner("Treinando modelos de Deep Learning..."):
                                    # Get training data based on selection
                                    if data_source == "Dados atuais":
                                        if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                                            training_data = st.session_state.price_data
                                        else:
                                            st.error("Dados atuais não disponíveis. Inicialize o sistema e busque dados primeiro.")
                                            st.stop()
                                    else:  # Historical data
                                        # Potentially use price data from different source or time range
                                        # For now, use price_data as a fallback
                                        if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                                            training_data = st.session_state.price_data
                                        else:
                                            st.error("Dados históricos não disponíveis.")
                                            st.stop()
                                    
                                    # Train models
                                    result = st.session_state.deep_learning.train_models(
                                        data=training_data,
                                        models=models_to_train,
                                        target_column=target_column,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        patience=patience,
                                        train_ratio=train_ratio
                                    )
                                    
                                    if result.get("status") == "success":
                                        st.success(result.get("message", "Treinamento concluído com sucesso"))
                                        
                                        # Display evaluation metrics if available
                                        if "results" in result and "evaluation" in result["results"]:
                                            st.markdown("#### Métricas de Avaliação")
                                            
                                            eval_metrics = result["results"]["evaluation"]
                                            
                                            for model_name, metrics in eval_metrics.items():
                                                if metrics:
                                                    st.markdown(f"**Modelo {model_name.upper()}:**")
                                                    st.markdown(f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                                                    st.markdown(f"- MAE: {metrics.get('mae', 'N/A'):.4f}")
                                                    st.markdown(f"- R²: {metrics.get('r2', 'N/A'):.4f}")
                                    else:
                                        st.error(result.get("message", "Erro durante o treinamento"))
                else:
                    st.warning("Modelos Deep Learning não foram inicializados. Reinicie a aplicação.")
            
            # Predictions Tab
            with dl_tab3:
                st.markdown("### Previsões de Deep Learning")
                
                if st.session_state.deep_learning:
                    # Check if models are available
                    models_status = st.session_state.deep_learning.check_models_available()
                    
                    if models_status.get('lstm', False) or models_status.get('transformer', False):
                        # Form for generating predictions
                        with st.form("dl_prediction_form"):
                            st.markdown("#### Configurações de Previsão")
                            
                            # Model selection
                            model_type = st.selectbox(
                                "Modelo para previsão",
                                options=[
                                    "ensemble", 
                                    "lstm", 
                                    "transformer"
                                ],
                                index=0,
                                help="Escolha qual modelo usar para previsão. Ensemble combina ambos os modelos."
                            )
                            
                            # Submit button
                            submitted = st.form_submit_button("Gerar Previsão")
                            
                            if submitted:
                                with st.spinner("Gerando previsão..."):
                                    # Check if price data is available
                                    if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                                        # Generate forecast
                                        forecast = st.session_state.deep_learning.get_forecast(
                                            data=st.session_state.price_data,
                                            model_type=model_type
                                        )
                                        
                                        if forecast.get("status") == "success":
                                            # Display forecast results
                                            st.markdown("#### Resultados da Previsão")
                                            
                                            # Create tabs for different visualizations
                                            pred_tab1, pred_tab2 = st.tabs(["Gráfico", "Detalhes"])
                                            
                                            with pred_tab1:
                                                # Plot forecast
                                                fig = st.session_state.deep_learning.plot_forecast(
                                                    data=st.session_state.price_data,
                                                    forecast_result=forecast,
                                                    last_n_periods=30,
                                                    use_plotly=True
                                                )
                                                
                                                if fig:
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.error("Erro ao gerar gráfico de previsão")
                                            
                                            with pred_tab2:
                                                # Display forecast details
                                                st.markdown("**Valores Previstos:**")
                                                
                                                forecast_values = forecast.get("forecast", [])
                                                trend = forecast.get("trend", "neutral")
                                                signal = forecast.get("signal", "neutral")
                                                
                                                # Create forecast table
                                                forecast_df = pd.DataFrame({
                                                    "Período": [f"P+{i+1}" for i in range(len(forecast_values))],
                                                    "Previsão": forecast_values
                                                })
                                                
                                                st.dataframe(forecast_df)
                                                
                                                # Display trend and signal
                                                trend_color = "green" if trend == "up" else "red" if trend == "down" else "gray"
                                                signal_color = "green" if signal == "buy" else "red" if signal == "sell" else "gray"
                                                
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    st.markdown(f"**Tendência:** <span style='color:{trend_color};'>{trend.upper()}</span>", unsafe_allow_html=True)
                                                
                                                with col2:
                                                    st.markdown(f"**Sinal de Trading:** <span style='color:{signal_color};'>{signal.upper()}</span>", unsafe_allow_html=True)
                                                
                                                # Generate trading signal
                                                trading_signal = st.session_state.deep_learning.generate_trading_signal(
                                                    data=st.session_state.price_data,
                                                    model_type=model_type
                                                )
                                                
                                                st.markdown("#### Análise de Trading")
                                                st.markdown(f"**Confiança:** {trading_signal.get('confidence', 0)*100:.1f}%")
                                                
                                                # Trading recommendation
                                                if trading_signal.get('signal') == "buy":
                                                    st.success("✅ RECOMENDAÇÃO: COMPRAR")
                                                elif trading_signal.get('signal') == "sell":
                                                    st.error("❌ RECOMENDAÇÃO: VENDER")
                                                else:
                                                    st.info("⚠️ RECOMENDAÇÃO: AGUARDAR")
                                        else:
                                            st.error(forecast.get("message", "Erro ao gerar previsão"))
                                    else:
                                        st.error("Dados não disponíveis. Inicialize o sistema e busque dados primeiro.")
                        
                        # Historical predictions
                        with st.expander("Histórico de Previsões", expanded=False):
                            prediction_history = st.session_state.deep_learning.get_prediction_history()
                            
                            if prediction_history:
                                st.markdown(f"**Últimas {len(prediction_history)} previsões geradas:**")
                                
                                for i, pred in enumerate(reversed(prediction_history[:5])):  # Show last 5
                                    timestamp = pred.get("timestamp", "Unknown time")
                                    model = pred.get("model_type", "unknown")
                                    
                                    st.markdown(f"**{i+1}. Previsão de {timestamp}** (Modelo: {model})")
                            else:
                                st.info("Nenhuma previsão no histórico")
                    else:
                        st.warning("Nenhum modelo de Deep Learning disponível. Treine os modelos primeiro.")
                else:
                    st.warning("Modelos Deep Learning não foram inicializados. Reinicie a aplicação.")
            
            # Market Regime Detection Tab
            with ml_tab1:
                st.subheader("Market Regime Analysis")
                
                # Display current market regime if available
                if hasattr(st.session_state, "market_regime_detector") and st.session_state.market_regime_detector:
                    regime_summary = st.session_state.market_regime_detector.get_regime_summary()
                    
                    # Current regime with color coding
                    current_regime = regime_summary.get("current_regime", "undefined")
                    regime_color = {
                        "trending_up": "green",
                        "trending_down": "red",
                        "ranging": "blue",
                        "volatile": "orange",
                        "breakout_up": "green",
                        "breakout_down": "red",
                        "mean_reverting": "purple",
                        "momentum": "teal",
                        "undefined": "gray"
                    }.get(current_regime, "gray")
                    
                    # Display current regime with styled header
                    st.markdown(f"""
                    ### Regime Atual: <span style='color:{regime_color};'>{current_regime.upper()}</span>
                    """, unsafe_allow_html=True)
                    
                    # Display regime stability
                    stability = regime_summary.get("stability", 0) * 100
                    st.metric("Estabilidade do Regime", f"{stability:.1f}%", 
                             delta=None)
                    
                    # Display regime history
                    st.subheader("Histórico de Regimes")
                    history = regime_summary.get("regime_history", [])
                    history_str = " → ".join([h.upper() for h in history])
                    st.info(f"Evolução recente: {history_str}")
                    
                    # Display regime distribution as chart
                    st.subheader("Distribuição de Regimes")
                    distribution = regime_summary.get("regime_distribution", {})
                    if distribution:
                        # Convert to dataframe for chart
                        dist_data = {
                            "Regime": list(distribution.keys()),
                            "Ocorrências": list(distribution.values())
                        }
                        dist_df = pd.DataFrame(dist_data)
                        st.bar_chart(dist_df.set_index("Regime"))
                    
                    # Display recommendations
                    st.subheader("Recomendações para o Regime Atual")
                    recommendations = regime_summary.get("recommendations", [])
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                        
                    # Allow manual regime change for testing
                    with st.expander("Detecção Manual de Regime (Para Testes)", expanded=False):
                        st.write("Use essa opção para forçar a detecção de um regime específico para testes.")
                        if st.button("Detectar Regime Agora"):
                            try:
                                if hasattr(st.session_state, "market_data") and st.session_state.market_data is not None:
                                    with st.spinner("Detectando regime de mercado..."):
                                        new_regime = st.session_state.market_regime_detector.detect_regime(
                                            st.session_state.market_data
                                        )
                                        st.success(f"Regime detectado: {new_regime.value}")
                                        st.session_state.add_notification(
                                            f"Regime de mercado detectado: {new_regime.value}",
                                            "info"
                                        )
                                        st.rerun()
                                else:
                                    st.error("Dados de mercado não disponíveis. Verifique a conexão com a plataforma.")
                            except Exception as e:
                                st.error(f"Erro ao detectar regime: {str(e)}")
                else:
                    st.warning("Detector de regime de mercado não inicializado.")
                    if st.button("Inicializar Detector de Regime"):
                        try:
                            from market_regime import MarketRegimeDetector
                            st.session_state.market_regime_detector = MarketRegimeDetector()
                            st.success("Detector de regime de mercado inicializado!")
                            st.session_state.add_notification(
                                "Detector de regime de mercado inicializado",
                                "success"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao inicializar detector de regime: {str(e)}")
            
            # Parameter Auto-Optimization Tab
            with ml_tab2:
                st.subheader("Otimização Automática de Parâmetros")
                
                if hasattr(st.session_state, "auto_optimizer") and st.session_state.auto_optimizer:
                    optimization_summary = st.session_state.auto_optimizer.get_optimization_summary()
                    
                    # Display optimization stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total de Otimizações", 
                                 optimization_summary.get("total_optimizations", 0))
                        st.metric("Trades Desde Última Otimização", 
                                 optimization_summary.get("trade_count_since_last", 0))
                    with col2:
                        best_score = optimization_summary.get("best_score", 0)
                        st.metric("Melhor Score", f"{best_score:.4f}" if best_score else "N/A")
                        
                        # Check if optimization is due soon
                        trade_count = optimization_summary.get("trade_count_since_last", 0)
                        if hasattr(st.session_state.auto_optimizer, "optimization_interval"):
                            opt_interval = st.session_state.auto_optimizer.optimization_interval
                            progress = min(100, int(trade_count / opt_interval * 100))
                            st.progress(progress / 100, text=f"Próxima otimização: {progress}%")
                    
                    # Display current parameters
                    st.subheader("Parâmetros Atuais")
                    current_params = optimization_summary.get("current_parameters", {})
                    if current_params:
                        params_df = pd.DataFrame({
                            "Parâmetro": list(current_params.keys()),
                            "Valor": list(current_params.values())
                        })
                        st.dataframe(params_df)
                    else:
                        st.info("Nenhum parâmetro atual disponível.")
                    
                    # Display best parameters
                    with st.expander("Melhores Parâmetros Encontrados", expanded=False):
                        best_params = optimization_summary.get("best_parameters", {})
                        if best_params:
                            params_df = pd.DataFrame({
                                "Parâmetro": list(best_params.keys()),
                                "Valor": list(best_params.values())
                            })
                            st.dataframe(params_df)
                        else:
                            st.info("Nenhum parâmetro otimizado disponível.")
                    
                    # Last optimization details
                    with st.expander("Detalhes da Última Otimização", expanded=False):
                        last_opt = optimization_summary.get("last_optimization", {})
                        if last_opt:
                            st.write(f"Data: {last_opt.get('timestamp', 'N/A')}")
                            st.write(f"Método: {last_opt.get('method', 'N/A')}")
                            st.write(f"Score: {last_opt.get('score', 'N/A')}")
                            st.write(f"Duração: {last_opt.get('duration', 'N/A')} segundos")
                            
                            # Show parameters found
                            if "parameters" in last_opt and last_opt["parameters"]:
                                st.subheader("Parâmetros Encontrados")
                                opt_params = last_opt["parameters"]
                                opt_params_df = pd.DataFrame({
                                    "Parâmetro": list(opt_params.keys()),
                                    "Valor": list(opt_params.values())
                                })
                                st.dataframe(opt_params_df)
                        else:
                            st.info("Nenhuma otimização realizada ainda.")
                    
                    # Allow manual optimization
                    st.subheader("Otimização Manual")
                    if st.button("Executar Otimização Agora"):
                        try:
                            if hasattr(st.session_state, "market_data") and st.session_state.market_data is not None:
                                with st.spinner("Otimizando parâmetros... Isso pode levar alguns minutos."):
                                    result = st.session_state.auto_optimizer.optimize(st.session_state.market_data)
                                    st.success("Otimização concluída com sucesso!")
                                    st.session_state.add_notification(
                                        f"Otimização de parâmetros concluída. Score: {result.get('score', 'N/A')}",
                                        "success"
                                    )
                                    st.rerun()
                            else:
                                st.error("Dados de mercado não disponíveis. Verifique a conexão com a plataforma.")
                        except Exception as e:
                            st.error(f"Erro durante a otimização: {str(e)}")
                else:
                    st.warning("Otimizador de parâmetros não inicializado.")
                    if st.button("Inicializar Otimizador"):
                        try:
                            from auto_optimizer import AutoOptimizer
                            # Inicialização básica - seria necessário configurar função de backtesting e parâmetros
                            st.info("Função de inicialização do otimizador em implementação...")
                            st.session_state.add_notification(
                                "A inicialização do otimizador requer configuração adicional",
                                "warning"
                            )
                        except Exception as e:
                            st.error(f"Erro ao inicializar otimizador: {str(e)}")
            
            # Performance Monitoring Tab
            with ml_tab3:
                st.subheader("Monitoramento de Performance")
                
                if hasattr(st.session_state, "performance_monitor") and st.session_state.performance_monitor:
                    # Get performance metrics
                    metrics = st.session_state.performance_monitor.get_performance_metrics()
                    status = st.session_state.performance_monitor.get_status_summary()
                    
                    # Display system status
                    system_status = status.get("status", "unknown")
                    status_color = {
                        "active": "green",
                        "paused": "orange",
                        "suspended": "red",
                        "warning": "yellow",
                        "unknown": "gray"
                    }.get(system_status.lower(), "gray")
                    
                    st.markdown(f"""
                    ### Status do Sistema: <span style='color:{status_color};'>{system_status.upper()}</span>
                    """, unsafe_allow_html=True)
                    
                    if "health_score" in metrics:
                        health_score = metrics["health_score"] * 100
                        st.metric("Saúde do Sistema", f"{health_score:.1f}%", 
                                 delta=None)
                    
                    # Display key metrics
                    st.subheader("Métricas Principais")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if "profit_factor" in metrics:
                            st.metric("Fator de Lucro", 
                                     f"{metrics['profit_factor']:.2f}",
                                     delta=None)
                        if "win_rate" in metrics:
                            st.metric("Taxa de Acerto", 
                                     f"{metrics['win_rate']*100:.1f}%",
                                     delta=None)
                    
                    with col2:
                        if "sharpe_ratio" in metrics:
                            st.metric("Índice Sharpe", 
                                     f"{metrics['sharpe_ratio']:.2f}",
                                     delta=None)
                        if "avg_profit_per_trade" in metrics:
                            st.metric("Lucro Médio/Trade", 
                                     f"R$ {metrics['avg_profit_per_trade']:.2f}",
                                     delta=None)
                    
                    with col3:
                        if "max_drawdown" in metrics:
                            st.metric("Drawdown Máximo", 
                                     f"{metrics['max_drawdown']*100:.1f}%",
                                     delta=None)
                        if "profit_loss" in metrics:
                            st.metric("Resultado Líquido", 
                                     f"R$ {metrics['profit_loss']:.2f}",
                                     delta=None)
                    
                    # Trade statistics
                    trade_stats = st.session_state.performance_monitor.get_trade_statistics()
                    if trade_stats:
                        st.subheader("Estatísticas de Trades")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total de Trades", trade_stats.get("total_trades", 0))
                            st.metric("Trades Vencedores", trade_stats.get("winning_trades", 0))
                            st.metric("Melhor Trade", f"R$ {trade_stats.get('best_trade', 0):.2f}")
                            
                        with col2:
                            st.metric("Trades Ativos", trade_stats.get("active_trades", 0))
                            st.metric("Trades Perdedores", trade_stats.get("losing_trades", 0))
                            st.metric("Pior Trade", f"R$ {trade_stats.get('worst_trade', 0):.2f}")
                    
                    # System notifications
                    st.subheader("Alertas do Sistema")
                    notifications = st.session_state.performance_monitor.get_notifications(10)
                    if notifications:
                        for notification in notifications:
                            notification_level = notification.get("level", "info").lower()
                            message = notification.get("message", "")
                            timestamp = notification.get("timestamp", "")
                            
                            if notification_level == "error":
                                st.error(f"{timestamp} - {message}")
                            elif notification_level == "warning":
                                st.warning(f"{timestamp} - {message}")
                            elif notification_level == "success":
                                st.success(f"{timestamp} - {message}")
                            else:
                                st.info(f"{timestamp} - {message}")
                    else:
                        st.info("Nenhuma notificação recente do sistema.")
                    
                    # Allow performance monitoring reset
                    with st.expander("Ações do Monitor de Performance", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Atualizar Métricas"):
                                try:
                                    st.session_state.performance_monitor.update_metrics()
                                    st.success("Métricas atualizadas com sucesso!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Erro ao atualizar métricas: {str(e)}")
                        
                        with col2:
                            if st.button("Salvar Relatório"):
                                try:
                                    filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                    if st.session_state.performance_monitor.save_performance_report(filename):
                                        st.success(f"Relatório salvo como {filename}")
                                    else:
                                        st.error("Erro ao salvar relatório")
                                except Exception as e:
                                    st.error(f"Erro ao salvar relatório: {str(e)}")
                else:
                    st.warning("Monitor de performance não inicializado.")
                    if st.button("Inicializar Monitor"):
                        try:
                            from performance_monitor import PerformanceMonitor
                            st.session_state.performance_monitor = PerformanceMonitor()
                            st.success("Monitor de performance inicializado!")
                            st.session_state.add_notification(
                                "Monitor de performance inicializado",
                                "success"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao inicializar monitor: {str(e)}")
                
                # Equity Curve Section
                st.subheader("Curva de Equity")
                equity_curve = None
                if hasattr(st.session_state, "performance_monitor") and st.session_state.performance_monitor:
                    try:
                        equity_curve = st.session_state.performance_monitor.get_equity_curve()
                    except:
                        pass
                        
                if equity_curve:
                    st.image(f"data:image/png;base64,{equity_curve}")
                else:
                    st.info("Curva de equity não disponível ainda.")
            
            # Action buttons
            st.markdown("### Actions")
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("Save Performance Report"):
                    if st.session_state.performance_monitor:
                        try:
                            saved = st.session_state.performance_monitor.save_performance_report()
                            if saved:
                                st.success("Performance report saved successfully")
                            else:
                                st.error("Failed to save performance report")
                        except Exception as e:
                            st.error(f"Error saving report: {str(e)}")
                    else:
                        st.error("Performance monitor not initialized")
            
            with action_col2:
                if st.button("Train Market Regime Model"):
                    if st.session_state.market_regime_detector and st.session_state.price_data is not None:
                        try:
                            with st.spinner("Training market regime classifier..."):
                                st.session_state.market_regime_detector.train_classifier(st.session_state.price_data)
                                st.success("Market regime classifier trained successfully")
                        except Exception as e:
                            st.error(f"Error training regime classifier: {str(e)}")
                    else:
                        st.error("Market regime detector not initialized or no data available")
            
            with action_col3:
                if st.button("Reset Performance Monitor"):
                    if st.session_state.performance_monitor:
                        try:
                            # Reset to initial equity value
                            initial_equity = 10000.0  # Default starting equity
                            st.session_state.performance_monitor.initialize(initial_equity)
                            st.success("Performance monitor reset successfully")
                        except Exception as e:
                            st.error(f"Error resetting performance monitor: {str(e)}")
                    else:
                        st.error("Performance monitor not initialized")
                        
        # Configuration Tab
        with tab5:
            st.subheader("Configurações do Sistema")
            
            config_tab1, config_tab2, config_tab3, config_tab4, config_tab5, config_tab6, config_tab7 = st.tabs([
                "Conexão API", "Parâmetros de Estratégia", "Treinamento de Modelos", "Gerenciamento de Risco", "Análise de Notícias", "Contratos WINFUT", "Licenciamento"
            ])
            
            # API Connection Settings
            with config_tab1:
                st.markdown("### Configuração da API do Profit Pro")
                
                st.info("""
                Para operar automaticamente, você precisa configurar a conexão com a API do Profit Pro da sua corretora.
                Informe abaixo os dados de acesso fornecidos pela sua corretora.
                """)
                
                with st.expander("Como obter suas credenciais da API", expanded=True):
                    st.markdown("""
                    1. **Acesse sua corretora**: Entre em contato com sua corretora para solicitar acesso à API
                    2. **Ative o acesso à API**: No site da corretora, vá até a seção de API ou Configurações
                    3. **Gere suas credenciais**: Crie uma chave de API e um segredo para autenticação
                    4. **Configure o URL**: Use o URL fornecido pela corretora para a API do Profit Pro
                    
                    ⚠️ **Importante**: Nunca compartilhe suas credenciais de API com terceiros. 
                    Estas credenciais dão acesso à sua conta e podem ser usadas para realizar operações.
                    """)
                
                # Adicionar opção de modo de simulação (usar valor inicial da API)
                api = st.session_state.profit_api if hasattr(st.session_state, 'profit_api') else ProfitProAPI()
                simulation_mode = st.checkbox(
                    "Usar modo de simulação",
                    value=api.simulation_mode,
                    help="Ativa o modo de simulação para testar o robô sem realizar operações reais"
                )
                
                if simulation_mode:
                    st.success("""
                    **Modo de simulação ativado!** 
                    
                    Neste modo, todas as operações serão simuladas e nenhuma ordem real será enviada.
                    Perfeito para testar estratégias e configurações sem risco financeiro.
                    """)
                
                # Connection type selector
                conn_type = st.radio(
                    "Tipo de Conexão",
                    options=["API REST", "DLL/Socket"],
                    horizontal=True,
                    index=0 if not PROFIT_PRO_USE_DLL else 1,
                    help="Escolha o tipo de conexão com o Profit Pro"
                )
                
                # Create columns for better visual organization
                col1, col2 = st.columns(2)
                
                if conn_type == "API REST":
                    with col1:
                        # API URL
                        api_url = st.text_input(
                            "URL da API",
                            value=PROFIT_API_URL,
                            help="URL base da API do Profit Pro (ex: https://api.corretora.com.br)"
                        )
                    
                    with col2:
                        # Trading symbol
                        symbol = st.text_input(
                            "Símbolo de Negociação",
                            value=SYMBOL,
                            help="Símbolo do contrato para negociação (ex: WINFUT, WINM25)"
                        )
                    
                    # API Key
                    api_key = st.text_input(
                        "Chave da API (API Key)",
                        value=PROFIT_API_KEY,
                        type="password",
                        help="Chave de API fornecida pela corretora"
                    )
                    
                    # API Secret
                    api_secret = st.text_input(
                        "Segredo da API (API Secret)",
                        value=PROFIT_API_SECRET,
                        type="password",
                        help="Segredo de API fornecido pela corretora"
                    )
                    
                    # Initialize variables used by DLL interface
                    host = PROFIT_PRO_HOST
                    port = PROFIT_PRO_PORT
                    dll_path = PROFIT_PRO_DLL_PATH
                    log_path = PROFIT_PRO_LOG_PATH
                    
                else:  # DLL/Socket
                    # Initialize variables used by REST API
                    api_url = PROFIT_API_URL
                    api_key = PROFIT_API_KEY
                    api_secret = PROFIT_API_SECRET
                    
                    col1a, col1b = st.columns(2)
                    with col1a:
                        host = st.text_input(
                            "Host",
                            value=PROFIT_PRO_HOST,
                            help="Endereço do servidor Profit Pro (geralmente localhost)"
                        )
                    
                    with col1b:
                        port = st.number_input(
                            "Porta",
                            min_value=1,
                            max_value=65535,
                            value=PROFIT_PRO_PORT,
                            help="Porta de comunicação do Profit Pro"
                        )
                        
                    with col2:
                        # Trading symbol
                        symbol = st.text_input(
                            "Símbolo de Negociação",
                            value=SYMBOL,
                            help="Símbolo do contrato para negociação (ex: WINFUT, WINM25)"
                        )
                        
                    # DLL path
                    dll_path = st.text_input(
                        "Caminho para DLL do Profit Pro (opcional)",
                        value=PROFIT_PRO_DLL_PATH,
                        help="Caminho completo para o arquivo DLL do Profit Pro (ex: C:/Program Files/ProfitPro/API/ProfitPro.dll)"
                    )
                    
                    # Log path
                    log_path = st.text_input(
                        "Diretório para Logs (opcional)",
                        value=PROFIT_PRO_LOG_PATH,
                        help="Diretório onde os logs da DLL serão salvos"
                    )
                
                # API connection status
                st.markdown("#### Status da Conexão")
                
                # Test connection button
                col1, col2 = st.columns([1, 3])
                with col1:
                    test_conn = st.button("Testar Conexão", use_container_width=True)
                
                if test_conn:
                    with st.spinner("Testando conexão com a plataforma..."):
                        try:
                            # Test Profit Pro connection
                            if conn_type == "API REST":
                                # REST API mode
                                temp_api = ProfitProAPI(
                                    use_dll=False,
                                    api_url=api_url,
                                    api_key=api_key,
                                    api_secret=api_secret,
                                    host=host if host else None,
                                    port=port if port > 0 else None,
                                    symbol=symbol,
                                    simulation_mode=simulation_mode
                                )
                                
                                # Configure authentication headers
                                if temp_api.session:
                                    temp_api.session.headers.update({
                                        'X-API-Key': api_key,
                                        'X-API-Secret': api_secret,
                                        'Content-Type': 'application/json'
                                    })
                            else:  # DLL/Socket
                                # DLL mode
                                temp_api = ProfitProAPI(
                                    use_dll=True,
                                    dll_path=dll_path,
                                    dll_version=PROFIT_PRO_DLL_VERSION,
                                    host=host if host else None,
                                    port=port if port > 0 else None,
                                    symbol=symbol,
                                    simulation_mode=simulation_mode
                                )
                            
                            # Test connection
                            connected = temp_api.connect()
                            
                            if connected:
                                st.success("✅ Conexão estabelecida com sucesso!")
                                
                                # Try to fetch account info
                                try:
                                    account_info = temp_api.get_account_info()
                                    if account_info and 'balance' in account_info:
                                        st.info(f"Saldo da conta: R$ {account_info['balance']:,.2f}")
                                except:
                                    pass
                            else:
                                st.error("""
                                ❌ Falha na conexão com o Profit Pro. Verifique suas credenciais.
                                
                                Possíveis causas:
                                - URL incorreto
                                - Chave de API ou Secret incorretos
                                - Servidor da API fora do ar
                                - Firewall bloqueando conexão
                                """)
                        except Exception as e:
                            st.error(f"""
                            ❌ Erro ao conectar: {str(e)}
                            
                            Observação: Se você está testando o aplicativo sem uma conexão real,
                            este erro é esperado. Para operações reais, você precisará configurar
                            suas credenciais de API válidas.
                            """)
                
                # Save API settings
                if st.button("Salvar Configurações da API"):
                    try:
                        # Set use_metatrader to false
                        os.environ["USE_METATRADER"] = "false"
                        
                        # Save simulation mode setting
                        os.environ["PROFIT_PRO_SIMULATION_MODE"] = str(simulation_mode).lower()
                        
                        # Atualizar o modo de simulação para a API atual, se estiver inicializada
                        if hasattr(st.session_state, 'profit_api'):
                            st.session_state.profit_api.simulation_mode = simulation_mode
                        
                        if conn_type == "API REST":
                            os.environ["PROFIT_API_URL"] = api_url
                            os.environ["PROFIT_API_KEY"] = api_key
                            os.environ["PROFIT_API_SECRET"] = api_secret
                            os.environ["PROFIT_PRO_USE_DLL"] = "false"
                        else:  # DLL/Socket
                            os.environ["PROFIT_PRO_HOST"] = host
                            os.environ["PROFIT_PRO_PORT"] = str(port)
                            os.environ["PROFIT_PRO_USE_DLL"] = "true"
                            if dll_path:
                                os.environ["PROFIT_PRO_DLL_PATH"] = dll_path
                            if log_path:
                                os.environ["PROFIT_PRO_LOG_PATH"] = log_path
                        
                        # Always save symbol
                        os.environ["SYMBOL"] = symbol
                        
                        st.success(f"✅ Configurações da API salvas com sucesso!")
                        st.info("ℹ️ As configurações serão aplicadas na próxima inicialização do sistema.")
                    except Exception as e:
                        st.error(f"❌ Erro ao salvar configurações: {str(e)}")
            
            # Strategy Parameters
            with config_tab2:
                st.write("Parâmetros da Estratégia de Trading")
                
                # Confidence threshold
                confidence = st.slider(
                    "Limiar de Confiança do Sinal",
                    min_value=0.5,
                    max_value=0.95,
                    value=STRATEGY_PARAMS["confidence_threshold"],
                    step=0.05
                )
                
                # Minimum volume
                min_volume = st.number_input(
                    "Volume Mínimo para Sinais Válidos",
                    min_value=100,
                    max_value=10000,
                    value=STRATEGY_PARAMS["min_volume"],
                    step=100
                )
                
                # Use market regime
                use_market_regime = st.checkbox(
                    "Considerar Regime de Mercado",
                    value=STRATEGY_PARAMS["use_market_regime"]
                )
                
                # Ensemble weights
                st.write("Pesos do Ensemble de Modelos")
                rf_weight = st.slider(
                    "Peso do Random Forest",
                    min_value=0.0,
                    max_value=1.0,
                    value=STRATEGY_PARAMS["ensemble_weights"]["random_forest"],
                    step=0.1
                )
                xgb_weight = st.slider(
                    "Peso do XGBoost",
                    min_value=0.0,
                    max_value=1.0,
                    value=STRATEGY_PARAMS["ensemble_weights"]["xgboost"],
                    step=0.1
                )
                
                # Technical indicators parameters
                with st.expander("Parâmetros de Indicadores Técnicos", expanded=False):
                    sma_fast = st.number_input(
                        "Período da Média Móvel Rápida",
                        min_value=3,
                        max_value=50,
                        value=TECHNICAL_INDICATORS["sma_fast"],
                        step=1
                    )
                    
                    sma_slow = st.number_input(
                        "Período da Média Móvel Lenta",
                        min_value=10,
                        max_value=200,
                        value=TECHNICAL_INDICATORS["sma_slow"],
                        step=1
                    )
                    
                    rsi_period = st.number_input(
                        "Período do RSI",
                        min_value=7,
                        max_value=30,
                        value=TECHNICAL_INDICATORS["rsi_period"],
                        step=1
                    )
                    
                    macd_fast = st.number_input(
                        "Período Rápido do MACD",
                        min_value=8,
                        max_value=20,
                        value=TECHNICAL_INDICATORS["macd_fast"],
                        step=1
                    )
                    
                    macd_slow = st.number_input(
                        "Período Lento do MACD",
                        min_value=20,
                        max_value=40,
                        value=TECHNICAL_INDICATORS["macd_slow"],
                        step=1
                    )
                
                # Save strategy parameters
                if st.button("Salvar Parâmetros da Estratégia"):
                    try:
                        # Update strategy parameters
                        st.session_state.strategy.confidence_threshold = confidence
                        st.session_state.strategy.min_volume = min_volume
                        st.session_state.strategy.use_market_regime = use_market_regime
                        
                        # Update ensemble weights
                        new_weights = {
                            "random_forest": rf_weight,
                            "xgboost": xgb_weight
                        }
                        STRATEGY_PARAMS["ensemble_weights"] = new_weights
                        
                        # Update technical indicators
                        TECHNICAL_INDICATORS["sma_fast"] = sma_fast
                        TECHNICAL_INDICATORS["sma_slow"] = sma_slow
                        TECHNICAL_INDICATORS["rsi_period"] = rsi_period
                        TECHNICAL_INDICATORS["macd_fast"] = macd_fast
                        TECHNICAL_INDICATORS["macd_slow"] = macd_slow
                        
                        st.session_state.add_notification("Parâmetros da estratégia salvos", "success")
                        st.success("Parâmetros da estratégia salvos com sucesso")
                    except Exception as e:
                        st.session_state.add_notification(f"Erro ao salvar parâmetros: {str(e)}", "error")
                        st.error(f"Erro ao salvar parâmetros: {str(e)}")
            
            # Model Training
            with config_tab3:
                st.subheader("Parâmetros de Treinamento do Modelo")
                
                # Data source selection
                st.session_state["model_data_source"] = st.radio(
                    "Fonte de Dados para Treinamento",
                    options=["profit_pro", "investing"],
                    format_func=lambda x: "Profit Pro" if x == "profit_pro" else "Investing.com",
                    horizontal=True,
                    key="model_source"
                )
                
                # Timeframe selection for Investing.com
                if st.session_state["model_data_source"] == "investing":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.session_state["model_timeframe"] = st.selectbox(
                            "Timeframe para Dados Históricos",
                            options=["1m", "5m", "15m", "30m", "1h", "1d"],
                            index=2,  # Default to 15m
                            key="model_tf"
                        )
                    
                    with col2:
                        st.session_state["model_days"] = st.slider(
                            "Dias de Histórico",
                            min_value=5,
                            max_value=365,
                            value=30,
                            key="model_hist_days"
                        )
                
                # Model parameters
                with st.expander("Parâmetros do Random Forest", expanded=False):
                    n_estimators_rf = st.number_input(
                        "Número de Estimadores",
                        min_value=50,
                        max_value=500,
                        value=ML_PARAMS["random_forest"]["n_estimators"],
                        step=50
                    )
                    
                    max_depth_rf = st.number_input(
                        "Profundidade Máxima",
                        min_value=3,
                        max_value=20,
                        value=ML_PARAMS["random_forest"]["max_depth"],
                        step=1
                    )
                
                with st.expander("Parâmetros do XGBoost", expanded=False):
                    n_estimators_xgb = st.number_input(
                        "Número de Estimadores",
                        min_value=50,
                        max_value=500,
                        value=ML_PARAMS["xgboost"]["n_estimators"],
                        step=50,
                        key="xgb_n_estimators"
                    )
                    
                    max_depth_xgb = st.number_input(
                        "Profundidade Máxima",
                        min_value=3,
                        max_value=20,
                        value=ML_PARAMS["xgboost"]["max_depth"],
                        step=1,
                        key="xgb_max_depth"
                    )
                    
                    learning_rate = st.number_input(
                        "Taxa de Aprendizado",
                        min_value=0.01,
                        max_value=0.3,
                        value=ML_PARAMS["xgboost"]["learning_rate"],
                        step=0.01
                    )
                
                # General ML parameters
                prediction_horizon = st.number_input(
                    "Horizonte de Previsão (barras)",
                    min_value=1,
                    max_value=20,
                    value=ML_PARAMS["prediction_horizon"],
                    step=1
                )
                
                lookback_period = st.number_input(
                    "Período de Retrospectiva (barras)",
                    min_value=10,
                    max_value=100,
                    value=ML_PARAMS["lookback_period"],
                    step=5
                )
                
                # Save ML parameters
                if st.button("Atualizar Parâmetros ML"):
                    try:
                        # Update Random Forest parameters
                        ML_PARAMS["random_forest"]["n_estimators"] = n_estimators_rf
                        ML_PARAMS["random_forest"]["max_depth"] = max_depth_rf
                        
                        # Update XGBoost parameters
                        ML_PARAMS["xgboost"]["n_estimators"] = n_estimators_xgb
                        ML_PARAMS["xgboost"]["max_depth"] = max_depth_xgb
                        ML_PARAMS["xgboost"]["learning_rate"] = learning_rate
                        
                        # Update general ML parameters
                        ML_PARAMS["prediction_horizon"] = prediction_horizon
                        ML_PARAMS["lookback_period"] = lookback_period
                        
                        st.session_state.data_processor.lookback_period = lookback_period
                        
                        st.session_state.add_notification("Parâmetros ML atualizados", "success")
                        st.success("Parâmetros ML atualizados com sucesso")
                    except Exception as e:
                        st.session_state.add_notification(f"Erro ao atualizar parâmetros ML: {str(e)}", "error")
                        st.error(f"Erro ao atualizar parâmetros ML: {str(e)}")
                
                # Train models button
                if st.button("Treinar Modelos", type="primary"):
                    train_models()
                
                # Model evaluation metrics
                if hasattr(st.session_state, 'model_evaluation'):
                    st.subheader("Avaliação do Modelo")
                    
                    for model_name, metrics in st.session_state.model_evaluation.items():
                        st.write(f"Desempenho do {model_name.upper()}:")
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric("Acurácia", f"{metrics['accuracy']:.4f}")
                        
                        with col_m2:
                            st.metric("Precisão", f"{metrics['precision']:.4f}")
                        
                        with col_m3:
                            st.metric("Recall", f"{metrics['recall']:.4f}")
                        
                        with col_m4:
                            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            
            # Risk Management
            with config_tab4:
                st.write("Parâmetros de Gerenciamento de Risco")
                
                # Stop loss and take profit
                stop_loss_ticks = st.number_input(
                    "Stop Loss (ticks)",
                    min_value=10,
                    max_value=500,
                    value=RISK_MANAGEMENT["stop_loss_ticks"],
                    step=10
                )
                
                take_profit_ticks = st.number_input(
                    "Take Profit (ticks)",
                    min_value=10,
                    max_value=500,
                    value=RISK_MANAGEMENT["take_profit_ticks"],
                    step=10
                )
                
            # News Analysis
            with config_tab5:
                st.write("Configurações de Análise de Notícias")
                
                # Enable/disable news analysis
                news_enabled = st.checkbox(
                    "Ativar Análise de Notícias",
                    value=st.session_state.news_analysis_enabled,
                    help="Analisa notícias econômicas que possam impactar o desempenho do mercado"
                )
                
                # News sources configuration
                st.subheader("Fontes de Notícias")
                news_sources_text = st.text_area(
                    "Fontes de Notícias (uma URL por linha)",
                    value="\n".join(NEWS_SOURCES) if NEWS_SOURCES else "",
                    height=100,
                    help="Digite as URLs das fontes de notícias, uma por linha (ex: https://www.infomoney.com.br)"
                )
                
                # Keywords configuration
                st.subheader("Palavras-chave para Monitorar")
                keywords_text = st.text_area(
                    "Palavras-chave (uma por linha)",
                    value="\n".join(NEWS_KEYWORDS) if NEWS_KEYWORDS else "",
                    height=100,
                    help="Digite palavras-chave para monitorar nas notícias, uma por linha (ex: taxa de juros, Banco Central, inflação)"
                )
                
                # Update frequency
                update_interval = st.number_input(
                    "Intervalo de Atualização (segundos)",
                    min_value=60,
                    max_value=3600,
                    value=NEWS_UPDATE_INTERVAL,
                    step=60,
                    help="Com que frequência verificar novas notícias (em segundos)"
                )
                
                # Maximum articles to analyze
                max_articles = st.number_input(
                    "Número Máximo de Artigos para Analisar",
                    min_value=10,
                    max_value=200,
                    value=NEWS_MAX_ARTICLES,
                    step=10,
                    help="Número máximo de artigos a serem analisados de uma vez"
                )
                
                # Save news analysis settings
                if st.button("Salvar Configurações de Análise de Notícias"):
                    try:
                        # Parse news sources and keywords
                        news_sources_list = [url.strip() for url in news_sources_text.split("\n") if url.strip()]
                        keywords_list = [kw.strip() for kw in keywords_text.split("\n") if kw.strip()]
                        
                        # Update session state
                        st.session_state.news_analysis_enabled = news_enabled
                        
                        # Save to environment variables
                        os.environ["NEWS_ANALYSIS_ENABLED"] = str(news_enabled).lower()
                        os.environ["NEWS_UPDATE_INTERVAL"] = str(update_interval)
                        os.environ["NEWS_MAX_ARTICLES"] = str(max_articles)
                        
                        # Save lists as comma-separated strings
                        os.environ["NEWS_SOURCES"] = ",".join(news_sources_list)
                        os.environ["NEWS_KEYWORDS"] = ",".join(keywords_list)
                        
                        # If analyzer is already running, update it or restart it
                        if st.session_state.news_analyzer:
                            # Stop current analyzer
                            st.session_state.news_analyzer.stop()
                            
                            if news_enabled:
                                # Create new analyzer with updated settings
                                st.session_state.news_analyzer = NewsAnalyzer(
                                    news_sources=news_sources_list if news_sources_list else None,
                                    keywords=keywords_list if keywords_list else None,
                                    update_interval=update_interval,
                                    max_articles=max_articles
                                )
                                # Start the analyzer
                                st.session_state.news_analyzer.start()
                                st.session_state.add_notification("Analisador de notícias reiniciado com novas configurações", "success")
                            else:
                                st.session_state.news_analyzer = None
                                st.session_state.add_notification("Analisador de notícias desativado", "info")
                        elif news_enabled:
                            # Create and start new analyzer
                            st.session_state.news_analyzer = NewsAnalyzer(
                                news_sources=news_sources_list if news_sources_list else None,
                                keywords=keywords_list if keywords_list else None,
                                update_interval=update_interval,
                                max_articles=max_articles
                            )
                            st.session_state.news_analyzer.start()
                            st.session_state.add_notification("Analisador de notícias iniciado", "success")
                        
                        st.success("Configurações de análise de notícias salvas com sucesso")
                    except Exception as e:
                        st.session_state.add_notification(f"Erro ao salvar configurações de análise de notícias: {str(e)}", "error")
                        st.error(f"Erro ao salvar configurações de análise de notícias: {str(e)}")
                        
                # Display current news analysis status if enabled
                if st.session_state.news_analyzer:
                    st.subheader("Status Atual da Análise de Notícias")
                    
                    try:
                        # Get latest news and impact scores
                        latest_news = st.session_state.news_analyzer.get_latest_news(limit=5)
                        impact_scores = st.session_state.news_analyzer.get_impact_scores()
                        
                        if latest_news:
                            st.write("Notícias Analisadas Recentemente:")
                            for news in latest_news:
                                st.markdown(f"""
                                **{news.get('title', 'Sem título')}**  
                                Fonte: {news.get('source', 'Desconhecida')}  
                                Data: {news.get('date', 'Desconhecida')}  
                                Sentimento: {news.get('sentiment', {}).get('compound', 0):.2f}
                                """)
                                
                        if impact_scores:
                            st.write("Pontuação de Impacto das Palavras-chave:")
                            impact_df = pd.DataFrame(
                                [[k, v] for k, v in impact_scores.items()],
                                columns=['Palavra-chave', 'Pontuação de Impacto']
                            ).sort_values('Pontuação de Impacto', ascending=False)
                            
                            st.dataframe(impact_df, use_container_width=True)
                    except Exception as e:
                        st.write(f"Erro ao recuperar dados de análise de notícias: {str(e)}")
                
            # WINFUT Contracts
            with config_tab6:
                st.markdown("### Configuração de Contratos Mini Índice (WINFUT)")
                
                st.info("""
                Os contratos do Mini Índice (WINFUT) vencem a cada dois meses (ciclo par) 
                seguindo o código de letras:
                - **G**: Fevereiro
                - **J**: Abril
                - **M**: Junho
                - **Q**: Agosto
                - **V**: Outubro
                - **Z**: Dezembro
                
                Por exemplo, WINM25 é o contrato com vencimento em Junho de 2025.
                """)
                
                # Current contract display
                st.markdown(f"**Contrato Atual Detectado:** `{st.session_state.current_contract}`")
                
                # Refresh current contract
                if st.button("Atualizar Contrato Atual"):
                    with st.spinner("Detectando contrato atual..."):
                        st.session_state.current_contract = get_current_winfut_contract()
                        st.session_state.available_contracts = get_available_winfut_contracts()
                        st.success(f"Contrato atualizado para: {st.session_state.current_contract}")
                
                # Available contracts
                st.markdown("### Contratos Disponíveis")
                available_contracts = st.session_state.available_contracts
                
                # Display available contracts
                st.write("Escolha o contrato para negociação:")
                
                selected_contract = st.selectbox(
                    "Contrato para Negociação",
                    options=available_contracts,
                    index=available_contracts.index(st.session_state.current_contract) if st.session_state.current_contract in available_contracts else 0
                )
                
                # Set selected contract as current
                if st.button("Definir como Contrato Atual"):
                    st.session_state.current_contract = selected_contract
                    st.success(f"Contrato atual definido para: {selected_contract}")
                    
                # Custom contract input
                st.markdown("### Contrato Personalizado")
                custom_contract = st.text_input(
                    "Código de Contrato Personalizado", 
                    value="",
                    help="Digite o código de um contrato personalizado, por exemplo WINV25"
                )
                
                if st.button("Usar Contrato Personalizado") and custom_contract:
                    # Validate contract format (should start with WIN)
                    if custom_contract.upper().startswith("WIN") and len(custom_contract) >= 5:
                        st.session_state.current_contract = custom_contract.upper()
                        st.success(f"Usando contrato personalizado: {custom_contract.upper()}")
                    else:
                        st.error("Formato de contrato inválido. O código deve começar com 'WIN' seguido da letra do mês e do ano.")
                
                # Information about contract codes
                with st.expander("Sobre os Vencimentos de Contratos", expanded=False):
                    st.markdown("""
                    ### Código de Vencimentos Mini Índice
                    
                    Os contratos de Mini Índice da B3 seguem um padrão de codificação:
                    
                    | Mês | Código | Vencimento |
                    |-----|--------|------------|
                    | Fevereiro | G | Quarta-feira mais próxima ao dia 15 |
                    | Abril | J | Quarta-feira mais próxima ao dia 15 |
                    | Junho | M | Quarta-feira mais próxima ao dia 15 |
                    | Agosto | Q | Quarta-feira mais próxima ao dia 15 |
                    | Outubro | V | Quarta-feira mais próxima ao dia 15 |
                    | Dezembro | Z | Quarta-feira mais próxima ao dia 15 |
                    
                    **Exemplo:** WINM25 representa o contrato de Mini Índice com vencimento em Junho de 2025.
                    
                    ### Liquidez e Rolagem
                    
                    A maior liquidez costuma estar no contrato com vencimento mais próximo.
                    Recomenda-se fazer a rolagem de contratos (mudar do contrato atual para o próximo)
                    cerca de 5 a 7 dias antes do vencimento.
                    """)
                        
                trailing_stop_ticks = st.number_input(
                    "Trailing Stop (ticks)",
                    min_value=10,
                    max_value=500,
                    value=RISK_MANAGEMENT["trailing_stop_ticks"],
                    step=10
                )
                
                # Daily limits
                max_daily_loss = st.number_input(
                    "Perda Diária Máxima (pontos)",
                    min_value=100,
                    max_value=5000,
                    value=RISK_MANAGEMENT["max_daily_loss"],
                    step=100
                )
                
                max_daily_trades = st.number_input(
                    "Operações Diárias Máximas",
                    min_value=1,
                    max_value=50,
                    value=RISK_MANAGEMENT["max_daily_trades"],
                    step=1
                )
                
                # Risk per trade
                risk_per_trade = st.slider(
                    "Risco por Operação (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=RISK_MANAGEMENT["risk_per_trade"] * 100,
                    step=0.5
                ) / 100
                
                # Save risk parameters
                if st.button("Salvar Parâmetros de Risco"):
                    try:
                        # Update risk parameters
                        RISK_MANAGEMENT["stop_loss_ticks"] = stop_loss_ticks
                        RISK_MANAGEMENT["take_profit_ticks"] = take_profit_ticks
                        RISK_MANAGEMENT["trailing_stop_ticks"] = trailing_stop_ticks
                        RISK_MANAGEMENT["max_daily_loss"] = max_daily_loss
                        RISK_MANAGEMENT["max_daily_trades"] = max_daily_trades
                        RISK_MANAGEMENT["risk_per_trade"] = risk_per_trade
                        
                        # Update risk manager
                        st.session_state.risk_manager.stop_loss_ticks = stop_loss_ticks
                        st.session_state.risk_manager.take_profit_ticks = take_profit_ticks
                        st.session_state.risk_manager.trailing_stop_ticks = trailing_stop_ticks
                        st.session_state.risk_manager.max_daily_loss = max_daily_loss
                        st.session_state.risk_manager.max_daily_trades = max_daily_trades
                        st.session_state.risk_manager.risk_per_trade = risk_per_trade
                        
                        st.session_state.add_notification("Parâmetros de risco salvos", "success")
                        st.success("Parâmetros de risco salvos com sucesso")
                    except Exception as e:
                        st.session_state.add_notification(f"Erro ao salvar parâmetros de risco: {str(e)}", "error")
                        st.error(f"Erro ao salvar parâmetros de risco: {str(e)}")
        
        # Contratos Tab (B3 Contract Information)
        with tab6:
            st.title("Informações de Contratos WINFUT")
            st.write("Nesta seção você pode consultar informações detalhadas sobre os contratos futuros de mini índice (WINFUT) disponíveis na B3.")
            
            # Verificar se o web scraper está inicializado
            if st.session_state.web_scraper is not None:
                # Interface com duas colunas
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Contratos Ativos")
                    
                    # Botão para atualizar a lista de contratos
                    if st.button("Atualizar Lista de Contratos"):
                        try:
                            with st.spinner("Buscando contratos ativos na B3..."):
                                active_contracts = st.session_state.web_scraper.get_active_contracts()
                                
                                if active_contracts:
                                    st.session_state.active_contracts = active_contracts
                                    st.session_state.add_notification(
                                        f"Lista de contratos atualizada: {len(active_contracts)} contratos encontrados",
                                        "success"
                                    )
                                else:
                                    st.warning("Não foi possível obter a lista de contratos ativos.")
                        except Exception as e:
                            st.error(f"Erro ao buscar contratos: {str(e)}")
                    
                    # Exibir contratos ativos armazenados na sessão
                    if "active_contracts" in st.session_state and st.session_state.active_contracts:
                        # Criar um dataframe para exibição formatada
                        contracts_data = []
                        
                        # Mapeamento de códigos de mês para nomes e números
                        month_codes = {
                            'F': {'nome': 'Janeiro', 'num': 1},
                            'G': {'nome': 'Janeiro', 'num': 1},
                            'H': {'nome': 'Março', 'num': 3},
                            'J': {'nome': 'Abril', 'num': 4},
                            'K': {'nome': 'Maio', 'num': 5},
                            'M': {'nome': 'Julho', 'num': 7},
                            'N': {'nome': 'Julho', 'num': 7},
                            'Q': {'nome': 'Agosto', 'num': 8},
                            'U': {'nome': 'Setembro', 'num': 9},
                            'V': {'nome': 'Outubro', 'num': 10},
                            'X': {'nome': 'Dezembro', 'num': 12},
                            'Z': {'nome': 'Dezembro', 'num': 12}
                        }
                        
                        for contract_code in st.session_state.active_contracts:
                            try:
                                # Extrair código do mês e ano
                                if len(contract_code) >= 5:
                                    month_code = contract_code[3]
                                    year_str = contract_code[4:]
                                    
                                    # Obter informação do mês
                                    month_info = month_codes.get(month_code, {'nome': 'Desconhecido', 'num': 1})
                                    month_name = month_info['nome']
                                    month_num = month_info['num']
                                    
                                    # Converter para números
                                    year = 2000 + int(year_str)  # Assumindo anos 2000+
                                    
                                    # Calcular a data de vencimento (geralmente terceira sexta-feira do mês)
                                    import calendar
                                    
                                    # Encontrar a terceira sexta-feira do mês
                                    c = calendar.monthcalendar(year, month_num)
                                    friday_count = 0
                                    third_friday_day = None
                                    
                                    for week in c:
                                        if week[calendar.FRIDAY] != 0:
                                            friday_count += 1
                                            if friday_count == 3:
                                                third_friday_day = week[calendar.FRIDAY]
                                                break
                                    
                                    if third_friday_day:
                                        vencimento = datetime.datetime(year, month_num, third_friday_day)
                                        vencimento_str = vencimento.strftime("%d/%m/%Y")
                                        
                                        # Calcular dias até o vencimento
                                        hoje = datetime.datetime.now()
                                        dias_ate_vencimento = (vencimento - hoje).days
                                    else:
                                        vencimento_str = "N/A"
                                        dias_ate_vencimento = "N/A"
                                else:
                                    month_name = "Desconhecido"
                                    vencimento_str = "N/A"
                                    dias_ate_vencimento = "N/A"
                            except Exception as e:
                                logger.error(f"Erro ao processar contrato {contract_code}: {str(e)}")
                                month_name = "Erro"
                                vencimento_str = "Erro"
                                dias_ate_vencimento = "Erro"
                            
                            # Adicionar ao dataset para exibição
                            contracts_data.append({
                                "Código": contract_code,
                                "Vencimento": vencimento_str,
                                "Mês": month_name,
                                "Dias Restantes": dias_ate_vencimento
                            })
                        
                        if contracts_data:
                            df_contracts = pd.DataFrame(contracts_data)
                            st.dataframe(df_contracts, use_container_width=True)
                            
                            # Seleção de contrato para detalhamento
                            contract_codes = [c["Código"] for c in contracts_data]
                            selected_contract = st.selectbox(
                                "Selecione um contrato para ver detalhes",
                                options=contract_codes
                            )
                            
                            if st.button("Ver Detalhes do Contrato"):
                                try:
                                    with st.spinner(f"Buscando detalhes do contrato {selected_contract}..."):
                                        contract_details = st.session_state.web_scraper.get_contract_details(selected_contract)
                                        st.session_state.current_contract_details = contract_details
                                except Exception as e:
                                    st.error(f"Erro ao obter detalhes do contrato: {str(e)}")
                    else:
                        st.info("Clique em 'Atualizar Lista de Contratos' para carregar os contratos disponíveis.")
                
                with col2:
                    st.subheader("Detalhes do Contrato")
                    
                    # Exibir detalhes do contrato selecionado
                    if "current_contract_details" in st.session_state and st.session_state.current_contract_details:
                        contract = st.session_state.current_contract_details
                        
                        # Verificar se há erro nos detalhes do contrato
                        if "error" in contract:
                            st.error(contract["error"])
                        else:
                            # Exibir dados do contrato em formato de card/métrica
                            st.markdown(f"### {contract.get('nome_completo', 'Mini Índice Futuro')}")
                            st.markdown(f"""
                            **Código:** {contract.get('código', 'N/A')}  
                            **Vencimento:** {contract.get('vencimento', 'N/A')} ({contract.get('vencimento_dia_semana', 'N/A')})  
                            **Dias até o vencimento:** {contract.get('dias_até_vencimento', 'N/A')}  
                            """)
                            
                            # Métricas de preço em duas colunas
                            precio_col1, precio_col2 = st.columns(2)
                            with precio_col1:
                                st.metric("Último", contract.get('último', 'N/A'))
                                st.metric("Máxima", contract.get('máxima', 'N/A'))
                            with precio_col2:
                                st.metric("Variação", contract.get('variação', 'N/A'))
                                st.metric("Mínima", contract.get('mínima', 'N/A'))
                            
                            # Informações gerais sobre o contrato
                            st.markdown("### Informações do Contrato")
                            st.markdown(f"""
                            **Multiplicador:** {contract.get('multiplicador', 'N/A')}  
                            **Margem Aproximada:** {contract.get('margem_aproximada', 'N/A')}  
                            **Horário de Negociação:** {contract.get('horário_negociação', 'N/A')}  
                            **Volume:** {contract.get('volume', 'N/A')}  
                            
                            *Última atualização: {contract.get('timestamp', 'N/A')}*
                            """)
                            
                            # Notícias relacionadas ao contrato
                            st.subheader("Notícias Relacionadas")
                            
                            if st.button("Buscar Notícias"):
                                try:
                                    with st.spinner(f"Buscando notícias sobre o contrato {contract.get('código', '')}..."):
                                        news = st.session_state.web_scraper.search_news_for_contract(
                                            contract.get('código', ''), 
                                            limit=5
                                        )
                                        
                                        if news:
                                            st.session_state.contract_news = news
                                        else:
                                            st.warning("Não foram encontradas notícias para este contrato.")
                                except Exception as e:
                                    st.error(f"Erro ao buscar notícias: {str(e)}")
                            
                            # Exibir notícias se disponíveis
                            if "contract_news" in st.session_state and st.session_state.contract_news:
                                for i, news_item in enumerate(st.session_state.contract_news):
                                    relevance = news_item.get('relevância', 0)
                                    relevance_color = "green" if relevance > 7 else "orange" if relevance > 4 else "gray"
                                    
                                    with st.container():
                                        st.markdown(f"""
                                        **{news_item.get('título', 'Sem título')}**  
                                        *{news_item.get('fonte', 'Desconhecida')} - {news_item.get('data', 'Data desconhecida')}*  
                                        Relevância: <span style='color:{relevance_color}'>{relevance:.1f}/10</span>
                                        """, unsafe_allow_html=True)
                                        
                                        # Link para a notícia
                                        if news_item.get('link'):
                                            st.markdown(f"[Ler a notícia completa]({news_item.get('link')})")
                                        
                                        st.markdown("---")
                    else:
                        st.info("Selecione um contrato à esquerda e clique em 'Ver Detalhes do Contrato' para visualizar informações detalhadas.")
                
                # Calendário Econômico (abaixo das duas colunas)
                st.subheader("Calendário Econômico")
                
                if st.button("Buscar Eventos Econômicos"):
                    try:
                        with st.spinner("Buscando calendário econômico..."):
                            economic_events = st.session_state.web_scraper.get_economic_calendar()
                            
                            if economic_events:
                                st.session_state.economic_events = economic_events
                                st.session_state.add_notification(
                                    f"Calendário econômico atualizado: {len(economic_events)} eventos encontrados",
                                    "success"
                                )
                            else:
                                st.warning("Não foi possível obter o calendário econômico.")
                    except Exception as e:
                        st.error(f"Erro ao buscar calendário econômico: {str(e)}")
                
                # Exibir eventos econômicos se disponíveis
                if "economic_events" in st.session_state and st.session_state.economic_events:
                    # Criar dataframe para exibição formatada
                    events_data = []
                    for event in st.session_state.economic_events[:10]:  # Limitar a 10 eventos
                        events_data.append({
                            "Data": event.get("data", "N/A"),
                            "Hora": event.get("hora", "N/A"),
                            "Evento": event.get("evento", "N/A"),
                            "Impacto": event.get("impacto", "Médio")
                        })
                    
                    if events_data:
                        df_events = pd.DataFrame(events_data)
                        st.dataframe(df_events, use_container_width=True)
            else:
                st.warning("O Web Scraper de contratos da B3 não está inicializado. Reinicie a aplicação.")
            
        # Licensing Tab
        with tab7:
            
            st.markdown("### Licenciamento e Telemetria")
            
            st.info("""
            Configure sua licença e opções de telemetria nesta seção. 
            O envio de dados de performance é necessário para o modelo de compartilhamento de lucros.
            """)
            
            # Create license and telemetry settings UI
            license_col1, license_col2 = st.columns(2)
            
            with license_col1:
                # User ID (read-only if already exists)
                if 'performance_tracker' in st.session_state and st.session_state.performance_tracker and st.session_state.performance_tracker.user_id:
                    user_id = st.text_input(
                        "ID do Usuário",
                        value=st.session_state.performance_tracker.user_id,
                        disabled=True,
                        help="Identificador único do usuário (gerado automaticamente)"
                    )
                else:
                    user_id = st.text_input(
                        "ID do Usuário (opcional)",
                        value="",
                        help="Identificador único do usuário (deixe em branco para gerar automaticamente)"
                    )
            
            with license_col2:
                # License key
                if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                    current_license = st.session_state.performance_tracker.license_key or ""
                else:
                    current_license = os.environ.get("LICENSE_KEY", "")
                    
                license_key = st.text_input(
                    "Chave de Licença",
                    value=current_license,
                    help="Chave de licença para ativar recursos premium"
                )
            
            # API Key for telemetry
            if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                current_api_key = st.session_state.performance_tracker.api_key or ""
            else:
                current_api_key = os.environ.get("ANALYTICS_API_KEY", "")
                
            api_key = st.text_input(
                "Chave de API para Telemetria",
                value=current_api_key,
                type="password",
                help="Chave para autenticação com o servidor de telemetria"
            )
            
            # API Endpoint for telemetry
            if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                current_endpoint = st.session_state.performance_tracker.api_endpoint
            else:
                current_endpoint = os.environ.get("TELEMETRY_API_ENDPOINT", "https://api.winfutrobot.com.br/telemetry")
                
            api_endpoint = st.text_input(
                "Endpoint da API de Telemetria",
                value=current_endpoint,
                help="URL do servidor para onde os dados de telemetria serão enviados"
            )
            
            # Commission settings
            st.markdown("#### Configurações de Comissão")
            
            # Get current commission rate
            if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                current_rate = st.session_state.performance_tracker.commission_rate
            else:
                current_rate = 0.20  # Default 20%
            
            commission_rate = st.slider(
                "Taxa de Comissão",
                min_value=0.0,
                max_value=0.5,
                value=current_rate,
                step=0.01,
                format="%.2f",
                help="Porcentagem do lucro devida como comissão (modelo de compartilhamento de lucros)"
            )
            
            # Telemetry settings
            st.markdown("#### Configurações de Telemetria")
            
            # Get current send interval
            if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                current_interval = st.session_state.performance_tracker.send_interval
            else:
                current_interval = 24  # Default 24 hours
            
            send_interval = st.number_input(
                "Intervalo de Envio (horas)",
                min_value=1,
                max_value=168,  # Up to 7 days
                value=current_interval,
                help="Intervalo em horas para envio automático de dados de performance"
            )
            
            # Enable/disable telemetry
            enable_telemetry = st.toggle(
                "Ativar Telemetria",
                value=True,
                help="Habilita o envio de dados de performance para o servidor"
            )
            
            # Save license and telemetry settings button
            if st.button("Salvar Configurações de Licenciamento"):
                try:
                    # Update environment variables
                    if user_id:
                        os.environ["USER_ID"] = user_id
                    
                    if license_key:
                        os.environ["LICENSE_KEY"] = license_key
                    
                    if api_key:
                        os.environ["ANALYTICS_API_KEY"] = api_key
                        
                    if api_endpoint:
                        os.environ["TELEMETRY_API_ENDPOINT"] = api_endpoint
                    
                    # Helper function to update configuration
                    def update_config_file(config_updates):
                        # For now, just log the updates (you can implement actual config saving later)
                        logger.info(f"Updating configuration with: {config_updates}")
                        # In a real implementation, this would update a config file
                    
                    # If tracker exists, update it
                    if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                        if user_id:
                            st.session_state.performance_tracker.user_id = user_id
                        
                        if license_key:
                            st.session_state.performance_tracker.license_key = license_key
                        
                        if api_key:
                            st.session_state.performance_tracker.api_key = api_key
                            
                        if api_endpoint:
                            st.session_state.performance_tracker.api_endpoint = api_endpoint
                        
                        # Update commission rate
                        st.session_state.performance_tracker.update_commission_rate(commission_rate)
                        
                        # Update send interval
                        st.session_state.performance_tracker.send_interval = send_interval
                        
                        # Update config
                        update_config_file({
                            "user_id": user_id if user_id else st.session_state.performance_tracker.user_id,
                            "license_key": license_key,
                            "analytics_api_key": api_key,
                            "telemetry_api_endpoint": api_endpoint,
                            "commission_rate": commission_rate,
                            "telemetry_interval": send_interval,
                            "telemetry_enabled": enable_telemetry
                        })
                        
                        st.success("Configurações de licenciamento salvas com sucesso!")
                    else:
                        # Initialize tracker with new settings
                        from user_analytics import initialize_analytics
                        
                        performance_tracker, usage_collector = initialize_analytics(
                            user_id=user_id,
                            license_key=license_key,
                            api_key=api_key,
                            api_endpoint=api_endpoint
                        )
                        
                        # Update settings
                        if performance_tracker:
                            performance_tracker.update_commission_rate(commission_rate)
                            performance_tracker.send_interval = send_interval
                            
                            # Store in session state
                            st.session_state.performance_tracker = performance_tracker
                            st.session_state.usage_collector = usage_collector
                            
                            # Update config
                            update_config_file({
                                "user_id": user_id if user_id else performance_tracker.user_id,
                                "license_key": license_key,
                                "analytics_api_key": api_key,
                                "telemetry_api_endpoint": api_endpoint,
                                "commission_rate": commission_rate,
                                "telemetry_interval": send_interval,
                                "telemetry_enabled": enable_telemetry
                            })
                            
                            st.success("Configurações de licenciamento salvas com sucesso!")
                        else:
                            st.error("Falha ao inicializar telemetria. Verifique as configurações.")
                except Exception as e:
                    st.error(f"Erro ao salvar configurações: {str(e)}")
            
            # Performance data and reports
            st.markdown("#### Dados de Performance")
            
            if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                tracker = st.session_state.performance_tracker
                
                # Display commission due
                commission_summary = tracker.get_commission_summary()
                
                st.metric(
                    label="Lucro Total Registrado",
                    value=f"R$ {commission_summary['total_profit']:,.2f}"
                )
                
                st.metric(
                    label="Comissão Devida",
                    value=f"R$ {commission_summary['commission_due']:,.2f}",
                    delta=f"{commission_summary['commission_rate']:.2%} do lucro"
                )
                
                # Period selector for reports
                report_period = st.selectbox(
                    "Período para Relatório",
                    options=["7 dias", "14 dias", "30 dias", "60 dias", "90 dias"],
                    index=2  # Default to 30 days
                )
                
                # Convert period to days
                if report_period == "7 dias":
                    days = 7
                elif report_period == "14 dias":
                    days = 14
                elif report_period == "30 dias":
                    days = 30
                elif report_period == "60 dias":
                    days = 60
                else:
                    days = 90
                
                # Generate report button
                report_col1, report_col2, report_col3 = st.columns(3)
                
                with report_col1:
                    if st.button("Gerar Relatório de Performance", use_container_width=True):
                        with st.spinner("Gerando relatório..."):
                            try:
                                # Generate report with equity curve
                                report = tracker.generate_performance_report(
                                    days=days, 
                                    include_equity_curve=True
                                )
                                
                                if report:
                                    # Store in session state
                                    st.session_state.performance_report = report
                                    
                                    # Display success message
                                    st.success("Relatório gerado com sucesso!")
                                    
                                    # Update usage statistics
                                    if 'usage_collector' in st.session_state and st.session_state.usage_collector:
                                        st.session_state.usage_collector.record_feature_usage("generate_report")
                            except Exception as e:
                                st.error(f"Erro ao gerar relatório: {str(e)}")
                
                with report_col2:
                    if st.button("Salvar Relatório (JSON)", use_container_width=True):
                        try:
                            # Save report to file
                            filename = f"relatorio_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            success = tracker.save_report(filename=filename, format="json")
                            
                            if success:
                                st.success(f"Relatório salvo como: {filename}")
                            else:
                                st.error("Falha ao salvar relatório")
                        except Exception as e:
                            st.error(f"Erro ao salvar relatório: {str(e)}")
                
                with report_col3:
                    if st.button("Salvar Relatório (HTML)", use_container_width=True):
                        try:
                            # Save report to file
                            filename = f"relatorio_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                            success = tracker.save_report(filename=filename, format="html")
                            
                            if success:
                                st.success(f"Relatório salvo como: {filename}")
                            else:
                                st.error("Falha ao salvar relatório")
                        except Exception as e:
                            st.error(f"Erro ao salvar relatório: {str(e)}")
                
                # Manual telemetry actions
                st.markdown("#### Ações de Telemetria")
                
                telemetry_col1, telemetry_col2 = st.columns(2)
                
                with telemetry_col1:
                    if st.button("Enviar Dados Agora", use_container_width=True):
                        with st.spinner("Enviando dados para o servidor..."):
                            try:
                                # Force send data
                                success = tracker.send_performance_data(force=True)
                                
                                if success:
                                    st.success("Dados enviados com sucesso!")
                                else:
                                    st.warning("Sem dados para enviar ou falha no envio.")
                            except Exception as e:
                                st.error(f"Erro ao enviar dados: {str(e)}")
                
                with telemetry_col2:
                    if st.button("Verificar Status de Licença", use_container_width=True):
                        with st.spinner("Verificando status da licença..."):
                            try:
                                # Send data (which also checks license)
                                success = tracker.send_performance_data(force=True)
                                
                                if success:
                                    if tracker.license_key:
                                        st.success(f"Licença ativa: {tracker.license_key}")
                                    else:
                                        st.warning("Nenhuma licença configurada")
                                else:
                                    st.error("Falha ao verificar licença. Servidor indisponível.")
                            except Exception as e:
                                st.error(f"Erro ao verificar licença: {str(e)}")
                
                # Display performance summary if report exists
                if 'performance_report' in st.session_state:
                    report = st.session_state.performance_report
                    
                    st.markdown("#### Resumo de Performance")
                    
                    # Create metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            label="Trades Totais",
                            value=report["performance"]["trades"]
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="Taxa de Acerto",
                            value=f"{report['performance']['win_rate']:.2%}"
                        )
                    
                    with metric_col3:
                        st.metric(
                            label="Resultado Líquido",
                            value=f"R$ {report['performance']['net_pnl']:,.2f}"
                        )
                    
                    with metric_col4:
                        st.metric(
                            label="Fator de Lucro",
                            value=f"{report['performance']['profit_factor']:.2f}"
                        )
                    
                    # Display equity curve if available
                    if "equity_curve" in report:
                        st.markdown("#### Curva de Patrimônio")
                        st.image(f"data:image/png;base64,{report['equity_curve']}")
            else:
                st.warning("""
                Telemetria não está inicializada. Salve as configurações 
                de licenciamento para ativar o rastreamento de performance.
                """)
        
        # Logs Tab
        with tab8:
            st.subheader("Trading Logs")
            
            # Log filter
            
        # Análise Avançada Tab
        with tab9:
            st.subheader("Análise Avançada de Mercado")
            
            # Subtabs for different advanced analyses
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
                "Indicadores Avançados", "Correlações", "Padrões de Candlestick", "Fluxo de Ordens"
            ])
            
            # Tab 1: Advanced Technical Indicators
            with analysis_tab1:
                st.markdown("### Indicadores Técnicos Avançados")
                
                # Get data to analyze
                data_source = st.radio(
                    "Fonte de dados",
                    options=["Dados atuais", "Buscar novos dados"],
                    horizontal=True
                )
                
                df_to_analyze = None
                
                if data_source == "Dados atuais":
                    # Use existing data
                    if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                        df_to_analyze = st.session_state.price_data.copy()
                        st.success(f"Usando dados existentes ({len(df_to_analyze)} períodos)")
                    else:
                        st.warning("Não há dados disponíveis. Selecione 'Buscar novos dados' para obter dados para análise.")
                else:
                    # Fetch new data
                    st.markdown("#### Selecione intervalo para buscar dados")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        timeframe = st.selectbox(
                            "Intervalo",
                            options=["1m", "5m", "15m", "30m", "1h", "1d"],
                            index=2  # Default to 15m
                        )
                    
                    with col2:
                        days = st.slider(
                            "Dias de história",
                            min_value=1,
                            max_value=30,
                            value=5
                        )
                    
                    if st.button("Buscar dados para análise"):
                        with st.spinner(f"Buscando {days} dias de dados WINFUT ({timeframe})..."):
                            try:
                                start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                                
                                # Use current contract
                                current_contract = st.session_state.current_contract
                                
                                # Attempt to get data from investing.com collector
                                if st.session_state.investing_collector is not None:
                                    historical_data = st.session_state.investing_collector.get_historical_data(
                                        symbol=current_contract,
                                        interval=timeframe,
                                        start_date=start_date,
                                        use_cache=False
                                    )
                                    
                                    if not historical_data.empty:
                                        df_to_analyze = historical_data.copy()
                                        st.success(f"Dados obtidos com sucesso: {len(df_to_analyze)} períodos")
                                    else:
                                        st.error("Não foi possível obter dados para o período selecionado.")
                                else:
                                    st.error("Collector de dados não está inicializado.")
                            except Exception as e:
                                st.error(f"Erro ao buscar dados: {str(e)}")
                
                # If we have data to analyze, calculate and display indicators
                if df_to_analyze is not None and not df_to_analyze.empty:
                    # Calculate advanced indicators
                    if st.session_state.technical_indicators is not None:
                        with st.spinner("Calculando indicadores avançados..."):
                            advanced_indicators = TechnicalIndicators.calculate_all(df_to_analyze)
                            
                            # Display available indicator categories
                            indicator_categories = {
                                "moving_averages": "Médias Móveis Avançadas",
                                "momentum": "Indicadores de Momentum",
                                "volatility": "Indicadores de Volatilidade",
                                "trend": "Indicadores de Tendência",
                                "oscillators": "Osciladores",
                                "volume": "Indicadores de Volume",
                                "cycle": "Indicadores de Ciclo"
                            }
                            
                            selected_category = st.selectbox(
                                "Categoria de indicadores",
                                options=list(indicator_categories.keys()),
                                format_func=lambda x: indicator_categories[x]
                            )
                            
                            # Map category to columns
                            category_columns = {
                                "moving_averages": [col for col in advanced_indicators.columns if any(x in col for x in ['sma_', 'ema_', 'wma_', 'hma_', 'vwma_', 'tma_'])],
                                "momentum": [col for col in advanced_indicators.columns if any(x in col for x in ['rsi_', 'cci', 'mfi', 'roc_', 'williams_r', 'momentum'])],
                                "volatility": [col for col in advanced_indicators.columns if any(x in col for x in ['bb_', 'atr_', 'natr', 'hist_vol_', 'chaikin_vol'])],
                                "trend": [col for col in advanced_indicators.columns if any(x in col for x in ['adx_', 'aroon_', 'vortex_', 'kst', 'trix', 'mass_index'])],
                                "oscillators": [col for col in advanced_indicators.columns if any(x in col for x in ['stoch_', 'macd', 'willr'])],
                                "volume": [col for col in advanced_indicators.columns if any(x in col for x in ['obv', 'vwap', 'volume_'])],
                                "cycle": [col for col in advanced_indicators.columns if any(x in col for x in ['fib_', 'pp_'])]
                            }
                            
                            # Get columns for selected category
                            selected_columns = category_columns.get(selected_category, [])
                            
                            if selected_columns:
                                # Plot indicators
                                st.markdown(f"#### {indicator_categories[selected_category]}")
                                
                                # Plot price with indicators
                                fig = go.Figure()
                                
                                # Add price (candlestick)
                                fig.add_trace(go.Candlestick(
                                    x=advanced_indicators.index,
                                    open=advanced_indicators['open'],
                                    high=advanced_indicators['high'],
                                    low=advanced_indicators['low'],
                                    close=advanced_indicators['close'],
                                    name="OHLC"
                                ))
                                
                                # Add selected indicators
                                for col in selected_columns[:5]:  # Limit to 5 indicators
                                    if col in advanced_indicators.columns:
                                        fig.add_trace(go.Scatter(
                                            x=advanced_indicators.index,
                                            y=advanced_indicators[col],
                                            mode='lines',
                                            name=col
                                        ))
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"{indicator_categories[selected_category]} - {current_contract}",
                                    xaxis_title="Data",
                                    yaxis_title="Preço/Valor",
                                    height=600,
                                    xaxis_rangeslider_visible=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display indicator values
                                st.markdown("#### Valores dos indicadores (últimos 5 períodos)")
                                st.dataframe(advanced_indicators[selected_columns].tail(5))
                            else:
                                st.info(f"Nenhum indicador disponível para a categoria {indicator_categories[selected_category]}")
                    else:
                        st.warning("Componente de indicadores técnicos avançados não está inicializado.")
                
            # Tab 2: Correlation Analysis
            with analysis_tab2:
                st.markdown("### Análise de Correlações")
                
                if st.session_state.correlation_analyzer is not None:
                    # Instructions
                    st.info("Esta análise permite identificar correlações entre o WINFUT e outros ativos financeiros.")
                    
                    # Get data for analysis
                    corr_source = st.radio(
                        "Fonte de dados para correlação",
                        options=["Dados atuais", "Buscar novos dados"],
                        horizontal=True,
                        key="corr_source"
                    )
                    
                    winfut_data = None
                    other_assets_data = {}
                    
                    if corr_source == "Dados atuais":
                        # Use existing data
                        if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                            winfut_data = st.session_state.price_data.copy()
                            st.success(f"Usando dados WINFUT existentes ({len(winfut_data)} períodos)")
                            
                            # For other assets, we still need to fetch
                            st.info("Para outros ativos, ainda é necessário buscar dados.")
                        else:
                            st.warning("Não há dados WINFUT disponíveis. Selecione 'Buscar novos dados'.")
                    
                    # Select assets to correlate with
                    st.markdown("#### Selecione ativos para correlacionar com WINFUT")
                    
                    default_assets = st.session_state.correlation_analyzer.default_assets
                    
                    selected_assets = st.multiselect(
                        "Ativos",
                        options=default_assets,
                        default=default_assets[:5]  # Default to first 5
                    )
                    
                    # Parameters for analysis
                    st.markdown("#### Parâmetros de análise")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        days = st.slider(
                            "Dias de história",
                            min_value=5,
                            max_value=60,
                            value=30,
                            key="corr_days"
                        )
                    
                    with col2:
                        method = st.selectbox(
                            "Método de correlação",
                            options=["pearson", "spearman", "kendall"],
                            index=0,
                            key="corr_method"
                        )
                    
                    with col3:
                        timeframe = st.selectbox(
                            "Intervalo",
                            options=["1h", "4h", "1d"],
                            index=2,  # Default to daily
                            key="corr_timeframe"
                        )
                    
                    # Button to fetch data and calculate correlations
                    if st.button("Calcular Correlações"):
                        with st.spinner("Buscando dados e calculando correlações..."):
                            try:
                                # Fetch WINFUT data if needed
                                if winfut_data is None or corr_source == "Buscar novos dados":
                                    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                                    
                                    # Current contract
                                    current_contract = st.session_state.current_contract
                                    
                                    if st.session_state.investing_collector is not None:
                                        winfut_data = st.session_state.investing_collector.get_historical_data(
                                            symbol=current_contract,
                                            interval=timeframe,
                                            start_date=start_date,
                                            use_cache=False
                                        )
                                
                                # If we have WINFUT data, get data for other assets
                                if winfut_data is not None and not winfut_data.empty:
                                    st.success(f"Dados WINFUT obtidos: {len(winfut_data)} períodos")
                                    
                                    # Fetch data for selected assets
                                    for asset in selected_assets:
                                        try:
                                            asset_data = st.session_state.investing_collector.get_historical_data(
                                                symbol=asset,
                                                interval=timeframe,
                                                start_date=datetime.datetime.now() - datetime.timedelta(days=days),
                                                use_cache=False
                                            )
                                            
                                            if not asset_data.empty:
                                                other_assets_data[asset] = asset_data
                                                st.success(f"Dados obtidos para {asset}: {len(asset_data)} períodos")
                                        except Exception as e:
                                            st.warning(f"Erro ao buscar dados para {asset}: {str(e)}")
                                    
                                    # Calculate correlations if we have data
                                    if other_assets_data:
                                        # Calculate correlation matrix
                                        correlation_matrix = st.session_state.correlation_analyzer.calculate_correlation(
                                            winfut_data, 
                                            other_assets_data,
                                            method=method
                                        )
                                        
                                        if not correlation_matrix.empty:
                                            # Display correlation heatmap
                                            st.markdown("#### Matriz de Correlação")
                                            
                                            fig = st.session_state.correlation_analyzer.plot_correlation_matrix(
                                                correlation_matrix,
                                                title=f"Correlações com WINFUT ({timeframe}, {days} dias)"
                                            )
                                            st.pyplot(fig)
                                            
                                            # Get most correlated assets
                                            high_corr = st.session_state.correlation_analyzer.get_correlated_assets(
                                                correlation_matrix,
                                                min_correlation=0.5
                                            )
                                            
                                            # Get negatively correlated assets
                                            neg_corr = st.session_state.correlation_analyzer.get_correlated_assets(
                                                correlation_matrix,
                                                min_correlation=0.5
                                            )
                                            neg_corr = {k: v for k, v in neg_corr.items() if v < 0}
                                            
                                            # Display correlation results
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.markdown("#### Ativos mais correlacionados")
                                                
                                                if high_corr:
                                                    for asset, corr_value in high_corr.items():
                                                        if corr_value > 0:
                                                            st.markdown(f"- **{asset}**: {corr_value:.4f}")
                                                else:
                                                    st.info("Nenhum ativo com alta correlação positiva.")
                                            
                                            with col2:
                                                st.markdown("#### Ativos com correlação negativa")
                                                
                                                if neg_corr:
                                                    for asset, corr_value in neg_corr.items():
                                                        st.markdown(f"- **{asset}**: {corr_value:.4f}")
                                                else:
                                                    st.info("Nenhum ativo com correlação negativa significativa.")
                                            
                                            # Lead-lag relationships
                                            st.markdown("#### Relações de Lead-Lag")
                                            st.info("Esta análise mostra quais ativos tendem a se mover antes do WINFUT (valores negativos de lag) ou depois do WINFUT (valores positivos de lag).")
                                            
                                            # Calculate lead-lag
                                            lead_lag_data = st.session_state.correlation_analyzer.get_lead_lag_relationships(
                                                winfut_data,
                                                other_assets_data,
                                                max_lag=5
                                            )
                                            
                                            if lead_lag_data:
                                                # Plot lead-lag heatmap
                                                fig = st.session_state.correlation_analyzer.plot_lead_lag_heatmap(
                                                    lead_lag_data,
                                                    title=f"Análise Lead-Lag com WINFUT ({timeframe}, {days} dias)"
                                                )
                                                st.pyplot(fig)
                                                
                                                # Find predictive assets
                                                predictive_assets = {}
                                                for asset, lag_data in lead_lag_data.items():
                                                    neg_lags = {lag: corr for lag, corr in lag_data.items() if lag < 0}
                                                    if neg_lags:
                                                        best_lag = max(neg_lags.items(), key=lambda x: abs(x[1]))
                                                        if abs(best_lag[1]) >= 0.5:  # Only if correlation is significant
                                                            predictive_assets[asset] = {'lag': best_lag[0], 'correlation': best_lag[1]}
                                                
                                                # Display predictive assets
                                                if predictive_assets:
                                                    st.markdown("#### Ativos com potencial preditivo")
                                                    
                                                    for asset, data in predictive_assets.items():
                                                        direction = "positiva" if data['correlation'] > 0 else "negativa"
                                                        st.markdown(f"- **{asset}**: se move {abs(data['lag'])} períodos antes do WINFUT (correlação {direction}: {data['correlation']:.4f})")
                                                else:
                                                    st.info("Nenhum ativo com potencial preditivo significativo identificado.")
                                            else:
                                                st.warning("Não foi possível calcular relações de lead-lag.")
                                            
                                            # Correlation changes over time
                                            st.markdown("#### Mudanças de Correlação ao Longo do Tempo")
                                            
                                            # Calculate correlation changes
                                            corr_changes = st.session_state.correlation_analyzer.get_correlation_changes(
                                                winfut_data,
                                                other_assets_data,
                                                window_size=20,
                                                step=1
                                            )
                                            
                                            if corr_changes:
                                                # Plot correlation changes
                                                fig = st.session_state.correlation_analyzer.plot_correlation_changes(
                                                    corr_changes,
                                                    title=f"Mudanças de Correlação ao Longo do Tempo (janela de 20 períodos)"
                                                )
                                                st.pyplot(fig)
                                                
                                                # Calculate correlation stability
                                                stability_measures = {}
                                                for asset, corr_data in corr_changes.items():
                                                    if corr_data:
                                                        corrs = [c[1] for c in corr_data]
                                                        stability_measures[asset] = {
                                                            'variability': np.std(corrs),
                                                            'mean': np.mean(corrs),
                                                            'trend': np.corrcoef(range(len(corrs)), corrs)[0, 1] if len(corrs) > 1 else 0
                                                        }
                                                
                                                # Display most stable correlations
                                                if stability_measures:
                                                    st.markdown("#### Estabilidade de Correlações")
                                                    
                                                    # Sort by variability (ascending)
                                                    stable_assets = dict(sorted(stability_measures.items(), key=lambda x: x[1]['variability']))
                                                    
                                                    for asset, data in list(stable_assets.items())[:3]:
                                                        st.markdown(f"- **{asset}**: correlação média de {data['mean']:.4f} (variabilidade: {data['variability']:.4f})")
                                                
                                            else:
                                                st.warning("Não foi possível calcular mudanças de correlação.")
                                        else:
                                            st.error("Não foi possível calcular matriz de correlação.")
                                    else:
                                        st.error("Não foi possível obter dados para os ativos selecionados.")
                                else:
                                    st.error("Não foi possível obter dados WINFUT.")
                            except Exception as e:
                                st.error(f"Erro na análise de correlação: {str(e)}")
                else:
                    st.warning("Componente de análise de correlação não está inicializado.")
            
            # Tab 3: Candlestick Patterns
            with analysis_tab3:
                st.markdown("### Análise de Padrões de Candlestick")
                
                if st.session_state.pattern_detector is not None:
                    # Instructions
                    st.info("Esta análise identifica padrões de candlestick e formações gráficas no WINFUT.")
                    
                    # Get data for analysis
                    pattern_source = st.radio(
                        "Fonte de dados para análise de padrões",
                        options=["Dados atuais", "Buscar novos dados"],
                        horizontal=True,
                        key="pattern_source"
                    )
                    
                    pattern_data = None
                    
                    if pattern_source == "Dados atuais":
                        # Use existing data
                        if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                            pattern_data = st.session_state.price_data.copy()
                            st.success(f"Usando dados existentes ({len(pattern_data)} períodos)")
                        else:
                            st.warning("Não há dados disponíveis. Selecione 'Buscar novos dados'.")
                    else:
                        # Parameters for fetching new data
                        st.markdown("#### Parâmetros para busca de dados")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            pattern_timeframe = st.selectbox(
                                "Intervalo",
                                options=["1m", "5m", "15m", "30m", "1h", "1d"],
                                index=2,  # Default to 15m
                                key="pattern_timeframe"
                            )
                        
                        with col2:
                            pattern_days = st.slider(
                                "Dias de história",
                                min_value=1,
                                max_value=30,
                                value=5,
                                key="pattern_days"
                            )
                        
                        # Button to fetch data
                        if st.button("Buscar dados para análise de padrões"):
                            with st.spinner(f"Buscando {pattern_days} dias de dados WINFUT ({pattern_timeframe})..."):
                                try:
                                    start_date = datetime.datetime.now() - datetime.timedelta(days=pattern_days)
                                    
                                    # Current contract
                                    current_contract = st.session_state.current_contract
                                    
                                    if st.session_state.investing_collector is not None:
                                        pattern_data = st.session_state.investing_collector.get_historical_data(
                                            symbol=current_contract,
                                            interval=pattern_timeframe,
                                            start_date=start_date,
                                            use_cache=False
                                        )
                                        
                                        if not pattern_data.empty:
                                            st.success(f"Dados obtidos: {len(pattern_data)} períodos")
                                        else:
                                            st.error("Não foi possível obter dados para o período selecionado.")
                                    else:
                                        st.error("Collector de dados não está inicializado.")
                                except Exception as e:
                                    st.error(f"Erro ao buscar dados: {str(e)}")
                    
                    # If we have data, detect patterns
                    if pattern_data is not None and not pattern_data.empty:
                        # Analysis options
                        st.markdown("#### Opções de análise")
                        
                        analysis_type = st.radio(
                            "Tipo de análise",
                            options=["Padrões de Candlestick", "Formações Gráficas", "Relatório Completo"],
                            horizontal=True,
                            key="pattern_analysis_type"
                        )
                        
                        if analysis_type == "Padrões de Candlestick" or analysis_type == "Relatório Completo":
                            # Calculate candlestick patterns
                            with st.spinner("Detectando padrões de candlestick..."):
                                candlestick_signals = st.session_state.pattern_detector.detect_candlestick_patterns(pattern_data)
                                
                                if not candlestick_signals.empty:
                                    # Display recent patterns
                                    st.markdown("#### Padrões de Candlestick Recentes")
                                    
                                    # Get the most recent patterns
                                    recent_patterns = []
                                    
                                    for idx, row in candlestick_signals.iloc[-10:].iterrows():
                                        pattern_name, value, signal_type = st.session_state.pattern_detector.get_strongest_pattern(row)
                                        
                                        if pattern_name != "Nenhum padrão":
                                            recent_patterns.append({
                                                'data': idx,
                                                'padrão': pattern_name,
                                                'tipo': signal_type,
                                                'valor': value
                                            })
                                    
                                    if recent_patterns:
                                        # Display as table
                                        recent_df = pd.DataFrame(recent_patterns)
                                        st.dataframe(recent_df)
                                        
                                        # Plot candlestick chart with patterns
                                        fig = st.session_state.pattern_detector.plot_candlestick_patterns(
                                            pattern_data,
                                            candlestick_signals,
                                            window_size=30
                                        )
                                        st.pyplot(fig)
                                    else:
                                        st.info("Nenhum padrão de candlestick significativo detectado nos dados mais recentes.")
                                else:
                                    st.warning("Não foi possível detectar padrões de candlestick.")
                        
                        if analysis_type == "Formações Gráficas" or analysis_type == "Relatório Completo":
                            # Calculate chart patterns
                            with st.spinner("Detectando formações gráficas..."):
                                chart_signals = st.session_state.pattern_detector.detect_chart_patterns(pattern_data)
                                
                                if not chart_signals.empty:
                                    # Display detected chart patterns
                                    st.markdown("#### Formações Gráficas Detectadas")
                                    
                                    # Find recent chart patterns
                                    recent_chart_patterns = []
                                    
                                    # Chart pattern categories
                                    chart_categories = {
                                        "head_and_shoulders": "Cabeça e Ombros (baixista)",
                                        "inverse_head_and_shoulders": "Cabeça e Ombros Invertido (altista)",
                                        "double_top": "Topo Duplo (baixista)",
                                        "double_bottom": "Fundo Duplo (altista)",
                                        "triple_top": "Topo Triplo (baixista)",
                                        "triple_bottom": "Fundo Triplo (altista)",
                                        "rising_wedge": "Cunha Ascendente (baixista)",
                                        "falling_wedge": "Cunha Descendente (altista)",
                                        "ascending_triangle": "Triângulo Ascendente (altista)",
                                        "descending_triangle": "Triângulo Descendente (baixista)",
                                        "symmetric_triangle": "Triângulo Simétrico (neutro)",
                                        "rectangle": "Retângulo (neutro)",
                                        "flag_pole": "Bandeira (continuação)",
                                        "pennant": "Flâmula (continuação)"
                                    }
                                    
                                    for idx, row in chart_signals.iloc[-20:].iterrows():
                                        for pattern in st.session_state.pattern_detector.chart_patterns:
                                            if row[pattern] != 0:
                                                pattern_type = "altista" if row[pattern] > 0 else "baixista" if row[pattern] < 0 else "neutro"
                                                recent_chart_patterns.append({
                                                    'data': idx,
                                                    'padrão': chart_categories.get(pattern, pattern),
                                                    'tipo': pattern_type,
                                                    'valor': row[pattern]
                                                })
                                    
                                    if recent_chart_patterns:
                                        # Display as table
                                        chart_df = pd.DataFrame(recent_chart_patterns)
                                        st.dataframe(chart_df)
                                        
                                        # Plot chart with patterns
                                        fig = st.session_state.pattern_detector.plot_chart_patterns(
                                            pattern_data,
                                            chart_signals,
                                            window_size=50
                                        )
                                        st.pyplot(fig)
                                    else:
                                        st.info("Nenhuma formação gráfica significativa detectada nos dados recentes.")
                                else:
                                    st.warning("Não foi possível detectar formações gráficas.")
                        
                        if analysis_type == "Relatório Completo":
                            # Generate complete pattern report
                            with st.spinner("Gerando relatório completo de padrões..."):
                                pattern_report = st.session_state.pattern_detector.generate_pattern_report(pattern_data)
                                
                                if pattern_report and 'erro' not in pattern_report:
                                    # Display pattern statistics
                                    st.markdown("#### Estatísticas de Padrões")
                                    
                                    stats = pattern_report.get('statistics', {})
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Padrões de Alta (Candlestick)", stats.get('num_bullish_candles', 0))
                                        st.metric("Padrões de Baixa (Candlestick)", stats.get('num_bearish_candles', 0))
                                    
                                    with col2:
                                        st.metric("Padrões de Alta (Gráfico)", stats.get('num_bullish_chart', 0))
                                        st.metric("Padrões de Baixa (Gráfico)", stats.get('num_bearish_chart', 0))
                                    
                                    # Display trend analysis
                                    trend_analysis = pattern_report.get('trend_analysis', {})
                                    
                                    if trend_analysis:
                                        st.markdown(f"#### Análise de Tendência: **{trend_analysis.get('tendencia', 'neutro')}**")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.metric("Score Altista", trend_analysis.get('score_altista', 0))
                                        
                                        with col2:
                                            st.metric("Score Baixista", trend_analysis.get('score_baixista', 0))
                                    
                                    # Display recommendations
                                    recommendations = pattern_report.get('recommendations', [])
                                    
                                    if recommendations:
                                        st.markdown("#### Recomendações")
                                        
                                        for rec in recommendations:
                                            st.info(rec)
                                else:
                                    st.error(f"Erro ao gerar relatório: {pattern_report.get('erro', 'Erro desconhecido')}")
                else:
                    st.warning("Componente de detecção de padrões não está inicializado.")
            
            # Tab 4: Order Flow Analysis
            with analysis_tab4:
                st.markdown("### Análise de Fluxo de Ordens e Book de Ofertas")
                
                # Initialize OrderFlowAnalyzer if not already done
                if 'order_flow_analyzer' not in st.session_state:
                    try:
                        from order_flow_analyzer import OrderFlowAnalyzer
                        st.session_state.order_flow_analyzer = OrderFlowAnalyzer()
                        logger.info("OrderFlowAnalyzer inicializado com sucesso")
                    except Exception as e:
                        st.error(f"Erro ao inicializar OrderFlowAnalyzer: {str(e)}")
                        st.session_state.order_flow_analyzer = None
                
                # Check if component is initialized
                if st.session_state.order_flow_analyzer is not None:
                    # Instructions
                    st.info("Esta análise permite visualizar o fluxo de ordens, profundidade de mercado e distribuição de volume por níveis de preço.")
                    
                    # Analysis options
                    analysis_type = st.radio(
                        "Tipo de análise",
                        options=["Volume Profile", "Footprint Chart", "Livro de Ofertas (DOM)", "Desequilíbrios de Volume"],
                        horizontal=True,
                        key="order_flow_analysis_type"
                    )
                    
                    # Get data for analysis based on selected type
                    flow_source = st.radio(
                        "Fonte de dados",
                        options=["Dados atuais", "Buscar novos dados"],
                        horizontal=True,
                        key="flow_source"
                    )
                    
                    flow_data = None
                    tick_data = None
                    dom_data = None
                    
                    if flow_source == "Dados atuais":
                        # Use existing data
                        if st.session_state.price_data is not None and not st.session_state.price_data.empty:
                            flow_data = st.session_state.price_data.copy()
                            st.success(f"Usando dados existentes ({len(flow_data)} períodos)")
                        else:
                            st.warning("Não há dados disponíveis. Selecione 'Buscar novos dados' para obter dados para análise.")
                    else:
                        # Parameters for fetching new data
                        st.markdown("#### Parâmetros para busca de dados")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            flow_timeframe = st.selectbox(
                                "Intervalo",
                                options=["1m", "5m", "15m", "30m", "1h", "1d"],
                                index=1,  # Default to 5m for more detailed analysis
                                key="flow_timeframe"
                            )
                        
                        with col2:
                            flow_days = st.slider(
                                "Dias de história",
                                min_value=1,
                                max_value=10,
                                value=1,  # Default to 1 day for finer detail
                                key="flow_days"
                            )
                        
                        # Button to fetch data
                        if st.button("Buscar dados para análise de fluxo", key="fetch_flow_data"):
                            with st.spinner(f"Buscando {flow_days} dias de dados WINFUT ({flow_timeframe})..."):
                                try:
                                    start_date = datetime.datetime.now() - datetime.timedelta(days=flow_days)
                                    
                                    # Current contract
                                    current_contract = st.session_state.current_contract
                                    
                                    if st.session_state.investing_collector is not None:
                                        flow_data = st.session_state.investing_collector.get_historical_data(
                                            symbol=current_contract,
                                            interval=flow_timeframe,
                                            start_date=start_date,
                                            use_cache=False
                                        )
                                        
                                        if not flow_data.empty:
                                            st.success(f"Dados obtidos: {len(flow_data)} períodos")
                                            
                                            # Generate simulated tick data for demonstration
                                            # In a real system, this would come from the trading API
                                            if analysis_type in ["Footprint Chart"]:
                                                with st.spinner("Gerando dados de tick simulados para demonstração..."):
                                                    tick_data = []
                                                    
                                                    # For each candle, generate simulated ticks
                                                    for idx, row in flow_data.iterrows():
                                                        # Number of ticks for this candle (based on volume)
                                                        num_ticks = max(10, int(row['volume'] / 100))
                                                        
                                                        # Generate timestamps within candle period
                                                        if isinstance(idx, pd.Timestamp):
                                                            # If we have the next candle, use it to define the end
                                                            if idx == flow_data.index[-1]:  # Last candle
                                                                next_idx = idx + (flow_data.index[1] - flow_data.index[0])
                                                            else:
                                                                next_idx = flow_data.index[flow_data.index.get_loc(idx) + 1]
                                                                
                                                            # Generate evenly spaced timestamps
                                                            timestamps = pd.date_range(start=idx, end=next_idx - pd.Timedelta(microseconds=1), periods=num_ticks)
                                                            
                                                            # Generate prices around OHLC values
                                                            # This is just a simplified simulation for demonstration
                                                            price_min = row['low']
                                                            price_max = row['high']
                                                            
                                                            for i, ts in enumerate(timestamps):
                                                                # Determine if buy or sell
                                                                is_buy = np.random.choice([True, False])
                                                                
                                                                # Simulate price
                                                                price = price_min + np.random.random() * (price_max - price_min)
                                                                
                                                                # Simulate volume (larger at start, middle, and end of candle)
                                                                if i < num_ticks * 0.2 or i > num_ticks * 0.8 or (i > num_ticks * 0.45 and i < num_ticks * 0.55):
                                                                    volume = np.random.randint(5, 20)
                                                                else:
                                                                    volume = np.random.randint(1, 10)
                                                                    
                                                                tick_data.append({
                                                                    'timestamp': ts,
                                                                    'price': price,
                                                                    'volume': volume,
                                                                    'type': 'buy' if is_buy else 'sell'
                                                                })
                                                    
                                                    # Convert to DataFrame
                                                    tick_data = pd.DataFrame(tick_data)
                                                    st.success(f"Dados de tick simulados gerados: {len(tick_data)} ticks")
                                                            
                                            # Generate simulated DOM data for demonstration
                                            # In a real system, this would come from the trading API
                                            if analysis_type in ["Livro de Ofertas (DOM)"]:
                                                with st.spinner("Gerando dados de DOM simulados para demonstração..."):
                                                    # Get last price
                                                    last_price = flow_data['close'].iloc[-1]
                                                    
                                                    # Create simulated DOM data
                                                    bids = []
                                                    asks = []
                                                    
                                                    # Generate 10 levels of bids and asks
                                                    tick_size = 5  # Minimum price increment
                                                    
                                                    for i in range(10):
                                                        # Bid prices (below last price)
                                                        bid_price = last_price - (i + 1) * tick_size
                                                        
                                                        # Simulate volumes (decreasing as we move away from mid price)
                                                        # with some randomness
                                                        base_volume = max(1, int(100 * (0.9 ** i)))
                                                        random_factor = 0.7 + np.random.random() * 0.6
                                                        bid_volume = int(base_volume * random_factor)
                                                        
                                                        bids.append([bid_price, bid_volume])
                                                        
                                                        # Ask prices (above last price)
                                                        ask_price = last_price + (i + 1) * tick_size
                                                        
                                                        # Simulate volumes with similar pattern
                                                        random_factor = 0.7 + np.random.random() * 0.6
                                                        ask_volume = int(base_volume * random_factor)
                                                        
                                                        asks.append([ask_price, ask_volume])
                                                    
                                                    # Create DOM data structure
                                                    dom_data = {
                                                        'bids': bids,
                                                        'asks': asks,
                                                        'timestamp': datetime.datetime.now()
                                                    }
                                                    
                                                    # Process DOM data
                                                    dom_data = st.session_state.order_flow_analyzer.process_dom_data(dom_data)
                                                    st.success("Dados de DOM simulados gerados com sucesso")
                                                    
                                                    # Create DOM history for time series analysis
                                                    # Simulate 100 historical DOM snapshots
                                                    dom_history = []
                                                    base_timestamp = datetime.datetime.now() - datetime.timedelta(hours=1)
                                                    
                                                    for i in range(100):
                                                        timestamp = base_timestamp + datetime.timedelta(seconds=i * 36)
                                                        
                                                        # Vary mid price slightly for simulation
                                                        price_drift = (np.random.random() - 0.5) * 0.0001 * last_price
                                                        
                                                        # Copy basic structure but vary values
                                                        history_entry = {
                                                            'timestamp': timestamp,
                                                            'mid_price': dom_data['mid_price'] + price_drift * i,
                                                            'spread': dom_data['spread'] * (0.95 + np.random.random() * 0.1),
                                                            'best_bid': dom_data['best_bid'] + price_drift * i,
                                                            'best_ask': dom_data['best_ask'] + price_drift * i,
                                                            'best_bid_volume': int(dom_data['best_bid_volume'] * (0.9 + np.random.random() * 0.2)),
                                                            'best_ask_volume': int(dom_data['best_ask_volume'] * (0.9 + np.random.random() * 0.2)),
                                                            'total_bid_volume': int(dom_data['total_bid_volume'] * (0.95 + np.random.random() * 0.1)),
                                                            'total_ask_volume': int(dom_data['total_ask_volume'] * (0.95 + np.random.random() * 0.1)),
                                                            'imbalance': dom_data['imbalance'] * (0.8 + np.random.random() * 0.4),
                                                            'buy_sell_ratio': dom_data['buy_sell_ratio'] * (0.9 + np.random.random() * 0.2)
                                                        }
                                                        dom_history.append(history_entry)
                                                    
                                                    # Convert to DataFrame
                                                    dom_history_df = pd.DataFrame(dom_history)
                                                    dom_history_df = dom_history_df.set_index('timestamp')
                                        else:
                                            st.error("Não foi possível obter dados para o período selecionado.")
                                    else:
                                        st.error("Collector de dados não está inicializado.")
                                except Exception as e:
                                    st.error(f"Erro ao buscar dados: {str(e)}")
                    
                    # If we have data to analyze, proceed with selected analysis
                    if flow_data is not None and not flow_data.empty:
                        if analysis_type == "Volume Profile":
                            st.markdown("### Volume Profile (Perfil de Volume por Preço)")
                            
                            # Parameters for Volume Profile
                            st.markdown("#### Parâmetros de análise")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                num_bins = st.slider(
                                    "Número de níveis de preço",
                                    min_value=20,
                                    max_value=100,
                                    value=50,
                                    key="vp_bins"
                                )
                            
                            with col2:
                                show_price = st.checkbox(
                                    "Mostrar preço junto ao perfil",
                                    value=True,
                                    key="vp_show_price"
                                )
                            
                            # Button to calculate and display Volume Profile
                            if st.button("Calcular Perfil de Volume", key="calc_volume_profile"):
                                with st.spinner("Calculando perfil de volume..."):
                                    try:
                                        # Calculate Volume Profile
                                        volume_profile = st.session_state.order_flow_analyzer.calculate_volume_profile(
                                            flow_data,
                                            num_bins=num_bins
                                        )
                                        
                                        if volume_profile:
                                            # Create Plotly figure
                                            price_data_for_plot = flow_data if show_price else None
                                            
                                            fig = st.session_state.order_flow_analyzer.create_plotly_volume_profile(
                                                volume_profile,
                                                price_data=price_data_for_plot
                                            )
                                            
                                            # Display plot
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Display statistics
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric(
                                                    "Ponto de Controle (POC)",
                                                    f"{volume_profile['poc_price']:.2f}"
                                                )
                                            
                                            with col2:
                                                st.metric(
                                                    "Value Area Low",
                                                    f"{volume_profile['va_low_price']:.2f}"
                                                )
                                            
                                            with col3:
                                                st.metric(
                                                    "Value Area High",
                                                    f"{volume_profile['va_high_price']:.2f}"
                                                )
                                                
                                            # Explanation
                                            with st.expander("O que é Volume Profile?"):
                                                st.markdown("""
                                                **Volume Profile** é uma ferramenta de análise que mostra a distribuição 
                                                de volume negociado em diferentes níveis de preço. Os componentes principais são:
                                                
                                                - **Point of Control (POC)**: Nível de preço com o maior volume negociado.
                                                - **Value Area**: Região que contém 70% do volume total negociado.
                                                - **Value Area High/Low**: Limites superior e inferior da Value Area.
                                                
                                                Traders usam o Volume Profile para identificar:
                                                - **Zonas de suporte/resistência**: Níveis com alto volume tendem a atuar como suporte/resistência.
                                                - **Áreas de valor justo**: A Value Area representa onde o mercado considera o preço "justo".
                                                - **Desequilíbrios**: Áreas com volume baixo ou inexistente indicam possíveis zonas de movimento rápido de preço.
                                                """)
                                        else:
                                            st.error("Não foi possível calcular o perfil de volume.")
                                    except Exception as e:
                                        st.error(f"Erro ao calcular perfil de volume: {str(e)}")
                        
                        elif analysis_type == "Footprint Chart":
                            st.markdown("### Footprint Chart (Volume por Nível de Preço em Cada Candle)")
                            
                            if tick_data is not None and not tick_data.empty:
                                # Parameters for Footprint
                                st.markdown("#### Parâmetros de análise")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    num_price_levels = st.slider(
                                        "Níveis de preço por candle",
                                        min_value=5,
                                        max_value=20,
                                        value=10,
                                        key="fp_levels"
                                    )
                                
                                with col2:
                                    num_candles = st.slider(
                                        "Número de candles",
                                        min_value=5,
                                        max_value=30,
                                        value=10,
                                        key="fp_candles"
                                    )
                                
                                # Button to calculate and display Footprint
                                if st.button("Gerar Footprint Chart", key="gen_footprint"):
                                    with st.spinner("Calculando dados de footprint..."):
                                        try:
                                            # Calculate Footprint data
                                            footprint_data = st.session_state.order_flow_analyzer.calculate_footprint(
                                                flow_data.tail(num_candles),
                                                tick_data,
                                                num_price_levels=num_price_levels
                                            )
                                            
                                            if footprint_data and 'footprint_data' in footprint_data:
                                                # Plot Footprint Chart
                                                fig = st.session_state.order_flow_analyzer.plot_footprint_chart(
                                                    footprint_data,
                                                    num_candles=num_candles
                                                )
                                                
                                                # Display plot
                                                st.pyplot(fig)
                                                
                                                # Explanation
                                                with st.expander("O que é Footprint Chart?"):
                                                    st.markdown("""
                                                    **Footprint Chart** (ou gráfico de pegada) é uma representação avançada de gráfico de velas que 
                                                    mostra a distribuição de volume dentro de cada candle por nível de preço, separando volumes de 
                                                    compra e venda.
                                                    
                                                    Principais características:
                                                    - **Volume por nível de preço**: Mostra exatamente onde está o interesse real dos traders.
                                                    - **Compras vs. Vendas**: Identifica qual lado (comprador ou vendedor) foi mais agressivo em cada nível.
                                                    - **Delta**: Diferença entre volume de compra e venda, indicando pressão compradora ou vendedora.
                                                    - **Point of Control (POC)**: Nível de preço dentro do candle com maior volume.
                                                    
                                                    Traders usam Footprint Charts para:
                                                    - Identificar desequilíbrios de volume dentro de cada candle
                                                    - Encontrar pontos precisos de entrada e saída
                                                    - Analisar absorção de pressão compradora/vendedora
                                                    - Identificar pontos de exaustão de movimento
                                                    """)
                                            else:
                                                st.error("Não foi possível calcular dados de footprint.")
                                        except Exception as e:
                                            st.error(f"Erro ao gerar Footprint Chart: {str(e)}")
                            else:
                                st.warning("Dados de tick não disponíveis. Selecione 'Buscar novos dados' para gerar dados de tick simulados.")
                                st.info("Em um sistema real, esses dados viriam diretamente da API de trading com dados de tick reais.")
                        
                        elif analysis_type == "Livro de Ofertas (DOM)":
                            st.markdown("### Análise de DOM (Depth of Market)")
                            
                            if dom_data is not None:
                                # Analysis options for DOM
                                dom_analysis_type = st.radio(
                                    "Visualização de DOM",
                                    options=["Heatmap Atual", "Série Temporal", "Vazios de Liquidez", "Níveis de Suporte/Resistência"],
                                    horizontal=True,
                                    key="dom_visualization"
                                )
                                
                                if dom_analysis_type == "Heatmap Atual":
                                    # Display current DOM heatmap
                                    with st.spinner("Gerando heatmap de DOM..."):
                                        try:
                                            fig = st.session_state.order_flow_analyzer.create_plotly_dom_heatmap(dom_data)
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Display key metrics
                                            col1, col2, col3, col4 = st.columns(4)
                                            
                                            with col1:
                                                st.metric(
                                                    "Preço Médio",
                                                    f"{dom_data['mid_price']:.2f}"
                                                )
                                            
                                            with col2:
                                                st.metric(
                                                    "Spread",
                                                    f"{dom_data['spread']:.2f}"
                                                )
                                            
                                            with col3:
                                                st.metric(
                                                    "Desequilíbrio",
                                                    f"{dom_data['imbalance']:.2%}"
                                                )
                                            
                                            with col4:
                                                st.metric(
                                                    "Razão Compra/Venda",
                                                    f"{dom_data['buy_sell_ratio']:.2f}"
                                                )
                                        except Exception as e:
                                            st.error(f"Erro ao gerar heatmap de DOM: {str(e)}")
                                
                                elif dom_analysis_type == "Série Temporal":
                                    # Display DOM time series
                                    if 'dom_history_df' in locals() and not dom_history_df.empty:
                                        with st.spinner("Gerando série temporal de DOM..."):
                                            try:
                                                # Select metrics to display
                                                metrics = st.multiselect(
                                                    "Métricas para visualizar",
                                                    options=['mid_price', 'spread', 'imbalance', 'buy_sell_ratio', 
                                                           'best_bid_volume', 'best_ask_volume', 'total_bid_volume', 'total_ask_volume'],
                                                    default=['mid_price', 'imbalance', 'buy_sell_ratio'],
                                                    key="dom_metrics"
                                                )
                                                
                                                if metrics:
                                                    # Create time series plot
                                                    fig = st.session_state.order_flow_analyzer.create_plotly_dom_time_series(
                                                        dom_history_df,
                                                        metrics=metrics
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.info("Selecione pelo menos uma métrica para visualizar.")
                                            except Exception as e:
                                                st.error(f"Erro ao gerar série temporal de DOM: {str(e)}")
                                    else:
                                        st.warning("Histórico de DOM não disponível.")
                                
                                elif dom_analysis_type == "Vazios de Liquidez":
                                    # Detect and display liquidity voids
                                    with st.spinner("Detectando vazios de liquidez..."):
                                        try:
                                            # Parameters for liquidity void detection
                                            min_gap_percent = st.slider(
                                                "Diferença mínima para vazios (%)",
                                                min_value=0.05,
                                                max_value=1.0,
                                                value=0.1,
                                                step=0.05,
                                                format="%.2f%%",
                                                key="liq_void_threshold"
                                            )
                                            
                                            # Detect liquidity voids
                                            liquidity_voids = st.session_state.order_flow_analyzer.detect_liquidity_voids(
                                                dom_data,
                                                min_gap_percent=min_gap_percent
                                            )
                                            
                                            if liquidity_voids:
                                                # Plot DOM with liquidity voids
                                                fig = st.session_state.order_flow_analyzer.plot_dom_with_liquidity_voids(
                                                    dom_data,
                                                    liquidity_voids
                                                )
                                                st.pyplot(fig)
                                                
                                                # Display detected liquidity voids
                                                st.markdown(f"#### Vazios de Liquidez Detectados ({len(liquidity_voids)})")
                                                
                                                for i, void in enumerate(liquidity_voids):
                                                    void_type = "Spread" if void['type'] == 'spread' else "Compra" if void['type'] == 'bid' else "Venda"
                                                    st.markdown(f"**{i+1}. {void_type}**: {void['price_low']:.2f} - {void['price_high']:.2f} ({void['gap_percent']:.2f}%)")
                                            else:
                                                st.info("Nenhum vazio de liquidez significativo detectado com o limiar atual.")
                                        except Exception as e:
                                            st.error(f"Erro ao detectar vazios de liquidez: {str(e)}")
                                
                                elif dom_analysis_type == "Níveis de Suporte/Resistência":
                                    # Identify support/resistance from DOM
                                    if 'dom_history_df' in locals() and not dom_history_df.empty:
                                        with st.spinner("Identificando níveis de suporte e resistência..."):
                                            try:
                                                # Parameters
                                                volume_threshold = st.slider(
                                                    "Limiar de volume (%)",
                                                    min_value=0.5,
                                                    max_value=5.0,
                                                    value=0.75,
                                                    step=0.25,
                                                    key="sr_threshold"
                                                )
                                                
                                                # Identify levels
                                                sr_levels = st.session_state.order_flow_analyzer.identify_support_resistance_from_dom(
                                                    dom_history_df,
                                                    volume_threshold=volume_threshold
                                                )
                                                
                                                if sr_levels and (sr_levels['support'] or sr_levels['resistance']):
                                                    # Display levels
                                                    col1, col2 = st.columns(2)
                                                    
                                                    with col1:
                                                        st.markdown("#### Níveis de Suporte")
                                                        if sr_levels['support']:
                                                            for level in sr_levels['support']:
                                                                st.markdown(f"- {level:.2f}")
                                                        else:
                                                            st.info("Nenhum nível de suporte identificado.")
                                                    
                                                    with col2:
                                                        st.markdown("#### Níveis de Resistência")
                                                        if sr_levels['resistance']:
                                                            for level in sr_levels['resistance']:
                                                                st.markdown(f"- {level:.2f}")
                                                        else:
                                                            st.info("Nenhum nível de resistência identificado.")
                                                    
                                                    # Create custom chart with price and levels
                                                    fig = go.Figure()
                                                    
                                                    # Add price series
                                                    fig.add_trace(go.Scatter(
                                                        x=dom_history_df.index,
                                                        y=dom_history_df['mid_price'],
                                                        mode='lines',
                                                        name='Preço Médio',
                                                        line=dict(color='black', width=1.5)
                                                    ))
                                                    
                                                    # Add support levels
                                                    for level in sr_levels['support']:
                                                        fig.add_shape(
                                                            type="line",
                                                            x0=dom_history_df.index[0],
                                                            x1=dom_history_df.index[-1],
                                                            y0=level,
                                                            y1=level,
                                                            line=dict(color="green", width=2, dash="dash"),
                                                        )
                                                    
                                                    # Add resistance levels
                                                    for level in sr_levels['resistance']:
                                                        fig.add_shape(
                                                            type="line",
                                                            x0=dom_history_df.index[0],
                                                            x1=dom_history_df.index[-1],
                                                            y0=level,
                                                            y1=level,
                                                            line=dict(color="red", width=2, dash="dash"),
                                                        )
                                                    
                                                    # Update layout
                                                    fig.update_layout(
                                                        title="Preço com Níveis de Suporte e Resistência",
                                                        xaxis_title="Tempo",
                                                        yaxis_title="Preço",
                                                        height=500
                                                    )
                                                    
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.info("Nenhum nível significativo detectado com o limiar atual.")
                                            except Exception as e:
                                                st.error(f"Erro ao identificar níveis de suporte e resistência: {str(e)}")
                                    else:
                                        st.warning("Histórico de DOM não disponível.")
                                
                                # Explanation of DOM analysis
                                with st.expander("O que é Análise de DOM?"):
                                    st.markdown("""
                                    **DOM (Depth of Market)** ou Profundidade de Mercado é uma representação do livro de ofertas, 
                                    mostrando ordens pendentes de compra (bids) e venda (asks) em diversos níveis de preço.
                                    
                                    A análise de DOM fornece insights sobre:
                                    - **Pressão compradora vs. vendedora**: Desequilíbrios no volume disponível de cada lado.
                                    - **Liquidez**: Quantidade de volume disponível para execução em cada nível de preço.
                                    - **Vazios de liquidez**: Áreas com pouca ou nenhuma ordem, onde o preço pode se mover rapidamente.
                                    - **Suporte e resistência**: Níveis com grande concentração de ordens.
                                    
                                    Traders usam a análise de DOM para:
                                    - Identificar direção provável do próximo movimento
                                    - Detectar manipulações de mercado
                                    - Encontrar pontos ótimos de entrada e saída
                                    - Antecipar movimentos bruscos de preço
                                    """)
                            else:
                                st.warning("Dados de DOM não disponíveis. Selecione 'Buscar novos dados' para gerar dados de DOM simulados.")
                                st.info("Em um sistema real, esses dados viriam diretamente da API de trading com dados de DOM em tempo real.")
                        
                        elif analysis_type == "Desequilíbrios de Volume":
                            st.markdown("### Análise de Desequilíbrios de Volume")
                            
                            # Parameters for volume imbalance detection
                            st.markdown("#### Parâmetros de análise")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                threshold = st.slider(
                                    "Limite de desequilíbrio (múltiplo da média)",
                                    min_value=1.5,
                                    max_value=5.0,
                                    value=2.0,
                                    step=0.5,
                                    key="vi_threshold"
                                )
                            
                            with col2:
                                window_size = st.slider(
                                    "Janela para média móvel",
                                    min_value=10,
                                    max_value=50,
                                    value=20,
                                    key="vi_window"
                                )
                            
                            # Button to detect volume imbalances
                            if st.button("Detectar Desequilíbrios", key="detect_imbalances"):
                                with st.spinner("Analisando desequilíbrios de volume..."):
                                    try:
                                        # Detect volume imbalances
                                        imbalances = st.session_state.order_flow_analyzer.detect_volume_imbalances(
                                            flow_data,
                                            threshold=threshold,
                                            window_size=window_size
                                        )
                                        
                                        if not imbalances.empty:
                                            # Display detected imbalances
                                            st.markdown(f"#### Desequilíbrios Detectados ({len(imbalances)})")
                                            
                                            # Display recent imbalances in table
                                            imbalances_display = imbalances.copy()
                                            imbalances_display = imbalances_display.reset_index()
                                            
                                            # Format for display
                                            if 'timestamp' in imbalances_display.columns:
                                                imbalances_display['timestamp'] = imbalances_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                                                imbalances_display = imbalances_display.rename(columns={'timestamp': 'Data/Hora'})
                                            
                                            display_columns = [
                                                'Data/Hora' if 'Data/Hora' in imbalances_display.columns else 'index',
                                                'close', 'volume', 'volume_ratio', 'imbalance_type', 'imbalance_strength'
                                            ]
                                            
                                            # Rename columns for display
                                            column_rename = {
                                                'close': 'Fechamento',
                                                'volume': 'Volume',
                                                'volume_ratio': 'Rel. Volume',
                                                'imbalance_type': 'Tipo',
                                                'imbalance_strength': 'Intensidade'
                                            }
                                            
                                            imbalances_display = imbalances_display.rename(columns=column_rename)
                                            
                                            # Select and reorder columns for display
                                            display_columns = [col for col in display_columns if col in imbalances_display.columns]
                                            
                                            st.dataframe(imbalances_display[display_columns])
                                            
                                            # Plot price chart with imbalances highlighted
                                            fig = go.Figure()
                                            
                                            # Add price series
                                            fig.add_trace(go.Candlestick(
                                                x=flow_data.index,
                                                open=flow_data['open'],
                                                high=flow_data['high'],
                                                low=flow_data['low'],
                                                close=flow_data['close'],
                                                name="OHLC"
                                            ))
                                            
                                            # Add buy imbalances as markers
                                            buy_imbalances = imbalances[imbalances['imbalance_direction'] > 0]
                                            if not buy_imbalances.empty:
                                                fig.add_trace(go.Scatter(
                                                    x=buy_imbalances.index,
                                                    y=buy_imbalances['low'] * 0.999,  # Slightly below for visibility
                                                    mode='markers',
                                                    marker=dict(
                                                        symbol='triangle-up',
                                                        size=12,
                                                        color='green',
                                                        line=dict(width=1, color='darkgreen')
                                                    ),
                                                    name='Desequilíbrio de Compra'
                                                ))
                                            
                                            # Add sell imbalances as markers
                                            sell_imbalances = imbalances[imbalances['imbalance_direction'] < 0]
                                            if not sell_imbalances.empty:
                                                fig.add_trace(go.Scatter(
                                                    x=sell_imbalances.index,
                                                    y=sell_imbalances['high'] * 1.001,  # Slightly above for visibility
                                                    mode='markers',
                                                    marker=dict(
                                                        symbol='triangle-down',
                                                        size=12,
                                                        color='red',
                                                        line=dict(width=1, color='darkred')
                                                    ),
                                                    name='Desequilíbrio de Venda'
                                                ))
                                            
                                            # Update layout
                                            fig.update_layout(
                                                title='Desequilíbrios de Volume',
                                                xaxis_title='Data',
                                                yaxis_title='Preço',
                                                height=600,
                                                xaxis_rangeslider_visible=False
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.info("Nenhum desequilíbrio de volume significativo detectado com os parâmetros atuais.")
                                    except Exception as e:
                                        st.error(f"Erro ao detectar desequilíbrios de volume: {str(e)}")
                            
                            # Explanation
                            with st.expander("O que são Desequilíbrios de Volume?"):
                                st.markdown("""
                                **Desequilíbrios de Volume** são candles ou períodos que mostram um volume negociado 
                                significativamente acima da média, indicando um interesse incomum dos traders naquele nível de preço.
                                
                                Principais características:
                                - **Picos de volume**: Volume muito acima da média móvel do período analisado.
                                - **Direção**: O desequilíbrio pode ser de compra (candle de alta) ou venda (candle de baixa).
                                - **Intensidade**: Quanto maior o desvio em relação à média, mais significativo o desequilíbrio.
                                
                                Traders usam desequilíbrios de volume para:
                                - Identificar possíveis reversões ou continuações de tendência
                                - Detectar entrada de grandes players no mercado
                                - Identificar níveis de preço importantes
                                - Antecipar movimentos futuros com base no interesse atual
                                """)
                else:
                    st.warning("Componente de análise de fluxo de ordens não está inicializado.")
            log_filter = st.selectbox(
                "Filter Logs",
                options=["All", "Info", "Warning", "Error", "Trades", "Signals"]
            )
            
            # Display logs
            log_file = "winfut_robot.log"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    logs = f.readlines()
                
                # Filter logs
                filtered_logs = []
                for log in logs:
                    if log_filter == "All":
                        filtered_logs.append(log)
                    elif log_filter == "Info" and "INFO" in log:
                        filtered_logs.append(log)
                    elif log_filter == "Warning" and "WARNING" in log:
                        filtered_logs.append(log)
                    elif log_filter == "Error" and "ERROR" in log:
                        filtered_logs.append(log)
                    elif log_filter == "Trades" and ("order" in log.lower() or "position" in log.lower() or "trade" in log.lower()):
                        filtered_logs.append(log)
                    elif log_filter == "Signals" and "signal" in log.lower():
                        filtered_logs.append(log)
                
                # Display logs in a text area
                st.text_area("Log Output", value="".join(filtered_logs[-500:]), height=400)
            else:
                st.info("No log file found")
            
            # Full notification history
            with st.expander("Notification History", expanded=False):
                if st.session_state.notifications:
                    for notification in reversed(st.session_state.notifications):
                        notification_type = notification["type"]
                        if notification_type == "error":
                            st.error(f"{notification['time']} - {notification['message']}")
                        elif notification_type == "warning":
                            st.warning(f"{notification['time']} - {notification['message']}")
                        elif notification_type == "success":
                            st.success(f"{notification['time']} - {notification['message']}")
                        else:
                            st.info(f"{notification['time']} - {notification['message']}")
                else:
                    st.info("No notifications")
    else:
        # Initial startup screen
        st.write("### Welcome to the WINFUT Automated Trading Robot")
        st.write("This system provides automated trading for WINFUT futures contracts using machine learning.")
        st.write("Please click 'Initialize Trading System' to get started.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Usando ícones do Streamlit em vez de imagem externa
            st.markdown("# 📈 WINFUT Trading Robot")
            st.markdown("### 🤖 Automação de negociações no mercado futuro")
        
        st.write("#### Features:")
        st.write("- Automated trading using machine learning algorithms")
        st.write("- Real-time market data analysis")
        st.write("- Advanced backtesting capabilities")
        st.write("- Risk management controls")
        st.write("- Performance tracking and reporting")


# Auto-update function to keep data fresh
def auto_update():
    """Background function to update data periodically"""
    while True:
        try:
            # Verifica se estas chaves existem no session_state primeiro
            initialized = st.session_state.get("initialized", False)
            trader_running = st.session_state.get("trader_running", False)
            
            if initialized and trader_running:
                fetch_market_data()
                # Only rerun in streamlit if trader is running
                try:
                    st.rerun()
                except:
                    pass
        except Exception as e:
            # Log error but keep thread running
            print(f"Error in auto_update thread: {str(e)}")
            pass
        
        time.sleep(60)  # Update every 60 seconds


# Start auto-update in a separate thread
if __name__ == "__main__":
    main()
    
    # Start automatic updates in a separate thread if not already running
    if "update_thread_started" not in st.session_state:
        st.session_state.update_thread_started = False
        
    if not st.session_state.update_thread_started:
        update_thread = threading.Thread(target=auto_update, daemon=True)
        update_thread.start()
        st.session_state.update_thread_started = True
