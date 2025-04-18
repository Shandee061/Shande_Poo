"""
Module for processing market data and generating features for ML models.
"""
import numpy as np
import pandas as pd
import talib
import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging

from config import TECHNICAL_INDICATORS, ML_PARAMS
from technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to process and prepare data for the trading strategy and ML models"""
    
    def __init__(self):
        self.lookback_period = ML_PARAMS["lookback_period"]
        # Technical indicator parameters are directly read from config.py
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators to the data frame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying original data
        result = df.copy()
        
        try:
            # Primeiro, tente usar o método tradicional com TA-Lib para indicadores básicos
            try:
                # Simple Moving Averages
                result['sma_fast'] = talib.SMA(result['close'].values, timeperiod=TECHNICAL_INDICATORS["sma_fast"])
                result['sma_slow'] = talib.SMA(result['close'].values, timeperiod=TECHNICAL_INDICATORS["sma_slow"])
                
                # Exponential Moving Averages
                result['ema_fast'] = talib.EMA(result['close'].values, timeperiod=TECHNICAL_INDICATORS["ema_fast"])
                result['ema_slow'] = talib.EMA(result['close'].values, timeperiod=TECHNICAL_INDICATORS["ema_slow"])
                result['ema_200'] = talib.EMA(result['close'].values, timeperiod=TECHNICAL_INDICATORS["ema_200"])
                
                # Relative Strength Index (RSI)
                result['rsi'] = talib.RSI(result['close'].values, timeperiod=TECHNICAL_INDICATORS["rsi_period"])
                
                # Moving Average Convergence Divergence (MACD)
                macd, macd_signal, macd_hist = talib.MACD(
                    result['close'].values, 
                    fastperiod=TECHNICAL_INDICATORS["macd_fast"], 
                    slowperiod=TECHNICAL_INDICATORS["macd_slow"], 
                    signalperiod=TECHNICAL_INDICATORS["macd_signal"]
                )
                result['macd'] = macd
                result['macd_signal'] = macd_signal
                result['macd_hist'] = macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    result['close'].values,
                    timeperiod=TECHNICAL_INDICATORS["bbands_period"],
                    nbdevup=TECHNICAL_INDICATORS["bbands_std_dev"],
                    nbdevdn=TECHNICAL_INDICATORS["bbands_std_dev"]
                )
                result['bb_upper'] = bb_upper
                result['bb_middle'] = bb_middle
                result['bb_lower'] = bb_lower
                result['bb_width'] = (bb_upper - bb_lower) / bb_middle
                
                # Stochastic Oscillator
                result['slowk'], result['slowd'] = talib.STOCH(
                    result['high'].values,
                    result['low'].values,
                    result['close'].values,
                    fastk_period=TECHNICAL_INDICATORS["stoch_k_period"],
                    slowk_period=TECHNICAL_INDICATORS["stoch_d_period"],
                    slowk_matype=0,
                    slowd_period=TECHNICAL_INDICATORS["stoch_d_period"],
                    slowd_matype=0
                )
                
                # Average Directional Index (ADX)
                result['adx'] = talib.ADX(
                    result['high'].values,
                    result['low'].values,
                    result['close'].values,
                    timeperiod=TECHNICAL_INDICATORS["adx_period"]
                )
                
                # Commodity Channel Index (CCI)
                result['cci'] = talib.CCI(
                    result['high'].values,
                    result['low'].values,
                    result['close'].values,
                    timeperiod=TECHNICAL_INDICATORS["cci_period"]
                )
                
                # Average True Range (ATR)
                result['atr'] = talib.ATR(
                    result['high'].values,
                    result['low'].values,
                    result['close'].values,
                    timeperiod=TECHNICAL_INDICATORS["atr_period"]
                )
                
                # On-Balance Volume (OBV)
                result['obv'] = talib.OBV(
                    result['close'].values,
                    result['volume'].values
                )
                
                # Momentum
                result['momentum'] = talib.MOM(result['close'].values, timeperiod=10)
                
                # Rate of Change (ROC)
                result['roc'] = talib.ROC(result['close'].values, timeperiod=10)
                
                # Williams %R
                result['willr'] = talib.WILLR(
                    result['high'].values,
                    result['low'].values,
                    result['close'].values,
                    timeperiod=14
                )
                
                logger.info("TA-Lib indicadores calculados com sucesso.")
            except Exception as talib_error:
                logger.warning(f"Erro ao calcular indicadores com TA-Lib: {str(talib_error)}. Usando alternativas.")
            
            # Agora, adicione indicadores avançados com nossa implementação personalizada
            try:
                # Aplicar nossa classe de indicadores técnicos avançados
                logger.info("Calculando indicadores técnicos avançados...")
                
                # Calcular todos os indicadores avançados
                result = TechnicalIndicators.calculate_all(result)
                
                # Verificar o número de indicadores adicionados
                num_indicators = len(result.columns) - len(df.columns)
                logger.info(f"Adicionados {num_indicators} indicadores técnicos avançados.")
                
            except Exception as adv_error:
                logger.error(f"Erro ao calcular indicadores avançados: {str(adv_error)}")
            
            # Calcular indicadores mais básicos que usam apenas pandas
            # Calculate price change percentages
            result['price_change_1'] = result['close'].pct_change(1)
            result['price_change_5'] = result['close'].pct_change(5)
            result['price_change_10'] = result['close'].pct_change(10)
            
            # Volume changes
            result['volume_change_1'] = result['volume'].pct_change(1)
            result['volume_change_5'] = result['volume'].pct_change(5)
            
            # Volatility (Standard Deviation of closing prices)
            result['volatility'] = result['close'].rolling(window=20).std()
            
            # Moving Average Crossovers (se existirem)
            if 'sma_fast' in result.columns and 'sma_slow' in result.columns:
                result['sma_cross'] = (result['sma_fast'] > result['sma_slow']).astype(int)
                
            if 'ema_fast' in result.columns and 'ema_slow' in result.columns:
                result['ema_cross'] = (result['ema_fast'] > result['ema_slow']).astype(int)
            
            # MACD Crossover (se existir)
            if 'macd' in result.columns and 'macd_signal' in result.columns:
                result['macd_cross'] = (result['macd'] > result['macd_signal']).astype(int)
            
            # Price position relative to Bollinger Bands (se existirem)
            if all(col in result.columns for col in ['bb_upper', 'bb_lower', 'close']):
                result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
            
            # Price to SMA ratio (se existirem)
            if 'sma_fast' in result.columns:
                result['price_to_sma_fast'] = result['close'] / result['sma_fast']
                
            if 'sma_slow' in result.columns:
                result['price_to_sma_slow'] = result['close'] / result['sma_slow']
            
        except Exception as e:
            logger.error(f"Erro geral ao adicionar indicadores técnicos: {str(e)}")
            
        # Fill NaN values that result from calculations
        result.fillna(method='bfill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares features and targets for ML models.
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            
        Returns:
            Tuple of (features_df, target_series)
        """
        data = df.copy()
        
        # Add future price changes as target
        horizon = ML_PARAMS["prediction_horizon"]
        data[f'future_return_{horizon}'] = data['close'].pct_change(horizon).shift(-horizon)
        
        # Create binary target (1 for price increase, 0 for decrease)
        data['target'] = (data[f'future_return_{horizon}'] > 0).astype(int)
        
        # Drop rows with NaN values from calculations
        data.dropna(inplace=True)
        
        # Select features (all except target and OHLCV)
        feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 
                                                                      'target', f'future_return_{horizon}']]
        
        features = data[feature_columns]
        target = data['target']
        
        return features, target
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using Min-Max scaling.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Normalized features DataFrame
        """
        result = features.copy()
        
        for column in result.columns:
            min_val = result[column].min()
            max_val = result[column].max()
            if max_val > min_val:
                result[column] = (result[column] - min_val) / (max_val - min_val)
            else:
                result[column] = 0  # Handle constant features
        
        return result
    
    def create_rolling_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates rolling window features for time series forecasting.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with rolling window features
        """
        window_size = self.lookback_period
        result = df.copy()
        
        # Select only numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate rolling statistics
        for col in numerical_columns:
            # Rolling mean
            result[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
            
            # Rolling std
            result[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
            
            # Rolling min
            result[f'{col}_rolling_min'] = df[col].rolling(window=window_size).min()
            
            # Rolling max
            result[f'{col}_rolling_max'] = df[col].rolling(window=window_size).max()
        
        # Fill NaN values
        result.fillna(method='bfill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def add_order_book_features(self, df: pd.DataFrame, order_book_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Adds order book features to the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            order_book_data: List of order book snapshots
            
        Returns:
            DataFrame with added order book features
        """
        result = df.copy()
        
        if not order_book_data:
            logger.warning("No order book data available")
            return result
        
        # Convert order book data to DataFrame
        ob_df = pd.DataFrame(order_book_data)
        ob_df['datetime'] = pd.to_datetime(ob_df['timestamp'])
        ob_df.set_index('datetime', inplace=True)
        
        # Merge with price data based on timestamp
        result = pd.merge_asof(
            result, 
            ob_df, 
            left_index=True, 
            right_index=True, 
            direction='forward'
        )
        
        # Add order book features
        if 'bids' in result.columns and 'asks' in result.columns:
            # Calculate bid-ask spread
            result['bid_ask_spread'] = result.apply(
                lambda row: float(row['asks'][0]['price']) - float(row['bids'][0]['price']) 
                if isinstance(row['asks'], list) and isinstance(row['bids'], list) 
                and len(row['asks']) > 0 and len(row['bids']) > 0 
                else None, 
                axis=1
            )
            
            # Calculate bid and ask volumes (first 5 levels)
            result['bid_volume'] = result.apply(
                lambda row: sum([float(bid['quantity']) for bid in row['bids'][:5]]) 
                if isinstance(row['bids'], list) and len(row['bids']) >= 5 
                else None, 
                axis=1
            )
            
            result['ask_volume'] = result.apply(
                lambda row: sum([float(ask['quantity']) for ask in row['asks'][:5]]) 
                if isinstance(row['asks'], list) and len(row['asks']) >= 5 
                else None, 
                axis=1
            )
            
            # Calculate volume imbalance
            result['volume_imbalance'] = (result['bid_volume'] - result['ask_volume']) / (result['bid_volume'] + result['ask_volume'])
            
            # Drop the original bids and asks columns
            result.drop(['bids', 'asks', 'timestamp'], axis=1, errors='ignore', inplace=True)
            
        # Fill NaN values
        result.fillna(method='bfill', inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds time-based features like hour of day, day of week, etc.
        
        Args:
            df: DataFrame with datetime index or datetime column
            
        Returns:
            DataFrame with added time features
        """
        result = df.copy()
        
        # Verificar se o DataFrame tem um índice de datetime ou uma coluna datetime
        index_is_datetime = isinstance(result.index, pd.DatetimeIndex)
        has_datetime_column = 'datetime' in result.columns
        
        # Se o índice não for datetime e não houver coluna datetime, verificar se há coluna 'time'
        if not index_is_datetime and not has_datetime_column and 'time' in result.columns:
            # Renomear 'time' para 'datetime' se for uma coluna de datetime
            if pd.api.types.is_datetime64_dtype(result['time']):
                result.rename(columns={'time': 'datetime'}, inplace=True)
                has_datetime_column = True
        
        # Se o índice for datetime, resetar para obter coluna datetime
        if index_is_datetime:
            # Reset index para obter datetime como coluna
            result.reset_index(inplace=True)
            has_datetime_column = True
        
        # Se não houver coluna datetime ainda, tentar criar a partir do índice
        if not has_datetime_column:
            try:
                # Tentar criar uma coluna datetime a partir do índice
                result['datetime'] = pd.to_datetime(result.index)
            except:
                # Se falhar, criar uma série temporal básica a partir da hora atual
                logger.warning("Não foi possível criar coluna datetime. Criando série temporal simples.")
                n = len(result)
                now = datetime.datetime.now()
                # Criar timestamps com intervalos de 1 minuto
                result['datetime'] = [now - datetime.timedelta(minutes=n-i-1) for i in range(n)]
        
        # Extrair recursos temporais
        result['hour'] = result['datetime'].dt.hour
        result['minute'] = result['datetime'].dt.minute
        result['day_of_week'] = result['datetime'].dt.dayofweek
        result['day_of_month'] = result['datetime'].dt.day
        result['week_of_year'] = result['datetime'].dt.isocalendar().week
        result['month'] = result['datetime'].dt.month
        
        # Criar recursos cíclicos para hora (convertendo tempo para coordenadas circulares)
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        
        # Criar recursos cíclicos para dia da semana
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # Sinalizadores para mercado aberto, fechamento e períodos de almoço
        result['is_market_open'] = ((result['hour'] >= 9) & (result['hour'] < 17) | 
                                 ((result['hour'] == 17) & (result['minute'] <= 55))).astype(int)
        
        result['is_market_opening'] = ((result['hour'] == 9) & (result['minute'] < 30)).astype(int)
        result['is_market_closing'] = ((result['hour'] == 17) & (result['minute'] > 25)).astype(int)
        result['is_lunch_period'] = ((result['hour'] == 12) | (result['hour'] == 13) & (result['minute'] < 30)).astype(int)
        
        # Configura datetime de volta como índice
        result.set_index('datetime', inplace=True)
        
        return result
    
    def process_data(self, df: pd.DataFrame, order_book_data: List[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete data processing pipeline.
        
        Args:
            df: Raw OHLCV data
            order_book_data: Optional order book data
            
        Returns:
            Tuple of (processed_features, target)
        """
        try:
            # Verificar se o DataFrame está vazio
            if df is None or df.empty:
                logger.warning("DataFrame vazio ou None fornecido para processar")
                return pd.DataFrame(), pd.Series()
                
            # Verificar se temos as colunas OHLCV necessárias
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Colunas necessárias ausentes no DataFrame: {missing_columns}")
                # Tentar criar as colunas ausentes com valores padrão
                for col in missing_columns:
                    if col == 'open' and 'close' in df.columns:
                        df[col] = df['close']
                    elif col == 'high' and 'close' in df.columns:
                        df[col] = df['close'] * 1.001  # 0.1% acima do fechamento
                    elif col == 'low' and 'close' in df.columns:
                        df[col] = df['close'] * 0.999  # 0.1% abaixo do fechamento
                    else:
                        df[col] = 0  # Valor padrão, deve ser evitado
            
            # Verificar a coluna de volume
            if 'volume' not in df.columns:
                logger.warning("Coluna 'volume' ausente, criando com valores padrão")
                df['volume'] = 1000  # Valor de volume padrão
            
            # 1. Add technical indicators
            data = self.add_technical_indicators(df)
            
            # 2. Add time features
            data = self.add_time_features(data)
            
            # 3. Add order book features if available
            if order_book_data:
                data = self.add_order_book_features(data, order_book_data)
            
            # 4. Create rolling window features
            data = self.create_rolling_window_features(data)
            
            # 5. Prepare features and target
            features, target = self.prepare_features(data)
            
            # 6. Normalize features
            features = self.normalize_features(features)
            
            return features, target
            
        except Exception as e:
            logger.error(f"Erro no pipeline de processamento de dados: {str(e)}")
            # Retornar DataFrame e Series vazios em caso de falha
            return pd.DataFrame(), pd.Series()
