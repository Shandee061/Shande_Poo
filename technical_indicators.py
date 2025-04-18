"""
Módulo de indicadores técnicos avançados para análise de mercado.

Este módulo implementa diversos indicadores técnicos avançados que podem ser utilizados
para análise de mercado e tomada de decisão de trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import logging

# Setup logger
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Classe para cálculo de indicadores técnicos avançados.
    
    Esta classe implementa diversos indicadores além dos básicos (médias móveis, RSI, MACD)
    para fornecer uma análise técnica mais abrangente.
    """
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos disponíveis.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com todos os indicadores adicionados
        """
        if df.empty:
            return df
            
        # Verificar se temos as colunas necessárias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Coluna {col} não encontrada no DataFrame. Indicadores não serão calculados.")
                return df
        
        # Guardar as colunas originais para verificação de erros
        original_columns = df.columns.tolist()
        
        # Calcular os indicadores
        try:
            # Indicadores básicos
            df = TechnicalIndicators.add_moving_averages(df)
            df = TechnicalIndicators.add_rsi(df)
            df = TechnicalIndicators.add_macd(df)
            
            # Indicadores avançados
            df = TechnicalIndicators.add_bollinger_bands(df)
            df = TechnicalIndicators.add_keltner_channel(df)
            df = TechnicalIndicators.add_ichimoku_cloud(df)
            df = TechnicalIndicators.add_stochastic_oscillator(df)
            df = TechnicalIndicators.add_adx(df)
            df = TechnicalIndicators.add_atr(df)
            df = TechnicalIndicators.add_obv(df)
            df = TechnicalIndicators.add_fibonacci_retracement(df)
            df = TechnicalIndicators.add_pivot_points(df)
            df = TechnicalIndicators.add_vwap(df)
            df = TechnicalIndicators.add_momentum_indicators(df)
            df = TechnicalIndicators.add_volatility_indicators(df)
            df = TechnicalIndicators.add_trend_indicators(df)
            
            return df
            
        except Exception as e:
            # Em caso de erro, retornar DataFrame original
            logger.error(f"Erro ao calcular indicadores técnicos: {str(e)}")
            # Manter apenas as colunas originais
            return df[original_columns]
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona diversas médias móveis ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com médias móveis adicionadas
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Definir períodos para médias móveis
        sma_periods = [5, 9, 20, 50, 100, 200]
        ema_periods = [5, 9, 13, 21, 34, 55, 89]  # Sequência de Fibonacci
        wma_periods = [5, 10, 20]  # Weighted Moving Average
        hull_periods = [9, 16, 25]  # Hull Moving Average
        
        # Calcular SMA (Simple Moving Average)
        for period in sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Calcular EMA (Exponential Moving Average)
        for period in ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Calcular WMA (Weighted Moving Average)
        for period in wma_periods:
            weights = np.arange(1, period + 1)
            df[f'wma_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        
        # Calcular Hull Moving Average
        for period in hull_periods:
            half_period = int(period / 2)
            sqrt_period = int(np.sqrt(period))
            
            wma1 = df['close'].rolling(window=half_period).apply(
                lambda x: np.sum(np.arange(1, half_period + 1) * x) / np.sum(np.arange(1, half_period + 1)), 
                raw=True
            )
            wma2 = df['close'].rolling(window=period).apply(
                lambda x: np.sum(np.arange(1, period + 1) * x) / np.sum(np.arange(1, period + 1)), 
                raw=True
            )
            
            # HMA = WMA[2*WMA(n/2) - WMA(n)], sqrt(n)]
            df[f'hma_{period}'] = (2 * wma1 - wma2).rolling(window=sqrt_period).apply(
                lambda x: np.sum(np.arange(1, sqrt_period + 1) * x) / np.sum(np.arange(1, sqrt_period + 1)), 
                raw=True
            )
        
        # Média móvel triangular (TMA)
        period = 20
        df[f'tma_{period}'] = df[f'sma_{period}'].rolling(window=period).mean()
        
        # Média móvel de volume ponderado (VWMA)
        period = 20
        df[f'vwma_{period}'] = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: List[int] = [6, 14, 21]) -> pd.DataFrame:
        """
        Adiciona o Índice de Força Relativa (RSI) ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            periods: Lista de períodos para cálculo do RSI
            
        Returns:
            DataFrame com RSI adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        for period in periods:
            # Calcular diferenças diárias
            delta = df['close'].diff()
            
            # Separar ganhos (positivos) e perdas (negativos)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calcular médias móveis de ganhos e perdas
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calcular RS (Relative Strength) e RSI
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Adiciona o MACD (Moving Average Convergence Divergence) ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            fast_period: Período da média móvel rápida
            slow_period: Período da média móvel lenta
            signal_period: Período da linha de sinal
            
        Returns:
            DataFrame com MACD adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular EMAs
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calcular MACD e sinal
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Adiciona Bandas de Bollinger ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            period: Período para cálculo da média móvel
            std_dev: Número de desvios padrão para as bandas
            
        Returns:
            DataFrame com Bandas de Bollinger adicionadas
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular média móvel
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Calcular desvio padrão
        rolling_std = df['close'].rolling(window=period).std()
        
        # Calcular bandas superior e inferior
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Calcular largura das bandas (usado como indicador de volatilidade)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Porcentagem B (Posição do preço em relação às bandas)
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def add_keltner_channel(df: pd.DataFrame, period: int = 20, atr_multiplier: float = 2.0) -> pd.DataFrame:
        """
        Adiciona Canal de Keltner ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            period: Período para cálculo da média móvel
            atr_multiplier: Multiplicador do ATR para as bandas
            
        Returns:
            DataFrame com Canal de Keltner adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular ATR (Average True Range)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calcular média móvel como linha central
        df['kc_middle'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Calcular linhas superior e inferior
        df['kc_upper'] = df['kc_middle'] + (atr * atr_multiplier)
        df['kc_lower'] = df['kc_middle'] - (atr * atr_multiplier)
        
        # Calcular largura do canal
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        return df
    
    @staticmethod
    def add_ichimoku_cloud(df: pd.DataFrame, 
                           tenkan_period: int = 9, 
                           kijun_period: int = 26, 
                           senkou_span_b_period: int = 52,
                           displacement: int = 26) -> pd.DataFrame:
        """
        Adiciona Ichimoku Cloud (Nuvem de Ichimoku) ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            tenkan_period: Período para Tenkan-sen (linha de conversão)
            kijun_period: Período para Kijun-sen (linha base)
            senkou_span_b_period: Período para Senkou Span B (um limite da nuvem)
            displacement: Período de deslocamento para projeção
            
        Returns:
            DataFrame com Ichimoku Cloud adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular Tenkan-sen (Conversion Line): Média de máxima e mínima para período
        high_tenkan = df['high'].rolling(window=tenkan_period).max()
        low_tenkan = df['low'].rolling(window=tenkan_period).min()
        df['ichimoku_tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Calcular Kijun-sen (Base Line): Média de máxima e mínima para período
        high_kijun = df['high'].rolling(window=kijun_period).max()
        low_kijun = df['low'].rolling(window=kijun_period).min()
        df['ichimoku_kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Calcular Senkou Span A (Leading Span A): Média de Tenkan-sen e Kijun-sen deslocada
        df['ichimoku_senkou_span_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(displacement)
        
        # Calcular Senkou Span B (Leading Span B): Média de máxima e mínima para período longo, deslocada
        high_senkou_b = df['high'].rolling(window=senkou_span_b_period).max()
        low_senkou_b = df['low'].rolling(window=senkou_span_b_period).min()
        df['ichimoku_senkou_span_b'] = ((high_senkou_b + low_senkou_b) / 2).shift(displacement)
        
        # Calcular Chikou Span (Lagging Span): Preço atual deslocado para trás
        df['ichimoku_chikou_span'] = df['close'].shift(-displacement)
        
        return df
    
    @staticmethod
    def add_stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
        """
        Adiciona Oscilador Estocástico ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            k_period: Período para cálculo da linha %K
            d_period: Período para cálculo da linha %D
            slowing: Período de suavização
            
        Returns:
            DataFrame com Oscilador Estocástico adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular mínimas e máximas para o período
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # Calcular %K com suavização
        k_fast = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_k'] = k_fast.rolling(window=slowing).mean()
        
        # Calcular %D (média móvel de %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Adiciona o Índice Direcional Médio (ADX) ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            period: Período para cálculo do ADX
            
        Returns:
            DataFrame com ADX adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular True Range (TR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calcular movimentos direcionais (+DM e -DM)
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        df['+dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        
        df['-dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Calcular médias móveis suavizadas
        df['tr_{period}'] = df['tr'].rolling(window=period).mean()
        df['+dm_{period}'] = df['+dm'].rolling(window=period).mean()
        df['-dm_{period}'] = df['-dm'].rolling(window=period).mean()
        
        # Calcular indicadores direcionais
        df['+di_{period}'] = 100 * df[f'+dm_{period}'] / df[f'tr_{period}']
        df['-di_{period}'] = 100 * df[f'-dm_{period}'] / df[f'tr_{period}']
        
        # Calcular diferença direcional e soma direcional
        df['di_diff'] = abs(df[f'+di_{period}'] - df[f'-di_{period}'])
        df['di_sum'] = df[f'+di_{period}'] + df[f'-di_{period}']
        
        # Calcular DX e ADX
        df['dx'] = 100 * df['di_diff'] / df['di_sum']
        df[f'adx_{period}'] = df['dx'].rolling(window=period).mean()
        
        # Limpar colunas intermediárias
        cols_to_drop = ['tr1', 'tr2', 'tr3', 'up_move', 'down_move', 
                        '+dm', '-dm', 'tr_{period}', '+dm_{period}', '-dm_{period}',
                        'di_diff', 'di_sum', 'dx']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
        """
        Adiciona Average True Range (ATR) ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            periods: Lista de períodos para cálculo do ATR
            
        Returns:
            DataFrame com ATR adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular o True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calcular ATR para cada período
        for period in periods:
            df[f'atr_{period}'] = tr.rolling(window=period).mean()
            
            # Adicionar ATR em percentual (em relação ao preço)
            df[f'atr_percent_{period}'] = 100 * df[f'atr_{period}'] / df['close']
        
        return df
    
    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona On Balance Volume (OBV) ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com OBV adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Inicializar a série OBV
        df['obv'] = 0
        
        # Calcular mudança diária no preço
        price_change = df['close'].diff()
        
        # Aplicar regras de cálculo do OBV
        # Se preço subiu, adicionar volume
        # Se preço desceu, subtrair volume
        # Se preço manteve, não alterar OBV
        for i in range(1, len(df)):
            if price_change.iloc[i] > 0:
                df.loc[df.index[i], 'obv'] = df.loc[df.index[i-1], 'obv'] + df.loc[df.index[i], 'volume']
            elif price_change.iloc[i] < 0:
                df.loc[df.index[i], 'obv'] = df.loc[df.index[i-1], 'obv'] - df.loc[df.index[i], 'volume']
            else:
                df.loc[df.index[i], 'obv'] = df.loc[df.index[i-1], 'obv']
        
        # Adicionar média móvel do OBV
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        return df
    
    @staticmethod
    def add_fibonacci_retracement(df: pd.DataFrame, period: int = 120) -> pd.DataFrame:
        """
        Adiciona níveis de Retracement de Fibonacci com base em máximos e mínimos recentes.
        
        Args:
            df: DataFrame com dados OHLCV
            period: Período para identificar máximo e mínimo
            
        Returns:
            DataFrame com níveis de Fibonacci adicionados
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Calcular níveis de Fibonacci utilizando janelas deslizantes
        for i in range(period, len(df)):
            # Selecionar janela de dados
            window = df.iloc[i-period:i]
            
            # Encontrar máximo e mínimo no período
            max_price = window['high'].max()
            min_price = window['low'].min()
            
            # Calcular a diferença entre máximo e mínimo
            diff = max_price - min_price
            
            # Calcular os níveis de Fibonacci (depende da tendência)
            if window['close'].iloc[-1] >= window['close'].iloc[0]:  # Tendência de alta
                df.loc[df.index[i], 'fib_0'] = min_price
                df.loc[df.index[i], 'fib_0.236'] = min_price + 0.236 * diff
                df.loc[df.index[i], 'fib_0.382'] = min_price + 0.382 * diff
                df.loc[df.index[i], 'fib_0.5'] = min_price + 0.5 * diff
                df.loc[df.index[i], 'fib_0.618'] = min_price + 0.618 * diff
                df.loc[df.index[i], 'fib_0.786'] = min_price + 0.786 * diff
                df.loc[df.index[i], 'fib_1'] = max_price
            else:  # Tendência de baixa
                df.loc[df.index[i], 'fib_0'] = max_price
                df.loc[df.index[i], 'fib_0.236'] = max_price - 0.236 * diff
                df.loc[df.index[i], 'fib_0.382'] = max_price - 0.382 * diff
                df.loc[df.index[i], 'fib_0.5'] = max_price - 0.5 * diff
                df.loc[df.index[i], 'fib_0.618'] = max_price - 0.618 * diff
                df.loc[df.index[i], 'fib_0.786'] = max_price - 0.786 * diff
                df.loc[df.index[i], 'fib_1'] = min_price
        
        return df
    
    @staticmethod
    def add_pivot_points(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Adiciona Pontos de Pivô (Pivot Points) calculados diariamente.
        
        Args:
            df: DataFrame com dados OHLCV
            method: Método de cálculo ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
            
        Returns:
            DataFrame com Pontos de Pivô adicionados
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Adicionar colunas para Pontos de Pivô
        pivot_columns = ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']
        for col in pivot_columns:
            df[f'pp_{col}'] = np.nan
        
        # Calcular pontos de pivô para cada dia
        # Assuma que o DataFrame está em ordem cronológica
        for i in range(1, len(df)):
            prev_high = df['high'].iloc[i-1]
            prev_low = df['low'].iloc[i-1]
            prev_close = df['close'].iloc[i-1]
            
            if method == 'standard':
                # Pivot Point padrão
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = 2 * pivot - prev_low
                s1 = 2 * pivot - prev_high
                r2 = pivot + (prev_high - prev_low)
                s2 = pivot - (prev_high - prev_low)
                r3 = pivot + 2 * (prev_high - prev_low)
                s3 = pivot - 2 * (prev_high - prev_low)
                
            elif method == 'fibonacci':
                # Pivot Point Fibonacci
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = pivot + 0.382 * (prev_high - prev_low)
                s1 = pivot - 0.382 * (prev_high - prev_low)
                r2 = pivot + 0.618 * (prev_high - prev_low)
                s2 = pivot - 0.618 * (prev_high - prev_low)
                r3 = pivot + 1.0 * (prev_high - prev_low)
                s3 = pivot - 1.0 * (prev_high - prev_low)
                
            elif method == 'woodie':
                # Woodie Pivot Points
                pivot = (prev_high + prev_low + 2 * prev_close) / 4
                r1 = 2 * pivot - prev_low
                s1 = 2 * pivot - prev_high
                r2 = pivot + (prev_high - prev_low)
                s2 = pivot - (prev_high - prev_low)
                r3 = r1 + (prev_high - prev_low)
                s3 = s1 - (prev_high - prev_low)
                
            elif method == 'camarilla':
                # Camarilla Pivot Points
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
                s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
                r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
                s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
                r3 = prev_close + 1.1 * (prev_high - prev_low) / 4
                s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
                
            elif method == 'demark':
                # DeMark Pivot Points
                # Para o método DeMark, precisamos do preço de abertura, que pode não estar disponível
                # Vamos usar preço de fechamento anterior como aproximação se não tivermos preço de abertura
                x = prev_high + prev_low + 2 * prev_close
                pivot = x / 4
                r1 = x / 2 - prev_low
                s1 = x / 2 - prev_high
                # DeMark não define R2, R3, S2, S3 tradicionalmente
                r2 = r1
                r3 = r1
                s2 = s1
                s3 = s1
            
            else:
                # Método padrão como fallback
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = 2 * pivot - prev_low
                s1 = 2 * pivot - prev_high
                r2 = pivot + (prev_high - prev_low)
                s2 = pivot - (prev_high - prev_low)
                r3 = pivot + 2 * (prev_high - prev_low)
                s3 = pivot - 2 * (prev_high - prev_low)
            
            # Atribuir valores ao DataFrame
            df.loc[df.index[i], 'pp_pivot'] = pivot
            df.loc[df.index[i], 'pp_r1'] = r1
            df.loc[df.index[i], 'pp_r2'] = r2
            df.loc[df.index[i], 'pp_r3'] = r3
            df.loc[df.index[i], 'pp_s1'] = s1
            df.loc[df.index[i], 'pp_s2'] = s2
            df.loc[df.index[i], 'pp_s3'] = s3
        
        return df
    
    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona Volume Weighted Average Price (VWAP) ao DataFrame.
        Assumi que o DataFrame contém uma coluna de data/hora e representa dados de apenas um dia.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com VWAP adicionado
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Verificar se temos coluna datetime para verificar dias
        has_datetime = False
        if 'datetime' in df.columns or (df.index.name == 'datetime') or isinstance(df.index, pd.DatetimeIndex):
            has_datetime = True
            
        if has_datetime:
            # Agrupar por dia
            df['date'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['datetime']).dt.date
            
            # Calcular típico (typical price) e volume típico acumulado
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['tp_x_vol'] = df['typical_price'] * df['volume']
            
            # Calcular VWAP para cada dia
            df['cum_tp_x_vol'] = df.groupby('date')['tp_x_vol'].cumsum()
            df['cum_vol'] = df.groupby('date')['volume'].cumsum()
            df['vwap'] = df['cum_tp_x_vol'] / df['cum_vol']
            
            # Limpar colunas intermediárias
            df = df.drop(columns=['date', 'typical_price', 'tp_x_vol', 'cum_tp_x_vol', 'cum_vol'], errors='ignore')
        else:
            # Se não tivermos data, calcular VWAP para todo o DataFrame
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['tp_x_vol'] = df['typical_price'] * df['volume']
            df['cum_tp_x_vol'] = df['tp_x_vol'].cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = df['cum_tp_x_vol'] / df['cum_vol']
            
            # Limpar colunas intermediárias
            df = df.drop(columns=['typical_price', 'tp_x_vol', 'cum_tp_x_vol', 'cum_vol'], errors='ignore')
            
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores de momentum ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com indicadores de momentum adicionados
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # ROC (Rate of Change)
        for period in [9, 14, 21]:
            df[f'roc_{period}'] = 100 * (df['close'] / df['close'].shift(period) - 1)
        
        # MFI (Money Flow Index) - similar ao RSI, mas leva em conta volume
        period = 14
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Determinar dinheiro positivo e negativo
        pos_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
        neg_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
        
        # Calcular razão de fluxo de dinheiro
        pos_flow_sum = pd.Series(pos_flow).rolling(window=period).sum()
        neg_flow_sum = pd.Series(neg_flow).rolling(window=period).sum()
        
        # Calcular MFI
        money_ratio = pos_flow_sum / neg_flow_sum
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # CCI (Commodity Channel Index)
        period = 20
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_dev = abs(typical_price - typical_price.rolling(window=period).mean()).rolling(window=period).mean()
        df['cci'] = (typical_price - typical_price.rolling(window=period).mean()) / (0.015 * mean_dev)
        
        # Williams %R
        period = 14
        df['williams_r'] = -100 * (df['high'].rolling(window=period).max() - df['close']) / (df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min())
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores de volatilidade ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com indicadores de volatilidade adicionados
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Historical Volatility (HV)
        for period in [10, 20, 30]:
            # Retornos logarítmicos
            log_returns = np.log(df['close'] / df['close'].shift(1))
            # Desvio padrão dos retornos (volatilidade histórica)
            df[f'hist_vol_{period}'] = log_returns.rolling(window=period).std() * np.sqrt(252)  # Anualizado
        
        # Normalized Average True Range (NATR)
        period = 14
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        df['natr'] = 100 * atr / df['close']  # Normalizar pelo preço de fechamento
        
        # Chaikin Volatility
        period = 10
        atr_period = 10
        volatility_change = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
        df['chaikin_vol'] = 100 * (volatility_change.rolling(window=atr_period).mean() / volatility_change.rolling(window=atr_period).mean().shift(atr_period) - 1)
        
        return df
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores de tendência ao DataFrame.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com indicadores de tendência adicionados
        """
        # Clonar o DataFrame para evitar modificações no original
        df = df.copy()
        
        # Aroon Indicator
        period = 25
        df['aroon_up'] = 100 * (period - df['high'].rolling(window=period + 1).apply(lambda x: x.argmax(), raw=True)) / period
        df['aroon_down'] = 100 * (period - df['low'].rolling(window=period + 1).apply(lambda x: x.argmin(), raw=True)) / period
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        
        # TRIX (Triple Exponential Average)
        period = 15
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        df['trix'] = 100 * (ema3 / ema3.shift(1) - 1)
        
        # Mass Index
        period = 9
        ema_high_low = (df['high'] - df['low']).ewm(span=period, adjust=False).mean()
        ema_ema_high_low = ema_high_low.ewm(span=period, adjust=False).mean()
        mass = ema_high_low / ema_ema_high_low
        df['mass_index'] = mass.rolling(window=25).sum()
        
        # Vortex Indicator
        period = 14
        plus_vm = abs(df['high'] - df['low'].shift())
        minus_vm = abs(df['low'] - df['high'].shift())
        plus_vm_sum = plus_vm.rolling(window=period).sum()
        minus_vm_sum = minus_vm.rolling(window=period).sum()
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).sum()
        df['vortex_pos'] = plus_vm_sum / atr
        df['vortex_neg'] = minus_vm_sum / atr
        df['vortex_diff'] = df['vortex_pos'] - df['vortex_neg']
        
        # KST (Know Sure Thing)
        rcma1 = 100 * (df['close'] / df['close'].shift(10) - 1).rolling(window=10).mean()
        rcma2 = 100 * (df['close'] / df['close'].shift(15) - 1).rolling(window=10).mean()
        rcma3 = 100 * (df['close'] / df['close'].shift(20) - 1).rolling(window=10).mean()
        rcma4 = 100 * (df['close'] / df['close'].shift(30) - 1).rolling(window=15).mean()
        df['kst'] = 1 * rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4
        df['kst_signal'] = df['kst'].rolling(window=9).mean()
        
        return df