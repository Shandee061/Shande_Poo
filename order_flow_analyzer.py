"""
Módulo para análise de fluxo de ordens e profundidade de mercado.

Este módulo implementa ferramentas para análise do livro de ofertas (DOM - Depth of Market)
e fluxo de ordens, permitindo identificar pressão compradora/vendedora, pontos de
acumulação/distribuição e outros padrões em diferentes níveis de preço.
"""

import os
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração de logging
logger = logging.getLogger('order_flow_analyzer')

class OrderFlowAnalyzer:
    """
    Classe para análise de fluxo de ordens e profundidade de mercado (DOM).
    
    Esta classe fornece métodos para analisar:
    - Volume por Preço (Volume Profile)
    - Desequilíbrio de Volume (Volume Imbalance)
    - Footprint Charts (análise de volume por candle)
    - Análise de DOM (Book de Ofertas)
    """
    
    def __init__(self, 
                 data_dir: str = 'market_data',
                 cache_dir: str = 'cache',
                 dom_levels: int = 10):
        """
        Inicializa o analisador de fluxo de ordens.
        
        Args:
            data_dir: Diretório para armazenar dados de mercado
            cache_dir: Diretório para cache de análises
            dom_levels: Número de níveis de profundidade do mercado a monitorar
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.dom_levels = dom_levels
        
        # Cria diretórios se não existirem
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dicionário para armazenar cache de DOM histórico
        self.dom_cache = {}
        
        # Configuração de logging
        logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging_handler = logging.StreamHandler()
        logging_handler.setFormatter(logging.Formatter(logging_format))
        
        logger.addHandler(logging_handler)
        logger.setLevel(logging.INFO)
        logger.info('OrderFlowAnalyzer inicializado')
        
    def process_tick_data(self, 
                         tick_data: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados de tick para extrair informações relevantes sobre o fluxo de ordens.
        
        Args:
            tick_data: DataFrame com dados de tick (timestamp, price, volume, etc)
            
        Returns:
            DataFrame processado com métricas de fluxo de ordens
        """
        if tick_data is None or tick_data.empty:
            logger.warning("Dados de tick vazios")
            return pd.DataFrame()
        
        # Certificar que as colunas necessárias existem
        required_columns = ['timestamp', 'price', 'volume', 'type']
        if not all(col in tick_data.columns for col in required_columns):
            logger.error(f"Dados de tick não contêm todas as colunas necessárias: {required_columns}")
            return pd.DataFrame()
        
        # Copiar dados para evitar modificar o original
        processed_data = tick_data.copy()
        
        # Converter para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(processed_data['timestamp']):
            processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
        
        # Classificar por timestamp
        processed_data = processed_data.sort_values('timestamp')
        
        # Adicionar delta de preço
        processed_data['price_delta'] = processed_data['price'].diff()
        
        # Calcular métricas de fluxo de ordens
        # 1. Volume delta (volume de compra - volume de venda)
        buy_mask = processed_data['type'] == 'buy'
        sell_mask = processed_data['type'] == 'sell'
        
        processed_data['buy_volume'] = np.where(buy_mask, processed_data['volume'], 0)
        processed_data['sell_volume'] = np.where(sell_mask, processed_data['volume'], 0)
        processed_data['volume_delta'] = processed_data['buy_volume'] - processed_data['sell_volume']
        
        # 2. Volume cumulativo
        processed_data['cumulative_volume'] = processed_data['volume'].cumsum()
        processed_data['cumulative_volume_delta'] = processed_data['volume_delta'].cumsum()
        
        # 3. Delta por nível de preço
        price_levels = processed_data.groupby('price').agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'volume': 'sum'
        })
        price_levels['volume_delta'] = price_levels['buy_volume'] - price_levels['sell_volume']
        
        # Anexar informação de delta por nível de preço
        processed_data = processed_data.join(
            price_levels[['volume_delta']].rename(columns={'volume_delta': 'price_level_delta'}),
            on='price'
        )
        
        return processed_data
    
    def calculate_volume_profile(self, 
                                price_data: pd.DataFrame, 
                                num_bins: int = 50,
                                session_start: Optional[datetime.time] = None,
                                session_end: Optional[datetime.time] = None) -> Dict[str, Any]:
        """
        Calcula o perfil de volume (Volume Profile) para um intervalo de preços.
        
        Args:
            price_data: DataFrame com dados OHLCV
            num_bins: Número de níveis de preço para análise
            session_start: Hora de início da sessão para análise (opcional)
            session_end: Hora de fim da sessão para análise (opcional)
            
        Returns:
            Dicionário com resultados da análise de perfil de volume
        """
        if price_data is None or price_data.empty:
            logger.warning("Dados de preço vazios para cálculo de Volume Profile")
            return {}
        
        # Copiar dados para evitar modificar o original
        df = price_data.copy()
        
        # Filtrar por sessão se especificado
        if session_start is not None and session_end is not None:
            if isinstance(df.index, pd.DatetimeIndex):
                mask = (df.index.time >= session_start) & (df.index.time <= session_end)
                df = df[mask]
        
        # Verificar se ainda temos dados após filtrar
        if df.empty:
            logger.warning("Sem dados após filtragem por sessão")
            return {}
        
        # Calcular range de preços
        price_high = df['high'].max()
        price_low = df['low'].min()
        price_range = price_high - price_low
        
        # Calcular tamanho do bin (intervalo de preço)
        bin_size = price_range / num_bins
        
        # Criar bins de preço
        price_bins = np.linspace(price_low, price_high, num_bins + 1)
        
        # Inicializar contadores de volume por bin
        volume_by_price = np.zeros(num_bins)
        buy_volume_by_price = np.zeros(num_bins)
        sell_volume_by_price = np.zeros(num_bins)
        
        # Para cada candle, distribuir o volume pelos bins de preço que o candle abrange
        for idx, row in df.iterrows():
            candle_high_bin = min(int((row['high'] - price_low) / bin_size), num_bins - 1)
            candle_low_bin = max(int((row['low'] - price_low) / bin_size), 0)
            
            # Determinar se o candle é de compra ou venda
            is_buy_candle = row['close'] >= row['open']
            
            # Distribuir volume entre os bins abrangidos pelo candle
            for bin_idx in range(candle_low_bin, candle_high_bin + 1):
                # Simplesmente dividir o volume igualmente pelos bins abrangidos
                bin_volume = row['volume'] / (candle_high_bin - candle_low_bin + 1)
                volume_by_price[bin_idx] += bin_volume
                
                if is_buy_candle:
                    buy_volume_by_price[bin_idx] += bin_volume
                else:
                    sell_volume_by_price[bin_idx] += bin_volume
        
        # Calcular ponto de controle (nível com maior volume)
        poc_idx = np.argmax(volume_by_price)
        poc_price = price_low + (poc_idx + 0.5) * bin_size
        
        # Calcular Value Area (70% do volume)
        total_volume = np.sum(volume_by_price)
        target_volume = 0.7 * total_volume
        
        # Começar do POC e expandir em ambas as direções
        cumulative_volume = volume_by_price[poc_idx]
        va_low_idx = va_high_idx = poc_idx
        
        while cumulative_volume < target_volume and (va_low_idx > 0 or va_high_idx < num_bins - 1):
            # Verificar qual direção tem mais volume
            vol_below = volume_by_price[va_low_idx - 1] if va_low_idx > 0 else 0
            vol_above = volume_by_price[va_high_idx + 1] if va_high_idx < num_bins - 1 else 0
            
            if vol_below >= vol_above and va_low_idx > 0:
                va_low_idx -= 1
                cumulative_volume += vol_below
            elif va_high_idx < num_bins - 1:
                va_high_idx += 1
                cumulative_volume += vol_above
            else:
                break
        
        # Calcular preços da Value Area
        va_low_price = price_low + va_low_idx * bin_size
        va_high_price = price_low + (va_high_idx + 1) * bin_size
        
        # Preparar resultados
        result = {
            'price_bins': price_bins,
            'volume_by_price': volume_by_price,
            'buy_volume_by_price': buy_volume_by_price,
            'sell_volume_by_price': sell_volume_by_price,
            'poc_price': poc_price,
            'va_low_price': va_low_price,
            'va_high_price': va_high_price,
            'bin_size': bin_size
        }
        
        return result
    
    def detect_volume_imbalances(self, 
                                price_data: pd.DataFrame,
                                threshold: float = 2.0,
                                window_size: int = 20) -> pd.DataFrame:
        """
        Detecta desequilíbrios de volume significativos.
        
        Args:
            price_data: DataFrame com dados OHLCV
            threshold: Limite para considerar um desequilíbrio significativo
                      (múltiplo do volume médio da janela)
            window_size: Tamanho da janela para cálculo de volume médio
            
        Returns:
            DataFrame com desequilíbrios de volume detectados
        """
        if price_data is None or price_data.empty:
            logger.warning("Dados de preço vazios para detecção de desequilíbrios")
            return pd.DataFrame()
        
        # Copiar dados para evitar modificar o original
        df = price_data.copy()
        
        # Calcular volume médio móvel
        df['volume_sma'] = df['volume'].rolling(window=window_size).mean()
        
        # Identificar candles com volume acima do limiar
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['is_imbalance'] = df['volume_ratio'] > threshold
        
        # Calcular a direção do desequilíbrio (compra ou venda)
        df['imbalance_direction'] = 0
        buy_imbalance = (df['is_imbalance']) & (df['close'] > df['open'])
        sell_imbalance = (df['is_imbalance']) & (df['close'] < df['open'])
        
        df.loc[buy_imbalance, 'imbalance_direction'] = 1
        df.loc[sell_imbalance, 'imbalance_direction'] = -1
        
        # Filtrar apenas os desequilíbrios
        imbalances = df[df['is_imbalance']].copy()
        
        # Adicionar classificação e descrição
        imbalances['imbalance_type'] = np.where(
            imbalances['imbalance_direction'] > 0,
            'Compra',
            'Venda'
        )
        
        imbalances['imbalance_strength'] = np.where(
            imbalances['volume_ratio'] > 3 * threshold,
            'Forte',
            np.where(
                imbalances['volume_ratio'] > 2 * threshold,
                'Moderado',
                'Fraco'
            )
        )
        
        return imbalances
    
    def calculate_footprint(self, 
                           price_data: pd.DataFrame, 
                           tick_data: pd.DataFrame,
                           num_price_levels: int = 10) -> Dict[str, Any]:
        """
        Calcula dados para visualização de footprint chart (perfil de volume por candle).
        
        Args:
            price_data: DataFrame com dados OHLCV em periodicidade desejada (ex: 5min)
            tick_data: DataFrame com dados de tick para o mesmo período
            num_price_levels: Número de níveis de preço a dividir dentro de cada candle
            
        Returns:
            Dicionário com dados para visualização de footprint
        """
        if price_data is None or price_data.empty or tick_data is None or tick_data.empty:
            logger.warning("Dados insuficientes para cálculo de footprint")
            return {}
        
        # Certificar que tick_data tem timestamp como índice
        if 'timestamp' in tick_data.columns and not pd.api.types.is_datetime64_any_dtype(tick_data.index):
            tick_data = tick_data.set_index('timestamp')
        
        # Resultados
        footprint_data = []
        
        # Para cada candle, calcular a distribuição de volume
        for idx, candle in price_data.iterrows():
            # Filtrar ticks dentro do período do candle
            if isinstance(idx, pd.Timestamp):
                # Se o índice for timestamp, podemos usar o próximo candle para definir o fim
                if idx == price_data.index[-1]:  # Último candle
                    next_idx = idx + (idx - price_data.index[-2])  # Estimar próximo período
                else:
                    next_idx = price_data.index[price_data.index.get_loc(idx) + 1]
                
                candle_ticks = tick_data.loc[idx:next_idx - pd.Timedelta(microseconds=1)]
            else:
                # Caso não seja timestamp, teremos que usar outra abordagem
                logger.warning("Índice de price_data não é timestamp. Usando abordagem alternativa.")
                # Implementação alternativa (depende da estrutura dos dados)
                continue
            
            if candle_ticks.empty:
                continue
            
            # Definir níveis de preço dentro do candle
            price_min = candle['low']
            price_max = candle['high']
            price_step = (price_max - price_min) / num_price_levels
            
            if price_step <= 0:
                price_step = 0.01  # valor mínimo para evitar divisão por zero
                
            price_levels = np.linspace(price_min, price_max, num_price_levels + 1)
            
            # Inicializar contadores
            buy_volume_by_level = np.zeros(num_price_levels)
            sell_volume_by_level = np.zeros(num_price_levels)
            
            # Agregar volume de ticks por nível de preço
            for tick_idx, tick in candle_ticks.iterrows():
                # Determinar o nível de preço
                if 'price' not in tick:
                    continue
                    
                level_idx = min(int((tick['price'] - price_min) / price_step), num_price_levels - 1)
                level_idx = max(0, level_idx)  # garantir que não seja negativo
                
                # Agregar volume
                if 'type' in tick and 'volume' in tick:
                    if tick['type'] == 'buy':
                        buy_volume_by_level[level_idx] += tick['volume']
                    else:
                        sell_volume_by_level[level_idx] += tick['volume']
                elif 'volume' in tick:
                    # Se não tiver tipo, inferir pelo movimento do preço
                    if tick_idx > candle_ticks.index[0]:
                        prev_price = candle_ticks.loc[candle_ticks.index[candle_ticks.index.get_loc(tick_idx) - 1], 'price']
                        if tick['price'] >= prev_price:
                            buy_volume_by_level[level_idx] += tick['volume']
                        else:
                            sell_volume_by_level[level_idx] += tick['volume']
                    else:
                        # Para o primeiro tick, usar o preço de abertura como referência
                        if tick['price'] >= candle['open']:
                            buy_volume_by_level[level_idx] += tick['volume']
                        else:
                            sell_volume_by_level[level_idx] += tick['volume']
            
            # Calcular delta e imbalance por nível
            total_buy_volume = buy_volume_by_level.sum()
            total_sell_volume = sell_volume_by_level.sum()
            delta_by_level = buy_volume_by_level - sell_volume_by_level
            
            # Determinar POC (Point of Control) - nível com maior volume
            poc_level = np.argmax(buy_volume_by_level + sell_volume_by_level)
            poc_price = price_min + (poc_level + 0.5) * price_step
            
            # Adicionar aos resultados
            footprint_data.append({
                'timestamp': idx,
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
                'price_levels': price_levels,
                'buy_volume': buy_volume_by_level,
                'sell_volume': sell_volume_by_level,
                'delta': delta_by_level,
                'total_buy': total_buy_volume,
                'total_sell': total_sell_volume,
                'delta_total': total_buy_volume - total_sell_volume,
                'poc_price': poc_price
            })
        
        return {
            'footprint_data': footprint_data,
            'num_price_levels': num_price_levels
        }
    
    def process_dom_data(self, 
                        dom_data: Dict[str, Any],
                        timestamp: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """
        Processa dados de DOM (Depth of Market) para análise.
        
        Args:
            dom_data: Dicionário com dados de DOM (bid/ask prices e volumes)
            timestamp: Timestamp da captura de DOM (opcional)
            
        Returns:
            Dicionário com dados de DOM processados
        """
        if not dom_data or ('bids' not in dom_data and 'asks' not in dom_data):
            logger.warning("Dados de DOM inválidos")
            return {}
        
        # Usar timestamp atual se não for fornecido
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        # Extrair dados
        bids = dom_data.get('bids', [])
        asks = dom_data.get('asks', [])
        
        # Converter para arrays NumPy para cálculos mais eficientes
        bid_prices = np.array([bid[0] for bid in bids])
        bid_volumes = np.array([bid[1] for bid in bids])
        ask_prices = np.array([ask[0] for ask in asks])
        ask_volumes = np.array([ask[1] for ask in asks])
        
        # Calcular métricas de DOM
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            # Preço médio atual (mid price)
            best_bid = bid_prices[0] if len(bid_prices) > 0 else 0
            best_ask = ask_prices[0] if len(ask_prices) > 0 else 0
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Volume nos melhores níveis
            best_bid_volume = bid_volumes[0] if len(bid_volumes) > 0 else 0
            best_ask_volume = ask_volumes[0] if len(ask_volumes) > 0 else 0
            
            # Volume total de compra e venda
            total_bid_volume = np.sum(bid_volumes)
            total_ask_volume = np.sum(ask_volumes)
            
            # Imbalance ratio (desequilíbrio)
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
            
            # Pressão compradora/vendedora (razão entre volumes)
            buy_sell_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
            
            # Cumulativo de volume (para visualização de "market profile")
            cum_bid_volume = np.cumsum(bid_volumes)
            cum_ask_volume = np.cumsum(ask_volumes)
            
            # Resultado processado
            processed_dom = {
                'timestamp': timestamp,
                'bids': bids,
                'asks': asks,
                'bid_prices': bid_prices,
                'bid_volumes': bid_volumes,
                'ask_prices': ask_prices,
                'ask_volumes': ask_volumes,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': mid_price,
                'spread': spread,
                'best_bid_volume': best_bid_volume,
                'best_ask_volume': best_ask_volume,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'imbalance': imbalance,
                'buy_sell_ratio': buy_sell_ratio,
                'cum_bid_volume': cum_bid_volume,
                'cum_ask_volume': cum_ask_volume
            }
            
            # Armazenar em cache
            timestamp_key = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            self.dom_cache[timestamp_key] = processed_dom
            
            return processed_dom
        else:
            logger.warning("DOM sem dados de bid/ask")
            return {}
    
    def dom_history_to_dataframe(self) -> pd.DataFrame:
        """
        Converte o histórico de DOM para DataFrame para análise temporal.
        
        Returns:
            DataFrame com série temporal de DOM
        """
        if not self.dom_cache:
            logger.warning("Cache de DOM vazio")
            return pd.DataFrame()
        
        # Extrair métricas principais
        dom_history = []
        
        for timestamp_key, dom_data in self.dom_cache.items():
            history_entry = {
                'timestamp': dom_data['timestamp'],
                'mid_price': dom_data['mid_price'],
                'spread': dom_data['spread'],
                'best_bid': dom_data['best_bid'],
                'best_ask': dom_data['best_ask'],
                'best_bid_volume': dom_data['best_bid_volume'],
                'best_ask_volume': dom_data['best_ask_volume'],
                'total_bid_volume': dom_data['total_bid_volume'],
                'total_ask_volume': dom_data['total_ask_volume'],
                'imbalance': dom_data['imbalance'],
                'buy_sell_ratio': dom_data['buy_sell_ratio']
            }
            dom_history.append(history_entry)
        
        # Converter para DataFrame
        dom_df = pd.DataFrame(dom_history)
        
        # Garantir que timestamp seja o índice e esteja ordenado
        if 'timestamp' in dom_df.columns:
            dom_df = dom_df.set_index('timestamp').sort_index()
        
        return dom_df
    
    def detect_liquidity_voids(self, 
                              dom_data: Dict[str, Any],
                              min_gap_percent: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detecta áreas de liquidez reduzida (gaps) no livro de ofertas.
        
        Args:
            dom_data: Dicionário com dados de DOM processados
            min_gap_percent: Percentual mínimo de diferença entre níveis para considerar um gap
            
        Returns:
            Lista de gaps encontrados
        """
        if not dom_data:
            return []
        
        bid_prices = dom_data.get('bid_prices', [])
        ask_prices = dom_data.get('ask_prices', [])
        bid_volumes = dom_data.get('bid_volumes', [])
        ask_volumes = dom_data.get('ask_volumes', [])
        
        # Lista para armazenar gaps encontrados
        liquidity_voids = []
        
        # Verificar gaps entre níveis de preço
        # Bids (compra)
        for i in range(1, len(bid_prices)):
            # Calcular diferença percentual
            price_gap = (bid_prices[i-1] - bid_prices[i]) / bid_prices[i-1]
            
            if price_gap > min_gap_percent:
                liquidity_voids.append({
                    'type': 'bid',
                    'price_high': bid_prices[i-1],
                    'price_low': bid_prices[i],
                    'gap_percent': price_gap * 100,
                    'volume_above': bid_volumes[i-1],
                    'volume_below': bid_volumes[i]
                })
        
        # Asks (venda)
        for i in range(1, len(ask_prices)):
            # Calcular diferença percentual
            price_gap = (ask_prices[i] - ask_prices[i-1]) / ask_prices[i-1]
            
            if price_gap > min_gap_percent:
                liquidity_voids.append({
                    'type': 'ask',
                    'price_low': ask_prices[i-1],
                    'price_high': ask_prices[i],
                    'gap_percent': price_gap * 100,
                    'volume_below': ask_volumes[i-1],
                    'volume_above': ask_volumes[i]
                })
        
        # Gap entre melhor bid e melhor ask (spread)
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            spread_gap = (ask_prices[0] - bid_prices[0]) / bid_prices[0]
            
            if spread_gap > min_gap_percent:
                liquidity_voids.append({
                    'type': 'spread',
                    'price_low': bid_prices[0],
                    'price_high': ask_prices[0],
                    'gap_percent': spread_gap * 100,
                    'volume_below': bid_volumes[0],
                    'volume_above': ask_volumes[0]
                })
        
        return liquidity_voids
    
    def identify_support_resistance_from_dom(self, 
                                           dom_history: pd.DataFrame,
                                           volume_threshold: float = 0.75,
                                           window_size: int = 50) -> Dict[str, List[float]]:
        """
        Identifica níveis de suporte e resistência baseados em acumulação de volume no DOM.
        
        Args:
            dom_history: DataFrame com histórico de DOM
            volume_threshold: Limite percentual para considerar acumulação significativa
            window_size: Tamanho da janela para análise
            
        Returns:
            Dicionário com níveis de suporte e resistência identificados
        """
        if dom_history.empty:
            logger.warning("Histórico de DOM vazio")
            return {'support': [], 'resistance': []}
        
        # Extrair últimos N registros para análise
        recent_dom = dom_history.tail(window_size)
        
        # Histograma de volumes acumulados por nível de preço
        volume_by_price = {}
        
        # Função para adicionar volume a um nível de preço (com aproximação)
        def add_volume_to_level(price_dict, price, volume):
            # Arredondar para evitar flutuações mínimas de preço
            rounded_price = round(price, 2)
            if rounded_price in price_dict:
                price_dict[rounded_price] += volume
            else:
                price_dict[rounded_price] = volume
        
        # Processar histórico de DOM
        for _, dom_entry in recent_dom.iterrows():
            # Obter dados de DOM para esta entrada
            timestamp_key = dom_entry.name.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dom_entry.name, pd.Timestamp) else str(dom_entry.name)
            
            dom_data = self.dom_cache.get(timestamp_key)
            if not dom_data:
                continue
                
            # Processar níveis de bid
            for i, (price, volume) in enumerate(zip(dom_data['bid_prices'], dom_data['bid_volumes'])):
                add_volume_to_level(volume_by_price, price, volume)
            
            # Processar níveis de ask
            for i, (price, volume) in enumerate(zip(dom_data['ask_prices'], dom_data['ask_volumes'])):
                add_volume_to_level(volume_by_price, price, volume)
        
        # Converter para DataFrame para facilitar análise
        volume_df = pd.DataFrame([
            {'price': price, 'volume': volume} 
            for price, volume in volume_by_price.items()
        ]).sort_values('price')
        
        if volume_df.empty:
            return {'support': [], 'resistance': []}
        
        # Normalizar volumes
        total_volume = volume_df['volume'].sum()
        volume_df['volume_ratio'] = volume_df['volume'] / total_volume
        
        # Selecionar níveis com acumulação significativa
        significant_levels = volume_df[volume_df['volume_ratio'] > volume_threshold / 100]
        
        # Último preço conhecido para classificar suporte/resistência
        last_price = dom_history['mid_price'].iloc[-1] if 'mid_price' in dom_history.columns else None
        
        # Classificar níveis como suporte ou resistência
        supports = []
        resistances = []
        
        if last_price is not None:
            for _, level in significant_levels.iterrows():
                if level['price'] < last_price:
                    supports.append(level['price'])
                else:
                    resistances.append(level['price'])
        
        return {
            'support': sorted(supports),
            'resistance': sorted(resistances)
        }
    
    # Métodos de visualização
    
    def plot_volume_profile(self, 
                           volume_profile_data: Dict[str, Any],
                           price_data: pd.DataFrame = None) -> plt.Figure:
        """
        Plota o perfil de volume (Volume Profile).
        
        Args:
            volume_profile_data: Resultado do método calculate_volume_profile
            price_data: Dados de preço para plotar junto (opcional)
            
        Returns:
            Figura matplotlib
        """
        if not volume_profile_data:
            logger.warning("Sem dados para plotar perfil de volume")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Sem dados disponíveis", ha='center', va='center')
            return fig
        
        # Extrair dados
        price_bins = volume_profile_data.get('price_bins', [])
        volume_by_price = volume_profile_data.get('volume_by_price', [])
        buy_volume_by_price = volume_profile_data.get('buy_volume_by_price', [])
        sell_volume_by_price = volume_profile_data.get('sell_volume_by_price', [])
        poc_price = volume_profile_data.get('poc_price', None)
        va_low_price = volume_profile_data.get('va_low_price', None)
        va_high_price = volume_profile_data.get('va_high_price', None)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot de preço se disponível
        if price_data is not None and not price_data.empty:
            ax2 = ax.twinx()
            ax2.plot(price_data.index, price_data['close'], color='black', alpha=0.6, linewidth=1)
        
        # Plotar volume profile horizontalmente
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        # Plotar volumes de compra e venda
        if len(buy_volume_by_price) == len(bin_centers):
            ax.barh(bin_centers, buy_volume_by_price, height=volume_profile_data.get('bin_size', 1), 
                    color='green', alpha=0.6, label='Volume de Compra')
        
        if len(sell_volume_by_price) == len(bin_centers):
            ax.barh(bin_centers, -sell_volume_by_price, height=volume_profile_data.get('bin_size', 1), 
                    color='red', alpha=0.6, label='Volume de Venda')
        
        # Plotar POC e Value Area
        if poc_price is not None:
            ax.axhline(y=poc_price, color='blue', linestyle='-', linewidth=2, 
                       label='POC (Point of Control)')
        
        if va_low_price is not None and va_high_price is not None:
            ax.axhline(y=va_low_price, color='purple', linestyle='--', linewidth=1, 
                      label='Value Area Low')
            ax.axhline(y=va_high_price, color='purple', linestyle='--', linewidth=1, 
                      label='Value Area High')
            
            # Destacar a Value Area
            ax.axhspan(va_low_price, va_high_price, alpha=0.1, color='purple')
        
        # Configurar gráfico
        ax.set_title('Perfil de Volume (Volume Profile)', fontsize=14)
        ax.set_xlabel('Volume', fontsize=12)
        ax.set_ylabel('Preço', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_footprint_chart(self, 
                            footprint_data: Dict[str, Any],
                            num_candles: int = 20) -> plt.Figure:
        """
        Plota um gráfico de footprint (distribuição de volume dentro dos candles).
        
        Args:
            footprint_data: Resultado do método calculate_footprint
            num_candles: Número de candles a mostrar
            
        Returns:
            Figura matplotlib
        """
        if not footprint_data or 'footprint_data' not in footprint_data or not footprint_data['footprint_data']:
            logger.warning("Sem dados para plotar footprint chart")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Sem dados disponíveis", ha='center', va='center')
            return fig
        
        # Extrair dados
        candles = footprint_data['footprint_data']
        
        # Limitar número de candles
        if len(candles) > num_candles:
            candles = candles[-num_candles:]
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Configuração inicial
        timestamps = [candle['timestamp'] for candle in candles]
        
        # Definir limites do eixo Y (preço)
        min_price = min([candle['low'] for candle in candles])
        max_price = max([candle['high'] for candle in candles])
        price_range = max_price - min_price
        
        ax.set_ylim(min_price - price_range * 0.05, max_price + price_range * 0.05)
        
        # Largura de cada candle
        candle_width = 0.8
        
        # Para cada candle
        for i, candle in enumerate(candles):
            # Definir posição X
            x_pos = i
            
            # Desenhar linha vertical do candle (high-low)
            ax.plot([x_pos, x_pos], [candle['low'], candle['high']], color='black', linewidth=1)
            
            # Desenhar corpo do candle
            candle_color = 'green' if candle['close'] >= candle['open'] else 'red'
            candle_bottom = min(candle['open'], candle['close'])
            candle_height = abs(candle['close'] - candle['open'])
            
            ax.add_patch(plt.Rectangle(
                (x_pos - candle_width / 2, candle_bottom),
                candle_width,
                candle_height,
                color=candle_color,
                alpha=0.3
            ))
            
            # Obter dados de footprint para este candle
            buy_volume = candle['buy_volume']
            sell_volume = candle['sell_volume']
            price_levels = candle['price_levels']
            delta = candle['delta']
            
            # Desenhar heatmap de volume dentro do candle
            for j in range(len(buy_volume)):
                # Níveis de preço para este bin
                level_bottom = price_levels[j]
                level_height = price_levels[j+1] - price_levels[j]
                
                # Calcular cores baseadas em delta
                if delta[j] > 0:  # Mais compra que venda
                    buy_alpha = min(0.8, abs(delta[j]) / max(1, max(abs(delta))))
                    ax.add_patch(plt.Rectangle(
                        (x_pos - candle_width / 2, level_bottom),
                        candle_width,
                        level_height,
                        color='green',
                        alpha=buy_alpha
                    ))
                elif delta[j] < 0:  # Mais venda que compra
                    sell_alpha = min(0.8, abs(delta[j]) / max(1, max(abs(delta))))
                    ax.add_patch(plt.Rectangle(
                        (x_pos - candle_width / 2, level_bottom),
                        candle_width,
                        level_height,
                        color='red',
                        alpha=sell_alpha
                    ))
                
                # Adicionar texto com volumes
                total_vol = buy_volume[j] + sell_volume[j]
                if total_vol > 0:
                    buy_text = f"{int(buy_volume[j])}"
                    sell_text = f"{int(sell_volume[j])}"
                    
                    text_y = level_bottom + level_height / 2
                    
                    # Texto volume de compra (esquerda)
                    ax.text(x_pos - candle_width / 4, text_y, buy_text, 
                           ha='center', va='center', fontsize=8, color='white')
                    
                    # Texto volume de venda (direita)
                    ax.text(x_pos + candle_width / 4, text_y, sell_text, 
                           ha='center', va='center', fontsize=8, color='white')
            
            # Marcar POC (Point of Control) - nível com maior volume
            ax.plot([x_pos - candle_width/2, x_pos + candle_width/2], 
                   [candle['poc_price'], candle['poc_price']],
                   color='blue', linewidth=1.5)
        
        # Configurar eixos
        ax.set_title('Footprint Chart (Distribuição de Volume por Candle)', fontsize=14)
        ax.set_xlabel('Tempo', fontsize=12)
        ax.set_ylabel('Preço', fontsize=12)
        
        # Configurar eixo X para mostrar timestamps
        ax.set_xticks(range(len(timestamps)))
        if all(isinstance(ts, (datetime.datetime, pd.Timestamp)) for ts in timestamps):
            x_labels = [ts.strftime('%H:%M') for ts in timestamps]
        else:
            x_labels = [str(ts) for ts in timestamps]
        ax.set_xticklabels(x_labels, rotation=45)
        
        ax.grid(True, alpha=0.3)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label='Volume de Compra'),
            Patch(facecolor='red', alpha=0.6, label='Volume de Venda'),
            Patch(facecolor='blue', alpha=0.6, label='POC (Nível com Maior Volume)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_dom_heatmap(self, 
                        dom_data: Dict[str, Any]) -> plt.Figure:
        """
        Plota um heatmap do DOM atual.
        
        Args:
            dom_data: Dados de DOM processados
            
        Returns:
            Figura matplotlib
        """
        if not dom_data:
            logger.warning("Sem dados para plotar heatmap de DOM")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Sem dados disponíveis", ha='center', va='center')
            return fig
        
        # Extrair dados
        bid_prices = dom_data.get('bid_prices', np.array([]))
        bid_volumes = dom_data.get('bid_volumes', np.array([]))
        ask_prices = dom_data.get('ask_prices', np.array([]))
        ask_volumes = dom_data.get('ask_volumes', np.array([]))
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalizar volumes para determinar intensidade da cor
        max_volume = max(np.max(bid_volumes) if len(bid_volumes) > 0 else 0,
                         np.max(ask_volumes) if len(ask_volumes) > 0 else 0)
        
        # Plotar volumes de compra (bid)
        for i, (price, volume) in enumerate(zip(bid_prices, bid_volumes)):
            normalized_volume = volume / max_volume if max_volume > 0 else 0
            alpha = min(0.9, normalized_volume)
            
            ax.barh(price, -volume, height=0.01, color='green', alpha=alpha)
            
            # Adicionar texto com volume
            if i < 10:  # Mostrar volume apenas para os 10 primeiros níveis
                ax.text(-volume*1.05, price, f"{int(volume)}", 
                       ha='right', va='center', fontsize=8)
        
        # Plotar volumes de venda (ask)
        for i, (price, volume) in enumerate(zip(ask_prices, ask_volumes)):
            normalized_volume = volume / max_volume if max_volume > 0 else 0
            alpha = min(0.9, normalized_volume)
            
            ax.barh(price, volume, height=0.01, color='red', alpha=alpha)
            
            # Adicionar texto com volume
            if i < 10:  # Mostrar volume apenas para os 10 primeiros níveis
                ax.text(volume*1.05, price, f"{int(volume)}", 
                       ha='left', va='center', fontsize=8)
        
        # Adicionar linha para o mid price
        if 'mid_price' in dom_data:
            ax.axhline(y=dom_data['mid_price'], color='blue', linestyle='--', 
                      linewidth=1, label=f"Mid Price: {dom_data['mid_price']:.2f}")
            
            # Mostrar spread
            if 'spread' in dom_data:
                ax.text(0, dom_data['mid_price'], 
                       f"Spread: {dom_data['spread']:.2f}", 
                       ha='center', va='bottom', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Configurar gráfico
        ax.set_title('Livro de Ofertas (DOM) - Heatmap', fontsize=14)
        ax.set_xlabel('Volume', fontsize=12)
        ax.set_ylabel('Preço', fontsize=12)
        
        # Mostrar razão de volumes e imbalance
        if 'buy_sell_ratio' in dom_data and 'imbalance' in dom_data:
            buy_sell_text = f"Buy/Sell: {dom_data['buy_sell_ratio']:.2f}"
            imbalance_text = f"Imbalance: {dom_data['imbalance']:.2%}"
            
            ax.text(0.02, 0.02, buy_sell_text + '\n' + imbalance_text,
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_dom_time_series(self, 
                            dom_history: pd.DataFrame, 
                            metrics: List[str] = None,
                            window_size: int = 100) -> plt.Figure:
        """
        Plota série temporal de métricas do DOM.
        
        Args:
            dom_history: DataFrame com histórico de DOM
            metrics: Lista de métricas para plotar
            window_size: Número de pontos a mostrar
            
        Returns:
            Figura matplotlib
        """
        if dom_history.empty:
            logger.warning("Histórico de DOM vazio")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Sem dados disponíveis", ha='center', va='center')
            return fig
        
        # Métricas padrão se não forem fornecidas
        if metrics is None:
            metrics = ['mid_price', 'imbalance', 'buy_sell_ratio']
        
        # Limitar ao tamanho da janela
        dom_recent = dom_history.tail(window_size)
        
        # Criar figura com subplots para cada métrica
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)
        
        if len(metrics) == 1:
            axes = [axes]
        
        # Plotar cada métrica
        for i, metric in enumerate(metrics):
            if metric in dom_recent.columns:
                ax = axes[i]
                
                # Plotar a série temporal
                ax.plot(dom_recent.index, dom_recent[metric], 'b-')
                
                # Adicionar título e labels
                ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12)
                ax.set_ylabel(metric, fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Para métricas específicas, adicionar visualizações especiais
                if metric == 'imbalance':
                    # Colorir áreas de desequilíbrio
                    ax.fill_between(dom_recent.index, 0, dom_recent[metric], 
                                   where=dom_recent[metric] > 0, color='green', alpha=0.3)
                    ax.fill_between(dom_recent.index, dom_recent[metric], 0, 
                                   where=dom_recent[metric] < 0, color='red', alpha=0.3)
                    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
                
                # Adicionar referência do último valor
                last_value = dom_recent[metric].iloc[-1] if len(dom_recent) > 0 else None
                if last_value is not None:
                    ax.axhline(y=last_value, color='red', linestyle='--', linewidth=1, alpha=0.5)
                    ax.text(dom_recent.index[-1], last_value, f" {last_value:.4f}", 
                           ha='left', va='center', fontsize=9)
            else:
                logger.warning(f"Métrica '{metric}' não encontrada no histórico de DOM")
        
        # Formatar eixo X para timestamps
        if isinstance(dom_recent.index, pd.DatetimeIndex):
            for ax in axes:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Título geral
        fig.suptitle('Análise Temporal do Livro de Ofertas (DOM)', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_dom_with_liquidity_voids(self, 
                                     dom_data: Dict[str, Any],
                                     liquidity_voids: List[Dict[str, Any]]) -> plt.Figure:
        """
        Plota o DOM com destaque para os vazios de liquidez.
        
        Args:
            dom_data: Dados de DOM processados
            liquidity_voids: Lista de vazios de liquidez detectados
            
        Returns:
            Figura matplotlib
        """
        # Criar gráfico base de DOM
        fig = self.plot_dom_heatmap(dom_data)
        
        # Se não há dados ou vazios de liquidez, retorna o gráfico base
        if not dom_data or not liquidity_voids:
            return fig
        
        # Obter eixo para adicionar destacques
        ax = fig.axes[0]
        
        # Destacar vazios de liquidez
        for void in liquidity_voids:
            price_low = void['price_low']
            price_high = void['price_high']
            gap_percent = void['gap_percent']
            void_type = void['type']
            
            # Cor baseada no tipo de vazio
            color = 'orange' if void_type == 'spread' else 'blue'
            
            # Destacar área
            ax.axhspan(price_low, price_high, alpha=0.2, color=color, hatch='/')
            
            # Adicionar anotação
            mid_price = (price_low + price_high) / 2
            ax.text(0, mid_price, f"{gap_percent:.1f}%", 
                   ha='center', va='center', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Atualizar título
        ax.set_title(f'Livro de Ofertas (DOM) com Vazios de Liquidez ({len(liquidity_voids)})', fontsize=14)
        
        # Adicionar legenda para vazios de liquidez
        from matplotlib.patches import Patch
        void_legend = Patch(facecolor='blue', alpha=0.2, hatch='/', label='Vazio de Liquidez')
        spread_legend = Patch(facecolor='orange', alpha=0.2, hatch='/', label='Spread (Bid-Ask)')
        
        # Combinar com legenda existente
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([void_legend, spread_legend])
        labels.extend(['Vazio de Liquidez', 'Spread (Bid-Ask)'])
        
        ax.legend(handles=handles, labels=labels, loc='upper left')
        
        plt.tight_layout()
        return fig
    
    # Métodos para Plotly (para integração com Streamlit)
    
    def create_plotly_volume_profile(self, 
                                   volume_profile_data: Dict[str, Any],
                                   price_data: pd.DataFrame = None) -> go.Figure:
        """
        Cria um gráfico de perfil de volume usando Plotly.
        
        Args:
            volume_profile_data: Resultado do método calculate_volume_profile
            price_data: Dados de preço para plotar junto (opcional)
            
        Returns:
            Figura Plotly
        """
        if not volume_profile_data:
            # Criar figura vazia com mensagem
            fig = go.Figure()
            fig.add_annotation(text="Sem dados disponíveis",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extrair dados
        price_bins = volume_profile_data.get('price_bins', [])
        volume_by_price = volume_profile_data.get('volume_by_price', [])
        buy_volume_by_price = volume_profile_data.get('buy_volume_by_price', [])
        sell_volume_by_price = volume_profile_data.get('sell_volume_by_price', [])
        poc_price = volume_profile_data.get('poc_price', None)
        va_low_price = volume_profile_data.get('va_low_price', None)
        va_high_price = volume_profile_data.get('va_high_price', None)
        
        # Criar bins centrais para plotagem
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        # Criado figura com subplots se tivermos dados de preço
        if price_data is not None and not price_data.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Adicionar série de preço
            fig.add_trace(
                go.Scatter(
                    x=price_data.index, 
                    y=price_data['close'],
                    name='Preço de Fechamento',
                    line=dict(color='black', width=1)
                ),
                secondary_y=True
            )
        else:
            fig = go.Figure()
        
        # Adicionar volumes de compra
        if len(buy_volume_by_price) == len(bin_centers):
            fig.add_trace(
                go.Bar(
                    x=buy_volume_by_price,
                    y=bin_centers,
                    orientation='h',
                    name='Volume de Compra',
                    marker=dict(color='rgba(0, 128, 0, 0.6)')
                ),
                secondary_y=False
            )
        
        # Adicionar volumes de venda (negativo para visualizar à esquerda)
        if len(sell_volume_by_price) == len(bin_centers):
            fig.add_trace(
                go.Bar(
                    x=-sell_volume_by_price,
                    y=bin_centers,
                    orientation='h',
                    name='Volume de Venda',
                    marker=dict(color='rgba(255, 0, 0, 0.6)')
                ),
                secondary_y=False
            )
        
        # Adicionar linhas para POC e Value Area
        if poc_price is not None:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[poc_price],
                    mode='markers+text',
                    name='POC (Point of Control)',
                    marker=dict(color='blue', symbol='triangle-right', size=12),
                    text=['POC'],
                    textposition='middle right'
                ),
                secondary_y=False
            )
        
        if va_low_price is not None and va_high_price is not None:
            # Value Area Low
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[va_low_price],
                    mode='markers+text',
                    name='Value Area Low',
                    marker=dict(color='purple', symbol='triangle-down', size=10),
                    text=['VAL'],
                    textposition='middle right'
                ),
                secondary_y=False
            )
            
            # Value Area High
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[va_high_price],
                    mode='markers+text',
                    name='Value Area High',
                    marker=dict(color='purple', symbol='triangle-up', size=10),
                    text=['VAH'],
                    textposition='middle right'
                ),
                secondary_y=False
            )
            
            # Adicionar área sombreada para Value Area
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[va_low_price, va_high_price],
                    fill='toself',
                    fillcolor='rgba(128, 0, 128, 0.1)',
                    line=dict(color='rgba(128, 0, 128, 0.1)'),
                    name='Value Area (70%)',
                    showlegend=False
                ),
                secondary_y=False
            )
        
        # Configurar layout
        fig.update_layout(
            title='Perfil de Volume (Volume Profile)',
            xaxis_title='Volume',
            yaxis_title='Preço',
            barmode='overlay',
            bargap=0,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1
            ),
            height=600
        )
        
        # Configurar eixos secundários se usados
        if price_data is not None and not price_data.empty:
            fig.update_yaxes(title_text="Preço", secondary_y=True)
            fig.update_yaxes(title_text="", secondary_y=False)
        
        return fig
    
    def create_plotly_dom_heatmap(self, 
                                dom_data: Dict[str, Any]) -> go.Figure:
        """
        Cria um heatmap de DOM usando Plotly.
        
        Args:
            dom_data: Dados de DOM processados
            
        Returns:
            Figura Plotly
        """
        if not dom_data:
            # Criar figura vazia com mensagem
            fig = go.Figure()
            fig.add_annotation(text="Sem dados disponíveis",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extrair dados
        bid_prices = dom_data.get('bid_prices', np.array([]))
        bid_volumes = dom_data.get('bid_volumes', np.array([]))
        ask_prices = dom_data.get('ask_prices', np.array([]))
        ask_volumes = dom_data.get('ask_volumes', np.array([]))
        
        # Criar figura
        fig = go.Figure()
        
        # Normalizar volumes para determinar intensidade da cor
        max_volume = max(np.max(bid_volumes) if len(bid_volumes) > 0 else 0,
                         np.max(ask_volumes) if len(ask_volumes) > 0 else 0)
        
        # Adicionar volumes de compra (bid)
        if len(bid_prices) > 0 and len(bid_volumes) > 0:
            fig.add_trace(
                go.Bar(
                    x=-bid_volumes,
                    y=bid_prices,
                    orientation='h',
                    name='Compra (Bid)',
                    marker=dict(
                        color='rgba(0, 128, 0, 0.6)',
                        line=dict(color='rgba(0, 128, 0, 1.0)', width=1)
                    ),
                    text=[f"{int(vol)}" for vol in bid_volumes],
                    textposition='outside',
                    hoverinfo='text',
                    hovertext=[f"Preço: {price:.2f}<br>Volume: {int(vol)}"
                              for price, vol in zip(bid_prices, bid_volumes)]
                )
            )
        
        # Adicionar volumes de venda (ask)
        if len(ask_prices) > 0 and len(ask_volumes) > 0:
            fig.add_trace(
                go.Bar(
                    x=ask_volumes,
                    y=ask_prices,
                    orientation='h',
                    name='Venda (Ask)',
                    marker=dict(
                        color='rgba(255, 0, 0, 0.6)',
                        line=dict(color='rgba(255, 0, 0, 1.0)', width=1)
                    ),
                    text=[f"{int(vol)}" for vol in ask_volumes],
                    textposition='outside',
                    hoverinfo='text',
                    hovertext=[f"Preço: {price:.2f}<br>Volume: {int(vol)}"
                              for price, vol in zip(ask_prices, ask_volumes)]
                )
            )
        
        # Adicionar linha para o mid price
        if 'mid_price' in dom_data:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[dom_data['mid_price']],
                    mode='markers+text',
                    name=f"Mid Price: {dom_data['mid_price']:.2f}",
                    marker=dict(color='blue', symbol='diamond', size=12),
                    text=[f"Mid: {dom_data['mid_price']:.2f}"],
                    textposition='middle right'
                )
            )
            
            # Adicionar anotação de spread
            if 'spread' in dom_data:
                fig.add_annotation(
                    x=0,
                    y=dom_data['mid_price'],
                    text=f"Spread: {dom_data['spread']:.2f}",
                    showarrow=False,
                    yshift=15,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1
                )
        
        # Adicionar informações de imbalance
        if 'buy_sell_ratio' in dom_data and 'imbalance' in dom_data:
            imbalance_color = 'green' if dom_data['imbalance'] > 0 else 'red'
            imbalance_text = f"Imbalance: {dom_data['imbalance']:.2%}"
            ratio_text = f"Buy/Sell: {dom_data['buy_sell_ratio']:.2f}"
            
            fig.add_annotation(
                x=0,
                y=min(ask_prices[0] if len(ask_prices) > 0 else 0, 
                      bid_prices[0] if len(bid_prices) > 0 else 0),
                text=f"{imbalance_text}<br>{ratio_text}",
                showarrow=False,
                yshift=-40,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=imbalance_color,
                borderwidth=2
            )
        
        # Configurar layout
        fig.update_layout(
            title='Livro de Ofertas (DOM) - Heatmap',
            xaxis_title='Volume',
            yaxis_title='Preço',
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1
            ),
            height=600,
            xaxis=dict(
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            )
        )
        
        return fig
    
    def create_plotly_dom_time_series(self, 
                                    dom_history: pd.DataFrame, 
                                    metrics: List[str] = None,
                                    window_size: int = 100) -> go.Figure:
        """
        Cria um gráfico de série temporal de métricas do DOM usando Plotly.
        
        Args:
            dom_history: DataFrame com histórico de DOM
            metrics: Lista de métricas para plotar
            window_size: Número de pontos a mostrar
            
        Returns:
            Figura Plotly
        """
        if dom_history.empty:
            # Criar figura vazia com mensagem
            fig = go.Figure()
            fig.add_annotation(text="Sem dados disponíveis",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Métricas padrão se não forem fornecidas
        if metrics is None:
            metrics = ['mid_price', 'imbalance', 'buy_sell_ratio']
        
        # Limitar ao tamanho da janela
        dom_recent = dom_history.tail(window_size)
        
        # Criar figura com subplots para cada métrica
        fig = make_subplots(rows=len(metrics), cols=1, 
                           shared_xaxes=True,
                           subplot_titles=[m.replace('_', ' ').title() for m in metrics],
                           vertical_spacing=0.04)
        
        # Adicionar cada métrica como um traço
        for i, metric in enumerate(metrics):
            if metric in dom_recent.columns:
                # Índice de linha para o subplot (1-based)
                row_idx = i + 1
                
                # Adicionar série temporal
                fig.add_trace(
                    go.Scatter(
                        x=dom_recent.index,
                        y=dom_recent[metric],
                        mode='lines',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color='blue', width=1.5)
                    ),
                    row=row_idx, col=1
                )
                
                # Para métricas específicas, adicionar visualizações especiais
                if metric == 'imbalance':
                    # Adicionar área preenchida para valores positivos (verde)
                    fig.add_trace(
                        go.Scatter(
                            x=dom_recent.index,
                            y=dom_recent[metric].clip(lower=0),
                            fill='tozeroy',
                            mode='none',
                            name='Imbalance Positivo',
                            fillcolor='rgba(0, 128, 0, 0.3)',
                            showlegend=False
                        ),
                        row=row_idx, col=1
                    )
                    
                    # Adicionar área preenchida para valores negativos (vermelho)
                    fig.add_trace(
                        go.Scatter(
                            x=dom_recent.index,
                            y=dom_recent[metric].clip(upper=0),
                            fill='tozeroy',
                            mode='none',
                            name='Imbalance Negativo',
                            fillcolor='rgba(255, 0, 0, 0.3)',
                            showlegend=False
                        ),
                        row=row_idx, col=1
                    )
                    
                    # Adicionar linha de zero
                    fig.add_trace(
                        go.Scatter(
                            x=dom_recent.index,
                            y=[0] * len(dom_recent),
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=row_idx, col=1
                    )
                
                # Adicionar referência do último valor
                last_value = dom_recent[metric].iloc[-1] if len(dom_recent) > 0 else None
                if last_value is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[dom_recent.index[-1]],
                            y=[last_value],
                            mode='markers+text',
                            marker=dict(color='red', size=8),
                            text=[f"{last_value:.4f}"],
                            textposition='middle right',
                            showlegend=False
                        ),
                        row=row_idx, col=1
                    )
                
                # Atualizar eixos y
                fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row_idx, col=1)
            else:
                logger.warning(f"Métrica '{metric}' não encontrada no histórico de DOM")
        
        # Configurar layout
        fig.update_layout(
            title='Análise Temporal do Livro de Ofertas (DOM)',
            height=250 * len(metrics),
            showlegend=False,
            xaxis=dict(
                title='Tempo',
                rangeslider=dict(visible=False)
            )
        )
        
        # Configurar formato dos eixos x para timestamps
        if isinstance(dom_recent.index, pd.DatetimeIndex):
            fig.update_xaxes(
                tickformat='%H:%M:%S',
                tickangle=45
            )
        
        return fig