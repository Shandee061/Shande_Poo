"""
Módulo para detecção de padrões de candlestick e formações de gráficos.

Este módulo implementa funcionalidades para identificar padrões de velas (candlesticks)
e formações gráficas comuns em gráficos de preços, fornecendo sinais de trading.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Union, Optional
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# Setup logger
logger = logging.getLogger(__name__)

class PatternDetector:
    """
    Classe para detecção e análise de padrões de candlestick e formações gráficas.
    
    Esta classe implementa métodos para identificar padrões comuns em gráficos de preços,
    fornecendo sinais de compra e venda baseados em padrões reconhecidos.
    """
    
    def __init__(self):
        """
        Inicializa o detector de padrões.
        """
        # Mapeia padrões de candlestick do TA-Lib para nomes descritivos
        self.candlestick_patterns = {
            'CDL2CROWS': '2 Corvos', 
            'CDL3BLACKCROWS': '3 Corvos Negros',
            'CDL3INSIDE': 'Três Dentro',
            'CDL3LINESTRIKE': 'Três Ataque em Linha',
            'CDL3OUTSIDE': 'Três Fora',
            'CDL3STARSINSOUTH': 'Três Estrelas no Sul',
            'CDL3WHITESOLDIERS': 'Três Soldados Brancos',
            'CDLABANDONEDBABY': 'Bebê Abandonado',
            'CDLADVANCEBLOCK': 'Bloco Avançado',
            'CDLBELTHOLD': 'Suporte em Cinto',
            'CDLBREAKAWAY': 'Ruptura',
            'CDLCLOSINGMARUBOZU': 'Marubozu de Fechamento',
            'CDLCONCEALBABYSWALL': 'Bebê na Parede',
            'CDLCOUNTERATTACK': 'Contra-Ataque',
            'CDLDARKCLOUDCOVER': 'Nuvem Escura',
            'CDLDOJI': 'Doji',
            'CDLDOJISTAR': 'Estrela Doji',
            'CDLDRAGONFLYDOJI': 'Doji Libélula',
            'CDLENGULFING': 'Engolfo',
            'CDLEVENINGDOJISTAR': 'Estrela Doji da Noite',
            'CDLEVENINGSTAR': 'Estrela da Noite',
            'CDLGAPSIDESIDEWHITE': 'Gap Lado a Lado Branco',
            'CDLGRAVESTONEDOJI': 'Doji Pedra Tumular',
            'CDLHAMMER': 'Martelo',
            'CDLHANGINGMAN': 'Homem Enforcado',
            'CDLHARAMI': 'Harami',
            'CDLHARAMICROSS': 'Cruz de Harami',
            'CDLHIGHWAVE': 'Onda Alta',
            'CDLHIKKAKE': 'Hikkake',
            'CDLHIKKAKEMOD': 'Hikkake Modificado',
            'CDLHOMINGPIGEON': 'Pombo-Correio',
            'CDLIDENTICAL3CROWS': '3 Corvos Idênticos',
            'CDLINNECK': 'Pescoço Para Dentro',
            'CDLINVERTEDHAMMER': 'Martelo Invertido',
            'CDLKICKING': 'Chute',
            'CDLKICKINGBYLENGTH': 'Chute por Comprimento',
            'CDLLADDERBOTTOM': 'Fundo de Escada',
            'CDLLONGLEGGEDDOJI': 'Doji de Pernas Longas',
            'CDLLONGLINE': 'Linha Longa',
            'CDLMARUBOZU': 'Marubozu',
            'CDLMATCHINGLOW': 'Baixa Coincidente',
            'CDLMATHOLD': 'Segurar',
            'CDLMORNINGDOJISTAR': 'Estrela Doji da Manhã',
            'CDLMORNINGSTAR': 'Estrela da Manhã',
            'CDLONNECK': 'Pescoço Para Fora',
            'CDLPIERCING': 'Perfurante',
            'CDLRICKSHAWMAN': 'Homem Riquixá',
            'CDLRISEFALL3METHODS': 'Métodos Subida/Descida 3',
            'CDLSEPARATINGLINES': 'Linhas de Separação',
            'CDLSHOOTINGSTAR': 'Estrela Cadente',
            'CDLSHORTLINE': 'Linha Curta',
            'CDLSPINNINGTOP': 'Pião',
            'CDLSTALLEDPATTERN': 'Padrão Parado',
            'CDLSTICKSANDWICH': 'Sanduíche de Velas',
            'CDLTAKURI': 'Takuri',
            'CDLTASUKIGAP': 'Gap Tasuki',
            'CDLTHRUSTING': 'Impulso',
            'CDLTRISTAR': 'Três Estrelas',
            'CDLUNIQUE3RIVER': 'Rio Único 3',
            'CDLUPSIDEGAP2CROWS': 'Gap Superior 2 Corvos',
            'CDLXSIDEGAP3METHODS': 'Gap Lateral Métodos 3'
        }
        
        # Classificar padrões por significado (alta, baixa, continuação)
        self.bullish_patterns = [
            'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLMORNINGDOJISTAR', 
            'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLPIERCING', 
            'CDLENGULFING', 'CDLHARAMI', 'CDLHARAMICROSS', 
            'CDLDRAGONFLYDOJI', 'CDLBELTHOLD', 'CDLHOMINGPIGEON',
            'CDLPIERCING', 'CDLTHRUSTING', 'CDLMATHOLD',
            'CDLRISEFALL3METHODS', 'CDLLADDERBOTTOM', 'CDLBREAKAWAY'
        ]
        
        self.bearish_patterns = [
            'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLEVENINGDOJISTAR', 
            'CDLHANGINGMAN', 'CDLSHOOTINGSTAR', 'CDLDARKCLOUDCOVER', 
            'CDLENGULFING', 'CDLHARAMI', 'CDLHARAMICROSS', 
            'CDLGRAVESTONEDOJI', 'CDLBELTHOLD', 'CDLHOMINGPIGEON',
            'CDLRISEFALL3METHODS', 'CDL2CROWS', 'CDLIDENTICAL3CROWS',
            'CDLUPSIDEGAP2CROWS', 'CDLCOUNTERATTACK', 'CDLBELTHOLD'
        ]
        
        self.continuation_patterns = [
            'CDLDOJI', 'CDLSPINNINGTOP', 'CDLHIGHWAVE',
            'CDLLONGLEGGEDDOJI', 'CDLSEPARATINGLINES', 'CDLLONGLINE',
            'CDLSHORTLINE', 'CDLTASUKIGAP', 'CDLGAPSIDESIDEWHITE'
        ]
        
        # Mapeia padrões para valores de confiabilidade (baseado em observações históricas)
        self.pattern_reliability = {
            'CDL3WHITESOLDIERS': 80,  # Alta confiabilidade para três soldados brancos
            'CDL3BLACKCROWS': 80,     # Alta confiabilidade para três corvos negros
            'CDLMORNINGSTAR': 75,     # Alta confiabilidade para estrela da manhã
            'CDLEVENINGSTAR': 75,     # Alta confiabilidade para estrela da noite
            'CDLENGULFING': 70,       # Boa confiabilidade para padrão de engolfo
            'CDLHARAMI': 60,          # Confiabilidade média para harami
            'CDLDOJI': 40,            # Baixa confiabilidade para doji isolado
            'CDLHAMMER': 65,          # Boa confiabilidade para martelo
            'CDLSHOOTINGSTAR': 65     # Boa confiabilidade para estrela cadente
        }
        
        # Define padrões gráficos mais complexos
        self.chart_patterns = [
            'head_and_shoulders', 'inverse_head_and_shoulders',
            'double_top', 'double_bottom',
            'triple_top', 'triple_bottom',
            'rising_wedge', 'falling_wedge',
            'ascending_triangle', 'descending_triangle', 'symmetric_triangle',
            'rectangle', 'flag_pole', 'pennant'
        ]
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica padrões de candlestick em um DataFrame de dados OHLC.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com sinais de padrões de candlestick
        """
        # Verificar se temos as colunas necessárias
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Coluna {col} não encontrada. Não é possível detectar padrões de candlestick.")
                return df
        
        # Criar um DataFrame para armazenar os resultados
        pattern_signals = pd.DataFrame(index=df.index)
        
        try:
            # Aplicar todas as funções de detecção de padrões do TA-Lib
            for pattern_name in self.candlestick_patterns:
                # Obter a função do TA-Lib
                pattern_function = getattr(talib, pattern_name)
                
                # Aplicar a função aos dados OHLC
                pattern_signals[pattern_name] = pattern_function(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                )
            
            # Adicionar nomes descritivos em português
            for pattern_code, pattern_name in self.candlestick_patterns.items():
                pattern_signals[f'{pattern_name}'] = pattern_signals[pattern_code]
            
            # Adicionar sinais agregados (alta, baixa, neutro)
            pattern_signals['bullish_patterns'] = 0
            pattern_signals['bearish_patterns'] = 0
            pattern_signals['continuation_patterns'] = 0
            
            # Contar padrões de alta
            for pattern in self.bullish_patterns:
                pattern_signals['bullish_patterns'] += (pattern_signals[pattern] > 0).astype(int)
            
            # Contar padrões de baixa
            for pattern in self.bearish_patterns:
                pattern_signals['bearish_patterns'] += (pattern_signals[pattern] < 0).astype(int)
            
            # Contar padrões de continuação
            for pattern in self.continuation_patterns:
                pattern_signals['continuation_patterns'] += (pattern_signals[pattern] != 0).astype(int)
            
            # Calcular um sinal geral
            pattern_signals['pattern_signal'] = pattern_signals['bullish_patterns'] - pattern_signals['bearish_patterns']
                
            logger.info(f"Detectados padrões de candlestick no DataFrame.")
            
        except Exception as e:
            logger.error(f"Erro ao detectar padrões de candlestick: {str(e)}")
        
        return pattern_signals
    
    def get_strongest_pattern(self, pattern_row: pd.Series) -> Tuple[str, int, str]:
        """
        Identifica o padrão mais forte em uma determinada linha de sinais.
        
        Args:
            pattern_row: Série pandas com sinais de padrões
            
        Returns:
            Tupla com (nome do padrão, valor do sinal, tipo de sinal)
        """
        # Filtrar apenas os padrões reais (excluir colunas agregadas)
        pattern_codes = [code for code in pattern_row.index if code in self.candlestick_patterns]
        
        if not pattern_codes:
            return ("Nenhum padrão", 0, "neutro")
        
        # Obter o padrão com o maior valor absoluto
        strongest_pattern = max(pattern_codes, key=lambda x: abs(pattern_row[x]) if not pd.isna(pattern_row[x]) else 0)
        pattern_value = pattern_row[strongest_pattern]
        
        if pd.isna(pattern_value) or pattern_value == 0:
            return ("Nenhum padrão", 0, "neutro")
        
        # Determinar o tipo de sinal
        if pattern_value > 0:
            signal_type = "bullish"
        elif pattern_value < 0:
            signal_type = "bearish"
        else:
            signal_type = "neutro"
        
        # Retornar nome do padrão em português
        pattern_name = self.candlestick_patterns.get(strongest_pattern, strongest_pattern)
        
        return (pattern_name, pattern_value, signal_type)
    
    def plot_candlestick_patterns(self, df: pd.DataFrame, pattern_signals: pd.DataFrame, 
                                 window_size: int = 30, end_date: Optional[str] = None) -> plt.Figure:
        """
        Plota um gráfico de candlestick com padrões identificados.
        
        Args:
            df: DataFrame com dados OHLCV
            pattern_signals: DataFrame com sinais de padrões de candlestick
            window_size: Número de barras para exibir
            end_date: Data final para a visualização (opcional)
            
        Returns:
            Figura matplotlib com visualização de padrões
        """
        try:
            import mplfinance as mpf
        except ImportError:
            logger.warning("Biblioteca mplfinance não encontrada. Usando implementação alternativa.")
            return self._plot_candlestick_basic(df, pattern_signals, window_size, end_date)
        
        # Verificar se temos dados suficientes
        if df.empty or pattern_signals.empty:
            logger.warning("Dados insuficientes para plotar padrões de candlestick.")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center')
            return fig
        
        # Preparar os dados para visualização
        if end_date:
            end_idx = df.index.get_loc(end_date) if end_date in df.index else len(df) - 1
        else:
            end_idx = len(df) - 1
        
        start_idx = max(0, end_idx - window_size + 1)
        plot_df = df.iloc[start_idx:end_idx+1].copy()
        
        # Converter para o formato esperado pelo mplfinance
        if not isinstance(plot_df.index, pd.DatetimeIndex):
            logger.warning("Índice não é datetime. Convertendo para visualização.")
            plot_df = plot_df.reset_index()
            if 'datetime' in plot_df.columns:
                plot_df.set_index('datetime', inplace=True)
            else:
                # Criar um índice datetime sintético
                plot_df.index = pd.date_range(start='2023-01-01', periods=len(plot_df), freq='D')
        
        # Garantir que temos as colunas necessárias no formato correto
        ohlc_cols = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        for old_col, new_col in ohlc_cols.items():
            if old_col in plot_df.columns:
                plot_df[new_col] = plot_df[old_col]
        
        if 'volume' in plot_df.columns:
            plot_df['Volume'] = plot_df['volume']
        
        # Filtrar apenas as colunas necessárias
        plot_df = plot_df[['Open', 'High', 'Low', 'Close'] + (['Volume'] if 'Volume' in plot_df.columns else [])]
        
        # Preparar marcadores para padrões
        markers = []
        
        # Criar anotações para padrões significativos
        plot_pattern_signals = pattern_signals.iloc[start_idx:end_idx+1].copy()
        
        for i, (idx, row) in enumerate(plot_pattern_signals.iterrows()):
            pattern_name, value, signal_type = self.get_strongest_pattern(row)
            
            if pattern_name != "Nenhum padrão":
                # Definir cores e marcadores com base no tipo de sinal
                color = 'green' if signal_type == 'bullish' else 'red' if signal_type == 'bearish' else 'blue'
                marker = '^' if signal_type == 'bullish' else 'v' if signal_type == 'bearish' else 'o'
                
                # Adicionar marcador
                markers.append(
                    mpf.make_addplot([plot_df['Low'].iloc[i] * 0.99] if i < len(plot_df) else [None], 
                                   type='scatter', 
                                   marker=marker, 
                                   markersize=100, 
                                   color=color)
                )
        
        # Configurar estilo
        style = mpf.make_mpf_style(base_mpf_style='yahoo', 
                                  gridstyle='', 
                                  y_on_right=False, 
                                  mavcolors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                                  facecolor='white',
                                  edgecolor='black',
                                  figcolor='white',
                                  marketcolors={'candle': {'up': 'white', 'down': 'black'},
                                               'edge': {'up': 'black', 'down': 'black'},
                                               'wick': {'up': 'black', 'down': 'black'},
                                               'ohlc': {'up': 'green', 'down': 'red'},
                                               'volume': {'up': 'green', 'down': 'red'},
                                               'vcedge': {'up': 'green', 'down': 'red'},
                                               'vcdopcod': False})
        
        # Criar o gráfico
        fig, axes = mpf.plot(plot_df, 
                           type='candle', 
                           style=style,
                           title='Padrões de Candlestick', 
                           figsize=(12, 8),
                           volume=True if 'Volume' in plot_df.columns else False,
                           addplot=markers if markers else None,
                           returnfig=True)
        
        return fig
    
    def _plot_candlestick_basic(self, df: pd.DataFrame, pattern_signals: pd.DataFrame, 
                              window_size: int = 30, end_date: Optional[str] = None) -> plt.Figure:
        """
        Implementação alternativa de visualização de padrões de candlestick.
        
        Args:
            df: DataFrame com dados OHLCV
            pattern_signals: DataFrame com sinais de padrões de candlestick
            window_size: Número de barras para exibir
            end_date: Data final para a visualização (opcional)
            
        Returns:
            Figura matplotlib com visualização de padrões
        """
        # Verificar se temos dados suficientes
        if df.empty or pattern_signals.empty:
            logger.warning("Dados insuficientes para plotar padrões de candlestick.")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center')
            return fig
        
        # Preparar os dados para visualização
        if end_date:
            try:
                end_idx = df.index.get_loc(end_date)
            except:
                end_idx = len(df) - 1
        else:
            end_idx = len(df) - 1
        
        start_idx = max(0, end_idx - window_size + 1)
        plot_df = df.iloc[start_idx:end_idx+1].copy()
        plot_pattern_signals = pattern_signals.iloc[start_idx:end_idx+1].copy()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Preparar dados para plotagem
        dates = np.arange(len(plot_df))
        
        # Plotar candlesticks
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            # Verificar se temos OHLC
            if all(col in row.index for col in ['open', 'high', 'low', 'close']):
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                # Determinar se é uma vela de alta ou baixa
                if close_price >= open_price:
                    # Vela de alta (branco/verde)
                    color = 'white'
                    edge_color = 'black'
                else:
                    # Vela de baixa (preto/vermelho)
                    color = 'black'
                    edge_color = 'black'
                
                # Plotar linha vertical (pavio)
                ax.plot([dates[i], dates[i]], [low_price, high_price], color=edge_color, linewidth=1)
                
                # Plotar corpo da vela
                rect = Rectangle((dates[i] - 0.4, min(open_price, close_price)), 
                               0.8, 
                               abs(close_price - open_price),
                               fill=True, 
                               color=color, 
                               ec=edge_color)
                ax.add_patch(rect)
                
                # Adicionar marcador para padrões
                if i < len(plot_pattern_signals):
                    pattern_row = plot_pattern_signals.iloc[i]
                    pattern_name, value, signal_type = self.get_strongest_pattern(pattern_row)
                    
                    if pattern_name != "Nenhum padrão":
                        # Definir cores e marcadores com base no tipo de sinal
                        marker_color = 'green' if signal_type == 'bullish' else 'red' if signal_type == 'bearish' else 'blue'
                        marker = '^' if signal_type == 'bullish' else 'v' if signal_type == 'bearish' else 'o'
                        
                        # Plotar marcador
                        ax.plot(dates[i], low_price * 0.99, marker=marker, markersize=10, color=marker_color)
                        
                        # Adicionar anotação com o nome do padrão
                        if i % 3 == 0:  # Mostrar apenas alguns para evitar sobreposição
                            ax.annotate(pattern_name, 
                                      xy=(dates[i], low_price * 0.97),
                                      xytext=(dates[i], low_price * 0.93),
                                      fontsize=8,
                                      color=marker_color,
                                      ha='center',
                                      arrowprops=dict(arrowstyle='->', color=marker_color))
        
        # Configurar eixos
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço')
        ax.set_title('Padrões de Candlestick')
        
        # Ajustar labels do eixo x para datas
        if isinstance(plot_df.index, pd.DatetimeIndex):
            date_labels = [idx.strftime('%Y-%m-%d') for idx in plot_df.index]
            plt.xticks(dates[::5], date_labels[::5], rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def detect_chart_patterns(self, df: pd.DataFrame, min_pattern_length: int = 5, 
                             max_pattern_length: int = 50) -> pd.DataFrame:
        """
        Identifica padrões gráficos mais complexos como cabeça-ombro, topos duplos, etc.
        
        Args:
            df: DataFrame com dados OHLCV
            min_pattern_length: Comprimento mínimo do padrão (em barras)
            max_pattern_length: Comprimento máximo do padrão (em barras)
            
        Returns:
            DataFrame com sinais de padrões gráficos
        """
        # Verificar se temos as colunas necessárias
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Coluna {col} não encontrada. Não é possível detectar padrões gráficos.")
                return pd.DataFrame(index=df.index)
        
        # Criar um DataFrame para armazenar os resultados
        chart_signals = pd.DataFrame(index=df.index)
        
        # Inicializar colunas para cada tipo de padrão
        for pattern in self.chart_patterns:
            chart_signals[pattern] = 0
        
        try:
            # Detectar topos e fundos
            highs, lows = self._find_peaks_and_troughs(df)
            
            # Detectar cada tipo de padrão
            chart_signals = self._detect_head_and_shoulders(df, chart_signals, highs, lows, min_pattern_length, max_pattern_length)
            chart_signals = self._detect_double_patterns(df, chart_signals, highs, lows, min_pattern_length, max_pattern_length)
            chart_signals = self._detect_triple_patterns(df, chart_signals, highs, lows, min_pattern_length, max_pattern_length)
            chart_signals = self._detect_triangles(df, chart_signals, highs, lows, min_pattern_length, max_pattern_length)
            chart_signals = self._detect_wedges(df, chart_signals, highs, lows, min_pattern_length, max_pattern_length)
            chart_signals = self._detect_rectangles(df, chart_signals, highs, lows, min_pattern_length, max_pattern_length)
            
            # Adicionar sinais agregados (alta, baixa, continuação)
            chart_signals['bullish_chart_patterns'] = 0
            chart_signals['bearish_chart_patterns'] = 0
            
            # Padrões de alta
            bullish_chart_patterns = [
                'inverse_head_and_shoulders', 'double_bottom', 'triple_bottom',
                'falling_wedge', 'ascending_triangle'
            ]
            
            # Padrões de baixa
            bearish_chart_patterns = [
                'head_and_shoulders', 'double_top', 'triple_top',
                'rising_wedge', 'descending_triangle'
            ]
            
            # Contar padrões de alta
            for pattern in bullish_chart_patterns:
                chart_signals['bullish_chart_patterns'] += (chart_signals[pattern] > 0).astype(int)
            
            # Contar padrões de baixa
            for pattern in bearish_chart_patterns:
                chart_signals['bearish_chart_patterns'] += (chart_signals[pattern] > 0).astype(int)
            
            # Calcular um sinal geral
            chart_signals['chart_pattern_signal'] = chart_signals['bullish_chart_patterns'] - chart_signals['bearish_chart_patterns']
            
            logger.info(f"Detectados padrões gráficos no DataFrame.")
            
        except Exception as e:
            logger.error(f"Erro ao detectar padrões gráficos: {str(e)}")
        
        return chart_signals
    
    def _find_peaks_and_troughs(self, df: pd.DataFrame, smoothing: int = 2,
                              threshold_pct: float = 0.005) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Identifica topos (highs) e fundos (lows) na série de preços.
        
        Args:
            df: DataFrame com dados OHLCV
            smoothing: Número de pontos para suavização (média móvel)
            threshold_pct: Limiar percentual para considerar um topo/fundo significativo
            
        Returns:
            Tupla com dicionários (índice: valor) para topos e fundos
        """
        # Usar máximas e mínimas para identificar topos e fundos
        highs = df['high'].rolling(window=smoothing).mean()
        lows = df['low'].rolling(window=smoothing).mean()
        
        # Dicionários para armazenar topos e fundos
        peak_dict = {}  # {índice: valor}
        trough_dict = {}  # {índice: valor}
        
        # Identificar topos
        for i in range(2, len(highs) - 2):
            if pd.isna(highs[i]):
                continue
                
            # Um ponto é um topo se for maior que os pontos adjacentes
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                # Verificar se a diferença é significativa
                if (highs[i] - min(highs[i-1], highs[i+1])) / highs[i] >= threshold_pct:
                    peak_dict[i] = highs[i]
        
        # Identificar fundos
        for i in range(2, len(lows) - 2):
            if pd.isna(lows[i]):
                continue
                
            # Um ponto é um fundo se for menor que os pontos adjacentes
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                # Verificar se a diferença é significativa
                if (max(lows[i-1], lows[i+1]) - lows[i]) / lows[i] >= threshold_pct:
                    trough_dict[i] = lows[i]
        
        return peak_dict, trough_dict
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame, signals: pd.DataFrame, 
                                 highs: Dict[int, float], lows: Dict[int, float],
                                 min_length: int, max_length: int) -> pd.DataFrame:
        """
        Detecta padrões de cabeça e ombros e cabeça e ombros invertido.
        
        Args:
            df: DataFrame com dados OHLCV
            signals: DataFrame com sinais de padrões
            highs: Dicionário com topos (índice: valor)
            lows: Dicionário com fundos (índice: valor)
            min_length: Comprimento mínimo do padrão
            max_length: Comprimento máximo do padrão
            
        Returns:
            DataFrame atualizado com sinais de padrões
        """
        result = signals.copy()
        
        # Obter listas ordenadas de topos e fundos
        peak_indices = sorted(highs.keys())
        trough_indices = sorted(lows.keys())
        
        # Detectar padrão de cabeça e ombros (topos)
        for i in range(len(peak_indices) - 2):
            # Verificar se temos 3 topos consecutivos
            idx_left = peak_indices[i]
            idx_head = peak_indices[i+1]
            idx_right = peak_indices[i+2]
            
            # Verificar se o padrão está dentro do tamanho permitido
            if idx_right - idx_left < min_length or idx_right - idx_left > max_length:
                continue
            
            # Verificar características do padrão de cabeça e ombros
            left_shoulder = highs[idx_left]
            head = highs[idx_head]
            right_shoulder = highs[idx_right]
            
            # O pico do meio deve ser maior que os outros dois
            # Os ombros devem estar aproximadamente na mesma altura
            if head > left_shoulder and head > right_shoulder and \
               abs(left_shoulder - right_shoulder) / left_shoulder < 0.1:
                
                # Verificar a linha de pescoço (neckline)
                # Encontrar os fundos entre os topos
                neck_left = None
                neck_right = None
                
                for j in trough_indices:
                    if idx_left < j < idx_head:
                        neck_left = j
                    if idx_head < j < idx_right:
                        neck_right = j
                        break
                
                if neck_left is not None and neck_right is not None:
                    # A linha do pescoço deve ser aproximadamente horizontal
                    if abs(lows[neck_left] - lows[neck_right]) / lows[neck_left] < 0.05:
                        # Marcar o sinal na última barra do padrão
                        result.loc[df.index[idx_right], 'head_and_shoulders'] = -100  # Sinal de baixa
        
        # Detectar padrão de cabeça e ombros invertido (fundos)
        for i in range(len(trough_indices) - 2):
            # Verificar se temos 3 fundos consecutivos
            idx_left = trough_indices[i]
            idx_head = trough_indices[i+1]
            idx_right = trough_indices[i+2]
            
            # Verificar se o padrão está dentro do tamanho permitido
            if idx_right - idx_left < min_length or idx_right - idx_left > max_length:
                continue
            
            # Verificar características do padrão de cabeça e ombros invertido
            left_shoulder = lows[idx_left]
            head = lows[idx_head]
            right_shoulder = lows[idx_right]
            
            # O fundo do meio deve ser menor que os outros dois
            # Os ombros devem estar aproximadamente na mesma altura
            if head < left_shoulder and head < right_shoulder and \
               abs(left_shoulder - right_shoulder) / left_shoulder < 0.1:
                
                # Verificar a linha de pescoço (neckline)
                # Encontrar os topos entre os fundos
                neck_left = None
                neck_right = None
                
                for j in peak_indices:
                    if idx_left < j < idx_head:
                        neck_left = j
                    if idx_head < j < idx_right:
                        neck_right = j
                        break
                
                if neck_left is not None and neck_right is not None:
                    # A linha do pescoço deve ser aproximadamente horizontal
                    if abs(highs[neck_left] - highs[neck_right]) / highs[neck_left] < 0.05:
                        # Marcar o sinal na última barra do padrão
                        result.loc[df.index[idx_right], 'inverse_head_and_shoulders'] = 100  # Sinal de alta
        
        return result
    
    def _detect_double_patterns(self, df: pd.DataFrame, signals: pd.DataFrame, 
                              highs: Dict[int, float], lows: Dict[int, float],
                              min_length: int, max_length: int) -> pd.DataFrame:
        """
        Detecta padrões de topo duplo e fundo duplo.
        
        Args:
            df: DataFrame com dados OHLCV
            signals: DataFrame com sinais de padrões
            highs: Dicionário com topos (índice: valor)
            lows: Dicionário com fundos (índice: valor)
            min_length: Comprimento mínimo do padrão
            max_length: Comprimento máximo do padrão
            
        Returns:
            DataFrame atualizado com sinais de padrões
        """
        result = signals.copy()
        
        # Obter listas ordenadas de topos e fundos
        peak_indices = sorted(highs.keys())
        trough_indices = sorted(lows.keys())
        
        # Detectar padrão de topo duplo
        for i in range(len(peak_indices) - 1):
            idx1 = peak_indices[i]
            idx2 = peak_indices[i+1]
            
            # Verificar se o padrão está dentro do tamanho permitido
            if idx2 - idx1 < min_length or idx2 - idx1 > max_length:
                continue
            
            # Os dois topos devem estar aproximadamente na mesma altura
            if abs(highs[idx1] - highs[idx2]) / highs[idx1] < 0.03:
                # Verificar se há um vale significativo entre os topos
                valley_found = False
                valley_idx = None
                
                for j in trough_indices:
                    if idx1 < j < idx2:
                        valley_found = True
                        valley_idx = j
                        break
                
                if valley_found and valley_idx is not None:
                    # Verificar se o vale é significativamente mais baixo que os topos
                    if (highs[idx1] - lows[valley_idx]) / highs[idx1] > 0.03:
                        # Marcar o sinal na última barra do padrão
                        result.loc[df.index[idx2], 'double_top'] = -100  # Sinal de baixa
        
        # Detectar padrão de fundo duplo
        for i in range(len(trough_indices) - 1):
            idx1 = trough_indices[i]
            idx2 = trough_indices[i+1]
            
            # Verificar se o padrão está dentro do tamanho permitido
            if idx2 - idx1 < min_length or idx2 - idx1 > max_length:
                continue
            
            # Os dois fundos devem estar aproximadamente na mesma altura
            if abs(lows[idx1] - lows[idx2]) / lows[idx1] < 0.03:
                # Verificar se há um pico significativo entre os fundos
                peak_found = False
                peak_idx = None
                
                for j in peak_indices:
                    if idx1 < j < idx2:
                        peak_found = True
                        peak_idx = j
                        break
                
                if peak_found and peak_idx is not None:
                    # Verificar se o pico é significativamente mais alto que os fundos
                    if (highs[peak_idx] - lows[idx1]) / lows[idx1] > 0.03:
                        # Marcar o sinal na última barra do padrão
                        result.loc[df.index[idx2], 'double_bottom'] = 100  # Sinal de alta
        
        return result
    
    def _detect_triple_patterns(self, df: pd.DataFrame, signals: pd.DataFrame, 
                              highs: Dict[int, float], lows: Dict[int, float],
                              min_length: int, max_length: int) -> pd.DataFrame:
        """
        Detecta padrões de topo triplo e fundo triplo.
        
        Args:
            df: DataFrame com dados OHLCV
            signals: DataFrame com sinais de padrões
            highs: Dicionário com topos (índice: valor)
            lows: Dicionário com fundos (índice: valor)
            min_length: Comprimento mínimo do padrão
            max_length: Comprimento máximo do padrão
            
        Returns:
            DataFrame atualizado com sinais de padrões
        """
        result = signals.copy()
        
        # Obter listas ordenadas de topos e fundos
        peak_indices = sorted(highs.keys())
        trough_indices = sorted(lows.keys())
        
        # Detectar padrão de topo triplo
        for i in range(len(peak_indices) - 2):
            idx1 = peak_indices[i]
            idx2 = peak_indices[i+1]
            idx3 = peak_indices[i+2]
            
            # Verificar se o padrão está dentro do tamanho permitido
            if idx3 - idx1 < min_length or idx3 - idx1 > max_length:
                continue
            
            # Os três topos devem estar aproximadamente na mesma altura
            if abs(highs[idx1] - highs[idx2]) / highs[idx1] < 0.03 and \
               abs(highs[idx2] - highs[idx3]) / highs[idx2] < 0.03:
                
                # Verificar se há vales significativos entre os topos
                valley1_found = False
                valley2_found = False
                
                for j in trough_indices:
                    if idx1 < j < idx2:
                        valley1_found = True
                    if idx2 < j < idx3:
                        valley2_found = True
                
                if valley1_found and valley2_found:
                    # Marcar o sinal na última barra do padrão
                    result.loc[df.index[idx3], 'triple_top'] = -100  # Sinal de baixa
        
        # Detectar padrão de fundo triplo
        for i in range(len(trough_indices) - 2):
            idx1 = trough_indices[i]
            idx2 = trough_indices[i+1]
            idx3 = trough_indices[i+2]
            
            # Verificar se o padrão está dentro do tamanho permitido
            if idx3 - idx1 < min_length or idx3 - idx1 > max_length:
                continue
            
            # Os três fundos devem estar aproximadamente na mesma altura
            if abs(lows[idx1] - lows[idx2]) / lows[idx1] < 0.03 and \
               abs(lows[idx2] - lows[idx3]) / lows[idx2] < 0.03:
                
                # Verificar se há picos significativos entre os fundos
                peak1_found = False
                peak2_found = False
                
                for j in peak_indices:
                    if idx1 < j < idx2:
                        peak1_found = True
                    if idx2 < j < idx3:
                        peak2_found = True
                
                if peak1_found and peak2_found:
                    # Marcar o sinal na última barra do padrão
                    result.loc[df.index[idx3], 'triple_bottom'] = 100  # Sinal de alta
        
        return result
    
    def _detect_triangles(self, df: pd.DataFrame, signals: pd.DataFrame, 
                        highs: Dict[int, float], lows: Dict[int, float],
                        min_length: int, max_length: int) -> pd.DataFrame:
        """
        Detecta padrões de triângulos (ascendente, descendente, simétrico).
        
        Args:
            df: DataFrame com dados OHLCV
            signals: DataFrame com sinais de padrões
            highs: Dicionário com topos (índice: valor)
            lows: Dicionário com fundos (índice: valor)
            min_length: Comprimento mínimo do padrão
            max_length: Comprimento máximo do padrão
            
        Returns:
            DataFrame atualizado com sinais de padrões
        """
        result = signals.copy()
        
        # Obter listas ordenadas de topos e fundos
        peak_indices = sorted(highs.keys())
        trough_indices = sorted(lows.keys())
        
        # Detectar padrão de triângulo ascendente (resistência horizontal, suporte ascendente)
        # Precisamos de pelo menos 2 topos na mesma altura e 2 fundos ascendentes
        for end_idx in range(len(df) - 1, 0, -1):
            # Encontrar os últimos 2-3 topos antes do índice final
            last_peaks = [idx for idx in peak_indices if idx < end_idx][-3:]
            
            if len(last_peaks) < 2:
                continue
            
            # Verificar se o padrão está dentro do tamanho permitido
            if last_peaks[-1] - last_peaks[0] < min_length or last_peaks[-1] - last_peaks[0] > max_length:
                continue
            
            # Verificar se os topos estão aproximadamente na mesma altura (resistência horizontal)
            top_values = [highs[idx] for idx in last_peaks]
            if max(top_values) - min(top_values) < 0.02 * min(top_values):
                # Encontrar os últimos 2-3 fundos antes do índice final
                last_troughs = [idx for idx in trough_indices if idx < end_idx][-3:]
                
                if len(last_troughs) < 2:
                    continue
                
                # Verificar se os fundos são ascendentes
                trough_values = [lows[idx] for idx in last_troughs]
                ascending = True
                for i in range(1, len(trough_values)):
                    if trough_values[i] <= trough_values[i-1]:
                        ascending = False
                        break
                
                if ascending:
                    # Marcar o sinal na barra atual (possível ponto de quebra)
                    result.loc[df.index[end_idx], 'ascending_triangle'] = 100  # Sinal de alta
        
        # Detectar padrão de triângulo descendente (suporte horizontal, resistência descendente)
        for end_idx in range(len(df) - 1, 0, -1):
            # Encontrar os últimos 2-3 fundos antes do índice final
            last_troughs = [idx for idx in trough_indices if idx < end_idx][-3:]
            
            if len(last_troughs) < 2:
                continue
            
            # Verificar se o padrão está dentro do tamanho permitido
            if last_troughs[-1] - last_troughs[0] < min_length or last_troughs[-1] - last_troughs[0] > max_length:
                continue
            
            # Verificar se os fundos estão aproximadamente na mesma altura (suporte horizontal)
            bottom_values = [lows[idx] for idx in last_troughs]
            if max(bottom_values) - min(bottom_values) < 0.02 * min(bottom_values):
                # Encontrar os últimos 2-3 topos antes do índice final
                last_peaks = [idx for idx in peak_indices if idx < end_idx][-3:]
                
                if len(last_peaks) < 2:
                    continue
                
                # Verificar se os topos são descendentes
                peak_values = [highs[idx] for idx in last_peaks]
                descending = True
                for i in range(1, len(peak_values)):
                    if peak_values[i] >= peak_values[i-1]:
                        descending = False
                        break
                
                if descending:
                    # Marcar o sinal na barra atual (possível ponto de quebra)
                    result.loc[df.index[end_idx], 'descending_triangle'] = -100  # Sinal de baixa
        
        # Detectar padrão de triângulo simétrico (topos descendentes, fundos ascendentes)
        for end_idx in range(len(df) - 1, 0, -1):
            # Encontrar os últimos 2-3 topos e fundos antes do índice final
            last_peaks = [idx for idx in peak_indices if idx < end_idx][-3:]
            last_troughs = [idx for idx in trough_indices if idx < end_idx][-3:]
            
            if len(last_peaks) < 2 or len(last_troughs) < 2:
                continue
            
            # Verificar se o padrão está dentro do tamanho permitido
            pattern_start = min(last_peaks[0], last_troughs[0])
            pattern_end = max(last_peaks[-1], last_troughs[-1])
            
            if pattern_end - pattern_start < min_length or pattern_end - pattern_start > max_length:
                continue
            
            # Verificar se os topos são descendentes
            peak_values = [highs[idx] for idx in last_peaks]
            peaks_descending = True
            for i in range(1, len(peak_values)):
                if peak_values[i] >= peak_values[i-1]:
                    peaks_descending = False
                    break
            
            # Verificar se os fundos são ascendentes
            trough_values = [lows[idx] for idx in last_troughs]
            troughs_ascending = True
            for i in range(1, len(trough_values)):
                if trough_values[i] <= trough_values[i-1]:
                    troughs_ascending = False
                    break
            
            if peaks_descending and troughs_ascending:
                # Marcar o sinal na barra atual (possível ponto de quebra)
                result.loc[df.index[end_idx], 'symmetric_triangle'] = 50  # Sinal neutro (pode quebrar para cima ou para baixo)
        
        return result
    
    def _detect_wedges(self, df: pd.DataFrame, signals: pd.DataFrame, 
                     highs: Dict[int, float], lows: Dict[int, float],
                     min_length: int, max_length: int) -> pd.DataFrame:
        """
        Detecta padrões de cunha (rising wedge, falling wedge).
        
        Args:
            df: DataFrame com dados OHLCV
            signals: DataFrame com sinais de padrões
            highs: Dicionário com topos (índice: valor)
            lows: Dicionário com fundos (índice: valor)
            min_length: Comprimento mínimo do padrão
            max_length: Comprimento máximo do padrão
            
        Returns:
            DataFrame atualizado com sinais de padrões
        """
        result = signals.copy()
        
        # Obter listas ordenadas de topos e fundos
        peak_indices = sorted(highs.keys())
        trough_indices = sorted(lows.keys())
        
        # Detectar padrão de cunha ascendente (rising wedge) - ambas linhas sobem, mas convergem (baixista)
        for end_idx in range(len(df) - 1, 0, -1):
            # Encontrar os últimos 2-3 topos e fundos antes do índice final
            last_peaks = [idx for idx in peak_indices if idx < end_idx][-3:]
            last_troughs = [idx for idx in trough_indices if idx < end_idx][-3:]
            
            if len(last_peaks) < 2 or len(last_troughs) < 2:
                continue
            
            # Verificar se o padrão está dentro do tamanho permitido
            pattern_start = min(last_peaks[0], last_troughs[0])
            pattern_end = max(last_peaks[-1], last_troughs[-1])
            
            if pattern_end - pattern_start < min_length or pattern_end - pattern_start > max_length:
                continue
            
            # Verificar se ambos topos e fundos são ascendentes
            peak_values = [highs[idx] for idx in last_peaks]
            trough_values = [lows[idx] for idx in last_troughs]
            
            peaks_ascending = all(peak_values[i] > peak_values[i-1] for i in range(1, len(peak_values)))
            troughs_ascending = all(trough_values[i] > trough_values[i-1] for i in range(1, len(trough_values)))
            
            if peaks_ascending and troughs_ascending:
                # Verificar convergência (a taxa de subida dos fundos é maior que a dos topos)
                if len(peak_values) >= 2 and len(trough_values) >= 2:
                    peak_slope = (peak_values[-1] - peak_values[0]) / (last_peaks[-1] - last_peaks[0])
                    trough_slope = (trough_values[-1] - trough_values[0]) / (last_troughs[-1] - last_troughs[0])
                    
                    if trough_slope > peak_slope:
                        # Marcar o sinal na barra atual (possível ponto de quebra)
                        result.loc[df.index[end_idx], 'rising_wedge'] = -100  # Sinal de baixa
        
        # Detectar padrão de cunha descendente (falling wedge) - ambas linhas descem, mas convergem (altista)
        for end_idx in range(len(df) - 1, 0, -1):
            # Encontrar os últimos 2-3 topos e fundos antes do índice final
            last_peaks = [idx for idx in peak_indices if idx < end_idx][-3:]
            last_troughs = [idx for idx in trough_indices if idx < end_idx][-3:]
            
            if len(last_peaks) < 2 or len(last_troughs) < 2:
                continue
            
            # Verificar se o padrão está dentro do tamanho permitido
            pattern_start = min(last_peaks[0], last_troughs[0])
            pattern_end = max(last_peaks[-1], last_troughs[-1])
            
            if pattern_end - pattern_start < min_length or pattern_end - pattern_start > max_length:
                continue
            
            # Verificar se ambos topos e fundos são descendentes
            peak_values = [highs[idx] for idx in last_peaks]
            trough_values = [lows[idx] for idx in last_troughs]
            
            peaks_descending = all(peak_values[i] < peak_values[i-1] for i in range(1, len(peak_values)))
            troughs_descending = all(trough_values[i] < trough_values[i-1] for i in range(1, len(trough_values)))
            
            if peaks_descending and troughs_descending:
                # Verificar convergência (a taxa de descida dos fundos é menor que a dos topos)
                if len(peak_values) >= 2 and len(trough_values) >= 2:
                    peak_slope = (peak_values[-1] - peak_values[0]) / (last_peaks[-1] - last_peaks[0])
                    trough_slope = (trough_values[-1] - trough_values[0]) / (last_troughs[-1] - last_troughs[0])
                    
                    if trough_slope > peak_slope:  # Ambos negativos, mas o trough_slope é menos negativo
                        # Marcar o sinal na barra atual (possível ponto de quebra)
                        result.loc[df.index[end_idx], 'falling_wedge'] = 100  # Sinal de alta
        
        return result
    
    def _detect_rectangles(self, df: pd.DataFrame, signals: pd.DataFrame, 
                         highs: Dict[int, float], lows: Dict[int, float],
                         min_length: int, max_length: int) -> pd.DataFrame:
        """
        Detecta padrões de retângulo e bandeiras.
        
        Args:
            df: DataFrame com dados OHLCV
            signals: DataFrame com sinais de padrões
            highs: Dicionário com topos (índice: valor)
            lows: Dicionário com fundos (índice: valor)
            min_length: Comprimento mínimo do padrão
            max_length: Comprimento máximo do padrão
            
        Returns:
            DataFrame atualizado com sinais de padrões
        """
        result = signals.copy()
        
        # Obter listas ordenadas de topos e fundos
        peak_indices = sorted(highs.keys())
        trough_indices = sorted(lows.keys())
        
        # Detectar padrão de retângulo (topos e fundos em faixas horizontais)
        for end_idx in range(len(df) - 1, 0, -1):
            # Encontrar os últimos 2-3 topos e fundos antes do índice final
            last_peaks = [idx for idx in peak_indices if idx < end_idx][-3:]
            last_troughs = [idx for idx in trough_indices if idx < end_idx][-3:]
            
            if len(last_peaks) < 2 or len(last_troughs) < 2:
                continue
            
            # Verificar se o padrão está dentro do tamanho permitido
            pattern_start = min(last_peaks[0], last_troughs[0])
            pattern_end = max(last_peaks[-1], last_troughs[-1])
            
            if pattern_end - pattern_start < min_length or pattern_end - pattern_start > max_length:
                continue
            
            # Verificar se os topos estão aproximadamente na mesma altura
            peak_values = [highs[idx] for idx in last_peaks]
            if max(peak_values) - min(peak_values) < 0.03 * min(peak_values):
                # Verificar se os fundos estão aproximadamente na mesma altura
                trough_values = [lows[idx] for idx in last_troughs]
                if max(trough_values) - min(trough_values) < 0.03 * min(trough_values):
                    # Calcular a amplitude do retângulo
                    rectangle_height = min(peak_values) - max(trough_values)
                    
                    # Verificar se a amplitude é significativa
                    if rectangle_height > 0 and rectangle_height / min(peak_values) > 0.02:
                        # Marcar o sinal na barra atual (possível ponto de quebra)
                        # O sinal depende da direção anterior
                        prev_trend = self._determine_prior_trend(df, pattern_start)
                        
                        if prev_trend > 0:  # Tendência anterior de alta
                            result.loc[df.index[end_idx], 'rectangle'] = 50  # Neutro, mas com viés de alta
                        elif prev_trend < 0:  # Tendência anterior de baixa
                            result.loc[df.index[end_idx], 'rectangle'] = -50  # Neutro, mas com viés de baixa
                        else:
                            result.loc[df.index[end_idx], 'rectangle'] = 0  # Neutro
        
        # Detectar padrão de bandeira (flag) e flâmula (pennant)
        for end_idx in range(len(df) - 1, 0, -1):
            # Primeiro, identificar um "mastro" (movimento forte e rápido)
            strong_move_found = False
            pole_start_idx = None
            pole_end_idx = None
            flag_type = None  # 'flag' ou 'pennant'
            
            # Verificar os últimos N períodos para encontrar um movimento forte
            for i in range(max(0, end_idx - 30), end_idx):
                # Calcular o movimento percentual em 5 barras
                if i >= 5:
                    pct_change = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
                    
                    # Um movimento forte seria > 5% em 5 barras
                    if abs(pct_change) > 0.05:
                        strong_move_found = True
                        pole_start_idx = i-5
                        pole_end_idx = i
                        break
            
            if strong_move_found:
                # Agora, verificar se depois do mastro temos uma consolidação (bandeira/flâmula)
                consolidation_start = pole_end_idx
                consolidation_end = end_idx
                
                # Verificar se a consolidação está dentro do tamanho permitido
                if consolidation_end - consolidation_start < min_length or consolidation_end - consolidation_start > max_length:
                    continue
                
                # Analisar o padrão de preço durante a consolidação
                highs_consolidation = df['high'].iloc[consolidation_start:consolidation_end+1]
                lows_consolidation = df['low'].iloc[consolidation_start:consolidation_end+1]
                
                # Calcular a tendência linear dos topos e fundos na consolidação
                x = np.arange(len(highs_consolidation))
                
                if len(x) > 2:  # Precisamos de pelo menos 3 pontos para uma linha significativa
                    # Tendência dos topos
                    slope_highs, _ = np.polyfit(x, highs_consolidation, 1)
                    
                    # Tendência dos fundos
                    slope_lows, _ = np.polyfit(x, lows_consolidation, 1)
                    
                    # Determinar o tipo de padrão com base nas tendências
                    if abs(slope_highs) < 0.001 and abs(slope_lows) < 0.001:
                        # Linhas horizontais → retângulo/bandeira
                        flag_type = 'flag'
                    elif (slope_highs < 0 and slope_lows > 0) or (slope_highs > 0 and slope_lows < 0):
                        # Linhas convergentes → flâmula
                        flag_type = 'pennant'
                    
                    # Determinar o sinal com base na direção do mastro
                    pole_direction = np.sign(df['close'].iloc[pole_end_idx] - df['close'].iloc[pole_start_idx])
                    
                    if flag_type == 'flag':
                        result.loc[df.index[end_idx], 'flag_pole'] = 100 * pole_direction
                    elif flag_type == 'pennant':
                        result.loc[df.index[end_idx], 'pennant'] = 100 * pole_direction
        
        return result
    
    def _determine_prior_trend(self, df: pd.DataFrame, start_idx: int, lookback: int = 20) -> int:
        """
        Determina a tendência anterior a um ponto específico.
        
        Args:
            df: DataFrame com dados OHLCV
            start_idx: Índice a partir do qual analisar a tendência anterior
            lookback: Número de barras para olhar para trás
            
        Returns:
            1 para tendência de alta, -1 para tendência de baixa, 0 para lateral
        """
        # Verificar se temos dados suficientes
        if start_idx < lookback:
            lookback = start_idx
        
        if lookback < 5:
            return 0  # Dados insuficientes
        
        # Pegar os preços de fechamento
        prices = df['close'].iloc[start_idx - lookback:start_idx].values
        
        # Calcular a tendência linear
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Determinar a direção da tendência com base na inclinação
        if slope > 0.001:  # Usar um limiar pequeno para evitar ruído
            return 1  # Tendência de alta
        elif slope < -0.001:
            return -1  # Tendência de baixa
        else:
            return 0  # Tendência lateral
    
    def plot_chart_patterns(self, df: pd.DataFrame, chart_signals: pd.DataFrame, 
                           window_size: int = 100, end_date: Optional[str] = None) -> plt.Figure:
        """
        Plota um gráfico com os padrões gráficos identificados.
        
        Args:
            df: DataFrame com dados OHLCV
            chart_signals: DataFrame com sinais de padrões gráficos
            window_size: Número de barras para exibir
            end_date: Data final para a visualização (opcional)
            
        Returns:
            Figura matplotlib com visualização de padrões
        """
        # Verificar se temos dados suficientes
        if df.empty or chart_signals.empty:
            logger.warning("Dados insuficientes para plotar padrões gráficos.")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center')
            return fig
        
        # Preparar os dados para visualização
        if end_date:
            try:
                end_idx = df.index.get_loc(end_date)
            except:
                end_idx = len(df) - 1
        else:
            end_idx = len(df) - 1
        
        start_idx = max(0, end_idx - window_size + 1)
        plot_df = df.iloc[start_idx:end_idx+1].copy()
        plot_signals = chart_signals.iloc[start_idx:end_idx+1].copy()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plotar os preços de fechamento
        ax.plot(plot_df.index, plot_df['close'], label='Preço de Fechamento')
        
        # Adicionar marcações para os padrões
        pattern_colors = {
            'head_and_shoulders': 'red',
            'inverse_head_and_shoulders': 'green',
            'double_top': 'red',
            'double_bottom': 'green',
            'triple_top': 'red',
            'triple_bottom': 'green',
            'rising_wedge': 'red',
            'falling_wedge': 'green',
            'ascending_triangle': 'blue',
            'descending_triangle': 'orange',
            'symmetric_triangle': 'purple',
            'rectangle': 'gray',
            'flag_pole': 'brown',
            'pennant': 'cyan'
        }
        
        # Dicionário para controle de anotações (evitar sobreposição)
        last_annotation = {}
        min_annotation_distance = 5  # Mínimo de barras entre anotações do mesmo tipo
        
        # Adicionar marcações para cada tipo de padrão
        for pattern in self.chart_patterns:
            last_annotation[pattern] = -min_annotation_distance * 2  # Inicializar fora do range
            
            for i, (idx, value) in enumerate(plot_signals[pattern].items()):
                if value != 0 and i - last_annotation[pattern] >= min_annotation_distance:
                    last_annotation[pattern] = i
                    
                    # Definir cor e marcador com base no valor
                    color = pattern_colors.get(pattern, 'black')
                    marker = '^' if value > 0 else 'v' if value < 0 else 'o'
                    
                    # Posição do marcador (acima para alta, abaixo para baixa)
                    if value > 0:
                        y_pos = plot_df['low'].iloc[i] * 0.99
                    else:
                        y_pos = plot_df['high'].iloc[i] * 1.01
                    
                    # Plotar marcador
                    ax.scatter(idx, y_pos, marker=marker, s=100, color=color, zorder=5)
                    
                    # Adicionar anotação com o nome do padrão
                    pattern_name = pattern.replace('_', ' ').title()
                    annotation_pos = y_pos * 0.98 if value < 0 else y_pos * 1.02
                    
                    ax.annotate(pattern_name, 
                              xy=(idx, y_pos),
                              xytext=(idx, annotation_pos),
                              fontsize=8,
                              ha='center',
                              color=color,
                              arrowprops=dict(arrowstyle='->', color=color))
        
        # Configurar eixos
        ax.set_title('Padrões Gráficos Detectados')
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço')
        ax.grid(True, alpha=0.3)
        
        # Formatar eixo x para datas
        if isinstance(plot_df.index, pd.DatetimeIndex):
            plt.gcf().autofmt_xdate()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        
        return fig
    
    def generate_pattern_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Gera um relatório completo de análise de padrões.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            Dicionário com resultados da análise de padrões
        """
        report = {}
        
        # Verificar dados
        if df.empty:
            logger.warning("Dados insuficientes para gerar relatório de padrões.")
            return {'erro': "Dados insuficientes para análise de padrões."}
        
        try:
            # 1. Detectar padrões de candlestick
            candlestick_signals = self.detect_candlestick_patterns(df)
            report['candlestick_signals'] = candlestick_signals
            
            # 2. Detectar padrões gráficos
            chart_signals = self.detect_chart_patterns(df)
            report['chart_signals'] = chart_signals
            
            # 3. Análise de padrões recentes (últimas 5 barras)
            recent_candles = candlestick_signals.iloc[-5:].copy()
            recent_chart = chart_signals.iloc[-5:].copy()
            
            # 4. Identificar padrões mais significativos recentes
            significant_patterns = []
            
            # Para padrões de candlestick
            for idx, row in recent_candles.iterrows():
                pattern_name, value, signal_type = self.get_strongest_pattern(row)
                
                if pattern_name != "Nenhum padrão":
                    signal_direction = "alta" if signal_type == "bullish" else "baixa" if signal_type == "bearish" else "neutro"
                    significant_patterns.append({
                        'data': idx,
                        'tipo': 'candlestick',
                        'padrao': pattern_name,
                        'direcao': signal_direction,
                        'forca': abs(value)
                    })
            
            # Para padrões gráficos
            for idx, row in recent_chart.iterrows():
                # Encontrar o padrão mais forte
                patterns_present = []
                for pattern in self.chart_patterns:
                    if row[pattern] != 0:
                        direction = "alta" if row[pattern] > 0 else "baixa" if row[pattern] < 0 else "neutro"
                        patterns_present.append({
                            'padrao': pattern.replace('_', ' ').title(),
                            'direcao': direction,
                            'forca': abs(row[pattern])
                        })
                
                if patterns_present:
                    # Ordenar por força e pegar o mais forte
                    strongest = max(patterns_present, key=lambda x: x['forca'])
                    significant_patterns.append({
                        'data': idx,
                        'tipo': 'grafico',
                        'padrao': strongest['padrao'],
                        'direcao': strongest['direcao'],
                        'forca': strongest['forca']
                    })
            
            report['significant_patterns'] = significant_patterns
            
            # 5. Estatísticas dos padrões
            stats = {
                'num_bullish_candles': (candlestick_signals['bullish_patterns'] > 0).sum(),
                'num_bearish_candles': (candlestick_signals['bearish_patterns'] > 0).sum(),
                'num_neutral_candles': (candlestick_signals['continuation_patterns'] > 0).sum(),
                'num_bullish_chart': (chart_signals['bullish_chart_patterns'] > 0).sum(),
                'num_bearish_chart': (chart_signals['bearish_chart_patterns'] > 0).sum(),
                'total_patterns_detected': len(significant_patterns)
            }
            report['statistics'] = stats
            
            # 6. Tendência geral
            bullish_score = stats['num_bullish_candles'] + stats['num_bullish_chart'] * 2
            bearish_score = stats['num_bearish_candles'] + stats['num_bearish_chart'] * 2
            
            if bullish_score > bearish_score * 1.5:
                trend = "fortemente altista"
            elif bullish_score > bearish_score * 1.1:
                trend = "altista"
            elif bearish_score > bullish_score * 1.5:
                trend = "fortemente baixista"
            elif bearish_score > bullish_score * 1.1:
                trend = "baixista"
            else:
                trend = "neutro/lateral"
            
            report['trend_analysis'] = {
                'tendencia': trend,
                'score_altista': bullish_score,
                'score_baixista': bearish_score
            }
            
            # 7. Recomendações com base na análise
            recommendations = self._generate_pattern_recommendations(report)
            report['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de padrões: {str(e)}")
            return {'erro': f"Erro ao gerar relatório: {str(e)}"}
    
    def _generate_pattern_recommendations(self, report_data: Dict[str, any]) -> List[str]:
        """
        Gera recomendações com base na análise de padrões.
        
        Args:
            report_data: Dicionário com resultados da análise de padrões
            
        Returns:
            Lista de recomendações
        """
        recommendations = []
        
        # Extrair dados importantes
        trend = report_data.get('trend_analysis', {}).get('tendencia', 'neutro')
        significant_patterns = report_data.get('significant_patterns', [])
        
        # 1. Recomendações baseadas na tendência geral
        if trend == "fortemente altista":
            recommendations.append(
                "A análise de padrões indica tendência fortemente altista. Considere posições de compra com stop abaixo dos suportes recentes."
            )
        elif trend == "altista":
            recommendations.append(
                "A análise de padrões sugere viés altista. Operações de compra em pullbacks são favorecidas."
            )
        elif trend == "fortemente baixista":
            recommendations.append(
                "A análise de padrões indica tendência fortemente baixista. Considere posições de venda com stop acima das resistências recentes."
            )
        elif trend == "baixista":
            recommendations.append(
                "A análise de padrões sugere viés baixista. Operações de venda em repiques são favorecidas."
            )
        else:
            recommendations.append(
                "A análise de padrões indica mercado neutro ou lateral. Operações de range (entre suporte e resistência) são favorecidas."
            )
        
        # 2. Recomendações com base nos padrões mais recentes
        recent_patterns = sorted(significant_patterns, key=lambda x: x['data'], reverse=True)[:3]
        
        if recent_patterns:
            pattern_desc = []
            
            for pattern in recent_patterns:
                direction = pattern['direcao']
                pattern_name = pattern['padrao']
                
                pattern_desc.append(f"{pattern_name} ({direction})")
            
            if len(pattern_desc) > 0:
                recommendations.append(
                    f"Padrões recentes mais significativos: {', '.join(pattern_desc)}."
                )
        
        # 3. Recomendações mais específicas com base nos padrões recentes
        if recent_patterns:
            latest_pattern = recent_patterns[0]
            
            if latest_pattern['direcao'] == 'alta':
                recommendations.append(
                    f"O padrão recente {latest_pattern['padrao']} sugere potencial movimento de alta. "
                    f"Considere entradas de compra com stop abaixo do mínimo recente."
                )
            elif latest_pattern['direcao'] == 'baixa':
                recommendations.append(
                    f"O padrão recente {latest_pattern['padrao']} sugere potencial movimento de baixa. "
                    f"Considere entradas de venda com stop acima do máximo recente."
                )
        
        # 4. Recomendação geral se não houver padrões significativos
        if not significant_patterns:
            recommendations.append(
                "Não foram detectados padrões significativos recentemente. "
                "Recomenda-se aguardar sinais mais claros antes de realizar operações direcionais."
            )
        
        return recommendations