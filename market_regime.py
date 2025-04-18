"""
Módulo para detecção automática de regimes de mercado.

Este módulo implementa algoritmos para classificar o estado atual do mercado
baseado em dados históricos de preço e volume, permitindo que o robô adapte
suas estratégias de acordo com o regime identificado.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enumeração dos possíveis regimes de mercado"""
    TRENDING_UP = "trending_up"          # Tendência de alta clara
    TRENDING_DOWN = "trending_down"      # Tendência de baixa clara
    RANGING = "ranging"                  # Mercado em lateralização
    VOLATILE = "volatile"                # Alta volatilidade sem direção clara
    BREAKOUT_UP = "breakout_up"          # Rompimento para cima
    BREAKOUT_DOWN = "breakout_down"      # Rompimento para baixo
    MEAN_REVERTING = "mean_reverting"    # Retorno à média
    MOMENTUM = "momentum"                # Movimento com momentum
    UNDEFINED = "undefined"              # Regime não identificado


class MarketRegimeDetector:
    """Classe para detectar regimes de mercado e adaptar estratégias"""
    
    def __init__(self, 
                 lookback_period: int = 60,
                 volatility_window: int = 20,
                 trend_threshold: float = 0.3,
                 volatility_threshold: float = 0.8,
                 range_threshold: float = 0.2,
                 use_kmeans: bool = True,
                 n_clusters: int = 5):
        """
        Inicializa o detector de regime de mercado.
        
        Args:
            lookback_period: Número de períodos para análise
            volatility_window: Janela para cálculo de volatilidade
            trend_threshold: Limiar para classificar como tendência
            volatility_threshold: Limiar para classificar como volátil
            range_threshold: Limiar para classificar como lateralização
            use_kmeans: Usar clustering para classificação de regimes
            n_clusters: Número de clusters para algoritmo KMeans
        """
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.range_threshold = range_threshold
        self.use_kmeans = use_kmeans
        self.n_clusters = n_clusters
        self.current_regime = MarketRegime.UNDEFINED
        self.regime_history = []
        self.kmeans = None
        self.scaler = StandardScaler()
        
        # Parâmetros por regime
        self.regime_params = {
            MarketRegime.TRENDING_UP: {
                "confidence_threshold": 0.7,
                "stop_loss_mult": 1.2,
                "take_profit_mult": 2.0,
                "indicators": {
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "macd_fast": 12,
                    "macd_slow": 26
                }
            },
            MarketRegime.TRENDING_DOWN: {
                "confidence_threshold": 0.7,
                "stop_loss_mult": 1.2,
                "take_profit_mult": 2.0,
                "indicators": {
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "macd_fast": 12,
                    "macd_slow": 26
                }
            },
            MarketRegime.RANGING: {
                "confidence_threshold": 0.8,  # Maior confiança para mercados laterais
                "stop_loss_mult": 1.0,
                "take_profit_mult": 1.5,
                "indicators": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "bb_period": 20,
                    "bb_std": 2.0
                }
            },
            MarketRegime.VOLATILE: {
                "confidence_threshold": 0.85,  # Maior confiança para mercados voláteis
                "stop_loss_mult": 1.5,         # Stop loss mais largo
                "take_profit_mult": 2.5,       # Take profit mais ambicioso
                "indicators": {
                    "atr_period": 14,
                    "volatility_factor": 2.0
                }
            },
            MarketRegime.BREAKOUT_UP: {
                "confidence_threshold": 0.75,
                "stop_loss_mult": 1.2,
                "take_profit_mult": 2.5,
                "indicators": {
                    "donchian_period": 20,
                    "volume_mult": 1.5
                }
            },
            MarketRegime.BREAKOUT_DOWN: {
                "confidence_threshold": 0.75,
                "stop_loss_mult": 1.2,
                "take_profit_mult": 2.5,
                "indicators": {
                    "donchian_period": 20,
                    "volume_mult": 1.5
                }
            },
            MarketRegime.MEAN_REVERTING: {
                "confidence_threshold": 0.8,
                "stop_loss_mult": 1.0,
                "take_profit_mult": 1.5,
                "indicators": {
                    "bollinger_period": 20,
                    "bollinger_dev": 2.0,
                    "mean_period": 50
                }
            },
            MarketRegime.MOMENTUM: {
                "confidence_threshold": 0.7,
                "stop_loss_mult": 1.3,
                "take_profit_mult": 2.0,
                "indicators": {
                    "momentum_period": 10,
                    "adx_period": 14,
                    "adx_threshold": 25
                }
            },
            MarketRegime.UNDEFINED: {
                "confidence_threshold": 0.8,
                "stop_loss_mult": 1.0,
                "take_profit_mult": 1.5,
                "indicators": {
                    "sma_fast": 10,
                    "sma_slow": 20,
                    "rsi_period": 14
                }
            }
        }
    
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detecta o regime atual do mercado baseado nos dados históricos.
        
        Args:
            data: DataFrame com dados OHLCV
            
        Returns:
            Regime de mercado identificado
        """
        if len(data) < self.lookback_period:
            logger.warning(f"Dados insuficientes para detecção de regime. Necessário {self.lookback_period} períodos, recebido {len(data)}")
            return MarketRegime.UNDEFINED
            
        # Usar os últimos N períodos conforme configurado
        df = data.tail(self.lookback_period).copy()
        
        # Calcular features para detecção de regime
        features = self._calculate_regime_features(df)
        
        if self.use_kmeans:
            # Usar machine learning (K-means) para classificar o regime
            regime = self._classify_regime_kmeans(features)
        else:
            # Usar método baseado em regras para classificar o regime
            regime = self._classify_regime_rules(features, df)
        
        # Registrar mudança de regime
        if regime != self.current_regime:
            logger.info(f"Regime de mercado alterado: {self.current_regime.value} -> {regime.value}")
            
        # Atualizar regime atual e histórico
        self.current_regime = regime
        self.regime_history.append(regime)
        
        # Limitar tamanho do histórico
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
            
        return regime
    
    def _calculate_regime_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula indicadores e características necessárias para identificação de regime.
        
        Args:
            df: DataFrame com dados de preço
            
        Returns:
            Dicionário com as características calculadas
        """
        # Preço de fechamento e volume
        close_prices = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else np.ones_like(close_prices)
        
        # Calcular retornos
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Calcular médias móveis
        sma_fast = df['close'].rolling(window=10).mean().iloc[-1]
        sma_slow = df['close'].rolling(window=30).mean().iloc[-1]
        
        # Calcular bandas de Bollinger
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        bb_width = (upper_band - lower_band) / sma_20
        
        # Calcular ATR (Average True Range) para volatilidade
        high = df['high'].values
        low = df['low'].values
        close_shift = np.concatenate(([close_prices[0]], close_prices[:-1]))
        tr1 = high - low
        tr2 = np.abs(high - close_shift)
        tr3 = np.abs(low - close_shift)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr[-self.volatility_window:])
        
        # Calcular RSI
        delta = df['close'].diff().dropna()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14).mean().iloc[-1]
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
            
        # Calcular inclinação (tendência)
        x = np.arange(len(close_prices))
        y = close_prices
        slope, _ = np.polyfit(x, y, 1)
        normalized_slope = slope / close_prices.mean()
        
        # Calcular retorno acumulado
        cum_return = (close_prices[-1] / close_prices[0]) - 1
        
        # Calcular volume relativo
        avg_volume = np.mean(volumes)
        rel_volume = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # Calcular range do preço
        price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
        
        # Calcular volatilidade
        volatility = np.std(returns) * np.sqrt(252)  # Anualizada
        
        # Calcular distância da média
        dist_from_mean = (close_prices[-1] - np.mean(close_prices)) / np.std(close_prices)
        
        # Calcular indicadores adicionais
        price_momentum = close_prices[-1] / close_prices[-10] - 1 if len(close_prices) >= 10 else 0
        avg_bar_size = np.mean((df['high'] - df['low']).tail(10)) / close_prices[-1]
        
        # Retornar características como dicionário
        return {
            "sma_diff": (sma_fast - sma_slow) / close_prices[-1],
            "bb_width": bb_width.iloc[-1],
            "atr_rel": atr / close_prices[-1],
            "rsi": rsi,
            "slope": normalized_slope,
            "cum_return": cum_return,
            "volatility": volatility,
            "rel_volume": rel_volume,
            "price_range": price_range,
            "dist_from_mean": dist_from_mean,
            "price_momentum": price_momentum,
            "avg_bar_size": avg_bar_size
        }
    
    def _classify_regime_rules(self, features: Dict[str, float], df: pd.DataFrame) -> MarketRegime:
        """
        Classifica o regime de mercado usando regras predefinidas.
        
        Args:
            features: Dicionário com características calculadas
            df: DataFrame com dados de preço
            
        Returns:
            Regime de mercado classificado
        """
        # Extrair características principais
        slope = features["slope"]
        volatility = features["volatility"]
        rsi = features["rsi"]
        sma_diff = features["sma_diff"]
        price_range = features["price_range"]
        dist_from_mean = features["dist_from_mean"]
        price_momentum = features["price_momentum"]
        rel_volume = features["rel_volume"]
        
        # Verificar breakouts
        if sma_diff > self.trend_threshold and rel_volume > 1.5 and price_momentum > 0.02:
            return MarketRegime.BREAKOUT_UP
        elif sma_diff < -self.trend_threshold and rel_volume > 1.5 and price_momentum < -0.02:
            return MarketRegime.BREAKOUT_DOWN
            
        # Verificar tendência
        if slope > self.trend_threshold and sma_diff > 0:
            return MarketRegime.TRENDING_UP
        elif slope < -self.trend_threshold and sma_diff < 0:
            return MarketRegime.TRENDING_DOWN
            
        # Verificar alta volatilidade
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE
            
        # Verificar lateralização
        if abs(slope) < self.range_threshold and price_range < self.range_threshold:
            return MarketRegime.RANGING
            
        # Verificar momentum
        if abs(price_momentum) > 0.01 and abs(dist_from_mean) > 1.0:
            return MarketRegime.MOMENTUM
            
        # Verificar retorno à média
        if abs(dist_from_mean) > 2.0:
            return MarketRegime.MEAN_REVERTING
            
        # Padrão - não foi possível classificar com confiança
        return MarketRegime.UNDEFINED
    
    def _classify_regime_kmeans(self, features: Dict[str, float]) -> MarketRegime:
        """
        Classifica o regime de mercado usando KMeans clustering.
        
        Args:
            features: Dicionário com características calculadas
            
        Returns:
            Regime de mercado classificado
        """
        # Converter características para array
        feature_vector = np.array([list(features.values())])
        
        # Inicializar modelo se necessário
        if self.kmeans is None:
            # Como precisamos de mais dados para treinar o modelo, 
            # usamos regras predefinidas nesse caso
            return self._map_cluster_to_regime(0, feature_vector[0])
        
        # Normalizar dados
        feature_vector = self.scaler.transform(feature_vector)
        
        # Predizer cluster
        cluster = self.kmeans.predict(feature_vector)[0]
        
        # Mapear cluster para regime
        return self._map_cluster_to_regime(cluster, feature_vector[0])
    
    def _map_cluster_to_regime(self, cluster: int, features: np.ndarray) -> MarketRegime:
        """
        Mapeia um cluster para um regime de mercado.
        
        Args:
            cluster: Número do cluster
            features: Vetor de características
            
        Returns:
            Regime de mercado correspondente
        """
        # Ordenar características por importância para esse cluster
        # Isso é para quando tivermos um modelo treinado
        
        # Por enquanto, usaremos regras baseadas nos valores das características
        if len(features) < 5:
            return MarketRegime.UNDEFINED
            
        # Extrair características principais
        slope_idx = 4  # índice da inclinação no vetor de características
        volatility_idx = 6  # índice da volatilidade no vetor de características
        price_range_idx = 8  # índice do range de preço no vetor de características
        
        slope = features[slope_idx]
        volatility = features[volatility_idx]
        price_range = features[price_range_idx]
        
        # Classificação baseada em características principais
        if slope > 0.5:
            return MarketRegime.TRENDING_UP
        elif slope < -0.5:
            return MarketRegime.TRENDING_DOWN
        elif volatility > 1.5:
            return MarketRegime.VOLATILE
        elif price_range < 0.2:
            return MarketRegime.RANGING
        else:
            return MarketRegime.UNDEFINED
    
    def train_classifier(self, historical_data: pd.DataFrame, n_regimes: Optional[int] = None) -> None:
        """
        Treina o classificador KMeans com dados históricos.
        
        Args:
            historical_data: DataFrame com dados históricos OHLCV
            n_regimes: Número de regimes/clusters (opcional)
        """
        if len(historical_data) < self.lookback_period * 5:
            logger.warning(f"Dados insuficientes para treinar o classificador. Necessário pelo menos {self.lookback_period * 5} períodos")
            return
            
        # Usar número de clusters fornecido ou o padrão
        n_clusters = n_regimes if n_regimes is not None else self.n_clusters
        
        # Preparar dados para treinamento
        all_features = []
        
        # Criar janelas deslizantes para extrair features
        for i in range(self.lookback_period, len(historical_data)):
            window = historical_data.iloc[i-self.lookback_period:i]
            features = self._calculate_regime_features(window)
            all_features.append(list(features.values()))
        
        if not all_features:
            logger.warning("Nenhuma característica calculada para treinamento")
            return
            
        # Converter para array numpy
        X = np.array(all_features)
        
        # Normalizar dados
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Treinar modelo KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)
        
        logger.info(f"Modelo KMeans treinado com {n_clusters} clusters e {len(X)} amostras")
    
    def get_regime_parameters(self) -> Dict[str, Any]:
        """
        Obtém os parâmetros de trading otimizados para o regime atual.
        
        Returns:
            Dicionário com parâmetros otimizados para o regime atual
        """
        return self.regime_params[self.current_regime]
    
    def adapt_strategy(self, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapta os parâmetros de estratégia com base no regime atual.
        
        Args:
            strategy_params: Parâmetros atuais da estratégia
            
        Returns:
            Parâmetros adaptados ao regime atual
        """
        # Obter parâmetros para o regime atual
        regime_params = self.get_regime_parameters()
        
        # Criar cópia dos parâmetros originais
        adapted_params = strategy_params.copy()
        
        # Ajustar parâmetros de confiança e risco
        adapted_params["confidence_threshold"] = regime_params["confidence_threshold"]
        
        # Ajustar stop loss e take profit
        if "stop_loss_ticks" in adapted_params:
            base_sl = adapted_params["stop_loss_ticks"]
            adapted_params["stop_loss_ticks"] = int(base_sl * regime_params["stop_loss_mult"])
            
        if "take_profit_ticks" in adapted_params:
            base_tp = adapted_params["take_profit_ticks"]
            adapted_params["take_profit_ticks"] = int(base_tp * regime_params["take_profit_mult"])
        
        # Ajustar parâmetros de indicadores técnicos
        if "technical_indicators" in adapted_params:
            for indicator, value in regime_params["indicators"].items():
                if indicator in adapted_params["technical_indicators"]:
                    adapted_params["technical_indicators"][indicator] = value
        
        # Log das adaptações
        logger.info(f"Estratégia adaptada para regime {self.current_regime.value}")
        
        return adapted_params
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo do regime atual e recomendações.
        
        Returns:
            Dicionário com informações sobre o regime atual
        """
        regime = self.current_regime
        
        # Contagens de regimes recentes
        regime_counts = {}
        for r in self.regime_history[-20:]:  # últimos 20 regimes
            regime_counts[r] = regime_counts.get(r, 0) + 1
            
        # Calcular estabilidade do regime (percentual do regime dominante)
        stability = max(regime_counts.values()) / len(self.regime_history[-20:]) if self.regime_history else 0
        
        # Definir recomendações baseadas no regime
        recommendations = {
            MarketRegime.TRENDING_UP: [
                "Favorece posições compradas",
                "Usar médias móveis direcionais",
                "Trailing stops recomendados"
            ],
            MarketRegime.TRENDING_DOWN: [
                "Favorece posições vendidas",
                "Usar médias móveis direcionais",
                "Trailing stops recomendados"
            ],
            MarketRegime.RANGING: [
                "Operar reversões nos extremos do range",
                "Usar osciladores (RSI, Estocástico)",
                "Stop loss próximo aos extremos do range"
            ],
            MarketRegime.VOLATILE: [
                "Reduzir tamanho das posições",
                "Usar stops mais largos",
                "Considerar operar apenas em direções confirmadas"
            ],
            MarketRegime.BREAKOUT_UP: [
                "Entrar em rompimentos confirmados",
                "Volume aumentado é confirmação adicional",
                "Considerar entradas rápidas após o rompimento"
            ],
            MarketRegime.BREAKOUT_DOWN: [
                "Entrar em rompimentos confirmados",
                "Volume aumentado é confirmação adicional",
                "Considerar entradas rápidas após o rompimento"
            ],
            MarketRegime.MEAN_REVERTING: [
                "Operar reversões à média",
                "Usar bandas de Bollinger",
                "Entrar apenas em condições extremas"
            ],
            MarketRegime.MOMENTUM: [
                "Seguir a direção dominante",
                "Utilizar indicadores de momentum",
                "Entrar em retração temporária na direção principal"
            ],
            MarketRegime.UNDEFINED: [
                "Condições de mercado incertas",
                "Reduzir tamanho das posições",
                "Aguardar configurações mais claras"
            ]
        }
        
        return {
            "current_regime": regime.value,
            "stability": stability,
            "regime_history": [r.value for r in self.regime_history[-5:]],
            "regime_distribution": {k.value: v for k, v in regime_counts.items()},
            "recommendations": recommendations[regime]
        }