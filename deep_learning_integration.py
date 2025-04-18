"""
Módulo para integração de modelos de deep learning com o sistema de trading.

Este módulo facilita a conexão entre os modelos de deep learning (LSTM e Transformer)
e o restante do sistema de trading WINFUT.
"""
import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta

# Importar gerenciador de modelos deep learning
from deep_learning_models import DeepLearningModels

# Configuração de logging
logger = logging.getLogger(__name__)

class DeepLearningIntegration:
    """Classe para integrar modelos de deep learning ao sistema de trading."""
    
    def __init__(self, 
                 model_dir: str = "models",
                 use_gpu: bool = False,
                 default_lookback: int = 60,
                 default_horizon: int = 5,
                 confidence_threshold: float = 0.7):
        """
        Inicializa a integração de deep learning.
        
        Args:
            model_dir: Diretório para salvar/carregar modelos
            use_gpu: Se True, tenta usar GPU para treinamento se disponível
            default_lookback: Janela de lookback padrão
            default_horizon: Horizonte de previsão padrão
            confidence_threshold: Limiar de confiança para sinais de trading
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.default_lookback = default_lookback
        self.default_horizon = default_horizon
        self.confidence_threshold = confidence_threshold
        
        # Inicializar gerenciador de modelos
        self.dl_models = DeepLearningModels(
            model_dir=model_dir,
            use_gpu=use_gpu,
            lookback_window=default_lookback,
            forecast_horizon=default_horizon
        )
        
        # Verificar se modelos existem
        self.models_available = self.load_models()
        
        # Estado de treinamento
        self.is_training = False
        self.last_training_results = None
        
        # Histórico de previsões
        self.prediction_history = []
        
        # Criar diretório de modelos se não existir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_models(self) -> bool:
        """
        Carrega modelos salvos.
        
        Returns:
            True se pelo menos um modelo for carregado com sucesso
        """
        try:
            success = self.dl_models.load_model("both")
            if success:
                logger.info("Pelo menos um modelo de deep learning carregado com sucesso")
            else:
                logger.warning("Nenhum modelo de deep learning encontrado")
            return success
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {str(e)}")
            return False
    
    def check_models_available(self) -> Dict[str, bool]:
        """
        Verifica quais modelos estão disponíveis.
        
        Returns:
            Dicionário indicando disponibilidade dos modelos
        """
        return {
            "lstm": self.dl_models.lstm_model is not None,
            "transformer": self.dl_models.transformer_model is not None,
            "tensorflow": self.dl_models.tensorflow_available,
            "torch": self.dl_models.torch_available
        }
    
    def prepare_training_data(self, 
                           data: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None,
                           target_column: str = "close",
                           lookback: Optional[int] = None,
                           horizon: Optional[int] = None,
                           train_ratio: float = 0.8) -> Tuple:
        """
        Prepara dados para treinamento dos modelos.
        
        Args:
            data: DataFrame com dados históricos
            feature_columns: Lista de colunas para usar como features
            target_column: Coluna alvo para previsão
            lookback: Janela de lookback (usa default se None)
            horizon: Horizonte de previsão (usa default se None)
            train_ratio: Proporção de dados para treinamento
            
        Returns:
            Tupla com dados de treinamento
        """
        try:
            # Verificar colunas obrigatórias
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in data.columns for col in required_columns):
                logger.error("Dados não contêm todas as colunas OHLCV necessárias")
                return None
                
            # Atualizar parâmetros se fornecidos
            if feature_columns:
                self.dl_models.feature_columns = feature_columns
                
            if target_column:
                self.dl_models.target_column = target_column
                
            if lookback:
                self.dl_models.lookback_window = lookback
                
            if horizon:
                self.dl_models.forecast_horizon = horizon
            
            # Preparar dados
            training_data = self.dl_models.prepare_data(
                data,
                train_ratio=train_ratio,
                normalize=True,
                return_scalers=True
            )
            
            logger.info(f"Dados preparados para treinamento: {len(training_data[0])} exemplos")
            return training_data
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {str(e)}")
            return None
    
    def train_models(self, 
                   data: pd.DataFrame,
                   models: List[str] = ["lstm", "transformer"],
                   feature_columns: Optional[List[str]] = None,
                   target_column: str = "close",
                   lookback: Optional[int] = None,
                   horizon: Optional[int] = None,
                   epochs: int = 50,
                   batch_size: int = 64,
                   patience: int = 10,
                   train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Treina modelos de deep learning.
        
        Args:
            data: DataFrame com dados históricos
            models: Lista de modelos para treinar ("lstm", "transformer")
            feature_columns: Lista de colunas para usar como features
            target_column: Coluna alvo para previsão
            lookback: Janela de lookback (usa default se None)
            horizon: Horizonte de previsão (usa default se None)
            epochs: Número de épocas para treinamento
            batch_size: Tamanho do batch
            patience: Número de épocas sem melhoria para early stopping
            train_ratio: Proporção de dados para treinamento
            
        Returns:
            Dicionário com resultados do treinamento
        """
        if self.is_training:
            logger.warning("Treinamento já em andamento. Aguarde a conclusão.")
            return {"status": "error", "message": "Treinamento já em andamento"}
            
        self.is_training = True
        results = {"status": "error", "message": ""}
        
        try:
            # Preparar dados
            train_data = self.prepare_training_data(
                data,
                feature_columns=feature_columns,
                target_column=target_column,
                lookback=lookback,
                horizon=horizon,
                train_ratio=train_ratio
            )
            
            if not train_data:
                self.is_training = False
                return {"status": "error", "message": "Falha ao preparar dados de treinamento"}
                
            X_train, y_train, X_test, y_test, feature_scaler, target_scaler = train_data
            
            # Resultados de treinamento
            training_results = {}
            
            # Treinar LSTM se solicitado
            if "lstm" in models and self.dl_models.tensorflow_available:
                # Construir modelo
                self.dl_models.build_lstm_model()
                
                # Treinar modelo
                lstm_results = self.dl_models.train_lstm(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    verbose=1
                )
                
                if lstm_results:
                    training_results["lstm"] = lstm_results
                    logger.info("Treinamento LSTM concluído com sucesso")
                else:
                    logger.error("Falha no treinamento LSTM")
            
            # Treinar Transformer se solicitado
            if "transformer" in models and self.dl_models.torch_available:
                # Construir modelo
                self.dl_models.build_transformer_model()
                
                # Treinar modelo
                transformer_results = self.dl_models.train_transformer(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    verbose=True
                )
                
                if transformer_results:
                    training_results["transformer"] = transformer_results
                    logger.info("Treinamento Transformer concluído com sucesso")
                else:
                    logger.error("Falha no treinamento Transformer")
            
            # Avaliar modelos
            evaluation = self.dl_models.evaluate_models(X_test, y_test)
            training_results["evaluation"] = evaluation
            
            # Salvar modelos
            self.dl_models.save_model("both")
            
            # Atualizar status de disponibilidade
            self.models_available = True
            
            # Armazenar resultados
            self.last_training_results = {
                "training_results": training_results,
                "feature_columns": self.dl_models.feature_columns,
                "target_column": self.dl_models.target_column,
                "lookback_window": self.dl_models.lookback_window,
                "forecast_horizon": self.dl_models.forecast_horizon,
                "data_shape": data.shape,
                "timestamp": datetime.now().isoformat()
            }
            
            results = {
                "status": "success",
                "message": "Treinamento concluído com sucesso",
                "results": self.last_training_results
            }
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {str(e)}")
            results = {"status": "error", "message": f"Erro durante treinamento: {str(e)}"}
            
        finally:
            self.is_training = False
            
        return results
    
    def get_forecast(self, 
                   data: pd.DataFrame,
                   model_type: str = "ensemble",
                   n_future: Optional[int] = None) -> Dict[str, Any]:
        """
        Gera previsão com modelos de deep learning.
        
        Args:
            data: DataFrame com dados atuais
            model_type: Tipo de modelo a usar ("lstm", "transformer" ou "ensemble")
            n_future: Número de períodos futuros (usa forecast_horizon se None)
            
        Returns:
            Dicionário com previsões
        """
        if not self.models_available:
            logger.warning("Nenhum modelo disponível. Treine modelos primeiro.")
            return {"status": "error", "message": "Nenhum modelo disponível"}
            
        try:
            # Fazer previsão
            forecast_result = self.dl_models.forecast_next_period(data, model_type)
            
            if not forecast_result:
                return {"status": "error", "message": "Falha ao gerar previsão"}
                
            # Adicionar ao histórico
            self.prediction_history.append({
                "timestamp": datetime.now(),
                "forecast": forecast_result["forecast"].tolist(),
                "model_type": model_type
            })
            
            # Manter no máximo 100 previsões no histórico
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
                
            # Criar resultado
            result = {
                "status": "success",
                "forecast": forecast_result["forecast"].tolist(),
                "timestamp": forecast_result["timestamp"],
                "model_type": model_type,
                "lookback_window": self.dl_models.lookback_window,
                "forecast_horizon": self.dl_models.forecast_horizon
            }
            
            # Adicionar informações sobre tendência
            trend_direction = self._analyze_trend(forecast_result["forecast"])
            result["trend"] = trend_direction
            
            # Gerar sinal de trading se a confiança for alta
            if self._calculate_signal_confidence(forecast_result["forecast"]) >= self.confidence_threshold:
                result["signal"] = "buy" if trend_direction == "up" else "sell" if trend_direction == "down" else "neutral"
            else:
                result["signal"] = "neutral"
                
            return result
            
        except Exception as e:
            logger.error(f"Erro ao gerar previsão: {str(e)}")
            return {"status": "error", "message": f"Erro ao gerar previsão: {str(e)}"}
    
    def _analyze_trend(self, forecast: np.ndarray) -> str:
        """
        Analisa tendência a partir da previsão.
        
        Args:
            forecast: Array com valores previstos
            
        Returns:
            Direção da tendência ("up", "down" ou "neutral")
        """
        # Calcular coeficiente angular da linha de tendência
        x = np.arange(len(forecast))
        slope, _ = np.polyfit(x, forecast, 1)
        
        # Determinar direção com base na inclinação
        if slope > 0.001:  # Tendência de alta
            return "up"
        elif slope < -0.001:  # Tendência de baixa
            return "down"
        else:  # Tendência lateral
            return "neutral"
    
    def _calculate_signal_confidence(self, forecast: np.ndarray) -> float:
        """
        Calcula a confiança do sinal de trading.
        
        Args:
            forecast: Array com valores previstos
            
        Returns:
            Valor de confiança (0-1)
        """
        # Calcular variação percentual do primeiro ao último valor
        pct_change = (forecast[-1] - forecast[0]) / forecast[0]
        
        # Calcular coeficiente angular normalizado
        x = np.arange(len(forecast))
        slope, _ = np.polyfit(x, forecast, 1)
        normalized_slope = slope / np.mean(forecast)
        
        # Calcular R² da linha de tendência
        p = np.poly1d([slope, _])
        trend_line = p(x)
        ss_tot = np.sum((forecast - np.mean(forecast)) ** 2)
        ss_res = np.sum((forecast - trend_line) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Combinação ponderada dos fatores
        confidence = (0.4 * abs(pct_change * 10)) + (0.4 * abs(normalized_slope * 100)) + (0.2 * r2)
        
        # Limitar a 1.0
        return min(max(confidence, 0.0), 1.0)
    
    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """
        Obtém o histórico de previsões.
        
        Returns:
            Lista com histórico de previsões
        """
        return self.prediction_history
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Obtém resumo dos modelos disponíveis.
        
        Returns:
            Dicionário com informações sobre os modelos
        """
        available_models = self.check_models_available()
        
        result = {
            "models_available": self.models_available,
            "available_types": available_models,
            "last_training": self.last_training_results.get("timestamp", "Nunca") if self.last_training_results else "Nunca",
            "lookback_window": self.dl_models.lookback_window,
            "forecast_horizon": self.dl_models.forecast_horizon,
            "feature_columns": self.dl_models.feature_columns,
            "target_column": self.dl_models.target_column,
            "confidence_threshold": self.confidence_threshold
        }
        
        return result
    
    def plot_forecast(self, 
                    data: pd.DataFrame, 
                    forecast_result: Dict[str, Any],
                    last_n_periods: int = 30,
                    use_plotly: bool = True) -> Any:
        """
        Plota dados históricos e previsão.
        
        Args:
            data: DataFrame com dados históricos
            forecast_result: Resultado da previsão
            last_n_periods: Número de períodos históricos para mostrar
            use_plotly: Se True, usa Plotly em vez de Matplotlib
            
        Returns:
            Figura do gráfico
        """
        if not forecast_result or forecast_result.get("status") != "success":
            logger.error("Nenhuma previsão válida para plotar")
            return None
            
        # Obter dados históricos
        historical_data = data.iloc[-last_n_periods:].copy()
        
        # Obter previsão
        forecast_values = forecast_result["forecast"]
        horizon = len(forecast_values)
        
        # Criar datas futuras
        last_date = historical_data.index[-1]
        future_dates = [last_date + timedelta(minutes=i+1) for i in range(horizon)]
        
        # Verificar se estamos usando Plotly
        if use_plotly:
            try:
                # Criar figura
                fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
                
                # Adicionar dados históricos
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=historical_data["close"],
                        mode="lines",
                        name="Histórico",
                        line=dict(color="blue", width=2)
                    )
                )
                
                # Adicionar previsão
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=forecast_values,
                        mode="lines",
                        name="Previsão",
                        line=dict(color="red", width=2, dash="dash")
                    )
                )
                
                # Adicionar área sombreada para intervalo de previsão
                # Supondo uma incerteza de 5% para cima e para baixo
                uncertainty = 0.05
                upper_bound = [val * (1 + uncertainty) for val in forecast_values]
                lower_bound = [val * (1 - uncertainty) for val in forecast_values]
                
                fig.add_trace(
                    go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        line=dict(color="rgba(255, 0, 0, 0)"),
                        name="Intervalo de Previsão"
                    )
                )
                
                # Adicionar último valor histórico e primeira previsão
                fig.add_trace(
                    go.Scatter(
                        x=[historical_data.index[-1], future_dates[0]],
                        y=[historical_data["close"].iloc[-1], forecast_values[0]],
                        mode="lines",
                        line=dict(color="green", width=1),
                        showlegend=False
                    )
                )
                
                # Adicionar tendência
                trend = forecast_result.get("trend", "neutral")
                color = "green" if trend == "up" else "red" if trend == "down" else "gray"
                
                fig.add_annotation(
                    x=future_dates[horizon//2],
                    y=max(forecast_values),
                    text=f"Tendência: {trend.upper()}",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor=color,
                    arrowsize=1,
                    arrowwidth=2,
                    ax=-50,
                    ay=-40
                )
                
                # Adicionar confiança
                confidence = self._calculate_signal_confidence(np.array(forecast_values)) * 100
                signal = forecast_result.get("signal", "neutral")
                signal_color = "green" if signal == "buy" else "red" if signal == "sell" else "gray"
                
                fig.add_annotation(
                    x=future_dates[horizon//2],
                    y=min(forecast_values),
                    text=f"Confiança: {confidence:.1f}% | Sinal: {signal.upper()}",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor=signal_color,
                    arrowsize=1,
                    arrowwidth=2,
                    ax=50,
                    ay=40
                )
                
                # Configurar layout
                model_type = forecast_result.get("model_type", "")
                timestamp = forecast_result.get("timestamp", "")
                
                fig.update_layout(
                    title=f"Previsão Deep Learning ({model_type.upper()}) - {timestamp}",
                    xaxis_title="Data",
                    yaxis_title="Preço",
                    legend_title="Séries",
                    template="plotly_white",
                    height=600
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Erro ao criar gráfico Plotly: {str(e)}")
                use_plotly = False
        
        # Fallback para Matplotlib
        if not use_plotly:
            try:
                plt.figure(figsize=(12, 6))
                
                # Plotar dados históricos
                plt.plot(historical_data.index, historical_data["close"], label="Histórico", color="blue")
                
                # Plotar previsão
                plt.plot(future_dates, forecast_values, label="Previsão", color="red", linestyle="--")
                
                # Conectar último ponto histórico com primeira previsão
                plt.plot([historical_data.index[-1], future_dates[0]], 
                       [historical_data["close"].iloc[-1], forecast_values[0]], 
                       color="green")
                
                # Adicionar área sombreada para intervalo de previsão
                uncertainty = 0.05
                upper_bound = [val * (1 + uncertainty) for val in forecast_values]
                lower_bound = [val * (1 - uncertainty) for val in forecast_values]
                plt.fill_between(future_dates, upper_bound, lower_bound, color="red", alpha=0.1)
                
                # Adicionar informações
                trend = forecast_result.get("trend", "neutral")
                confidence = self._calculate_signal_confidence(np.array(forecast_values)) * 100
                signal = forecast_result.get("signal", "neutral")
                model_type = forecast_result.get("model_type", "")
                timestamp = forecast_result.get("timestamp", "")
                
                plt.title(f"Previsão Deep Learning ({model_type.upper()}) - {timestamp}")
                plt.figtext(0.15, 0.05, f"Tendência: {trend.upper()} | Confiança: {confidence:.1f}% | Sinal: {signal.upper()}")
                
                plt.xlabel("Data")
                plt.ylabel("Preço")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                return plt.gcf()
                
            except Exception as e:
                logger.error(f"Erro ao criar gráfico Matplotlib: {str(e)}")
                return None
    
    def generate_trading_signal(self, 
                              data: pd.DataFrame,
                              model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Gera sinal de trading baseado em deep learning.
        
        Args:
            data: DataFrame com dados atuais
            model_type: Tipo de modelo a usar
            
        Returns:
            Dicionário com sinal de trading
        """
        # Obter previsão
        forecast = self.get_forecast(data, model_type)
        
        if forecast.get("status") != "success":
            return {"signal": "neutral", "confidence": 0.0, "reason": "Falha na previsão"}
            
        # Obter sinal
        signal = forecast.get("signal", "neutral")
        
        # Calcular confiança
        confidence = self._calculate_signal_confidence(np.array(forecast["forecast"]))
        
        # Criar resultado
        result = {
            "signal": signal,
            "confidence": confidence,
            "forecast": forecast["forecast"],
            "trend": forecast.get("trend", "neutral"),
            "timestamp": forecast.get("timestamp", datetime.now().isoformat()),
            "model_type": model_type
        }
        
        return result
    
    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Atualiza o limiar de confiança para sinais.
        
        Args:
            threshold: Novo valor de limiar (0-1)
        """
        self.confidence_threshold = max(min(threshold, 1.0), 0.0)
        logger.info(f"Limiar de confiança atualizado para {self.confidence_threshold}")
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Obtém parâmetros dos modelos.
        
        Returns:
            Dicionário com parâmetros
        """
        result = {
            "lookback_window": self.dl_models.lookback_window,
            "forecast_horizon": self.dl_models.forecast_horizon,
            "feature_columns": self.dl_models.feature_columns,
            "target_column": self.dl_models.target_column,
            "confidence_threshold": self.confidence_threshold,
            "use_gpu": self.dl_models.use_gpu
        }
        
        return result
    
    def update_model_parameters(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Atualiza parâmetros dos modelos.
        
        Args:
            params: Dicionário com parâmetros a atualizar
            
        Returns:
            Dicionário com status da atualização
        """
        try:
            # Atualizar lookback
            if "lookback_window" in params:
                self.dl_models.lookback_window = int(params["lookback_window"])
                
            # Atualizar horizonte
            if "forecast_horizon" in params:
                self.dl_models.forecast_horizon = int(params["forecast_horizon"])
                
            # Atualizar features
            if "feature_columns" in params and params["feature_columns"]:
                self.dl_models.feature_columns = params["feature_columns"]
                
            # Atualizar target
            if "target_column" in params and params["target_column"]:
                self.dl_models.target_column = params["target_column"]
                
            # Atualizar limiar de confiança
            if "confidence_threshold" in params:
                self.update_confidence_threshold(float(params["confidence_threshold"]))
            
            logger.info("Parâmetros atualizados com sucesso")
            return {"status": "success", "message": "Parâmetros atualizados com sucesso"}
            
        except Exception as e:
            logger.error(f"Erro ao atualizar parâmetros: {str(e)}")
            return {"status": "error", "message": f"Erro ao atualizar parâmetros: {str(e)}"}