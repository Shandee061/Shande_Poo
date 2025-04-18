"""
Módulo para modelos avançados de deep learning (LSTM e Transformer).

Este módulo implementa redes neurais profundas para previsão de séries temporais
financeiras, especificamente para o mercado de WINFUT.
"""
import os
import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configuração de logging
logger = logging.getLogger(__name__)

class DeepLearningModels:
    """Classe para gerenciar modelos de deep learning para previsão de mercado."""
    
    def __init__(self, 
                 model_dir: str = "models",
                 use_gpu: bool = False,
                 lookback_window: int = 60,
                 forecast_horizon: int = 5,
                 feature_columns: Optional[List[str]] = None,
                 target_column: str = "close"):
        """
        Inicializa o gerenciador de modelos de deep learning.
        
        Args:
            model_dir: Diretório para salvar/carregar modelos
            use_gpu: Se True, tenta usar GPU para treinamento se disponível
            lookback_window: Número de períodos históricos para usar como input
            forecast_horizon: Número de períodos para prever no futuro
            feature_columns: Colunas de características para usar no modelo (opcional)
            target_column: Coluna alvo para previsão
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.feature_columns = feature_columns if feature_columns else ["open", "high", "low", "close", "volume"]
        self.target_column = target_column
        
        # Atributos para armazenar modelos
        self.lstm_model = None
        self.transformer_model = None
        self.scaler = None
        
        # Verificar se bibliotecas estão disponíveis
        self.tensorflow_available = False
        self.torch_available = False
        
        # Tentar importar bibliotecas (pode falhar no ambiente Replit)
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            # Configurar para usar GPU ou CPU
            if self.use_gpu:
                # Permitir crescimento de memória da GPU conforme necessário
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"GPU disponível para TensorFlow: {len(gpus)} dispositivo(s)")
                    except RuntimeError as e:
                        logger.error(f"Erro ao configurar GPU: {str(e)}")
                else:
                    logger.warning("Nenhuma GPU disponível para TensorFlow")
                    self.use_gpu = False
            logger.info("TensorFlow importado com sucesso")
        except ImportError:
            logger.warning("TensorFlow não está disponível. Modelos LSTM não funcionarão.")
        
        try:
            import torch
            self.torch_available = True
            # Configurar para usar GPU ou CPU
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"PyTorch usando GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("PyTorch usando CPU")
            logger.info("PyTorch importado com sucesso")
        except ImportError:
            logger.warning("PyTorch não está disponível. Modelos Transformer não funcionarão.")
        
        # Criar diretório para modelos se não existir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, 
                   df: pd.DataFrame, 
                   train_ratio: float = 0.8,
                   normalize: bool = True,
                   return_scalers: bool = False) -> Tuple:
        """
        Prepara os dados para treinamento e teste.
        
        Args:
            df: DataFrame com dados históricos
            train_ratio: Proporção de dados para treinamento
            normalize: Se True, normaliza os dados
            return_scalers: Se True, retorna também os scalers
            
        Returns:
            Tupla com dados de treinamento e teste
        """
        # Verificar se colunas necessárias existem
        for col in self.feature_columns + [self.target_column]:
            if col not in df.columns:
                error_msg = f"Coluna {col} não encontrada no DataFrame"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Obter apenas as colunas necessárias
        data = df[self.feature_columns + [self.target_column] if self.target_column not in self.feature_columns else self.feature_columns].copy()
        
        # Remover linhas com NaN
        data = data.dropna()
        
        if len(data) <= self.lookback_window:
            error_msg = f"Dados insuficientes. São necessários mais de {self.lookback_window} pontos."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Normalizar dados se solicitado
        if normalize:
            try:
                from sklearn.preprocessing import MinMaxScaler
                
                # Um scaler para features e outro para target
                feature_scaler = MinMaxScaler()
                target_scaler = MinMaxScaler()
                
                # Normalizar features
                feature_data = data[self.feature_columns].values
                feature_data = feature_scaler.fit_transform(feature_data)
                
                # Normalizar target
                target_data = data[[self.target_column]].values
                target_data = target_scaler.fit_transform(target_data)
                
                # Armazenar scalers para uso posterior
                self.feature_scaler = feature_scaler
                self.target_scaler = target_scaler
                
                # Reconstruir DataFrame normalizado
                normalized_features = pd.DataFrame(feature_data, columns=self.feature_columns, index=data.index)
                normalized_target = pd.DataFrame(target_data, columns=[self.target_column], index=data.index)
                
                data = pd.concat([normalized_features, normalized_target], axis=1)
                logger.info("Dados normalizados com sucesso")
            except Exception as e:
                logger.error(f"Erro ao normalizar dados: {str(e)}")
                normalize = False
        
        # Criar sequências para treinamento
        X, y = self._create_sequences(data)
        
        # Dividir em treino e teste
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Dados preparados: {len(X_train)} exemplos de treinamento, {len(X_test)} exemplos de teste")
        
        if return_scalers and normalize:
            return X_train, y_train, X_test, y_test, feature_scaler, target_scaler
        else:
            return X_train, y_train, X_test, y_test
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências para treinamento de modelos de séries temporais.
        
        Args:
            data: DataFrame com dados normalizados
            
        Returns:
            Tupla com X (input) e y (target) como arrays numpy
        """
        X, y = [], []
        
        # Obter número de features
        feature_count = len(self.feature_columns)
        
        # Converter para array numpy
        values = data.values
        
        for i in range(len(data) - self.lookback_window - self.forecast_horizon + 1):
            # X: lookback_window timesteps com todas as features
            X.append(values[i:i+self.lookback_window, :feature_count])
            
            # y: forecast_horizon timesteps adiante, apenas a coluna target
            target_idx = data.columns.get_loc(self.target_column)
            y.append(values[i+self.lookback_window:i+self.lookback_window+self.forecast_horizon, target_idx])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self) -> None:
        """
        Constrói um modelo LSTM para previsão de séries temporais.
        """
        if not self.tensorflow_available:
            logger.error("TensorFlow não está disponível. Não é possível construir modelo LSTM.")
            return
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
            from tensorflow.keras.optimizers import Adam
            
            # Obtém a forma dos dados de entrada (lookback_window, num_features)
            input_shape = (self.lookback_window, len(self.feature_columns))
            
            # Construir modelo
            model = Sequential([
                # Primeira camada LSTM bidirecional
                Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.2),
                
                # Segunda camada LSTM
                Bidirectional(LSTM(64, return_sequences=False)),
                BatchNormalization(),
                Dropout(0.2),
                
                # Camada densa para redução de dimensionalidade
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.1),
                
                # Camada de saída (forecast_horizon valores)
                Dense(self.forecast_horizon)
            ])
            
            # Compilar modelo
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Resumo do modelo
            model.summary(print_fn=logger.info)
            
            self.lstm_model = model
            logger.info("Modelo LSTM construído com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo LSTM: {str(e)}")
    
    def build_transformer_model(self) -> None:
        """
        Constrói um modelo Transformer para previsão de séries temporais.
        """
        if not self.torch_available:
            logger.error("PyTorch não está disponível. Não é possível construir modelo Transformer.")
            return
            
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            class TimeSeriesTransformer(nn.Module):
                def __init__(self, 
                           input_dim: int, 
                           output_dim: int,
                           d_model: int = 64, 
                           nhead: int = 4, 
                           num_layers: int = 3, 
                           dim_feedforward: int = 256, 
                           dropout: float = 0.1,
                           device: torch.device = torch.device("cpu")):
                    """
                    Inicializa modelo Transformer para séries temporais.
                    
                    Args:
                        input_dim: Dimensão de entrada (número de features)
                        output_dim: Dimensão de saída (horizonte de previsão)
                        d_model: Dimensão do modelo
                        nhead: Número de cabeças de atenção
                        num_layers: Número de camadas do encoder transformer
                        dim_feedforward: Dimensão da camada feedforward
                        dropout: Taxa de dropout
                        device: Dispositivo para execução (cpu ou cuda)
                    """
                    super(TimeSeriesTransformer, self).__init__()
                    
                    self.input_dim = input_dim
                    self.output_dim = output_dim
                    self.d_model = d_model
                    self.device = device
                    
                    # Camada de embedding para converter input para dimensão do modelo
                    self.input_embedding = nn.Linear(input_dim, d_model)
                    
                    # Encoder Transformer
                    encoder_layers = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
                    
                    # Camada de saída
                    self.output_layer = nn.Linear(d_model, output_dim)
                    
                    # Posicional encoding
                    self.register_buffer("positional_encoding", self._generate_positional_encoding())
                    
                    # Mover para o dispositivo correto (CPU/GPU)
                    self.to(device)
                
                def _generate_positional_encoding(self):
                    """
                    Gera encoding posicional para transformer.
                    """
                    max_len = self.lookback_window
                    pe = torch.zeros(max_len, self.d_model)
                    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
                    
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    
                    return pe.unsqueeze(0)  # [1, max_len, d_model]
                
                def forward(self, x):
                    """
                    Forward pass.
                    
                    Args:
                        x: Input tensor de forma [batch_size, seq_len, input_dim]
                    
                    Returns:
                        Tensor de saída [batch_size, output_dim]
                    """
                    # Converter para dimensão do modelo
                    x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
                    
                    # Adicionar encoding posicional
                    x = x + self.positional_encoding[:, :x.size(1), :]
                    
                    # Passar pelo encoder transformer
                    x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
                    
                    # Usar apenas o último token da sequência para previsão
                    x = x[:, -1, :]  # [batch_size, d_model]
                    
                    # Camada de saída
                    output = self.output_layer(x)  # [batch_size, output_dim]
                    
                    return output
            
            # Criar modelo
            transformer = TimeSeriesTransformer(
                input_dim=len(self.feature_columns),
                output_dim=self.forecast_horizon,
                d_model=64,
                nhead=4,
                num_layers=3,
                dim_feedforward=256,
                dropout=0.1,
                device=self.device
            )
            
            # Definir otimizador
            optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
            
            # Definir função de perda
            criterion = nn.MSELoss()
            
            self.transformer_model = transformer
            self.transformer_optimizer = optimizer
            self.transformer_criterion = criterion
            
            logger.info("Modelo Transformer construído com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao construir modelo Transformer: {str(e)}")
    
    def train_lstm(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray, 
                 X_val: np.ndarray = None, 
                 y_val: np.ndarray = None,
                 epochs: int = 50, 
                 batch_size: int = 64,
                 patience: int = 10,
                 verbose: int = 1) -> Dict[str, Any]:
        """
        Treina o modelo LSTM.
        
        Args:
            X_train: Dados de entrada para treinamento
            y_train: Alvos para treinamento
            X_val: Dados de entrada para validação (opcional)
            y_val: Alvos para validação (opcional)
            epochs: Número de épocas para treinamento
            batch_size: Tamanho do batch
            patience: Número de épocas sem melhoria para early stopping
            verbose: Nível de detalhamento dos logs (0, 1, 2)
            
        Returns:
            Histórico de treinamento
        """
        if not self.tensorflow_available:
            logger.error("TensorFlow não está disponível. Não é possível treinar LSTM.")
            return None
        
        if self.lstm_model is None:
            logger.info("Modelo LSTM não inicializado. Construindo modelo...")
            self.build_lstm_model()
            
        if self.lstm_model is None:
            logger.error("Falha ao construir modelo LSTM.")
            return None
            
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            
            # Configurar callbacks
            callbacks = []
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Checkpoint para salvar melhor modelo
            model_path = os.path.join(self.model_dir, "lstm_model.h5")
            checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
            
            # Redução de learning rate
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Preparar conjuntos de validação
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            # Treinar modelo
            start_time = datetime.now()
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Treinamento do LSTM concluído em {training_time:.2f} segundos")
            
            # Salvar scaler
            if hasattr(self, 'feature_scaler') and hasattr(self, 'target_scaler'):
                joblib.dump(self.feature_scaler, os.path.join(self.model_dir, "lstm_feature_scaler.joblib"))
                joblib.dump(self.target_scaler, os.path.join(self.model_dir, "lstm_target_scaler.joblib"))
                logger.info("Scalers salvos com sucesso")
                
            # Retornar histórico de treinamento
            return {
                "history": history.history,
                "training_time": training_time,
                "model_path": model_path
            }
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo LSTM: {str(e)}")
            return None
    
    def train_transformer(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray, 
                        X_val: np.ndarray = None, 
                        y_val: np.ndarray = None,
                        epochs: int = 50, 
                        batch_size: int = 64,
                        patience: int = 10,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        Treina o modelo Transformer.
        
        Args:
            X_train: Dados de entrada para treinamento
            y_train: Alvos para treinamento
            X_val: Dados de entrada para validação (opcional)
            y_val: Alvos para validação (opcional)
            epochs: Número de épocas para treinamento
            batch_size: Tamanho do batch
            patience: Número de épocas sem melhoria para early stopping
            verbose: Se True, exibe progresso
            
        Returns:
            Histórico de treinamento
        """
        if not self.torch_available:
            logger.error("PyTorch não está disponível. Não é possível treinar Transformer.")
            return None
        
        if self.transformer_model is None:
            logger.info("Modelo Transformer não inicializado. Construindo modelo...")
            self.build_transformer_model()
            
        if self.transformer_model is None:
            logger.error("Falha ao construir modelo Transformer.")
            return None
            
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            # Converter numpy para tensores PyTorch
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            
            # Dataset e DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Validação, se fornecida
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
            else:
                val_loader = None
            
            # Treinamento
            history = {
                "train_loss": [],
                "val_loss": [] if val_loader else None,
                "train_mae": [],
                "val_mae": [] if val_loader else None
            }
            
            best_val_loss = float('inf')
            no_improve_epochs = 0
            model_path = os.path.join(self.model_dir, "transformer_model.pt")
            
            start_time = datetime.now()
            
            for epoch in range(epochs):
                # Modo de treinamento
                self.transformer_model.train()
                train_loss = 0.0
                train_mae = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Forward pass
                    self.transformer_optimizer.zero_grad()
                    output = self.transformer_model(data)
                    loss = self.transformer_criterion(output, target)
                    
                    # Backward pass e otimização
                    loss.backward()
                    self.transformer_optimizer.step()
                    
                    # Estatísticas
                    train_loss += loss.item()
                    train_mae += torch.mean(torch.abs(output - target)).item()
                    
                # Estatísticas médias
                train_loss /= len(train_loader)
                train_mae /= len(train_loader)
                history["train_loss"].append(train_loss)
                history["train_mae"].append(train_mae)
                
                # Validação
                if val_loader:
                    self.transformer_model.eval()
                    val_loss = 0.0
                    val_mae = 0.0
                    
                    with torch.no_grad():
                        for data, target in val_loader:
                            output = self.transformer_model(data)
                            loss = self.transformer_criterion(output, target)
                            val_loss += loss.item()
                            val_mae += torch.mean(torch.abs(output - target)).item()
                    
                    val_loss /= len(val_loader)
                    val_mae /= len(val_loader)
                    history["val_loss"].append(val_loss)
                    history["val_mae"].append(val_mae)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_epochs = 0
                        # Salvar melhor modelo
                        torch.save(self.transformer_model.state_dict(), model_path)
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= patience:
                            if verbose:
                                logger.info(f"Early stopping na época {epoch+1}")
                            break
                else:
                    # Salvar modelo se não há validação
                    if epoch % 5 == 0 or epoch == epochs - 1:
                        torch.save(self.transformer_model.state_dict(), model_path)
                
                if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                    log_msg = f"Época {epoch+1}/{epochs} - "
                    log_msg += f"Loss: {train_loss:.4f}, MAE: {train_mae:.4f}"
                    if val_loader:
                        log_msg += f" - Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
                    logger.info(log_msg)
            
            # Carregar melhor modelo
            if os.path.exists(model_path):
                self.transformer_model.load_state_dict(torch.load(model_path))
                
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Treinamento do Transformer concluído em {training_time:.2f} segundos")
            
            # Salvar scaler
            if hasattr(self, 'feature_scaler') and hasattr(self, 'target_scaler'):
                joblib.dump(self.feature_scaler, os.path.join(self.model_dir, "transformer_feature_scaler.joblib"))
                joblib.dump(self.target_scaler, os.path.join(self.model_dir, "transformer_target_scaler.joblib"))
                logger.info("Scalers salvos com sucesso")
                
            # Retornar histórico de treinamento
            return {
                "history": history,
                "training_time": training_time,
                "model_path": model_path
            }
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo Transformer: {str(e)}")
            return None
    
    def predict_lstm(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Faz previsões com o modelo LSTM.
        
        Args:
            X: Dados de entrada para previsão
            denormalize: Se True, desnormaliza as previsões
            
        Returns:
            Previsões
        """
        if not self.tensorflow_available:
            logger.error("TensorFlow não está disponível. Não é possível fazer previsões LSTM.")
            return None
            
        if self.lstm_model is None:
            logger.error("Modelo LSTM não inicializado ou treinado.")
            return None
            
        try:
            # Fazer previsões
            predictions = self.lstm_model.predict(X)
            
            # Desnormalizar se necessário
            if denormalize and hasattr(self, 'target_scaler'):
                predictions = self.target_scaler.inverse_transform(predictions)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsões com LSTM: {str(e)}")
            return None
    
    def predict_transformer(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Faz previsões com o modelo Transformer.
        
        Args:
            X: Dados de entrada para previsão
            denormalize: Se True, desnormaliza as previsões
            
        Returns:
            Previsões
        """
        if not self.torch_available:
            logger.error("PyTorch não está disponível. Não é possível fazer previsões Transformer.")
            return None
            
        if self.transformer_model is None:
            logger.error("Modelo Transformer não inicializado ou treinado.")
            return None
            
        try:
            import torch
            
            # Converter para tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Modo de avaliação
            self.transformer_model.eval()
            
            # Fazer previsões
            with torch.no_grad():
                predictions = self.transformer_model(X_tensor).cpu().numpy()
            
            # Desnormalizar se necessário
            if denormalize and hasattr(self, 'target_scaler'):
                predictions = self.target_scaler.inverse_transform(predictions)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsões com Transformer: {str(e)}")
            return None
    
    def evaluate_models(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      denormalize: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Avalia os modelos de deep learning.
        
        Args:
            X_test: Dados de teste
            y_test: Alvos de teste
            denormalize: Se True, desnormaliza as previsões
            
        Returns:
            Dicionário com métricas de avaliação
        """
        results = {
            "lstm": None,
            "transformer": None
        }
        
        # Avaliar LSTM
        if self.lstm_model is not None and self.tensorflow_available:
            try:
                # Previsões
                lstm_preds = self.predict_lstm(X_test, denormalize)
                
                # Desnormalizar alvos se necessário
                if denormalize and hasattr(self, 'target_scaler'):
                    y_true = self.target_scaler.inverse_transform(y_test)
                else:
                    y_true = y_test
                
                # Calcular métricas
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(y_true, lstm_preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, lstm_preds)
                r2 = r2_score(y_true.reshape(-1), lstm_preds.reshape(-1))
                
                results["lstm"] = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
                
                logger.info(f"Avaliação LSTM - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao avaliar modelo LSTM: {str(e)}")
        
        # Avaliar Transformer
        if self.transformer_model is not None and self.torch_available:
            try:
                # Previsões
                transformer_preds = self.predict_transformer(X_test, denormalize)
                
                # Desnormalizar alvos se necessário
                if denormalize and hasattr(self, 'target_scaler'):
                    y_true = self.target_scaler.inverse_transform(y_test)
                else:
                    y_true = y_test
                
                # Calcular métricas
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(y_true, transformer_preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, transformer_preds)
                r2 = r2_score(y_true.reshape(-1), transformer_preds.reshape(-1))
                
                results["transformer"] = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
                
                logger.info(f"Avaliação Transformer - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao avaliar modelo Transformer: {str(e)}")
        
        return results
    
    def save_model(self, model_type: str = "both") -> bool:
        """
        Salva os modelos de deep learning.
        
        Args:
            model_type: Tipo de modelo a salvar ("lstm", "transformer" ou "both")
            
        Returns:
            True se bem-sucedido, False caso contrário
        """
        success = True
        
        if model_type in ["lstm", "both"] and self.lstm_model is not None:
            try:
                model_path = os.path.join(self.model_dir, "lstm_model.h5")
                self.lstm_model.save(model_path)
                logger.info(f"Modelo LSTM salvo em {model_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar modelo LSTM: {str(e)}")
                success = False
        
        if model_type in ["transformer", "both"] and self.transformer_model is not None:
            try:
                import torch
                model_path = os.path.join(self.model_dir, "transformer_model.pt")
                torch.save(self.transformer_model.state_dict(), model_path)
                logger.info(f"Modelo Transformer salvo em {model_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar modelo Transformer: {str(e)}")
                success = False
        
        # Salvar metadados do modelo
        try:
            metadata = {
                "lookback_window": self.lookback_window,
                "forecast_horizon": self.forecast_horizon,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "date_saved": datetime.now().isoformat()
            }
            
            import json
            metadata_path = os.path.join(self.model_dir, "deep_learning_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Metadados salvos em {metadata_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {str(e)}")
            success = False
            
        return success
    
    def load_model(self, model_type: str = "both") -> bool:
        """
        Carrega modelos de deep learning salvos.
        
        Args:
            model_type: Tipo de modelo a carregar ("lstm", "transformer" ou "both")
            
        Returns:
            True se bem-sucedido, False caso contrário
        """
        success = True
        
        # Carregar metadados primeiro
        try:
            import json
            metadata_path = os.path.join(self.model_dir, "deep_learning_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Atualizar atributos
                self.lookback_window = metadata.get("lookback_window", self.lookback_window)
                self.forecast_horizon = metadata.get("forecast_horizon", self.forecast_horizon)
                self.feature_columns = metadata.get("feature_columns", self.feature_columns)
                self.target_column = metadata.get("target_column", self.target_column)
                
                logger.info(f"Metadados carregados de {metadata_path}")
        except Exception as e:
            logger.warning(f"Erro ao carregar metadados: {str(e)}")
        
        # Carregar scalers
        try:
            feature_scaler_path = os.path.join(self.model_dir, "lstm_feature_scaler.joblib")
            target_scaler_path = os.path.join(self.model_dir, "lstm_target_scaler.joblib")
            
            if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
                self.feature_scaler = joblib.load(feature_scaler_path)
                self.target_scaler = joblib.load(target_scaler_path)
                logger.info("Scalers carregados com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao carregar scalers: {str(e)}")
        
        # Carregar modelo LSTM
        if model_type in ["lstm", "both"] and self.tensorflow_available:
            try:
                from tensorflow.keras.models import load_model
                model_path = os.path.join(self.model_dir, "lstm_model.h5")
                
                if os.path.exists(model_path):
                    self.lstm_model = load_model(model_path)
                    logger.info(f"Modelo LSTM carregado de {model_path}")
                else:
                    logger.warning(f"Arquivo do modelo LSTM não encontrado em {model_path}")
                    success = False
            except Exception as e:
                logger.error(f"Erro ao carregar modelo LSTM: {str(e)}")
                success = False
        
        # Carregar modelo Transformer
        if model_type in ["transformer", "both"] and self.torch_available:
            try:
                import torch
                model_path = os.path.join(self.model_dir, "transformer_model.pt")
                
                if os.path.exists(model_path):
                    # Primeiro construir o modelo
                    self.build_transformer_model()
                    
                    # Depois carregar os pesos
                    if self.transformer_model is not None:
                        self.transformer_model.load_state_dict(torch.load(model_path, map_location=self.device))
                        logger.info(f"Modelo Transformer carregado de {model_path}")
                    else:
                        logger.error("Falha ao construir modelo Transformer")
                        success = False
                else:
                    logger.warning(f"Arquivo do modelo Transformer não encontrado em {model_path}")
                    success = False
            except Exception as e:
                logger.error(f"Erro ao carregar modelo Transformer: {str(e)}")
                success = False
                
        return success
    
    def get_model_prediction(self, X: np.ndarray, model_type: str = "ensemble") -> np.ndarray:
        """
        Obtém previsões dos modelos.
        
        Args:
            X: Dados de entrada
            model_type: Tipo de modelo a usar ("lstm", "transformer" ou "ensemble")
            
        Returns:
            Previsões
        """
        predictions = None
        
        if model_type == "lstm" or (model_type == "ensemble" and self.lstm_model is not None):
            lstm_preds = self.predict_lstm(X)
            predictions = lstm_preds
        
        if model_type == "transformer" or (model_type == "ensemble" and self.transformer_model is not None):
            transformer_preds = self.predict_transformer(X)
            
            if predictions is None:
                predictions = transformer_preds
            elif model_type == "ensemble":
                # Média ponderada das previsões (0.4 LSTM, 0.6 Transformer)
                predictions = 0.4 * predictions + 0.6 * transformer_preds
        
        return predictions
    
    def forecast_next_period(self, 
                           current_data: pd.DataFrame, 
                           model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Faz previsão para o próximo período.
        
        Args:
            current_data: DataFrame com dados atuais
            model_type: Tipo de modelo a usar ("lstm", "transformer" ou "ensemble")
            
        Returns:
            Dicionário com previsões
        """
        # Verificar se tem dados suficientes
        if len(current_data) < self.lookback_window:
            logger.error(f"Dados insuficientes. São necessários pelo menos {self.lookback_window} pontos.")
            return None
            
        try:
            # Preparar dados
            data = current_data[self.feature_columns + [self.target_column] if self.target_column not in self.feature_columns else self.feature_columns].copy()
            data = data.dropna()
            
            # Últimos lookback_window pontos
            last_window = data.iloc[-self.lookback_window:].copy()
            
            # Normalizar se necessário
            if hasattr(self, 'feature_scaler'):
                feature_data = last_window[self.feature_columns].values
                feature_data = self.feature_scaler.transform(feature_data)
                
                # Reconstruir DataFrame normalizado
                last_window = pd.DataFrame(feature_data, columns=self.feature_columns, index=last_window.index)
            
            # Converter para formato esperado pelos modelos
            X = np.array([last_window[self.feature_columns].values])
            
            # Fazer previsão
            predictions = self.get_model_prediction(X, model_type)
            
            # Desnormalizar se necessário
            if hasattr(self, 'target_scaler'):
                predictions = self.target_scaler.inverse_transform(predictions)
            
            # Criar resultado
            result = {
                "forecast": predictions[0],
                "forecast_horizon": self.forecast_horizon,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsão: {str(e)}")
            return None