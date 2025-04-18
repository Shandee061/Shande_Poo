"""
Módulo para auto-otimização de parâmetros de trading.

Este módulo implementa algoritmos para otimização automática de parâmetros
de trading baseado em resultados de backtesting e desempenho em tempo real,
permitindo que o robô se adapte às condições de mercado.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Callable, Optional
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime, timedelta
import random
from scipy.optimize import minimize
from scipy.stats import norm

# Importar módulo de regime de mercado
from market_regime import MarketRegime, MarketRegimeDetector

logger = logging.getLogger(__name__)

class StrategyParameter:
    """Classe para representar um parâmetro de estratégia com limites e tipo"""
    
    def __init__(self, 
                 name: str, 
                 default_value: Any, 
                 min_value: Any = None, 
                 max_value: Any = None, 
                 step: Any = None, 
                 param_type: str = "float",
                 description: str = ""):
        """
        Inicializa um parâmetro de estratégia.
        
        Args:
            name: Nome do parâmetro
            default_value: Valor padrão
            min_value: Valor mínimo (opcional)
            max_value: Valor máximo (opcional)
            step: Passo para ajuste de parâmetro (opcional)
            param_type: Tipo do parâmetro ('float', 'int', 'bool', 'categorical')
            description: Descrição do parâmetro
        """
        self.name = name
        self.default_value = default_value
        self.current_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.param_type = param_type
        self.description = description
        self.history = []  # Histórico de valores
        
    def set_value(self, value: Any) -> None:
        """
        Define um novo valor para o parâmetro.
        
        Args:
            value: Novo valor para o parâmetro
        """
        # Validar e converter valor conforme o tipo
        if self.param_type == "float":
            value = float(value)
            if self.min_value is not None:
                value = max(value, self.min_value)
            if self.max_value is not None:
                value = min(value, self.max_value)
                
        elif self.param_type == "int":
            value = int(value)
            if self.min_value is not None:
                value = max(value, self.min_value)
            if self.max_value is not None:
                value = min(value, self.max_value)
                
        elif self.param_type == "bool":
            value = bool(value)
            
        # Registrar valor anterior no histórico
        self.history.append((datetime.now(), self.current_value))
        
        # Limitar tamanho do histórico
        if len(self.history) > 100:
            self.history.pop(0)
            
        # Atualizar valor atual
        self.current_value = value
        
    def reset(self) -> None:
        """Redefine o parâmetro para seu valor padrão."""
        self.current_value = self.default_value
        
    def get_random_value(self) -> Any:
        """
        Gera um valor aleatório dentro dos limites do parâmetro.
        
        Returns:
            Valor aleatório para o parâmetro
        """
        if self.param_type == "float":
            if self.min_value is not None and self.max_value is not None:
                return random.uniform(self.min_value, self.max_value)
            return self.default_value
            
        elif self.param_type == "int":
            if self.min_value is not None and self.max_value is not None:
                return random.randint(self.min_value, self.max_value)
            return self.default_value
            
        elif self.param_type == "bool":
            return random.choice([True, False])
            
        # Para outros tipos, retorna o valor padrão
        return self.default_value


class AutoOptimizer:
    """Classe para otimização automática de parâmetros de trading"""
    
    def __init__(self, 
                 backtest_func: Callable, 
                 parameters: Dict[str, StrategyParameter],
                 evaluation_metric: str = "profit_factor",
                 optimization_interval: int = 50,  # Número de operações antes de otimizar
                 exploration_rate: float = 0.2,    # Taxa de exploração para novos parâmetros
                 use_regime_detection: bool = True,
                 models_dir: str = "models"):
        """
        Inicializa o otimizador automático.
        
        Args:
            backtest_func: Função de backtesting que recebe parâmetros e retorna resultados
            parameters: Dicionário de parâmetros a serem otimizados
            evaluation_metric: Métrica para avaliar resultados ('profit_factor', 'sharpe_ratio', etc)
            optimization_interval: Número de operações antes de realizar otimização
            exploration_rate: Taxa de exploração de novos valores (0-1)
            use_regime_detection: Usar detecção de regime de mercado
            models_dir: Diretório para salvar modelos treinados
        """
        self.backtest_func = backtest_func
        self.parameters = parameters
        self.evaluation_metric = evaluation_metric
        self.optimization_interval = optimization_interval
        self.exploration_rate = exploration_rate
        self.use_regime_detection = use_regime_detection
        self.models_dir = models_dir
        
        # Inicializar variáveis de controle
        self.trade_count = 0
        self.last_optimization = 0
        self.best_parameters = {}
        self.best_score = -float('inf')  # Para métricas onde maior é melhor
        
        # Histórico de otimizações
        self.optimization_history = []
        
        # Modelo ML para previsão de desempenho
        self.ml_model = None
        self.scaler = StandardScaler()
        
        # Detector de regime de mercado
        self.regime_detector = MarketRegimeDetector() if use_regime_detection else None
        
        # Dicionário para armazenar modelos específicos de regime
        self.regime_models = {}
        
        # Criar diretório para modelos se não existir
        os.makedirs(models_dir, exist_ok=True)
        
        # Inicializar melhores parâmetros com valores padrão
        for name, param in self.parameters.items():
            self.best_parameters[name] = param.default_value
    
    def update_trade_count(self, n: int = 1) -> bool:
        """
        Atualiza o contador de operações e verifica se é hora de otimizar.
        
        Args:
            n: Número de operações a incrementar
            
        Returns:
            True se é hora de otimizar, False caso contrário
        """
        self.trade_count += n
        
        # Verificar se é hora de otimizar
        if self.trade_count - self.last_optimization >= self.optimization_interval:
            self.last_optimization = self.trade_count
            return True
        
        return False
    
    def optimize(self, 
                historical_data: pd.DataFrame, 
                trade_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executa a otimização automática de parâmetros.
        
        Args:
            historical_data: DataFrame com dados históricos OHLCV
            trade_results: Lista com resultados de trades recentes (opcional)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        # Verificar dados suficientes
        if len(historical_data) < 100:
            logger.warning("Dados históricos insuficientes para otimização")
            return self._get_current_parameters()
        
        # Detectar regime de mercado atual se ativado
        current_regime = None
        if self.use_regime_detection and self.regime_detector:
            current_regime = self.regime_detector.detect_regime(historical_data)
            logger.info(f"Regime de mercado atual: {current_regime.value}")
        
        # Decidir método de otimização
        if random.random() < self.exploration_rate:
            # Exploração: usar métodos mais exploratórios
            method = random.choice(["random_search", "genetic_algorithm", "bayesian"])
        else:
            # Exploração: tentar métodos mais precisos
            method = random.choice(["grid_search", "gradient_descent", "ml_model"])
        
        logger.info(f"Otimizando parâmetros usando método: {method}")
        
        if method == "random_search":
            optimized_params = self._random_search(historical_data, n_trials=30, regime=current_regime)
        elif method == "grid_search":
            optimized_params = self._grid_search(historical_data, regime=current_regime)
        elif method == "gradient_descent":
            optimized_params = self._gradient_descent(historical_data, regime=current_regime)
        elif method == "genetic_algorithm":
            optimized_params = self._genetic_algorithm(historical_data, regime=current_regime)
        elif method == "bayesian":
            optimized_params = self._bayesian_optimization(historical_data, regime=current_regime)
        elif method == "ml_model":
            optimized_params = self._ml_prediction(historical_data, regime=current_regime)
        else:
            # Método padrão
            optimized_params = self._random_search(historical_data, n_trials=20, regime=current_regime)
        
        # Registrar otimização
        optimization_record = {
            "timestamp": datetime.now(),
            "method": method,
            "parameters": optimized_params,
            "regime": current_regime.value if current_regime else "undefined",
            "trade_count": self.trade_count
        }
        
        # Adicionar record ao histórico
        self.optimization_history.append(optimization_record)
        
        # Salvar parâmetros em arquivo (para referência)
        self._save_optimization_result(optimization_record)
        
        # Atualizar parâmetros atuais no objeto
        for name, value in optimized_params.items():
            if name in self.parameters:
                self.parameters[name].set_value(value)
        
        return optimized_params
    
    def _random_search(self, 
                      historical_data: pd.DataFrame, 
                      n_trials: int = 30, 
                      regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Otimiza parâmetros usando busca aleatória.
        
        Args:
            historical_data: DataFrame com dados históricos
            n_trials: Número de tentativas
            regime: Regime de mercado atual (opcional)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        best_score = -float('inf')
        best_params = self._get_current_parameters()
        
        for i in range(n_trials):
            # Gerar conjunto de parâmetros aleatórios
            trial_params = {}
            for name, param in self.parameters.items():
                trial_params[name] = param.get_random_value()
            
            # Executar backtesting com parâmetros
            try:
                result = self.backtest_func(historical_data, trial_params)
                score = self._get_evaluation_score(result)
                
                # Atualizar melhor resultado
                if score > best_score:
                    best_score = score
                    best_params = trial_params.copy()
                    logger.info(f"Nova melhor pontuação encontrada: {best_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Erro durante backtesting: {str(e)}")
        
        # Atualizar melhor score global se encontrou algo melhor
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_parameters = best_params.copy()
        
        # Misturar com parâmetros anteriores para evitar mudanças bruscas
        return self._blend_parameters(best_params, self._get_current_parameters(), blend_ratio=0.7)
    
    def _grid_search(self, 
                    historical_data: pd.DataFrame, 
                    regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Otimiza parâmetros usando busca em grade simplificada.
        
        Args:
            historical_data: DataFrame com dados históricos
            regime: Regime de mercado atual (opcional)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        # Para limitar a complexidade, selecionamos apenas um subconjunto de parâmetros
        selected_params = random.sample(list(self.parameters.keys()), 
                                       min(3, len(self.parameters)))
        
        # Definir grid para cada parâmetro
        param_grid = {}
        for name in selected_params:
            param = self.parameters[name]
            
            if param.param_type == "float":
                if param.min_value is not None and param.max_value is not None:
                    step = param.step if param.step else (param.max_value - param.min_value) / 5
                    values = np.arange(param.min_value, param.max_value + step/2, step)
                    param_grid[name] = values.tolist()
                else:
                    param_grid[name] = [param.current_value]
                    
            elif param.param_type == "int":
                if param.min_value is not None and param.max_value is not None:
                    step = param.step if param.step else max(1, (param.max_value - param.min_value) // 5)
                    values = np.arange(param.min_value, param.max_value + 1, step)
                    param_grid[name] = values.tolist()
                else:
                    param_grid[name] = [param.current_value]
                    
            elif param.param_type == "bool":
                param_grid[name] = [True, False]
                
            else:
                param_grid[name] = [param.current_value]
        
        # Executar busca em grade
        best_score = -float('inf')
        best_params = self._get_current_parameters()
        
        # Gerar combinações de parâmetros (limitando complexidade)
        from itertools import product
        all_param_combinations = list(product(*param_grid.values()))
        random.shuffle(all_param_combinations)  # Embaralhar para diversidade
        
        # Limitar número de combinações
        max_combinations = 20
        param_combinations = all_param_combinations[:max_combinations]
        
        # Testar cada combinação
        for i, values in enumerate(param_combinations):
            # Construir dicionário de parâmetros para esta combinação
            trial_params = best_params.copy()  # Começar com parâmetros atuais
            for j, name in enumerate(param_grid.keys()):
                trial_params[name] = values[j]
            
            # Executar backtesting
            try:
                result = self.backtest_func(historical_data, trial_params)
                score = self._get_evaluation_score(result)
                
                # Atualizar melhor resultado
                if score > best_score:
                    best_score = score
                    best_params = trial_params.copy()
                    logger.info(f"Nova melhor pontuação encontrada: {best_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Erro durante backtesting na combinação {i+1}/{len(param_combinations)}: {str(e)}")
        
        # Atualizar melhor score global
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_parameters = best_params.copy()
        
        # Misturar com parâmetros anteriores para suavizar mudanças
        return self._blend_parameters(best_params, self._get_current_parameters(), blend_ratio=0.6)
    
    def _gradient_descent(self, 
                         historical_data: pd.DataFrame, 
                         regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Otimiza parâmetros usando descida de gradiente simplificada.
        
        Args:
            historical_data: DataFrame com dados históricos
            regime: Regime de mercado atual (opcional)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        # Selecionamos parâmetros numéricos para otimizar
        numeric_params = []
        for name, param in self.parameters.items():
            if param.param_type in ["float", "int"] and param.min_value is not None and param.max_value is not None:
                numeric_params.append(name)
        
        if not numeric_params:
            logger.warning("Sem parâmetros numéricos para otimização por gradiente")
            return self._get_current_parameters()
        
        # Selecionar um subconjunto para otimizar
        selected_params = random.sample(numeric_params, min(3, len(numeric_params)))
        
        # Função para minimizar
        def objective(x):
            # Construir dicionário de parâmetros
            trial_params = self._get_current_parameters()
            for i, name in enumerate(selected_params):
                param = self.parameters[name]
                if param.param_type == "int":
                    trial_params[name] = int(round(x[i]))
                else:
                    trial_params[name] = float(x[i])
            
            # Executar backtesting
            try:
                result = self.backtest_func(historical_data, trial_params)
                score = self._get_evaluation_score(result)
                # Inverter sinal para minimização
                return -score
            except Exception as e:
                logger.error(f"Erro durante otimização: {str(e)}")
                return float('inf')  # Retorna valor alto para minimização
        
        # Definir limites e ponto inicial
        bounds = []
        x0 = []
        for name in selected_params:
            param = self.parameters[name]
            bounds.append((param.min_value, param.max_value))
            # Usar valor atual como ponto inicial
            x0.append(param.current_value)
        
        # Executar otimização
        try:
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            # Extrair e aplicar resultados
            if result.success:
                optimized_params = self._get_current_parameters()
                for i, name in enumerate(selected_params):
                    param = self.parameters[name]
                    if param.param_type == "int":
                        optimized_params[name] = int(round(result.x[i]))
                    else:
                        optimized_params[name] = float(result.x[i])
                
                # Verificar score final
                final_result = self.backtest_func(historical_data, optimized_params)
                final_score = self._get_evaluation_score(final_result)
                
                # Atualizar melhor score global
                if final_score > self.best_score:
                    self.best_score = final_score
                    self.best_parameters = optimized_params.copy()
                
                # Misturar com parâmetros anteriores para suavizar mudanças
                return self._blend_parameters(optimized_params, self._get_current_parameters(), blend_ratio=0.5)
            
        except Exception as e:
            logger.error(f"Erro na otimização por gradiente: {str(e)}")
        
        # Em caso de falha, retornar parâmetros atuais
        return self._get_current_parameters()
    
    def _genetic_algorithm(self, 
                          historical_data: pd.DataFrame, 
                          regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Otimiza parâmetros usando algoritmo genético simplificado.
        
        Args:
            historical_data: DataFrame com dados históricos
            regime: Regime de mercado atual (opcional)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        # Configurações do algoritmo genético
        population_size = 15
        generations = 5
        mutation_rate = 0.3
        crossover_rate = 0.7
        
        # Função para criar um indivíduo (conjunto de parâmetros)
        def create_individual():
            individual = {}
            for name, param in self.parameters.items():
                individual[name] = param.get_random_value()
            return individual
        
        # Função de avaliação (fitness)
        def evaluate(individual):
            try:
                result = self.backtest_func(historical_data, individual)
                return self._get_evaluation_score(result)
            except Exception as e:
                logger.error(f"Erro na avaliação: {str(e)}")
                return -float('inf')
        
        # Função de cruzamento (crossover)
        def crossover(parent1, parent2):
            child = {}
            for name in self.parameters:
                if random.random() < crossover_rate:
                    child[name] = parent1[name]
                else:
                    child[name] = parent2[name]
            return child
        
        # Função de mutação
        def mutate(individual):
            for name, param in self.parameters.items():
                if random.random() < mutation_rate:
                    individual[name] = param.get_random_value()
            return individual
        
        # Inicializar população
        population = []
        
        # Incluir melhores parâmetros conhecidos e atuais
        population.append(self._get_current_parameters())
        population.append(self.best_parameters.copy())
        
        # Criar indivíduos aleatórios para completar a população
        for _ in range(population_size - 2):
            population.append(create_individual())
        
        # Evolução por gerações
        best_individual = None
        best_fitness = -float('inf')
        
        for generation in range(generations):
            # Avaliar população
            fitness_scores = []
            for individual in population:
                fitness = evaluate(individual)
                fitness_scores.append((individual, fitness))
            
            # Ordenar por fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Atualizar melhor indivíduo
            if fitness_scores[0][1] > best_fitness:
                best_individual = fitness_scores[0][0].copy()
                best_fitness = fitness_scores[0][1]
                logger.info(f"Nova melhor solução: {best_fitness:.4f} (geração {generation+1})")
            
            # Seleção (elitismo + torneio)
            next_gen = [fitness_scores[0][0].copy()]  # Elitismo: o melhor passa direto
            
            # Seleção por torneio
            while len(next_gen) < population_size:
                # Selecionar 3 indivíduos aleatórios
                tournament = random.sample(fitness_scores, 3)
                tournament.sort(key=lambda x: x[1], reverse=True)
                
                # Escolher 2 melhores para crossover
                parent1 = tournament[0][0]
                parent2 = tournament[1][0]
                
                # Crossover
                child = crossover(parent1, parent2)
                
                # Mutação
                child = mutate(child)
                
                # Adicionar à próxima geração
                next_gen.append(child)
            
            # Atualizar população
            population = next_gen
        
        # Atualizar melhor global se necessário
        if best_fitness > self.best_score:
            self.best_score = best_fitness
            self.best_parameters = best_individual.copy()
        
        # Misturar com parâmetros atuais para suavizar mudanças
        return self._blend_parameters(best_individual, self._get_current_parameters(), blend_ratio=0.65)
    
    def _bayesian_optimization(self, 
                              historical_data: pd.DataFrame, 
                              regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Otimiza parâmetros usando uma versão simplificada de otimização bayesiana.
        
        Args:
            historical_data: DataFrame com dados históricos
            regime: Regime de mercado atual (opcional)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        # Para simplificação, implementamos uma versão básica inspirada em Bayesian optimization
        # usando um modelo normal para estimar a distribuição de probabilidade de bons parâmetros
        
        # Selecionamos apenas parâmetros numéricos
        numeric_params = []
        for name, param in self.parameters.items():
            if param.param_type in ["float", "int"] and param.min_value is not None and param.max_value is not None:
                numeric_params.append(name)
        
        if not numeric_params:
            logger.warning("Sem parâmetros numéricos para otimização bayesiana")
            return self._get_current_parameters()
        
        # Selecionar um subconjunto para otimizar
        selected_params = random.sample(numeric_params, min(3, len(numeric_params)))
        
        # Número de amostras iniciais e iterações
        n_initial_samples = 10
        n_iterations = 10
        
        # Função para avaliar um conjunto de parâmetros
        def evaluate_params(params_dict):
            try:
                result = self.backtest_func(historical_data, params_dict)
                return self._get_evaluation_score(result)
            except Exception as e:
                logger.error(f"Erro na avaliação: {str(e)}")
                return -float('inf')
        
        # Gerar amostras iniciais
        X_samples = []
        y_samples = []
        
        # Incluir parâmetros atuais
        current_params = self._get_current_parameters()
        X_current = [current_params[name] for name in selected_params]
        y_current = evaluate_params(current_params)
        X_samples.append(X_current)
        y_samples.append(y_current)
        
        # Gerar amostras aleatórias adicionais
        for _ in range(n_initial_samples - 1):
            params = current_params.copy()
            for name in selected_params:
                param = self.parameters[name]
                params[name] = param.get_random_value()
            
            X_sample = [params[name] for name in selected_params]
            y_sample = evaluate_params(params)
            
            X_samples.append(X_sample)
            y_samples.append(y_sample)
        
        # Converter para arrays
        X = np.array(X_samples)
        y = np.array(y_samples)
        
        # Normalizar
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-6  # Evitar divisão por zero
        X_norm = (X - X_mean) / X_std
        
        # Melhor resultado inicial
        best_idx = np.argmax(y)
        best_params = current_params.copy()
        for i, name in enumerate(selected_params):
            best_params[name] = X_samples[best_idx][i]
        best_score = y_samples[best_idx]
        
        # Iterações de otimização
        for iteration in range(n_iterations):
            # Ajustar modelo gaussiano
            mean = np.mean(y)
            std = np.std(y) + 1e-6
            
            # Função de aquisição (UCB - Upper Confidence Bound)
            def acquisition(x_norm, kappa=2.0):
                x = x_norm * X_std + X_mean  # Desnormalizar
                
                # Calcular distância às amostras existentes
                distances = np.sum((X_norm - x_norm)**2, axis=1)
                min_distance = np.min(distances)
                
                # Calcular valor esperado e incerteza
                weights = np.exp(-0.5 * distances / 0.1**2)  # Kernel RBF
                weights = weights / np.sum(weights)
                expected_value = np.sum(weights * y)
                uncertainty = kappa * (1.0 / (min_distance + 1e-6))
                
                # UCB combina valor esperado e incerteza
                return expected_value + uncertainty
            
            # Encontrar o próximo ponto para avaliar
            best_acq = -float('inf')
            next_point = None
            
            # Amostragem para próximo ponto
            for _ in range(100):
                # Gerar ponto candidato normalizado
                x_norm = np.random.randn(len(selected_params))
                
                # Limitar ao intervalo [-2, 2] em espaço normalizado
                x_norm = np.clip(x_norm, -2, 2)
                
                # Desnormalizar para espaço original
                x = x_norm * X_std + X_mean
                
                # Ajustar limites dos parâmetros
                for i, name in enumerate(selected_params):
                    param = self.parameters[name]
                    if param.min_value is not None:
                        x[i] = max(x[i], param.min_value)
                    if param.max_value is not None:
                        x[i] = min(x[i], param.max_value)
                
                # Renormalizar
                x_norm = (x - X_mean) / X_std
                
                # Calcular valor de aquisição
                acq_value = acquisition(x_norm)
                
                if acq_value > best_acq:
                    best_acq = acq_value
                    next_point = x
            
            if next_point is None:
                break
                
            # Avaliar o próximo ponto
            next_params = current_params.copy()
            for i, name in enumerate(selected_params):
                param = self.parameters[name]
                value = next_point[i]
                
                # Converter para tipo correto
                if param.param_type == "int":
                    value = int(round(value))
                elif param.param_type == "float":
                    value = float(value)
                
                next_params[name] = value
            
            next_score = evaluate_params(next_params)
            
            # Atualizar amostras
            X = np.vstack([X, next_point])
            y = np.append(y, next_score)
            
            # Atualizar normalização
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-6
            X_norm = (X - X_mean) / X_std
            
            # Atualizar melhor resultado
            if next_score > best_score:
                best_score = next_score
                best_params = next_params.copy()
                logger.info(f"Nova melhor solução bayesiana: {best_score:.4f} (iteração {iteration+1})")
        
        # Atualizar melhor global se necessário
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_parameters = best_params.copy()
        
        # Misturar com parâmetros atuais para suavizar mudanças
        return self._blend_parameters(best_params, self._get_current_parameters(), blend_ratio=0.55)
    
    def _ml_prediction(self, 
                      historical_data: pd.DataFrame, 
                      regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Prediz parâmetros ótimos usando um modelo ML previamente treinado.
        
        Args:
            historical_data: DataFrame com dados históricos
            regime: Regime de mercado atual (opcional)
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        # Verificar se temos modelo treinado para o regime atual
        if regime and self.use_regime_detection:
            model_key = regime.value
            if model_key in self.regime_models:
                logger.info(f"Usando modelo específico para regime {regime.value}")
                model = self.regime_models[model_key]
            else:
                # Se não temos modelo para este regime, usar método alternativo
                logger.warning(f"Sem modelo treinado para regime {regime.value}, usando busca aleatória")
                return self._random_search(historical_data, n_trials=20, regime=regime)
        else:
            # Usar modelo geral se temos um
            if self.ml_model:
                model = self.ml_model
            else:
                # Se não temos modelo, usar método alternativo
                logger.warning("Sem modelo ML treinado, usando busca aleatória")
                return self._random_search(historical_data, n_trials=20, regime=regime)
        
        try:
            # Extrair características do mercado atual
            features = self._extract_market_features(historical_data)
            features_array = np.array([list(features.values())])
            
            # Normalizar
            features_norm = self.scaler.transform(features_array)
            
            # Predizer parâmetros ótimos
            param_values = model.predict(features_norm)[0]
            
            # Aplicar aos parâmetros
            predicted_params = self._get_current_parameters()
            i = 0
            
            for name, param in self.parameters.items():
                if param.param_type in ["float", "int"] and param.min_value is not None and param.max_value is not None:
                    value = param_values[i]
                    
                    # Limitar aos limites do parâmetro
                    value = max(param.min_value, min(param.max_value, value))
                    
                    # Converter para tipo correto
                    if param.param_type == "int":
                        value = int(round(value))
                    
                    predicted_params[name] = value
                    i += 1
            
            # Validar através de backtesting
            result = self.backtest_func(historical_data, predicted_params)
            score = self._get_evaluation_score(result)
            
            # Verificar melhoria
            current_params = self._get_current_parameters()
            current_result = self.backtest_func(historical_data, current_params)
            current_score = self._get_evaluation_score(current_result)
            
            # Se pior que os parâmetros atuais, misturar mais fortemente
            if score < current_score:
                logger.warning("Predição ML pior que parâmetros atuais")
                blend_ratio = 0.3  # 30% novos, 70% atuais
            else:
                # Atualizar melhor global
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = predicted_params.copy()
                blend_ratio = 0.7  # 70% novos, 30% atuais
            
            # Misturar com parâmetros atuais
            return self._blend_parameters(predicted_params, current_params, blend_ratio)
            
        except Exception as e:
            logger.error(f"Erro na predição ML: {str(e)}")
            return self._get_current_parameters()
    
    def train_ml_model(self, 
                      historical_data_list: List[pd.DataFrame], 
                      parameter_sets: List[Dict[str, Any]], 
                      scores: List[float],
                      regimes: Optional[List[MarketRegime]] = None) -> None:
        """
        Treina um modelo ML para predizer parâmetros ótimos com base em características de mercado.
        
        Args:
            historical_data_list: Lista de DataFrames com dados históricos
            parameter_sets: Lista de conjuntos de parâmetros testados
            scores: Lista de pontuações/métricas para cada conjunto
            regimes: Lista opcional de regimes de mercado para cada amostra
        """
        if len(historical_data_list) != len(parameter_sets) or len(parameter_sets) != len(scores):
            logger.error("Dimensões inconsistentes para treinamento do modelo ML")
            return
        
        if len(historical_data_list) < 30:
            logger.warning("Dados insuficientes para treinar modelo ML de qualidade")
            return
        
        try:
            # Extrair características de cada período histórico
            X_features = []
            for data in historical_data_list:
                features = self._extract_market_features(data)
                X_features.append(list(features.values()))
            
            # Converter para arrays
            X = np.array(X_features)
            
            # Extrair valores dos parâmetros como target
            numeric_params = []
            for name, param in self.parameters.items():
                if param.param_type in ["float", "int"] and param.min_value is not None and param.max_value is not None:
                    numeric_params.append(name)
            
            if not numeric_params:
                logger.warning("Sem parâmetros numéricos para treinar modelo")
                return
            
            # Construir matriz Y com valores dos parâmetros
            Y = np.zeros((len(parameter_sets), len(numeric_params)))
            for i, params in enumerate(parameter_sets):
                for j, name in enumerate(numeric_params):
                    Y[i, j] = params.get(name, self.parameters[name].default_value)
            
            # Normalizar dados
            X_scaled = self.scaler.fit_transform(X)
            
            # Se temos informação de regime, treinar modelos por regime
            if regimes and self.use_regime_detection:
                # Agrupar dados por regime
                regime_data = {}
                for i, regime in enumerate(regimes):
                    regime_key = regime.value
                    if regime_key not in regime_data:
                        regime_data[regime_key] = {"X": [], "Y": [], "scores": []}
                    
                    regime_data[regime_key]["X"].append(X_scaled[i])
                    regime_data[regime_key]["Y"].append(Y[i])
                    regime_data[regime_key]["scores"].append(scores[i])
                
                # Treinar modelo para cada regime com dados suficientes
                for regime_key, data in regime_data.items():
                    if len(data["X"]) >= 15:  # Mínimo para treinamento razoável
                        X_regime = np.array(data["X"])
                        Y_regime = np.array(data["Y"])
                        sample_weights = np.array(data["scores"])
                        
                        # Normalizar pesos
                        sample_weights = (sample_weights - sample_weights.min()) / (sample_weights.max() - sample_weights.min() + 1e-8)
                        sample_weights = sample_weights + 0.1  # Evitar pesos zero
                        
                        # Treinar modelo
                        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
                        model.fit(X_regime, Y_regime, sample_weight=sample_weights)
                        
                        # Salvar modelo para este regime
                        self.regime_models[regime_key] = model
                        
                        # Salvar modelo em disco
                        model_path = os.path.join(self.models_dir, f"parameter_model_{regime_key}.joblib")
                        joblib.dump(model, model_path)
                        logger.info(f"Modelo para regime {regime_key} treinado e salvo em {model_path}")
            
            # Pesos baseados nos scores
            sample_weights = np.array(scores)
            sample_weights = (sample_weights - sample_weights.min()) / (sample_weights.max() - sample_weights.min() + 1e-8)
            sample_weights = sample_weights + 0.1  # Evitar pesos zero
            
            # Treinar modelo global
            self.ml_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            self.ml_model.fit(X_scaled, Y, sample_weight=sample_weights)
            
            # Salvar modelo global
            model_path = os.path.join(self.models_dir, "parameter_model.joblib")
            joblib.dump(self.ml_model, model_path)
            
            # Salvar scaler
            scaler_path = os.path.join(self.models_dir, "parameter_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Modelo ML global treinado com {len(X)} amostras e salvo em {model_path}")
            
        except Exception as e:
            logger.error(f"Erro no treinamento do modelo ML: {str(e)}")
    
    def load_models(self) -> bool:
        """
        Carrega modelos ML salvos anteriormente.
        
        Returns:
            True se modelos foram carregados com sucesso, False caso contrário
        """
        try:
            # Carregar modelo global
            model_path = os.path.join(self.models_dir, "parameter_model.joblib")
            if os.path.exists(model_path):
                self.ml_model = joblib.load(model_path)
                
                # Carregar scaler
                scaler_path = os.path.join(self.models_dir, "parameter_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                
                # Carregar modelos específicos de regime
                if self.use_regime_detection:
                    for regime in MarketRegime:
                        regime_path = os.path.join(self.models_dir, f"parameter_model_{regime.value}.joblib")
                        if os.path.exists(regime_path):
                            self.regime_models[regime.value] = joblib.load(regime_path)
                
                logger.info(f"Modelos ML carregados com sucesso: global + {len(self.regime_models)} regimes")
                return True
            else:
                logger.warning("Não foram encontrados modelos ML para carregar")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelos ML: {str(e)}")
            return False
    
    def _extract_market_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extrai características do mercado para alimentar o modelo ML.
        
        Args:
            df: DataFrame com dados de preço
            
        Returns:
            Dicionário com características extraídas
        """
        # Garantir dados suficientes
        if len(df) < 30:
            logger.warning("Dados insuficientes para extrair características de mercado")
            # Retornar valores padrão
            return {
                "volatility": 0.01,
                "trend": 0.0,
                "volume_change": 0.0,
                "rsi": 50.0,
                "bb_width": 0.02,
                "price_momentum": 0.0,
                "price_range": 0.01,
                "avg_bar_size": 0.01,
                "returns_skew": 0.0,
                "returns_kurtosis": 3.0,
                "autocorrelation": 0.0,
                "market_efficiency": 0.5
            }
        
        try:
            # Preço de fechamento e volume
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
            
            # Retornos
            returns = np.diff(close) / close[:-1]
            
            # Calcular volatilidade
            volatility = np.std(returns) * np.sqrt(252)  # Anualizada
            
            # Calcular tendência (inclinação de linha de regressão)
            x = np.arange(len(close))
            trend, _ = np.polyfit(x, close, 1)
            trend = trend / close.mean()  # Normalizada
            
            # Variação de volume
            volume_change = (volume[-1] / volume[-10] - 1) if len(volume) >= 10 else 0
            
            # RSI
            delta = df['close'].diff().dropna()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            avg_gain = gain.rolling(window=14).mean().iloc[-1]
            avg_loss = loss.rolling(window=14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            # Largura das Bandas de Bollinger
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            upper_band = sma_20 + 2 * std_20
            lower_band = sma_20 - 2 * std_20
            bb_width = ((upper_band - lower_band) / sma_20).iloc[-1]
            
            # Momentum de preço
            price_momentum = close[-1] / close[-10] - 1 if len(close) >= 10 else 0
            
            # Range de preço
            price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
            
            # Tamanho médio das barras
            avg_bar_size = (df['high'] - df['low']).mean() / close[-1]
            
            # Estatísticas de retornos
            returns_skew = float(pd.Series(returns).skew()) if len(returns) >= 30 else 0
            returns_kurtosis = float(pd.Series(returns).kurtosis()) if len(returns) >= 30 else 3
            
            # Autocorrelação de retornos (eficiência de mercado)
            autocorrelation = float(pd.Series(returns).autocorr()) if len(returns) >= 30 else 0
            
            # Índice de eficiência de mercado
            cum_abs_returns = np.sum(np.abs(returns))
            abs_cum_returns = np.abs(close[-1] / close[0] - 1)
            market_efficiency = abs_cum_returns / cum_abs_returns if cum_abs_returns > 0 else 0.5
            
            return {
                "volatility": volatility,
                "trend": trend,
                "volume_change": volume_change,
                "rsi": rsi,
                "bb_width": bb_width,
                "price_momentum": price_momentum,
                "price_range": price_range,
                "avg_bar_size": avg_bar_size,
                "returns_skew": returns_skew,
                "returns_kurtosis": returns_kurtosis,
                "autocorrelation": autocorrelation,
                "market_efficiency": market_efficiency
            }
            
        except Exception as e:
            logger.error(f"Erro ao extrair características de mercado: {str(e)}")
            # Retornar valores padrão em caso de erro
            return {
                "volatility": 0.01,
                "trend": 0.0,
                "volume_change": 0.0,
                "rsi": 50.0,
                "bb_width": 0.02,
                "price_momentum": 0.0,
                "price_range": 0.01,
                "avg_bar_size": 0.01,
                "returns_skew": 0.0,
                "returns_kurtosis": 3.0,
                "autocorrelation": 0.0,
                "market_efficiency": 0.5
            }
    
    def _get_evaluation_score(self, result: Dict[str, Any]) -> float:
        """
        Calcula a pontuação de avaliação com base nos resultados de backtesting.
        
        Args:
            result: Dicionário com resultados de backtesting
            
        Returns:
            Pontuação de avaliação (maior é melhor)
        """
        # Verificar existência de métricas necessárias
        if not result or "metrics" not in result:
            return -float('inf')
        
        metrics = result["metrics"]
        
        # Obter métrica conforme configuração
        if self.evaluation_metric == "profit_factor" and "profit_factor" in metrics:
            score = metrics["profit_factor"]
        elif self.evaluation_metric == "sharpe_ratio" and "sharpe_ratio" in metrics:
            score = metrics["sharpe_ratio"]
        elif self.evaluation_metric == "total_return" and "total_return" in metrics:
            score = metrics["total_return"]
        elif self.evaluation_metric == "win_rate" and "win_rate" in metrics:
            score = metrics["win_rate"]
        elif self.evaluation_metric == "adjusted_return" and "adjusted_return" in metrics:
            score = metrics["adjusted_return"]
        elif "combined_score" in metrics:
            # Usar pontuação combinada personalizada
            score = metrics["combined_score"]
        else:
            # Criar pontuação combinada default
            pf = metrics.get("profit_factor", 1.0)
            win_rate = metrics.get("win_rate", 0.5)
            ret = metrics.get("total_return", 0.0)
            drawdown = metrics.get("max_drawdown", 0.1)
            
            # Evitar divisão por zero
            drawdown = max(drawdown, 0.001)
            
            # Combinação de métricas
            score = (pf * win_rate * ret) / drawdown
        
        # Penalizar resultados negativos fortemente
        if metrics.get("total_return", 0) < 0:
            score *= 0.5
        
        return score
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """
        Obtém valores atuais dos parâmetros em formato de dicionário.
        
        Returns:
            Dicionário com valores atuais dos parâmetros
        """
        current_params = {}
        for name, param in self.parameters.items():
            current_params[name] = param.current_value
        return current_params
    
    def _blend_parameters(self, 
                         new_params: Dict[str, Any], 
                         old_params: Dict[str, Any], 
                         blend_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Combina novos parâmetros com antigos para suavizar transições.
        
        Args:
            new_params: Novos parâmetros
            old_params: Parâmetros antigos
            blend_ratio: Proporção dos novos parâmetros (0-1)
            
        Returns:
            Dicionário com parâmetros combinados
        """
        blended = {}
        
        for name, param in self.parameters.items():
            if name not in new_params or name not in old_params:
                blended[name] = param.default_value
                continue
                
            new_val = new_params[name]
            old_val = old_params[name]
            
            if param.param_type == "float":
                blended[name] = old_val + blend_ratio * (new_val - old_val)
            elif param.param_type == "int":
                blended_float = old_val + blend_ratio * (new_val - old_val)
                blended[name] = int(round(blended_float))
            elif param.param_type == "bool":
                # Para booleanos, usar probabilidade
                if random.random() < blend_ratio:
                    blended[name] = new_val
                else:
                    blended[name] = old_val
            else:
                # Para categóricos, manter novo valor
                blended[name] = new_val
        
        return blended
    
    def _save_optimization_result(self, result: Dict[str, Any]) -> None:
        """
        Salva um resultado de otimização em arquivo JSON.
        
        Args:
            result: Dicionário com resultado de otimização
        """
        try:
            # Criar diretório para resultados se não existir
            results_dir = os.path.join(self.models_dir, "optimization_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Criar nome de arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"optimization_{timestamp}.json")
            
            # Converter datetime para string
            result_copy = result.copy()
            result_copy["timestamp"] = result_copy["timestamp"].isoformat()
            
            # Salvar em arquivo
            with open(filename, 'w') as f:
                json.dump(result_copy, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar resultado de otimização: {str(e)}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo do processo de otimização.
        
        Returns:
            Dicionário com resumo de otimização
        """
        return {
            "total_optimizations": len(self.optimization_history),
            "last_optimization": self.optimization_history[-1] if self.optimization_history else None,
            "best_score": self.best_score,
            "best_parameters": self.best_parameters,
            "current_parameters": self._get_current_parameters(),
            "trade_count_since_last": self.trade_count - self.last_optimization
        }
    
    def get_parameter_evolution(self) -> Dict[str, List[Tuple[datetime, Any]]]:
        """
        Retorna o histórico de evolução dos parâmetros.
        
        Returns:
            Dicionário com histórico de valores por parâmetro
        """
        evolution = {}
        for name, param in self.parameters.items():
            evolution[name] = param.history
        return evolution