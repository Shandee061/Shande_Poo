"""
Módulo para análise de correlação entre o WINFUT e outros ativos do mercado.

Este módulo implementa funcionalidades para calcular, analisar e visualizar correlações
entre o futuro de índice WINFUT e outros ativos relevantes do mercado.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta

# Setup logger
logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Classe para análise de correlações entre ativos financeiros.
    
    Esta classe implementa métodos para calcular correlações entre o WINFUT e outros
    ativos do mercado, identificando padrões e relações que podem ser úteis para
    a tomada de decisão de trading.
    """
    
    def __init__(self):
        """
        Inicializa o analisador de correlação.
        """
        # Lista de ativos para correlacionar com WINFUT
        self.default_assets = [
            'IBOV',      # Índice Bovespa
            'DOLAR',     # Dólar Comercial
            'DI1',       # Contrato Futuro de Taxa DI
            'PETR4',     # Petrobras
            'VALE3',     # Vale
            'ITUB4',     # Itaú Unibanco
            'SPX',       # S&P 500 (EUA)
            'NDX',       # Nasdaq 100 (EUA)
            'DJI',       # Dow Jones (EUA)
            'VIX',       # Índice de Volatilidade (CBOE)
            'CL',        # Petróleo Bruto (WTI)
            'GC',        # Ouro
            'BR',        # Petróleo Bruto (Brent)
        ]
        
        # Períodos padrão para análise
        self.default_periods = {
            'intraday': '1d',      # Dados intraday (5min, 15min, etc)
            'daily_short': '10d',   # Curto prazo (10 dias)
            'daily_medium': '30d',  # Médio prazo (30 dias)
            'daily_long': '90d',    # Longo prazo (90 dias)
            'weekly': '52w',        # Semanal (1 ano)
            'monthly': '24m'        # Mensal (2 anos)
        }
        
    def calculate_correlation(self, winfut_data: pd.DataFrame, 
                             other_assets_data: Dict[str, pd.DataFrame], 
                             method: str = 'pearson', 
                             timeframe: str = 'daily_medium') -> pd.DataFrame:
        """
        Calcula a correlação entre o WINFUT e outros ativos.
        
        Args:
            winfut_data: DataFrame com dados do WINFUT
            other_assets_data: Dicionário com DataFrames dos outros ativos
            method: Método de correlação ('pearson', 'spearman', 'kendall')
            timeframe: Período de tempo para análise
            
        Returns:
            DataFrame com matriz de correlação
        """
        # Verificar se temos dados para analisar
        if winfut_data.empty:
            logger.warning("Dados do WINFUT vazios. Não é possível calcular correlações.")
            return pd.DataFrame()
        
        if not other_assets_data:
            logger.warning("Sem dados de outros ativos. Não é possível calcular correlações.")
            return pd.DataFrame()
        
        # Criar DataFrame combinando todos os ativos
        combined_data = pd.DataFrame()
        
        # Adicionar preço de fechamento do WINFUT
        if 'close' in winfut_data.columns:
            combined_data['WINFUT'] = winfut_data['close']
        else:
            logger.warning("Coluna 'close' não encontrada nos dados do WINFUT.")
            return pd.DataFrame()
        
        # Adicionar preços de fechamento dos outros ativos
        for asset_name, asset_data in other_assets_data.items():
            if not asset_data.empty and 'close' in asset_data.columns:
                # Alinhar índices de data para garantir a compatibilidade
                if isinstance(asset_data.index, pd.DatetimeIndex) and isinstance(combined_data.index, pd.DatetimeIndex):
                    # Usar o mesmo índice de data/hora
                    try:
                        combined_data[asset_name] = asset_data['close']
                    except Exception as e:
                        logger.error(f"Erro ao adicionar {asset_name} aos dados combinados: {str(e)}")
                else:
                    logger.warning(f"Índice de {asset_name} não é datetime. Tentando converter.")
                    try:
                        # Tentar converter para datetime se necessário
                        asset_data_copy = asset_data.copy()
                        winfut_data_copy = winfut_data.copy()
                        
                        if not isinstance(asset_data_copy.index, pd.DatetimeIndex):
                            if 'datetime' in asset_data_copy.columns:
                                asset_data_copy.set_index('datetime', inplace=True)
                            else:
                                logger.warning(f"Não foi possível converter índice de {asset_name} para datetime.")
                                continue
                        
                        if not isinstance(winfut_data_copy.index, pd.DatetimeIndex):
                            if 'datetime' in winfut_data_copy.columns:
                                winfut_data_copy.set_index('datetime', inplace=True)
                            else:
                                logger.warning("Não foi possível converter índice do WINFUT para datetime.")
                                continue
                        
                        # Realinhar os dados pelo índice de data
                        combined_data = pd.DataFrame(winfut_data_copy['close'])
                        combined_data.columns = ['WINFUT']
                        combined_data[asset_name] = asset_data_copy['close']
                        
                    except Exception as e:
                        logger.error(f"Erro ao converter e adicionar {asset_name}: {str(e)}")
        
        # Remover linhas com NaN
        combined_data.dropna(inplace=True)
        
        if combined_data.empty:
            logger.warning("Dados combinados vazios após remover NaN.")
            return pd.DataFrame()
        
        # Calcular correlação
        correlation_matrix = combined_data.corr(method=method)
        
        return correlation_matrix
    
    def get_correlated_assets(self, correlation_matrix: pd.DataFrame, 
                             min_correlation: float = 0.7, 
                             max_correlation: float = 1.0) -> Dict[str, float]:
        """
        Obtém ativos com correlação alta em relação ao WINFUT.
        
        Args:
            correlation_matrix: DataFrame com matriz de correlação
            min_correlation: Correlação mínima (positiva ou negativa) para considerar
            max_correlation: Correlação máxima (positiva ou negativa) para considerar
            
        Returns:
            Dicionário com ativos correlacionados e seus valores de correlação
        """
        if correlation_matrix.empty or 'WINFUT' not in correlation_matrix.columns:
            logger.warning("Matriz de correlação vazia ou sem WINFUT.")
            return {}
        
        # Extrair correlações com o WINFUT
        winfut_correlations = correlation_matrix['WINFUT']
        
        # Filtrar ativos com correlação alta (positiva ou negativa)
        high_corr_assets = {}
        
        for asset, corr_value in winfut_correlations.items():
            if asset != 'WINFUT' and (abs(corr_value) >= min_correlation and abs(corr_value) <= max_correlation):
                high_corr_assets[asset] = corr_value
        
        # Ordenar por magnitude da correlação (absoluta)
        high_corr_assets = dict(sorted(high_corr_assets.items(), 
                                      key=lambda item: abs(item[1]), 
                                      reverse=True))
        
        return high_corr_assets
    
    def get_lead_lag_relationships(self, winfut_data: pd.DataFrame, 
                                  other_assets_data: Dict[str, pd.DataFrame], 
                                  max_lag: int = 10) -> Dict[str, Dict[int, float]]:
        """
        Identifica relações de lead-lag entre o WINFUT e outros ativos.
        
        Args:
            winfut_data: DataFrame com dados do WINFUT
            other_assets_data: Dicionário com DataFrames dos outros ativos
            max_lag: Número máximo de períodos de lag para analisar
            
        Returns:
            Dicionário com ativos e suas correlações por lag
        """
        if winfut_data.empty or not other_assets_data:
            logger.warning("Dados insuficientes para análise lead-lag.")
            return {}
        
        # Extrair retornos percentuais do WINFUT
        if 'close' in winfut_data.columns:
            winfut_returns = winfut_data['close'].pct_change().dropna()
        else:
            logger.warning("Coluna 'close' não encontrada nos dados do WINFUT.")
            return {}
        
        lead_lag_results = {}
        
        # Para cada ativo, calcular correlações com diferentes lags
        for asset_name, asset_data in other_assets_data.items():
            if not asset_data.empty and 'close' in asset_data.columns:
                try:
                    # Calcular retornos do ativo
                    asset_returns = asset_data['close'].pct_change().dropna()
                    
                    # Inicializar dicionário para armazenar correlações por lag
                    lag_correlations = {}
                    
                    # Calcular correlações para diferentes lags
                    for lag in range(-max_lag, max_lag + 1):
                        if lag < 0:
                            # Ativo lidera (negativo significa que o ativo se move antes do WINFUT)
                            correlation = asset_returns.shift(-lag).corr(winfut_returns)
                        else:
                            # WINFUT lidera (positivo significa que o WINFUT se move antes do ativo)
                            correlation = asset_returns.corr(winfut_returns.shift(lag))
                        
                        lag_correlations[lag] = correlation
                    
                    lead_lag_results[asset_name] = lag_correlations
                    
                except Exception as e:
                    logger.error(f"Erro ao calcular lead-lag para {asset_name}: {str(e)}")
        
        return lead_lag_results
    
    def get_correlation_changes(self, winfut_data: pd.DataFrame, 
                               other_assets_data: Dict[str, pd.DataFrame], 
                               window_size: int = 30, 
                               step: int = 5) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
        """
        Analisa como as correlações mudam ao longo do tempo.
        
        Args:
            winfut_data: DataFrame com dados do WINFUT
            other_assets_data: Dicionário com DataFrames dos outros ativos
            window_size: Tamanho da janela rolante para cálculo da correlação
            step: Tamanho do passo para cálculo da correlação rolante
            
        Returns:
            Dicionário com ativos e suas correlações ao longo do tempo
        """
        if winfut_data.empty or not other_assets_data:
            logger.warning("Dados insuficientes para análise de mudanças de correlação.")
            return {}
        
        # Verificar se temos a coluna 'close' nos dados do WINFUT
        if 'close' not in winfut_data.columns:
            logger.warning("Coluna 'close' não encontrada nos dados do WINFUT.")
            return {}
        
        correlation_changes = {}
        
        # Para cada ativo, calcular correlações rolantes
        for asset_name, asset_data in other_assets_data.items():
            if not asset_data.empty and 'close' in asset_data.columns:
                try:
                    # Alinhar índices se necessário
                    if isinstance(winfut_data.index, pd.DatetimeIndex) and isinstance(asset_data.index, pd.DatetimeIndex):
                        combined_data = pd.DataFrame()
                        combined_data['WINFUT'] = winfut_data['close']
                        combined_data[asset_name] = asset_data['close']
                        
                        # Remover NaN
                        combined_data.dropna(inplace=True)
                        
                        if not combined_data.empty:
                            # Calcular correlações rolantes
                            rolling_corr = []
                            
                            for i in range(0, len(combined_data) - window_size, step):
                                window_data = combined_data.iloc[i:i+window_size]
                                corr = window_data['WINFUT'].corr(window_data[asset_name])
                                timestamp = window_data.index[-1]  # Usar o último timestamp da janela
                                rolling_corr.append((timestamp, corr))
                            
                            correlation_changes[asset_name] = rolling_corr
                    
                except Exception as e:
                    logger.error(f"Erro ao calcular mudanças de correlação para {asset_name}: {str(e)}")
        
        return correlation_changes
    
    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame, 
                               title: str = "Matriz de Correlação com WINFUT") -> plt.Figure:
        """
        Gera um heatmap de correlação entre o WINFUT e outros ativos.
        
        Args:
            correlation_matrix: DataFrame com matriz de correlação
            title: Título do gráfico
            
        Returns:
            Figura matplotlib com heatmap de correlação
        """
        if correlation_matrix.empty:
            logger.warning("Matriz de correlação vazia. Não é possível criar heatmap.")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center')
            return fig
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Criar máscara para triângulo superior (opcional)
        mask = np.zeros_like(correlation_matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        # Definir mapa de cores
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Criar heatmap
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True, fmt=".2f")
        
        # Configurar título e layout
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_lead_lag_heatmap(self, lead_lag_data: Dict[str, Dict[int, float]], 
                            title: str = "Análise Lead-Lag com WINFUT") -> plt.Figure:
        """
        Gera um heatmap das relações lead-lag entre o WINFUT e outros ativos.
        
        Args:
            lead_lag_data: Dicionário com ativos e suas correlações por lag
            title: Título do gráfico
            
        Returns:
            Figura matplotlib com heatmap de relações lead-lag
        """
        if not lead_lag_data:
            logger.warning("Sem dados lead-lag. Não é possível criar heatmap.")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center')
            return fig
        
        # Converter para DataFrame
        assets = list(lead_lag_data.keys())
        lags = sorted(list(lead_lag_data[assets[0]].keys()))
        
        # Criar matriz para o heatmap
        heatmap_data = []
        for asset in assets:
            asset_data = [lead_lag_data[asset][lag] for lag in lags]
            heatmap_data.append(asset_data)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=assets, columns=lags)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, len(assets) * 0.5 + 2))
        
        # Definir mapa de cores
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Criar heatmap
        sns.heatmap(heatmap_df, cmap=cmap, vmax=1, vmin=-1, center=0,
                   linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True, fmt=".2f")
        
        # Configurar título e labels
        plt.title(title, fontsize=14)
        plt.xlabel("Lag (Períodos)")
        plt.ylabel("Ativo")
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_changes(self, correlation_changes: Dict[str, List[Tuple[pd.Timestamp, float]]],
                               title: str = "Mudanças de Correlação ao Longo do Tempo") -> plt.Figure:
        """
        Gera um gráfico das mudanças de correlação ao longo do tempo.
        
        Args:
            correlation_changes: Dicionário com ativos e suas correlações ao longo do tempo
            title: Título do gráfico
            
        Returns:
            Figura matplotlib com gráfico de mudanças de correlação
        """
        if not correlation_changes:
            logger.warning("Sem dados de mudanças de correlação. Não é possível criar gráfico.")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center')
            return fig
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plotar cada ativo
        for asset, data in correlation_changes.items():
            if data:
                dates = [d[0] for d in data]
                corrs = [d[1] for d in data]
                ax.plot(dates, corrs, marker='o', markersize=4, label=asset)
        
        # Adicionar linha horizontal em y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Configurar título e labels
        plt.title(title, fontsize=14)
        plt.xlabel("Data")
        plt.ylabel("Correlação com WINFUT")
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        
        return fig
    
    def get_regime_based_correlations(self, winfut_data: pd.DataFrame, 
                                    other_assets_data: Dict[str, pd.DataFrame], 
                                    regimes: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Calcula correlações por regime de mercado.
        
        Args:
            winfut_data: DataFrame com dados do WINFUT
            other_assets_data: Dicionário com DataFrames dos outros ativos
            regimes: Série pandas com labels de regime para cada período
            
        Returns:
            Dicionário com correlações por regime
        """
        if winfut_data.empty or not other_assets_data or regimes.empty:
            logger.warning("Dados insuficientes para análise de correlação por regime.")
            return {}
        
        # Verificar se o índice de regimes é compatível com os dados
        if not isinstance(regimes.index, pd.DatetimeIndex):
            logger.warning("Índice de regimes deve ser datetime para análise por regime.")
            return {}
        
        # Adicionar regimes ao DataFrame do WINFUT
        winfut_with_regimes = winfut_data.copy()
        winfut_with_regimes['regime'] = regimes
        
        # Combinar com outros ativos
        combined_data = {}
        
        for asset_name, asset_data in other_assets_data.items():
            if not asset_data.empty and 'close' in asset_data.columns:
                try:
                    # Mesclar dados do ativo com regimes
                    asset_data_aligned = asset_data.copy()
                    asset_data_aligned['winfut'] = winfut_data['close']
                    asset_data_aligned['regime'] = regimes
                    
                    # Remover NaN
                    asset_data_aligned.dropna(inplace=True)
                    
                    combined_data[asset_name] = asset_data_aligned
                except Exception as e:
                    logger.error(f"Erro ao combinar dados para {asset_name}: {str(e)}")
        
        # Calcular correlações por regime
        regime_correlations = {}
        unique_regimes = regimes.unique()
        
        for regime in unique_regimes:
            regime_correlations[regime] = {}
            
            for asset_name, asset_data in combined_data.items():
                # Filtrar dados para o regime específico
                regime_data = asset_data[asset_data['regime'] == regime]
                
                if not regime_data.empty:
                    # Calcular correlação para este regime
                    corr = regime_data['close'].corr(regime_data['winfut'])
                    regime_correlations[regime][asset_name] = corr
        
        return regime_correlations
    
    def plot_regime_correlations(self, regime_correlations: Dict[str, Dict[str, float]],
                               title: str = "Correlações por Regime de Mercado") -> plt.Figure:
        """
        Gera um gráfico de barras das correlações por regime de mercado.
        
        Args:
            regime_correlations: Dicionário com correlações por regime
            title: Título do gráfico
            
        Returns:
            Figura matplotlib com gráfico de barras de correlações por regime
        """
        if not regime_correlations:
            logger.warning("Sem dados de correlação por regime. Não é possível criar gráfico.")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center')
            return fig
        
        # Extrair informações
        regimes = list(regime_correlations.keys())
        assets = set()
        for regime_data in regime_correlations.values():
            assets.update(regime_data.keys())
        assets = sorted(list(assets))
        
        # Preparar dados para visualização
        data = []
        for asset in assets:
            asset_data = []
            for regime in regimes:
                if asset in regime_correlations[regime]:
                    asset_data.append(regime_correlations[regime][asset])
                else:
                    asset_data.append(np.nan)
            data.append(asset_data)
        
        # Converter para DataFrame
        plot_df = pd.DataFrame(data, index=assets, columns=regimes)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, len(assets) * 0.5 + 2))
        
        # Plotar heatmap
        sns.heatmap(plot_df, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                   linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True, fmt=".2f")
        
        # Configurar título e labels
        plt.title(title, fontsize=14)
        plt.xlabel("Regime de Mercado")
        plt.ylabel("Ativo")
        plt.tight_layout()
        
        return fig
    
    def get_correlation_statistics(self, correlation_matrix: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcula estatísticas descritivas das correlações.
        
        Args:
            correlation_matrix: DataFrame com matriz de correlação
            
        Returns:
            Dicionário com estatísticas descritivas das correlações
        """
        if correlation_matrix.empty or 'WINFUT' not in correlation_matrix.columns:
            logger.warning("Matriz de correlação vazia ou sem WINFUT.")
            return {}
        
        # Extrair correlações com o WINFUT
        winfut_correlations = correlation_matrix['WINFUT'].drop('WINFUT')
        
        # Calcular estatísticas
        stats = {
            'media': winfut_correlations.mean(),
            'mediana': winfut_correlations.median(),
            'desvio_padrao': winfut_correlations.std(),
            'minimo': winfut_correlations.min(),
            'maximo': winfut_correlations.max(),
            'positivas': (winfut_correlations > 0).sum() / len(winfut_correlations),
            'negativas': (winfut_correlations < 0).sum() / len(winfut_correlations),
            'fortes_positivas': (winfut_correlations > 0.7).sum() / len(winfut_correlations),
            'fortes_negativas': (winfut_correlations < -0.7).sum() / len(winfut_correlations)
        }
        
        # Média por categoria de ativo
        category_mapping = {
            'indices_br': ['IBOV'],
            'acoes_br': ['PETR4', 'VALE3', 'ITUB4'],
            'cambio': ['DOLAR'],
            'juros': ['DI1'],
            'indices_us': ['SPX', 'NDX', 'DJI'],
            'volatilidade': ['VIX'],
            'commodities': ['CL', 'GC', 'BR']
        }
        
        category_stats = {}
        for category, assets in category_mapping.items():
            valid_assets = [asset for asset in assets if asset in winfut_correlations.index]
            if valid_assets:
                category_corrs = winfut_correlations.loc[valid_assets]
                category_stats[category] = {
                    'media': category_corrs.mean(),
                    'mediana': category_corrs.median(),
                    'desvio_padrao': category_corrs.std(),
                    'minimo': category_corrs.min(),
                    'maximo': category_corrs.max()
                }
        
        return {
            'geral': stats,
            'categorias': category_stats
        }
    
    def generate_correlation_report(self, winfut_data: pd.DataFrame,
                                  other_assets_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Gera um relatório completo de correlação.
        
        Args:
            winfut_data: DataFrame com dados do WINFUT
            other_assets_data: Dicionário com DataFrames dos outros ativos
            
        Returns:
            Dicionário com resultados da análise de correlação
        """
        report = {}
        
        # Verificar dados
        if winfut_data.empty or not other_assets_data:
            logger.warning("Dados insuficientes para gerar relatório de correlação.")
            return {'erro': "Dados insuficientes para análise de correlação."}
        
        try:
            # 1. Correlação geral
            correlation_matrix = self.calculate_correlation(
                winfut_data, other_assets_data, method='pearson', timeframe='daily_medium'
            )
            report['correlation_matrix'] = correlation_matrix
            
            if not correlation_matrix.empty:
                # 2. Ativos mais correlacionados
                high_corr_assets = self.get_correlated_assets(correlation_matrix, min_correlation=0.6)
                report['high_corr_assets'] = high_corr_assets
                
                # 3. Ativos com correlação negativa
                neg_corr_assets = self.get_correlated_assets(correlation_matrix, min_correlation=0.5, max_correlation=1.0)
                neg_corr_assets = {k: v for k, v in neg_corr_assets.items() if v < 0}
                report['negative_corr_assets'] = neg_corr_assets
                
                # 4. Estatísticas de correlação
                stats = self.get_correlation_statistics(correlation_matrix)
                report['statistics'] = stats
            
            # 5. Análise lead-lag
            lead_lag_data = self.get_lead_lag_relationships(winfut_data, other_assets_data, max_lag=5)
            report['lead_lag'] = lead_lag_data
            
            if lead_lag_data:
                # Identificar ativos preditivos (aqueles com maior correlação em lags negativos)
                predictive_assets = {}
                for asset, lag_data in lead_lag_data.items():
                    neg_lags = {lag: corr for lag, corr in lag_data.items() if lag < 0}
                    if neg_lags:
                        best_lag = max(neg_lags.items(), key=lambda x: abs(x[1]))
                        predictive_assets[asset] = {'lag': best_lag[0], 'correlation': best_lag[1]}
                
                report['predictive_assets'] = predictive_assets
            
            # 6. Mudanças temporais de correlação
            time_changes = self.get_correlation_changes(winfut_data, other_assets_data, window_size=20, step=1)
            
            if time_changes:
                # Identificar ativos com correlação estável vs. variável
                stability_measures = {}
                for asset, corr_data in time_changes.items():
                    if corr_data:
                        corrs = [c[1] for c in corr_data]
                        stability_measures[asset] = {
                            'variability': np.std(corrs),
                            'mean': np.mean(corrs),
                            'trend': np.corrcoef(range(len(corrs)), corrs)[0, 1] if len(corrs) > 1 else 0
                        }
                
                report['correlation_stability'] = stability_measures
            
            # 7. Recomendações com base na análise
            recommendations = self._generate_recommendations(report)
            report['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de correlação: {str(e)}")
            return {'erro': f"Erro ao gerar relatório: {str(e)}"}
    
    def _generate_recommendations(self, report_data: Dict[str, any]) -> List[str]:
        """
        Gera recomendações com base na análise de correlação.
        
        Args:
            report_data: Dicionário com resultados da análise de correlação
            
        Returns:
            Lista de recomendações
        """
        recommendations = []
        
        # 1. Recomendações com base em ativos altamente correlacionados
        if 'high_corr_assets' in report_data and report_data['high_corr_assets']:
            top_assets = dict(sorted(report_data['high_corr_assets'].items(), 
                                   key=lambda item: abs(item[1]), 
                                   reverse=True)[:3])
            
            if top_assets:
                asset_names = list(top_assets.keys())
                recommendations.append(
                    f"Monitore de perto {', '.join(asset_names)} por serem os ativos mais correlacionados com WINFUT."
                )
        
        # 2. Recomendações com base em correlações negativas (hedge)
        if 'negative_corr_assets' in report_data and report_data['negative_corr_assets']:
            neg_assets = dict(sorted(report_data['negative_corr_assets'].items(), 
                                   key=lambda item: item[1])[:2])
            
            if neg_assets:
                asset_names = list(neg_assets.keys())
                recommendations.append(
                    f"Considere {', '.join(asset_names)} como potenciais hedges para posições em WINFUT devido à correlação negativa."
                )
        
        # 3. Recomendações com base em relações lead-lag
        if 'predictive_assets' in report_data and report_data['predictive_assets']:
            pred_assets = {k: v for k, v in report_data['predictive_assets'].items() 
                          if abs(v['correlation']) > 0.5}
            
            if pred_assets:
                for asset, data in pred_assets.items():
                    lag = abs(data['lag'])
                    direction = "positiva" if data['correlation'] > 0 else "negativa"
                    recommendations.append(
                        f"Use {asset} como indicador preditivo para WINFUT com {lag} períodos de antecedência (correlação {direction})."
                    )
        
        # 4. Recomendações com base na estabilidade da correlação
        if 'correlation_stability' in report_data and report_data['correlation_stability']:
            stable_assets = {k: v for k, v in report_data['correlation_stability'].items() 
                           if v['variability'] < 0.2 and abs(v['mean']) > 0.5}
            
            if stable_assets:
                stable_names = list(stable_assets.keys())[:2]
                recommendations.append(
                    f"{', '.join(stable_names)} apresentam correlações estáveis com WINFUT, sendo bons indicadores para estratégias de longo prazo."
                )
            
            trend_assets = {k: v for k, v in report_data['correlation_stability'].items() 
                          if abs(v['trend']) > 0.7}
            
            if trend_assets:
                for asset, data in trend_assets.items():
                    direction = "aumentando" if data['trend'] > 0 else "diminuindo"
                    recommendations.append(
                        f"A correlação entre {asset} e WINFUT está {direction} ao longo do tempo, reavalie regularmente esta relação."
                    )
        
        # 5. Recomendação geral se não houver dados suficientes
        if not recommendations:
            recommendations.append(
                "Dados de correlação insuficientes para gerar recomendações específicas. Aumente o período de análise ou adicione mais ativos."
            )
        
        return recommendations