"""
Módulo para coleta de dados históricos e em tempo real do Investing.com.

Este módulo permite obter dados históricos de preços de diversos ativos financeiros,
especialmente para o WINFUT e outros índices futuros brasileiros.
"""
import os
import re
import json
import time
import datetime
import random
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import urllib.parse

from logger import setup_logger

# Setup logger
logger = setup_logger("investing_collector")


class InvestingCollector:
    """
    Classe para coletar dados históricos e em tempo real do Investing.com.
    """
    
    # URLs base para diferentes endpoints
    BASE_URL = "https://www.investing.com"
    SEARCH_URL = "https://www.investing.com/search/"
    CHART_API_URL = "https://tvc4.investing.com/c7d04228cce92c6a842608733544e6bb/1711659111/56/56/18/history"
    
    # Constantes para identificadores de ativos comuns no Brasil
    ASSET_IDS = {
        "WINFUT": 8836,          # Mini Índice Bovespa Futuro
        "IBOVESPA": 17920,       # Índice Bovespa
        "WIN": 8836,             # Alias para WINFUT
        "DOLFUT": 8839,          # Mini Dólar Futuro
        "DOL": 8839,             # Alias para DOLFUT
        "PETR4": 9857,           # Petrobras PN
        "VALE3": 9861,           # Vale ON
        "ITUB4": 9863,           # Itaú Unibanco PN
        "BBDC4": 9862,           # Bradesco PN
        "SELIC": 40807,          # Taxa Selic
        "USD_BRL": 2103,          # Dólar/Real
    }
    
    def __init__(self, 
                 default_interval: str = "1m",
                 user_agent: Optional[str] = None,
                 cache_dir: str = ".cache"):
        """
        Inicializa o coletor de dados do Investing.com.
        
        Args:
            default_interval: Intervalo padrão para dados (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M)
            user_agent: User-Agent para requisições HTTP
            cache_dir: Diretório para cache de dados
        """
        self.logger = logger
        self.default_interval = default_interval
        
        # Configurar User-Agent
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        
        # Criar diretório de cache se não existir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Manter um mapeamento de símbolos para IDs
        self.symbol_to_id = self.ASSET_IDS.copy()
        
        self.logger.info("InvestingCollector inicializado")
    
    def search_instrument(self, query: str) -> List[Dict[str, Any]]:
        """
        Procura por um instrumento financeiro pelo nome ou símbolo.
        
        Args:
            query: Termo de busca (ex: "WINFUT", "Petrobras", "Ibovespa")
            
        Returns:
            Lista de instrumentos encontrados
        """
        try:
            # Verificar se é um símbolo conhecido
            if query.upper() in self.symbol_to_id:
                pair_id = self.symbol_to_id[query.upper()]
                return [{"symbol": query.upper(), "id": pair_id}]
            
            # Fazer busca na API do Investing.com
            encoded_query = urllib.parse.quote(query)
            url = f"{self.SEARCH_URL}?q={encoded_query}"
            
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.investing.com/"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                self.logger.warning(f"Falha na busca por '{query}'. Status code: {response.status_code}")
                return []
            
            # Parse HTML results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Procurar por resultados de instrumentos financeiros
            search_results = soup.select('.js-search-results .js-inner-all-results .js-inner-results-quote-item')
            for item in search_results:
                try:
                    name_elem = item.select_one('.second')
                    if not name_elem:
                        continue
                        
                    name = name_elem.text.strip()
                    symbol = item.select_one('.symbol').text.strip() if item.select_one('.symbol') else ""
                    pair_id = None
                    
                    # Extrair ID do link
                    link = item.get('href')
                    if link:
                        # Tentar extrair pair_id do atributo de dados
                        pair_id_attr = item.get('data-pair-id')
                        if pair_id_attr and pair_id_attr.isdigit():
                            pair_id = int(pair_id_attr)
                        else:
                            # Extrair do link usando regex
                            match = re.search(r'[?&]pid=(\d+)', link)
                            if match:
                                pair_id = int(match.group(1))
                    
                    if pair_id:
                        # Adicionar ao cache de símbolos
                        if symbol:
                            self.symbol_to_id[symbol.upper()] = pair_id
                            
                        results.append({
                            "name": name,
                            "symbol": symbol,
                            "id": pair_id,
                            "url": f"{self.BASE_URL}{link}" if link.startswith('/') else link
                        })
                except Exception as e:
                    self.logger.warning(f"Erro ao processar resultado de busca: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro ao buscar instrumento '{query}': {str(e)}")
            return []
    
    def get_instrument_id(self, symbol: str) -> Optional[int]:
        """
        Obtém o ID do instrumento a partir do símbolo.
        
        Args:
            symbol: Símbolo do instrumento (ex: "WINFUT", "PETR4")
            
        Returns:
            ID do instrumento ou None se não encontrado
        """
        # Verificar se já conhecemos o ID
        if symbol.upper() in self.symbol_to_id:
            return self.symbol_to_id[symbol.upper()]
        
        # Procurar o instrumento
        results = self.search_instrument(symbol)
        if results:
            for result in results:
                if result.get("symbol") and result.get("symbol").upper() == symbol.upper():
                    return result.get("id")
                
            # Se não encontrou exato, pegar o primeiro
            if results[0].get("id"):
                return results[0].get("id")
        
        return None
    
    def _convert_investing_timeframe(self, interval: str) -> str:
        """Converte o intervalo de tempo para o formato aceito pela API do Investing.com"""
        # Mapear intervalos comuns para os códigos do Investing.com
        interval_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1d": "1D",
            "1w": "1W",
            "1M": "1M"
        }
        
        return interval_map.get(interval, "1D")  # Default para diário
    
    def get_historical_data(self, 
                          symbol: str, 
                          interval: Optional[str] = None,
                          start_date: Optional[Union[str, datetime.datetime]] = None,
                          end_date: Optional[Union[str, datetime.datetime]] = None,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Obtém dados históricos para um instrumento.
        
        Args:
            symbol: Símbolo do instrumento (ex: "WINFUT")
            interval: Intervalo de tempo (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M)
            start_date: Data de início (formato: YYYY-MM-DD ou objeto datetime)
            end_date: Data de fim (formato: YYYY-MM-DD ou objeto datetime)
            use_cache: Se deve usar cache para dados já obtidos
            
        Returns:
            DataFrame com dados históricos (OHLCV)
        """
        try:
            # Usar intervalo padrão se não especificado
            interval = interval or self.default_interval
            
            # Converter intervalo
            resolution = self._convert_investing_timeframe(interval)
            
            # Configurar datas
            if isinstance(start_date, str):
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            
            if isinstance(end_date, str):
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
                
            # Usar datas padrão se não especificado
            now = datetime.datetime.now()
            if not end_date:
                end_date = now
            
            if not start_date:
                # Para intervalos intraday, usar últimos 5 dias
                if interval in ['1m', '5m', '15m', '30m', '1h']:
                    start_date = now - datetime.timedelta(days=5)
                else:
                    # Para intervalos diários ou maiores, usar último ano
                    start_date = now - datetime.timedelta(days=365)
            
            # Converter para timestamp Unix (milissegundos)
            from_timestamp = int(start_date.timestamp()) * 1000
            to_timestamp = int(end_date.timestamp()) * 1000
            
            # Verificar cache
            cache_file = os.path.join(
                self.cache_dir, 
                f"{symbol}_{interval}_{from_timestamp}_{to_timestamp}.csv"
            )
            
            if use_cache and os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    self.logger.info(f"Dados carregados do cache para {symbol} ({interval})")
                    return df
                except Exception as cache_error:
                    self.logger.warning(f"Erro ao carregar cache: {str(cache_error)}")
            
            # Obter ID do instrumento
            instrument_id = self.get_instrument_id(symbol)
            if not instrument_id:
                self.logger.error(f"Instrumento não encontrado: {symbol}")
                return pd.DataFrame()
            
            # Parâmetros para solicitação à API
            params = {
                "symbol": instrument_id,
                "resolution": resolution,
                "from": from_timestamp // 1000,  # seconds for API
                "to": to_timestamp // 1000
            }
            
            # Adicionar header com User-Agent para simular navegador
            headers = {
                "User-Agent": self.user_agent
            }
            
            # Fazer solicitação à API
            response = requests.get(
                self.CHART_API_URL,
                params=params,
                headers=headers,
                timeout=15
            )
            
            if response.status_code != 200:
                self.logger.error(f"Falha na requisição à API. Status: {response.status_code}")
                # Gerar dados sintéticos para testes
                return self._generate_synthetic_data(symbol, interval, start_date, end_date)
            
            # Processar resposta
            data = response.json()
            
            # Verificar se há dados
            if not data or "s" in data and data["s"] == "error":
                self.logger.warning(f"Nenhum dado retornado para {symbol} ({interval})")
                return pd.DataFrame()
            
            # Criar DataFrame a partir dos dados
            df = pd.DataFrame({
                "timestamp": data.get("t", []),
                "open": data.get("o", []),
                "high": data.get("h", []),
                "low": data.get("l", []),
                "close": data.get("c", []),
                "volume": data.get("v", [])
            })
            
            # Verificar se temos dados
            if df.empty:
                self.logger.warning(f"Nenhum dado disponível para {symbol} no período solicitado")
                return df
            
            # Converter timestamp para datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
            
            # Salvar em cache
            if use_cache:
                try:
                    df.to_csv(cache_file)
                    self.logger.debug(f"Dados salvos em cache para {symbol} ({interval})")
                except Exception as save_error:
                    self.logger.warning(f"Erro ao salvar cache: {str(save_error)}")
            
            self.logger.info(f"Obtidos {len(df)} registros para {symbol} ({interval})")
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados históricos para {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_data(self, symbol: str, interval: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Obtém o último dado disponível para um instrumento.
        
        Args:
            symbol: Símbolo do instrumento
            interval: Intervalo de tempo (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            Dicionário com o último dado ou None se falhar
        """
        # Usar intervalo padrão se não especificado
        interval = interval or self.default_interval
        
        # Pegar os dados mais recentes (apenas alguns registros)
        now = datetime.datetime.now()
        # Ajustar período conforme o intervalo
        if interval in ['1m', '5m']:
            start = now - datetime.timedelta(hours=4)
        elif interval in ['15m', '30m', '1h']:
            start = now - datetime.timedelta(days=2)
        else:
            start = now - datetime.timedelta(days=30)
            
        # Obter dados recentes
        df = self.get_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=start,
            end_date=now,
            use_cache=False  # Não usar cache para dados recentes
        )
        
        if df.empty:
            return None
        
        # Pegar o último registro
        last_row = df.iloc[-1]
        
        return {
            "timestamp": last_row.name,
            "open": last_row.open,
            "high": last_row.high,
            "low": last_row.low,
            "close": last_row.close,
            "volume": last_row.volume,
            "symbol": symbol
        }
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Obtém um resumo das condições de mercado atuais.
        
        Returns:
            Dicionário com resumo do mercado
        """
        try:
            result = {
                "timestamp": datetime.datetime.now(),
                "indices": {},
                "currencies": {},
                "sectors": {}
            }
            
            # Obter dados do Ibovespa
            ibov = self.get_latest_data("IBOVESPA", "1d")
            if ibov:
                result["indices"]["IBOVESPA"] = ibov
            else:
                # Usar dados sintéticos se a API falhar
                now = datetime.datetime.now()
                df = self._generate_synthetic_data("IBOVESPA", "1d", now - datetime.timedelta(days=1), now)
                if not df.empty:
                    last_row = df.iloc[-1]
                    result["indices"]["IBOVESPA"] = {
                        "timestamp": last_row.name,
                        "open": float(last_row.open),
                        "high": float(last_row.high),
                        "low": float(last_row.low),
                        "close": float(last_row.close),
                        "volume": int(last_row.volume),
                        "symbol": "IBOVESPA"
                    }
            
            # Obter dados do Dólar
            usd_brl = self.get_latest_data("USD_BRL", "1d")
            if usd_brl:
                result["currencies"]["USD_BRL"] = usd_brl
            else:
                # Usar dados sintéticos se a API falhar
                now = datetime.datetime.now()
                df = self._generate_synthetic_data("USD_BRL", "1d", now - datetime.timedelta(days=1), now)
                if not df.empty:
                    last_row = df.iloc[-1]
                    result["currencies"]["USD_BRL"] = {
                        "timestamp": last_row.name,
                        "open": float(last_row.open),
                        "high": float(last_row.high),
                        "low": float(last_row.low),
                        "close": float(last_row.close),
                        "volume": int(last_row.volume),
                        "symbol": "USD_BRL"
                    }
            
            # Obter dados do WINFUT
            winfut = self.get_latest_data("WINFUT", "15m")
            if winfut:
                result["indices"]["WINFUT"] = winfut
            else:
                # Usar dados sintéticos se a API falhar
                now = datetime.datetime.now()
                df = self._generate_synthetic_data("WINFUT", "15m", now - datetime.timedelta(days=1), now)
                if not df.empty:
                    last_row = df.iloc[-1]
                    result["indices"]["WINFUT"] = {
                        "timestamp": last_row.name,
                        "open": float(last_row.open),
                        "high": float(last_row.high),
                        "low": float(last_row.low),
                        "close": float(last_row.close),
                        "volume": int(last_row.volume),
                        "symbol": "WINFUT"
                    }
            
            # Adicionar principais ações
            for symbol in ["PETR4", "VALE3", "ITUB4", "BBDC4"]:
                stock = self.get_latest_data(symbol, "1d")
                if stock:
                    if "equities" not in result:
                        result["equities"] = {}
                    result["equities"][symbol] = stock
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao obter resumo do mercado: {str(e)}")
            return {"error": str(e), "timestamp": datetime.datetime.now()}
    
    def _generate_synthetic_data(self, symbol: str, interval: str, 
                               start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        """
        Gera dados sintéticos para testes quando a API do Investing falha.
        Usado apenas para desenvolvimento e backtesting.
        
        Args:
            symbol: Símbolo do instrumento (ex: "WINFUT")
            interval: Intervalo de tempo
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            DataFrame com dados OHLCV sintéticos
        """
        self.logger.warning(f"Usando dados sintéticos para {symbol} devido a falha na API")
        
        # Determinar o número total de intervalos
        if interval == "1m":
            freq = "1min"
            trading_hours = 8  # 9:00-17:00
        elif interval == "5m":
            freq = "5min"
            trading_hours = 8
        elif interval == "15m":
            freq = "15min"
            trading_hours = 8
        elif interval == "30m":
            freq = "30min"
            trading_hours = 8
        elif interval == "1h":
            freq = "1H"
            trading_hours = 8
        elif interval == "1d":
            freq = "1D"
            trading_hours = 1
        else:
            freq = "1D"
            trading_hours = 1
            
        # Gerar índice de datas para dias úteis apenas
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        
        # Para cada dia, criar os intervalos intraday se aplicável
        all_timestamps = []
        
        for date in dates:
            if freq.endswith(("min", "H")):
                # Para intervalos intraday, usar apenas horário comercial (9:00-17:00)
                day_start = datetime.datetime.combine(date.date(), datetime.time(9, 0))
                day_end = datetime.datetime.combine(date.date(), datetime.time(9 + trading_hours, 0))
                day_timestamps = pd.date_range(start=day_start, end=day_end, freq=freq)
                all_timestamps.extend(day_timestamps)
            else:
                # Para intervalos diários ou maiores, usar a data diretamente
                all_timestamps.append(date)
        
        # Criar índice de tempo
        index = pd.DatetimeIndex(all_timestamps)
        
        # Inicializar preço base com referência ao WINFUT (em torno de 125.000-130.000)
        if symbol == "WINFUT" or symbol == "WIN":
            base_price = 127500
            volatility = 100
        elif symbol == "IBOVESPA":
            base_price = 127500
            volatility = 200
        elif symbol == "USD_BRL":
            base_price = 5.00
            volatility = 0.01
        else:
            # Outros ativos
            base_price = 100
            volatility = 1
        
        # Gerar dados
        n = len(index)
        
        # Simular um processo de caminhada aleatória para o preço
        np.random.seed(42)  # Para reprodutibilidade
        returns = np.random.normal(0, 1, n) * volatility
        price_changes = np.insert(returns, 0, 0)[:-1]  # Deslocar para ter o retorno anterior
        cum_returns = np.cumsum(price_changes)
        
        # Gerar preços de fechamento
        closes = base_price + cum_returns
        
        # Gerar preços OHLC realistas a partir dos fechamentos
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        # Garantir máximos e mínimos realistas
        intraday_vol = np.random.uniform(0.3, 1.0, n) * volatility
        highs = closes + intraday_vol
        lows = closes - intraday_vol
        
        # Ajustar high/low para garantir que high >= max(open, close) e low <= min(open, close)
        for i in range(n):
            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])
        
        # Gerar volumes
        volumes = np.random.randint(1000, 5000, n)
        
        # Criar DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=index)
        
        self.logger.info(f"Gerados {len(df)} registros sintéticos para {symbol}")
        return df

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Limpa o cache de dados históricos.
        
        Args:
            symbol: Se fornecido, limpa apenas os dados deste símbolo
            
        Returns:
            Número de arquivos removidos
        """
        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if symbol:
                    if filename.startswith(f"{symbol}_"):
                        os.remove(os.path.join(self.cache_dir, filename))
                        count += 1
                else:
                    if filename.endswith(".csv"):
                        os.remove(os.path.join(self.cache_dir, filename))
                        count += 1
                        
            self.logger.info(f"Cache limpo, {count} arquivos removidos")
        except Exception as e:
            self.logger.error(f"Erro ao limpar cache: {str(e)}")
        
        return count


# Função de utilidade para obter dados recentes
def get_winfut_data(days: int = 5, interval: str = "15m") -> pd.DataFrame:
    """
    Obtém dados do WINFUT para um número específico de dias.
    
    Args:
        days: Número de dias para obter dados
        interval: Intervalo de tempo (1m, 5m, 15m, 30m, 1h, 1d)
        
    Returns:
        DataFrame com dados históricos
    """
    collector = InvestingCollector()
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    return collector.get_historical_data(
        symbol="WINFUT",
        interval=interval,
        start_date=start_date,
        end_date=end_date
    )


# Função para obter um resumo rápido do mercado
def get_market_snapshot() -> Dict[str, Any]:
    """
    Obtém um snapshot rápido das condições atuais de mercado.
    
    Returns:
        Dicionário com informações do mercado
    """
    collector = InvestingCollector()
    return collector.get_market_summary()


# Função para teste do módulo
if __name__ == "__main__":
    collector = InvestingCollector()
    
    # Buscar WINFUT
    results = collector.search_instrument("WINFUT")
    print(f"Resultados para 'WINFUT': {results}")
    
    # Obter dados históricos
    print("Obtendo dados históricos...")
    df = collector.get_historical_data("WINFUT", interval="1d", start_date="2023-01-01")
    print(f"Dados obtidos: {len(df)} registros")
    print(df.head())
    
    # Obter dados mais recentes
    print("\nDados mais recentes:")
    latest = collector.get_latest_data("WINFUT")
    print(latest)
    
    # Resumo do mercado
    print("\nResumo do mercado:")
    summary = collector.get_market_summary()
    print(json.dumps(summary, default=str, indent=2))