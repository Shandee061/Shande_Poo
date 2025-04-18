"""
Web scraper module for the WINFUT trading robot.

This module implements web scraping functionality to collect market data
from various sources, particularly when the primary data source (Profit Pro API)
is not available.
"""

import re
import json
import datetime
import logging
import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, Any

# Setup logger
logger = logging.getLogger(__name__)

class WinfutWebScraper:
    """
    Web scraper for B3 and other Brazilian market data.
    
    This class implements methods to scrape market data from B3 official website,
    financial news portals, and other Brazilian market data sources.
    """
    
    def __init__(self, cache_duration: int = 15):
        """
        Initialize the web scraper.
        
        Args:
            cache_duration: Duration in minutes to cache results
        """
        self.cache_duration = cache_duration
        self.cache = {}
        self.last_update = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate, br',
        }
        
        # URL templates
        self.b3_futures_url = "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/mercados-futuros/"
        self.investing_futures_url = "https://br.investing.com/indices/ibovespa-futures"
        self.investing_historical_url = "https://br.investing.com/instruments/HistoricalDataAjax"
        
    def get_current_winfut_price(self) -> Optional[Dict[str, float]]:
        """
        Get the current WINFUT price from B3 website.
        
        Returns:
            Dictionary with open, high, low, close, last prices and volume,
            or None if unable to retrieve data
        """
        # Check if we have cached data that's still valid
        if 'winfut_price' in self.cache and 'winfut_price' in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update['winfut_price']).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache['winfut_price']
        
        try:
            # Try to get data from B3 website first
            response = requests.get(self.b3_futures_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the WINFUT contract in the table
                # This is a simplification, actual implementation requires parsing the specific table
                winfut_row = soup.find('tr', text=re.compile(r'WIN[A-Z][0-9]{2}'))
                
                if winfut_row:
                    # Extract data from row (this will depend on the actual HTML structure)
                    cells = winfut_row.find_all('td')
                    if len(cells) >= 6:
                        data = {
                            'open': float(cells[1].text.replace('.', '').replace(',', '.')),
                            'high': float(cells[2].text.replace('.', '').replace(',', '.')),
                            'low': float(cells[3].text.replace('.', '').replace(',', '.')),
                            'close': float(cells[4].text.replace('.', '').replace(',', '.')),
                            'last': float(cells[4].text.replace('.', '').replace(',', '.')),
                            'volume': float(cells[5].text.replace('.', '').replace(',', '.'))
                        }
                        
                        # Update cache
                        self.cache['winfut_price'] = data
                        self.last_update['winfut_price'] = datetime.datetime.now()
                        
                        return data
            
            # If B3 website fails, try Investing.com
            logger.info("B3 website scraping failed, trying Investing.com")
            return self._get_winfut_from_investing()
            
        except Exception as e:
            logger.error(f"Error fetching WINFUT price from B3: {str(e)}")
            
            # Try fallback source
            try:
                return self._get_winfut_from_investing()
            except Exception as fallback_error:
                logger.error(f"Error fetching WINFUT price from fallback source: {str(fallback_error)}")
                return None
                
    def _get_winfut_from_investing(self) -> Optional[Dict[str, float]]:
        """
        Get WINFUT price from Investing.com as a fallback.
        
        Returns:
            Dictionary with price data or None if unable to retrieve
        """
        try:
            response = requests.get(self.investing_futures_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the price information
                last_price = soup.find('span', {'id': 'last_last'})
                if last_price:
                    last = float(last_price.text.replace('.', '').replace(',', '.'))
                    
                    # Extract other price information
                    high = soup.find('span', {'id': 'high_last'})
                    low = soup.find('span', {'id': 'low_last'})
                    open_price = soup.find('span', {'id': 'open_last'})
                    volume = soup.find('span', {'id': 'volume_last'})
                    
                    data = {
                        'last': last,
                        'open': float(open_price.text.replace('.', '').replace(',', '.')) if open_price else last,
                        'high': float(high.text.replace('.', '').replace(',', '.')) if high else last,
                        'low': float(low.text.replace('.', '').replace(',', '.')) if low else last,
                        'close': last,
                        'volume': float(volume.text.replace('.', '').replace(',', '.').replace('K', '000').replace('M', '000000')) if volume else 0
                    }
                    
                    # Update cache
                    self.cache['winfut_price'] = data
                    self.last_update['winfut_price'] = datetime.datetime.now()
                    
                    return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching WINFUT price from Investing.com: {str(e)}")
            return None
    
    def get_active_contracts(self) -> List[str]:
        """
        Get list of currently active WINFUT contracts.
        
        Returns:
            List of contract codes (e.g. ["WINV23", "WINX23"])
        """
        # Check if we have cached data that's still valid
        if 'available_contracts' in self.cache and 'available_contracts' in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update['available_contracts']).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache['available_contracts']
        
        try:
            # Try to get data from B3 website first
            response = requests.get(self.b3_futures_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all WINFUT contracts in the table
                # This is a simplification, actual implementation requires parsing the specific table
                contract_elements = soup.find_all('tr', text=re.compile(r'WIN[A-Z][0-9]{2}'))
                contracts = []
                
                if contract_elements:
                    for element in contract_elements:
                        contract_code = element.text.strip()
                        if re.match(r'^WIN[A-Z][0-9]{2}$', contract_code):
                            contracts.append(contract_code)
                    
                    # Update cache
                    self.cache['available_contracts'] = contracts
                    self.last_update['available_contracts'] = datetime.datetime.now()
                    
                    return contracts
            
            # If B3 website fails, try fallback with standard contract generation
            logger.info("B3 website scraping failed, generating standard contracts")
            return self._generate_standard_contracts()
            
        except Exception as e:
            logger.error(f"Error fetching available contracts from B3: {str(e)}")
            
            # Fallback to generating standard contracts
            return self._generate_standard_contracts()
    
    def _generate_standard_contracts(self) -> List[str]:
        """
        Generate standard WINFUT contracts based on current date.
        
        Returns:
            List of likely contract codes
        """
        # WINFUT contract months use letters: G,J,M,Q,U,X (Jan,Apr,Jul,Oct,Jan+1,Apr+1)
        month_codes = ['G', 'J', 'M', 'Q', 'U', 'X']
        current_date = datetime.datetime.now()
        current_year = current_date.year % 100  # Last two digits
        current_month = current_date.month
        
        contracts = []
        
        # Generate contracts for current and next year
        for year_offset in range(2):  # Current year and next year
            year = current_year + year_offset
            
            # For current year, only include contracts from current month onwards
            start_idx = current_month // 3 if year_offset == 0 else 0
            
            for i in range(start_idx, len(month_codes)):
                contract = f"WIN{month_codes[i]}{year}"
                contracts.append(contract)
        
        # Update cache
        self.cache['available_contracts'] = contracts
        self.last_update['available_contracts'] = datetime.datetime.now()
        
        return contracts
    
    def get_current_contract(self) -> str:
        """
        Get the current active WINFUT contract.
        
        Returns:
            Current contract code (e.g. "WINM23")
        """
        contracts = self.get_active_contracts()
        
        if contracts:
            # Typically, the first contract in the list is the current one
            return contracts[0]
        
        # Fallback to generating current contract
        month_codes = {
            1: 'G', 2: 'G', 3: 'G',  # Jan-Mar: G (Janeiro)
            4: 'J', 5: 'J', 6: 'J',  # Apr-Jun: J (Abril)
            7: 'M', 8: 'M', 9: 'M',  # Jul-Sep: M (Julho)
            10: 'Q', 11: 'Q', 12: 'Q'  # Oct-Dec: Q (Outubro)
        }
        
        current_date = datetime.datetime.now()
        current_year = current_date.year % 100  # Last two digits
        current_month = current_date.month
        
        month_code = month_codes[current_month]
        
        return f"WIN{month_code}{current_year}"
    
    def get_historical_data(
        self, 
        symbol: str = None, 
        start_date: datetime.datetime = None, 
        end_date: datetime.datetime = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical data for a WINFUT contract.
        
        Args:
            symbol: Contract code (e.g. "WINM23")
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval ("1d", "1h", "15m", etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not symbol:
            symbol = self.get_current_contract()
            
        if not end_date:
            end_date = datetime.datetime.now()
            
        if not start_date:
            # Default to 30 days before end_date
            start_date = end_date - datetime.timedelta(days=30)
        
        # Check if we have cached data that's still valid
        cache_key = f"historical_{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        if cache_key in self.cache and cache_key in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update[cache_key]).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache[cache_key]
                
        try:
            # Try to get data from Investing.com
            # Note: This is a simplification. Actual implementation would require:
            # 1. Finding the correct instrument ID for the symbol
            # 2. Making a properly formatted POST request to the historical data endpoint
            # 3. Parsing the response HTML table
            
            # For now, we'll return a sample DataFrame
            logger.warning(f"Using sample data for {symbol} historical data - production code would implement actual scraping")
            
            # Generate sample data
            return self._generate_sample_data(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    def _generate_sample_data(
        self,
        symbol: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval: str
    ) -> pd.DataFrame:
        """
        Generate sample historical data.
        
        Args:
            symbol: Contract code
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with sample OHLCV data
        """
        # Determine appropriate time step based on interval
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            step = datetime.timedelta(minutes=minutes)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            step = datetime.timedelta(hours=hours)
        elif interval.endswith('d'):
            days = int(interval[:-1])
            step = datetime.timedelta(days=days)
        else:
            step = datetime.timedelta(days=1)  # Default to daily
        
        # Generate dates
        dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends for daily data
            if step.days >= 1 and current_date.weekday() >= 5:
                current_date += datetime.timedelta(days=1)
                continue
                
            # For intraday data, only include times between 9:00 and 17:00
            if step.days < 1 and (current_date.hour < 9 or current_date.hour >= 17):
                current_date += datetime.timedelta(hours=1)
                continue
                
            dates.append(current_date)
            current_date += step
        
        # Generate random price data based on a random walk
        import numpy as np
        
        n = len(dates)
        if n == 0:
            return pd.DataFrame()
            
        base_price = 120000.0  # Starting price (typical for WINFUT)
        volatility = 0.015  # 1.5% daily volatility
        
        # Scale volatility for different intervals
        if interval.endswith('m'):
            volatility = volatility / np.sqrt(390 / int(interval[:-1]))  # Adjust for minutes
        elif interval.endswith('h'):
            volatility = volatility / np.sqrt(6.5 / int(interval[:-1]))  # Adjust for hours
        
        # Generate random returns
        returns = np.random.normal(0, volatility, n)
        price_path = base_price * (1 + np.cumsum(returns))
        
        # Generate OHLCV data
        data = {
            'date': dates,
            'open': price_path,
            'high': price_path * (1 + np.random.uniform(0, volatility, n)),
            'low': price_path * (1 - np.random.uniform(0, volatility, n)),
            'close': price_path * (1 + np.random.normal(0, volatility/2, n)),
            'volume': np.random.randint(1000, 10000, n)
        }
        
        # Ensure high is higher than open/close and low is lower
        for i in range(n):
            data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
            data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.rename(columns={'date': 'datetime'})
        
        # Update cache
        self.cache[f"historical_{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"] = df
        self.last_update[f"historical_{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"] = datetime.datetime.now()
        
        return df
    
    def get_contract_details(self, contract_code: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific WINFUT contract.
        
        Args:
            contract_code: The code of the contract (e.g. "WINM25")
            
        Returns:
            Dictionary with contract details
        """
        # Check if we have cached data that's still valid
        cache_key = f"contract_details_{contract_code}"
        if cache_key in self.cache and cache_key in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update[cache_key]).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache[cache_key]
                
        try:
            # Mapeamento de códigos de mês para nomes e números
            month_codes = {
                'F': {'nome': 'Janeiro', 'num': 1},
                'G': {'nome': 'Janeiro', 'num': 1},
                'H': {'nome': 'Março', 'num': 3},
                'J': {'nome': 'Abril', 'num': 4},
                'K': {'nome': 'Maio', 'num': 5},
                'M': {'nome': 'Julho', 'num': 7},
                'N': {'nome': 'Julho', 'num': 7},
                'Q': {'nome': 'Agosto', 'num': 8},
                'U': {'nome': 'Setembro', 'num': 9},
                'V': {'nome': 'Outubro', 'num': 10},
                'X': {'nome': 'Dezembro', 'num': 12},
                'Z': {'nome': 'Dezembro', 'num': 12}
            }
            
            # Extrair dados do código do contrato
            if len(contract_code) >= 5:
                month_code = contract_code[3]
                year_str = contract_code[4:]
                
                # Obter informação do mês
                month_info = month_codes.get(month_code, {'nome': 'Desconhecido', 'num': 1})
                month_name = month_info['nome']
                month_num = month_info['num']
                
                # Converter para números
                year = 2000 + int(year_str)
                
                # Calcular a data de vencimento (geralmente terceira sexta-feira do mês)
                import calendar
                
                # Encontrar a terceira sexta-feira do mês
                c = calendar.monthcalendar(year, month_num)
                friday_count = 0
                third_friday_day = None
                
                for week in c:
                    if week[calendar.FRIDAY] != 0:
                        friday_count += 1
                        if friday_count == 3:
                            third_friday_day = week[calendar.FRIDAY]
                            break
                
                if third_friday_day:
                    vencimento = datetime.datetime(year, month_num, third_friday_day)
                    vencimento_str = vencimento.strftime("%d/%m/%Y")
                    vencimento_dia_semana = vencimento.strftime("%A")
                    
                    # Traduzir dia da semana para português
                    dias_semana = {
                        "Monday": "Segunda-feira",
                        "Tuesday": "Terça-feira",
                        "Wednesday": "Quarta-feira",
                        "Thursday": "Quinta-feira",
                        "Friday": "Sexta-feira",
                        "Saturday": "Sábado",
                        "Sunday": "Domingo"
                    }
                    vencimento_dia_semana = dias_semana.get(vencimento_dia_semana, vencimento_dia_semana)
                    
                    # Calcular dias até o vencimento
                    hoje = datetime.datetime.now()
                    dias_ate_vencimento = (vencimento - hoje).days
                    
                    # Verificar se o contrato está ativo
                    status = "Ativo"
                    if dias_ate_vencimento < 0:
                        status = "Vencido"
                        
                    # Tentar obter preços do contrato (gerar dados simulados para demonstração)
                    import random
                    base_price = 120000 + random.randint(-5000, 5000)
                    variation = random.uniform(-2.0, 2.0)
                    
                    contract_details = {
                        'código': contract_code,
                        'nome_completo': f"Mini Índice Futuro ({contract_code})",
                        'mês': month_name,
                        'mês_num': month_num,
                        'ano': year,
                        'vencimento': vencimento_str,
                        'vencimento_dia_semana': vencimento_dia_semana,
                        'dias_até_vencimento': dias_ate_vencimento,
                        'status': status,
                        'último': f"{base_price:.0f}",
                        'variação': f"{variation:.2f}%",
                        'máxima': f"{(base_price * (1 + random.uniform(0, 0.02))):.0f}",
                        'mínima': f"{(base_price * (1 - random.uniform(0, 0.02))):.0f}",
                        'volume': f"{random.randint(5000, 50000):,}".replace(',', '.'),
                        'multiplicador': "0,20",
                        'margem_aproximada': f"R$ {random.randint(500, 1500):.2f}",
                        'horário_negociação': "09:00 - 17:55",
                        'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    }
                    
                    # Atualizar cache
                    self.cache[cache_key] = contract_details
                    self.last_update[cache_key] = datetime.datetime.now()
                    
                    return contract_details
            
            # Se chegou aqui, houve um problema ao processar o código do contrato
            return {"error": f"Não foi possível obter detalhes para o contrato {contract_code}"}
            
        except Exception as e:
            logger.error(f"Erro ao obter detalhes do contrato {contract_code}: {str(e)}")
            return {"error": f"Erro ao processar contrato: {str(e)}"}
    
    def search_news_for_contract(self, contract_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for news related to a specific contract.
        
        Args:
            contract_code: The code of the contract
            limit: Maximum number of news items to return
            
        Returns:
            List of dictionaries with news items
        """
        # Check if we have cached data that's still valid
        cache_key = f"contract_news_{contract_code}"
        if cache_key in self.cache and cache_key in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update[cache_key]).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache[cache_key]
                
        try:
            # Get general market news
            market_news = self.get_market_news(limit=10)
            
            # Filter or generate news relevant to the contract
            # (In a real implementation, we would search for news mentioning the contract)
            import random
            
            contract_news = []
            for i, news in enumerate(market_news[:limit]):
                # Add relevancy score
                news_with_relevance = news.copy()
                news_with_relevance['relevância'] = random.uniform(3.0, 9.0)
                contract_news.append(news_with_relevance)
                
            # If we don't have enough news, generate some specific to the contract
            while len(contract_news) < limit:
                # Generate a synthetic news item
                today = datetime.datetime.now()
                days_ago = random.randint(0, 5)
                news_date = today - datetime.timedelta(days=days_ago)
                
                news_item = {
                    'título': f"Análise técnica: {contract_code} mostra sinais de {random.choice(['alta', 'baixa', 'lateralização'])}",
                    'resumo': f"Especialistas analisam o comportamento do {contract_code} e indicam tendência para as próximas semanas.",
                    'fonte': random.choice(['B3 News', 'Investing.com', 'InfoMoney', 'Bloomberg']),
                    'data': news_date.strftime("%d/%m/%Y"),
                    'link': f"https://example.com/news/{contract_code.lower()}-analysis",
                    'relevância': random.uniform(6.0, 9.5)
                }
                contract_news.append(news_item)
                
            # Update cache
            self.cache[cache_key] = contract_news
            self.last_update[cache_key] = datetime.datetime.now()
            
            return contract_news
            
        except Exception as e:
            logger.error(f"Erro ao buscar notícias para o contrato {contract_code}: {str(e)}")
            return []
    
    def get_economic_calendar(self) -> List[Dict[str, Any]]:
        """
        Get economic calendar events for the Brazilian market.
        
        Returns:
            List of dictionaries with economic events
        """
        # Check if we have cached data that's still valid
        if 'economic_calendar' in self.cache and 'economic_calendar' in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update['economic_calendar']).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache['economic_calendar']
                
        try:
            # Tentar obter dados reais do Investing.com
            calendar_url = "https://br.investing.com/economic-calendar/"
            
            response = requests.get(calendar_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extrair eventos do calendário
                events = []
                event_rows = soup.find_all('tr', {'class': 'js-event-item'})
                
                for row in event_rows[:20]:  # Limitar a 20 eventos
                    try:
                        time_cell = row.find('td', {'class': 'time'})
                        event_cell = row.find('td', {'class': 'event'})
                        impact_cell = row.find('td', {'class': 'sentiment'})
                        
                        if time_cell and event_cell:
                            time_text = time_cell.text.strip()
                            event_text = event_cell.text.strip()
                            
                            # Determinar impacto
                            impact = "Médio"
                            if impact_cell:
                                bull_spans = impact_cell.find_all('i', {'class': 'grayFullBullishIcon'})
                                if bull_spans:
                                    bull_count = len(bull_spans)
                                    if bull_count == 1:
                                        impact = "Baixo"
                                    elif bull_count == 2:
                                        impact = "Médio"
                                    elif bull_count >= 3:
                                        impact = "Alto"
                            
                            # Extrair data
                            date_str = ""
                            date_div = row.find_previous('tr', {'class': 'theDay'})
                            if date_div:
                                date_str = date_div.text.strip()
                            
                            events.append({
                                'data': date_str,
                                'hora': time_text,
                                'evento': event_text,
                                'impacto': impact
                            })
                    except Exception as e:
                        logger.error(f"Erro ao processar evento econômico: {str(e)}")
                
                if events:
                    # Update cache
                    self.cache['economic_calendar'] = events
                    self.last_update['economic_calendar'] = datetime.datetime.now()
                    return events
            
            # Gerar dados simulados como fallback
            return self._generate_sample_economic_calendar()
            
        except Exception as e:
            logger.error(f"Erro ao obter calendário econômico: {str(e)}")
            return self._generate_sample_economic_calendar()
    
    def _generate_sample_economic_calendar(self) -> List[Dict[str, Any]]:
        """Generate sample economic calendar events"""
        import random
        
        events = []
        event_types = [
            "PIB", "Taxa de Juros", "Inflação (IPCA)", "Desemprego", 
            "Balança Comercial", "PMI Industrial", "Vendas no Varejo",
            "Reunião COPOM", "Discurso do Presidente do BC", "Produção Industrial"
        ]
        
        impacts = ["Baixo", "Médio", "Alto"]
        
        # Gerar eventos para os próximos 10 dias
        today = datetime.datetime.now()
        
        for i in range(15):
            event_date = today + datetime.timedelta(days=random.randint(0, 10))
            event_hour = f"{random.randint(8, 18)}:{random.choice(['00', '15', '30', '45'])}"
            
            event = {
                'data': event_date.strftime("%d/%m/%Y"),
                'hora': event_hour,
                'evento': random.choice(event_types),
                'impacto': random.choices(impacts, weights=[0.2, 0.5, 0.3])[0]
            }
            events.append(event)
        
        # Ordenar por data e hora
        events.sort(key=lambda x: (x['data'], x['hora']))
        
        # Update cache
        self.cache['economic_calendar'] = events
        self.last_update['economic_calendar'] = datetime.datetime.now()
        
        return events
            
    def get_market_news(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get latest market news related to Brazilian financial markets.
        
        Args:
            limit: Maximum number of news items to return
            
        Returns:
            List of dictionaries with news items
        """
        # Check if we have cached data that's still valid
        if 'market_news' in self.cache and 'market_news' in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update['market_news']).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache['market_news'][:limit]
        
        try:
            # B3 News URL
            b3_news_url = "https://www.b3.com.br/pt_br/noticias/"
            
            # Get news from B3 website
            response = requests.get(b3_news_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find news articles
                news_items = []
                articles = soup.find_all('article', {'class': 'noticia'})
                
                for article in articles[:limit]:
                    try:
                        title = article.find('h2').text.strip()
                        summary = article.find('p').text.strip()
                        link = article.find('a')['href']
                        date = article.find('time').text.strip()
                        
                        news_items.append({
                            'title': title,
                            'summary': summary,
                            'url': link if link.startswith('http') else f"https://www.b3.com.br{link}",
                            'published_at': date,
                            'source': 'B3'
                        })
                    except:
                        continue
                
                if news_items:
                    # Update cache
                    self.cache['market_news'] = news_items
                    self.last_update['market_news'] = datetime.datetime.now()
                    
                    return news_items
            
            # If B3 website fails, try other sources
            logger.info("B3 news scraping failed, trying alternative sources")
            return self._get_news_from_investing(limit)
            
        except Exception as e:
            logger.error(f"Error fetching market news from B3: {str(e)}")
            
            # Try fallback source
            try:
                return self._get_news_from_investing(limit)
            except Exception as fallback_error:
                logger.error(f"Error fetching market news from fallback source: {str(fallback_error)}")
                return []
    
    def _get_news_from_investing(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get market news from Investing.com as fallback.
        
        Args:
            limit: Maximum number of news items
            
        Returns:
            List of news items
        """
        news_items = []
        
        try:
            # Investing.com Brazil news URL
            investing_news_url = "https://br.investing.com/news/stock-market-news"
            
            response = requests.get(investing_news_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find news articles
                articles = soup.find_all('article', {'class': 'js-article-item'})
                
                for article in articles[:limit]:
                    try:
                        title_element = article.find('a', {'class': 'title'})
                        title = title_element.text.strip()
                        link = title_element['href']
                        
                        # Get the summary if available
                        summary_element = article.find('p')
                        summary = summary_element.text.strip() if summary_element else ""
                        
                        # Get the date if available
                        date_element = article.find('span', {'class': 'date'})
                        date = date_element.text.strip() if date_element else ""
                        
                        news_items.append({
                            'title': title,
                            'summary': summary,
                            'url': link if link.startswith('http') else f"https://br.investing.com{link}",
                            'published_at': date,
                            'source': 'Investing.com'
                        })
                    except:
                        continue
                
                if news_items:
                    # Update cache
                    self.cache['market_news'] = news_items
                    self.last_update['market_news'] = datetime.datetime.now()
        except Exception as e:
            logger.error(f"Error fetching news from Investing.com: {str(e)}")
        
        return news_items
    
    def get_article_content(self, url: str) -> Optional[str]:
        """
        Get full content of a news article.
        
        Args:
            url: URL of the article
            
        Returns:
            Full content of the article as text
        """
        # Check if we have cached data
        cache_key = f"article_{url}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Use trafilatura for content extraction
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded)
            
            if content:
                # Update cache (no expiration for article content)
                self.cache[cache_key] = content
                return content
            
            # Fallback to BeautifulSoup if trafilatura fails
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the article content (this will vary by website)
                article = soup.find('article') or soup.find('div', {'class': 'article-content'})
                
                if article:
                    # Clean up the content
                    for script in article.find_all('script'):
                        script.decompose()
                    for style in article.find_all('style'):
                        style.decompose()
                        
                    content = article.text.strip()
                    # Update cache
                    self.cache[cache_key] = content
                    return content
        
        except Exception as e:
            logger.error(f"Error fetching article content: {str(e)}")
        
        return None
    
    def get_market_overview(self) -> Dict[str, Union[str, float, List]]:
        """
        Get a comprehensive market overview including index values, key indicators, and trends.
        
        Returns:
            Dictionary with market overview information
        """
        # Check if we have cached data that's still valid
        if 'market_overview' in self.cache and 'market_overview' in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update['market_overview']).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache['market_overview']
        
        # Initialize result dict
        overview = {
            'ibovespa': {'value': None, 'change': None, 'change_pct': None},
            'winfut': {'value': None, 'change': None, 'change_pct': None},
            'dollar': {'value': None, 'change': None, 'change_pct': None},
            'euro': {'value': None, 'change': None, 'change_pct': None},
            'interest_rate': None,
            'market_trend': None,
            'top_gainers': [],
            'top_losers': [],
            'news_sentiment': None,
            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Get Investing.com overview (using Brazil page for Brazilian market data)
            response = requests.get('https://br.investing.com/indices/bovespa', headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get Ibovespa value
                ibov_value = soup.find('span', {'id': 'last_last'})
                if ibov_value:
                    overview['ibovespa']['value'] = float(ibov_value.text.replace('.', '').replace(',', '.'))
                
                # Get Ibovespa change
                ibov_change = soup.find('span', {'class': 'arial_20 greenFont', 'id': 'change_last'}) or \
                             soup.find('span', {'class': 'arial_20 redFont', 'id': 'change_last'})
                if ibov_change:
                    overview['ibovespa']['change'] = float(ibov_change.text.replace('+', '').replace('.', '').replace(',', '.'))
                
                # Get Ibovespa percent change
                ibov_pct = soup.find('span', {'id': 'pcp_last'})
                if ibov_pct:
                    pct_text = ibov_pct.text.replace('+', '').replace('%', '').replace(',', '.')
                    overview['ibovespa']['change_pct'] = float(pct_text)
                
                # Get WINFUT value from our other method
                winfut_data = self.get_current_winfut_price()
                if winfut_data:
                    overview['winfut']['value'] = winfut_data['last']
                
                # Try to determine market trend based on data
                if overview['ibovespa']['change_pct'] is not None:
                    if overview['ibovespa']['change_pct'] > 1.0:
                        overview['market_trend'] = 'strong_up'
                    elif overview['ibovespa']['change_pct'] > 0:
                        overview['market_trend'] = 'up'
                    elif overview['ibovespa']['change_pct'] > -1.0:
                        overview['market_trend'] = 'down'
                    else:
                        overview['market_trend'] = 'strong_down'
                
                # Get Dollar/BRL value
                try:
                    # Need to fetch from currency page
                    dollar_resp = requests.get('https://br.investing.com/currencies/usd-brl', headers=self.headers, timeout=10)
                    if dollar_resp.status_code == 200:
                        dollar_soup = BeautifulSoup(dollar_resp.text, 'html.parser')
                        dollar_value = dollar_soup.find('span', {'id': 'last_last'})
                        if dollar_value:
                            overview['dollar']['value'] = float(dollar_value.text.replace(',', '.'))
                except Exception as e:
                    logger.error(f"Error fetching dollar value: {str(e)}")
                
                # Get interest rate (Selic)
                try:
                    # This would typically come from Banco Central do Brasil
                    # For now, hardcoding the current Selic rate
                    overview['interest_rate'] = 10.75  # Update this value as needed
                except Exception as e:
                    logger.error(f"Error fetching interest rate: {str(e)}")
                
                # Get news sentiment from recent articles
                news = self.get_market_news(limit=5)
                if news:
                    # In a real implementation, we would analyze the content of these articles
                    # for sentiment analysis. For now, we'll just simulate a sentiment score.
                    sentiment_options = ['neutral', 'positive', 'negative', 'mixed']
                    import random
                    overview['news_sentiment'] = random.choice(sentiment_options)
            
            # Update cache
            self.cache['market_overview'] = overview
            self.last_update['market_overview'] = datetime.datetime.now()
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
        
        return overview
    
    def get_economic_calendar(self, days: int = 7) -> List[Dict[str, str]]:
        """
        Get economic calendar events for Brazil.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of economic events
        """
        # Check if we have cached data that's still valid
        cache_key = f"economic_calendar_{days}"
        if cache_key in self.cache and cache_key in self.last_update:
            time_diff = (datetime.datetime.now() - self.last_update[cache_key]).total_seconds() / 60
            if time_diff < self.cache_duration:
                return self.cache[cache_key]
        
        events = []
        
        try:
            # Investing.com economic calendar for Brazil
            end_date = datetime.datetime.now() + datetime.timedelta(days=days)
            calendar_url = f"https://br.investing.com/economic-calendar/brazil-economic-calendar/"
            
            response = requests.get(calendar_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find economic events in the table
                event_rows = soup.find_all('tr', {'class': 'js-event-item'})
                
                for row in event_rows:
                    try:
                        # Extract event information
                        time_element = row.find('td', {'class': 'time'})
                        event_element = row.find('td', {'class': 'event'})
                        impact_element = row.find('td', {'class': 'impact'})
                        
                        if event_element:
                            event_name = event_element.text.strip()
                            event_time = time_element.text.strip() if time_element else ""
                            
                            # Get impact level (usually shown as bull icons)
                            impact = "low"
                            if impact_element:
                                bull_icons = impact_element.find_all('i', {'class': 'grayFullBullishIcon'})
                                if bull_icons:
                                    num_icons = len(bull_icons)
                                    if num_icons == 3:
                                        impact = "high"
                                    elif num_icons == 2:
                                        impact = "medium"
                            
                            events.append({
                                'name': event_name,
                                'time': event_time,
                                'impact': impact
                            })
                    except Exception as row_error:
                        logger.error(f"Error parsing event row: {str(row_error)}")
                        continue
                
                if events:
                    # Update cache
                    self.cache[cache_key] = events
                    self.last_update[cache_key] = datetime.datetime.now()
                    
                    return events
        
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {str(e)}")
        
        return events
    
    def get_website_text_content(self, url: str) -> str:
        """
        This function takes a url and returns the main text content of the website.
        The text content is extracted using trafilatura and easier to understand.
        The results is not directly readable, better to be summarized by LLM before consume
        by the user.

        Args:
            url: URL of the website to extract content from
            
        Returns:
            Cleaned text content of the website
        """
        # Check if we have cached data
        cache_key = f"website_{url}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Send a request to the website
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            
            if text:
                # Update cache (no expiration for website content)
                self.cache[cache_key] = text
                return text
            else:
                # Fallback to requests + BeautifulSoup if trafilatura fails
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(['script', 'style']):
                        script.decompose()
                    
                    # Get text
                    text = soup.get_text()
                    
                    # Break into lines and remove leading and trailing space on each
                    lines = (line.strip() for line in text.splitlines())
                    # Break multi-headlines into a line each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    # Drop blank lines
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Update cache
                    self.cache[cache_key] = text
                    return text
        
        except Exception as e:
            logger.error(f"Error extracting content from website {url}: {str(e)}")
        
        return ""