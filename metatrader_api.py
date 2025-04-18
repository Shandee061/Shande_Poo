"""
Módulo de API para integração com o MetaTrader 5.

Este módulo fornece uma interface para se comunicar com o MetaTrader 5
para obtenção de dados de mercado e execução de ordens.
"""

import os
import sys
import time
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

# Configuração de logging
logger = logging.getLogger("winfut_robot.metatrader_api")

class MetaTraderAPI:
    """
    Classe para integração com o MetaTrader 5.
    
    Esta classe oferece métodos para obter dados de mercado e executar ordens
    através do MetaTrader 5. Quando o MT5 não está disponível, opera em modo simulado.
    """
    
    def __init__(self, 
                 login: Optional[str] = None, 
                 password: Optional[str] = None, 
                 server: Optional[str] = None,
                 terminal_path: Optional[str] = None):
        """
        Inicializa a API do MetaTrader.
        
        Args:
            login: Login para autenticação no MT5
            password: Senha para autenticação no MT5
            server: Servidor de trading do MT5
            terminal_path: Caminho para o terminal do MT5
        """
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        self.initialized = False
        self.connected = False
        self.connection_error = None
        self.is_simulation = True  # Inicia em modo de simulação por padrão
        self.last_price = None
        self.symbol = "WINFUT"
        
        # Tentar importar o módulo MetaTrader5
        self.mt5_available = False
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.mt5_available = True
            logger.info("Módulo MetaTrader5 disponível para importação")
        except ImportError:
            self.mt5 = None
            logger.warning("Módulo MetaTrader5 não está disponível. Operando em modo simulado.")
        
        # Dados simulados para quando o MT5 não está disponível
        self._init_simulation_data()
    
    def _init_simulation_data(self):
        """Inicializa dados para simulação quando o MT5 não está disponível."""
        self.account_info_sim = {
            "balance": 10000.0,
            "equity": 10000.0,
            "margin": 0.0,
            "free_margin": 10000.0,
            "profit": 0.0,
            "daily_pnl": 0.0
        }
        
        # Criar dados OHLCV sintéticos para simulação
        self.sim_data = self._generate_sample_data("WINFUT", "1m", 1000)
        self.open_positions_sim = []
        self.order_history_sim = []
        self.order_id_counter = 1000
    
    def connect(self) -> bool:
        """
        Conecta ao terminal do MetaTrader.
        
        Returns:
            True se a conexão foi bem sucedida, False caso contrário
        """
        if not self.mt5_available:
            logger.warning("Operando em modo simulado sem conexão real com o MetaTrader")
            self.initialized = True
            self.connected = True
            self.is_simulation = True
            return True
        
        try:
            # Inicializar a conexão com o MT5
            logger.info("Tentando conectar ao MetaTrader 5...")
            if self.terminal_path:
                initialized = self.mt5.initialize(path=self.terminal_path)
            else:
                initialized = self.mt5.initialize()
            
            if not initialized:
                error = self.mt5.last_error()
                logger.error(f"Falha ao inicializar o MetaTrader 5: {error}")
                self.connection_error = f"Falha ao inicializar o MetaTrader 5: {error}"
                self.is_simulation = True
                return False
            
            # Se login foi fornecido, fazer login no servidor
            if self.login and self.password and self.server:
                authorized = self.mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server
                )
                
                if not authorized:
                    error = self.mt5.last_error()
                    logger.error(f"Falha na autenticação no MetaTrader 5: {error}")
                    self.connection_error = f"Falha na autenticação no MetaTrader 5: {error}"
                    self.mt5.shutdown()
                    self.is_simulation = True
                    return False
            
            # Verificar se o símbolo está disponível
            symbols = self.mt5.symbols_get()
            symbol_names = [s.name for s in symbols]
            
            if self.symbol not in symbol_names:
                logger.warning(f"Símbolo {self.symbol} não encontrado no MetaTrader. Verificando símbolos similares...")
                
                # Tentar encontrar símbolo similar
                win_symbols = [s for s in symbol_names if s.startswith("WIN")]
                if win_symbols:
                    self.symbol = win_symbols[0]
                    logger.info(f"Usando símbolo alternativo: {self.symbol}")
                else:
                    logger.error(f"Nenhum símbolo WINFUT encontrado no MetaTrader")
                    self.connection_error = "Nenhum símbolo WINFUT encontrado no MetaTrader"
                    self.mt5.shutdown()
                    self.is_simulation = True
                    return False
            
            logger.info(f"Conexão bem-sucedida com o MetaTrader 5. Símbolo: {self.symbol}")
            self.initialized = True
            self.connected = True
            self.is_simulation = False
            return True
            
        except Exception as e:
            logger.error(f"Erro ao conectar ao MetaTrader 5: {str(e)}")
            self.connection_error = f"Erro ao conectar ao MetaTrader 5: {str(e)}"
            self.is_simulation = True
            return False
    
    def disconnect(self) -> bool:
        """
        Desconecta do terminal do MetaTrader.
        
        Returns:
            True se a desconexão foi bem sucedida, False caso contrário
        """
        if not self.mt5_available or not self.initialized:
            self.connected = False
            return True
        
        try:
            # Encerrar a conexão com o MT5
            self.mt5.shutdown()
            self.connected = False
            logger.info("Desconectado do MetaTrader 5")
            return True
        except Exception as e:
            logger.error(f"Erro ao desconectar do MetaTrader 5: {str(e)}")
            return False
    
    def update_connection(self, login: str, password: str, server: str, terminal_path: Optional[str] = None) -> bool:
        """
        Atualiza os parâmetros de conexão e reconecta.
        
        Args:
            login: Novo login para autenticação
            password: Nova senha para autenticação
            server: Novo servidor de trading
            terminal_path: Novo caminho para o terminal do MT5
            
        Returns:
            True se a reconexão foi bem sucedida, False caso contrário
        """
        # Desconectar primeiro
        if self.connected:
            self.disconnect()
        
        # Atualizar parâmetros
        self.login = login
        self.password = password
        self.server = server
        
        if terminal_path:
            self.terminal_path = terminal_path
        
        # Reconectar
        return self.connect()
    
    def is_connected(self) -> bool:
        """
        Verifica se está conectado ao MetaTrader.
        
        Returns:
            True se conectado, False caso contrário
        """
        return self.connected
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Obtém informações da conta.
        
        Returns:
            Dicionário com informações da conta
        """
        if self.is_simulation:
            return self.account_info_sim
        
        try:
            # Obter informações da conta do MT5
            account_info = self.mt5.account_info()
            if account_info:
                # Converter para dicionário
                account_dict = {
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.margin_free,
                    "profit": account_info.profit,
                    # Calcular daily_pnl (não disponível diretamente no MT5)
                    "daily_pnl": account_info.profit
                }
                return account_dict
            else:
                logger.error("Não foi possível obter informações da conta do MetaTrader")
                return self.account_info_sim
        except Exception as e:
            logger.error(f"Erro ao obter informações da conta: {str(e)}")
            return self.account_info_sim
    
    def get_price_data(self, symbol: Optional[str] = None, timeframe: str = "1m", period: str = "1d", limit: int = 1000) -> pd.DataFrame:
        """
        Obtém dados de preços históricos.
        
        Args:
            symbol: Símbolo para obter dados (padrão: self.symbol)
            timeframe: Timeframe dos dados ('1m', '5m', '15m', '30m', '1h', '1d')
            period: Período dos dados ('1d', '1w', '1m', '1y')
            limit: Número máximo de barras a retornar
            
        Returns:
            DataFrame com dados OHLCV
        """
        if symbol is None:
            symbol = self.symbol
        
        if self.is_simulation:
            return self._get_simulation_price_data(symbol, timeframe, period, limit)
        
        try:
            # Converter timeframe para formato do MT5
            mt5_timeframe = self._convert_timeframe(timeframe)
            
            # Calcular datas de início e fim com base no período
            now = datetime.now()
            if period == "1d":
                from_date = now - timedelta(days=1)
            elif period == "1w":
                from_date = now - timedelta(weeks=1)
            elif period == "1m":
                from_date = now - timedelta(days=30)
            elif period == "1y":
                from_date = now - timedelta(days=365)
            else:
                from_date = now - timedelta(days=1)
            
            # Obter dados históricos do MT5
            rates = self.mt5.copy_rates_range(symbol, mt5_timeframe, from_date, now)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"Nenhum dado histórico disponível para {symbol}")
                return self._get_simulation_price_data(symbol, timeframe, period, limit)
            
            # Converter para DataFrame
            df = pd.DataFrame(rates)
            
            # Converter timestamp para datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Renomear colunas para padrão OHLCV
            df.rename(columns={
                'time': 'datetime',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            # Configurar índice de tempo
            df.set_index('datetime', inplace=True)
            
            # Limitar número de registros
            if len(df) > limit:
                df = df.tail(limit)
            
            # Salvar último preço
            if not df.empty:
                self.last_price = df['close'].iloc[-1]
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de preço do MetaTrader: {str(e)}")
            return self._get_simulation_price_data(symbol, timeframe, period, limit)
    
    def _convert_timeframe(self, timeframe: str) -> int:
        """
        Converte string de timeframe para constante do MT5.
        
        Args:
            timeframe: String de timeframe ('1m', '5m', '15m', '30m', '1h', '1d')
            
        Returns:
            Constante de timeframe do MT5
        """
        if not self.mt5_available:
            return 0
            
        timeframe_map = {
            '1m': self.mt5.TIMEFRAME_M1,
            '5m': self.mt5.TIMEFRAME_M5,
            '15m': self.mt5.TIMEFRAME_M15,
            '30m': self.mt5.TIMEFRAME_M30,
            '1h': self.mt5.TIMEFRAME_H1,
            '4h': self.mt5.TIMEFRAME_H4,
            '1d': self.mt5.TIMEFRAME_D1,
            '1w': self.mt5.TIMEFRAME_W1,
        }
        
        return timeframe_map.get(timeframe, self.mt5.TIMEFRAME_M15)
    
    def get_order_book(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém o livro de ofertas (DOM).
        
        Args:
            symbol: Símbolo para obter o livro de ofertas (padrão: self.symbol)
            
        Returns:
            Dicionário com dados do livro de ofertas
        """
        if symbol is None:
            symbol = self.symbol
        
        if self.is_simulation:
            return self._get_simulation_order_book(symbol)
        
        try:
            # Obter livro de ofertas do MT5
            book = self.mt5.market_book_get(symbol)
            
            if not book:
                logger.warning(f"Não foi possível obter o livro de ofertas para {symbol}")
                return self._get_simulation_order_book(symbol)
            
            # Processar livro de ofertas
            bids = []
            asks = []
            
            for item in book:
                if item.type == self.mt5.BOOK_TYPE_SELL:
                    asks.append({
                        'price': item.price,
                        'volume': item.volume
                    })
                elif item.type == self.mt5.BOOK_TYPE_BUY:
                    bids.append({
                        'price': item.price,
                        'volume': item.volume
                    })
            
            # Ordenar bids e asks
            bids = sorted(bids, key=lambda x: x['price'], reverse=True)
            asks = sorted(asks, key=lambda x: x['price'])
            
            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter livro de ofertas do MetaTrader: {str(e)}")
            return self._get_simulation_order_book(symbol)
    
    def get_times_and_trades(self, symbol: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Obtém histórico recente de negociações (times & trades).
        
        Args:
            symbol: Símbolo para obter negociações (padrão: self.symbol)
            limit: Número máximo de negociações a retornar
            
        Returns:
            DataFrame com histórico de negociações
        """
        if symbol is None:
            symbol = self.symbol
        
        if self.is_simulation:
            return self._get_simulation_times_and_trades(symbol, limit)
        
        try:
            # Obter último preço de ticks do MT5
            now = datetime.now()
            from_date = now - timedelta(minutes=30)  # últimos 30 minutos
            
            ticks = self.mt5.copy_ticks_range(symbol, from_date, now, self.mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0:
                logger.warning(f"Nenhum tick disponível para {symbol}")
                return self._get_simulation_times_and_trades(symbol, limit)
            
            # Converter para DataFrame
            df = pd.DataFrame(ticks)
            
            # Converter timestamp para datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Renomear colunas
            df.rename(columns={
                'time': 'datetime',
                'bid': 'bid',
                'ask': 'ask',
                'last': 'price',
                'volume': 'volume',
                'time_msc': 'time_ms'
            }, inplace=True)
            
            # Limitar número de registros
            if len(df) > limit:
                df = df.tail(limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter times & trades do MetaTrader: {str(e)}")
            return self._get_simulation_times_and_trades(symbol, limit)
    
    def place_market_order(self, symbol: Optional[str] = None, side: str = "BUY", 
                          quantity: float = 1.0, stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Coloca uma ordem de mercado.
        
        Args:
            symbol: Símbolo para a ordem (padrão: self.symbol)
            side: Direção da ordem ('BUY' ou 'SELL')
            quantity: Quantidade de contratos
            stop_loss: Preço de stop loss (opcional)
            take_profit: Preço de take profit (opcional)
            
        Returns:
            Dicionário com resultado da operação
        """
        if symbol is None:
            symbol = self.symbol
        
        if self.is_simulation:
            return self._place_simulation_order(symbol, "MARKET", side, quantity, None, stop_loss, take_profit)
        
        try:
            # Obter último preço
            last_tick = self.mt5.symbol_info_tick(symbol)
            if not last_tick:
                logger.error(f"Não foi possível obter o último preço para {symbol}")
                return {
                    "success": False,
                    "error": f"Não foi possível obter o último preço para {symbol}"
                }
            
            price = last_tick.ask if side == "BUY" else last_tick.bid
            
            # Definir tipo de ordem
            order_type = self.mt5.ORDER_TYPE_BUY if side == "BUY" else self.mt5.ORDER_TYPE_SELL
            
            # Preparar request de ordem
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(quantity),
                "type": order_type,
                "price": price,
                "sl": stop_loss if stop_loss else 0.0,
                "tp": take_profit if take_profit else 0.0,
                "deviation": 10,  # Desvio de preço permitido
                "magic": 234000,  # Identificador do robô
                "comment": "WINFUT Trading Robot",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_FOK,
            }
            
            # Enviar ordem
            result = self.mt5.order_send(request)
            
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erro ao enviar ordem: {result.retcode}, {result.comment}")
                return {
                    "success": False,
                    "error": f"Erro ao enviar ordem: {result.retcode}, {result.comment}"
                }
            
            logger.info(f"Ordem enviada com sucesso. Ticket: {result.order}")
            return {
                "success": True,
                "order_id": result.order,
                "price": price,
                "side": side,
                "quantity": quantity,
                "type": "MARKET"
            }
            
        except Exception as e:
            logger.error(f"Erro ao enviar ordem de mercado: {str(e)}")
            return {
                "success": False,
                "error": f"Erro ao enviar ordem de mercado: {str(e)}"
            }
    
    def place_limit_order(self, symbol: Optional[str] = None, side: str = "BUY", 
                         quantity: float = 1.0, price: float = 0.0, 
                         stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Coloca uma ordem limite.
        
        Args:
            symbol: Símbolo para a ordem (padrão: self.symbol)
            side: Direção da ordem ('BUY' ou 'SELL')
            quantity: Quantidade de contratos
            price: Preço limite
            stop_loss: Preço de stop loss (opcional)
            take_profit: Preço de take profit (opcional)
            
        Returns:
            Dicionário com resultado da operação
        """
        if symbol is None:
            symbol = self.symbol
        
        if self.is_simulation:
            return self._place_simulation_order(symbol, "LIMIT", side, quantity, price, stop_loss, take_profit)
        
        try:
            # Definir tipo de ordem
            order_type = self.mt5.ORDER_TYPE_BUY_LIMIT if side == "BUY" else self.mt5.ORDER_TYPE_SELL_LIMIT
            
            # Preparar request de ordem
            request = {
                "action": self.mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": float(quantity),
                "type": order_type,
                "price": price,
                "sl": stop_loss if stop_loss else 0.0,
                "tp": take_profit if take_profit else 0.0,
                "deviation": 10,  # Desvio de preço permitido
                "magic": 234000,  # Identificador do robô
                "comment": "WINFUT Trading Robot",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_FOK,
            }
            
            # Enviar ordem
            result = self.mt5.order_send(request)
            
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erro ao enviar ordem limite: {result.retcode}, {result.comment}")
                return {
                    "success": False,
                    "error": f"Erro ao enviar ordem limite: {result.retcode}, {result.comment}"
                }
            
            logger.info(f"Ordem limite enviada com sucesso. Ticket: {result.order}")
            return {
                "success": True,
                "order_id": result.order,
                "price": price,
                "side": side,
                "quantity": quantity,
                "type": "LIMIT"
            }
            
        except Exception as e:
            logger.error(f"Erro ao enviar ordem limite: {str(e)}")
            return {
                "success": False,
                "error": f"Erro ao enviar ordem limite: {str(e)}"
            }
    
    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """
        Cancela uma ordem pendente.
        
        Args:
            order_id: ID da ordem a ser cancelada
            
        Returns:
            Dicionário com resultado da operação
        """
        if self.is_simulation:
            return self._cancel_simulation_order(order_id)
        
        try:
            # Preparar request de cancelamento
            request = {
                "action": self.mt5.TRADE_ACTION_REMOVE,
                "order": order_id,
                "comment": "Order canceled by WINFUT Trading Robot"
            }
            
            # Enviar request
            result = self.mt5.order_send(request)
            
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erro ao cancelar ordem: {result.retcode}, {result.comment}")
                return {
                    "success": False,
                    "error": f"Erro ao cancelar ordem: {result.retcode}, {result.comment}"
                }
            
            logger.info(f"Ordem {order_id} cancelada com sucesso")
            return {
                "success": True,
                "order_id": order_id,
                "message": "Ordem cancelada com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro ao cancelar ordem: {str(e)}")
            return {
                "success": False,
                "error": f"Erro ao cancelar ordem: {str(e)}"
            }
    
    def close_position(self, position_id: int) -> Dict[str, Any]:
        """
        Fecha uma posição aberta.
        
        Args:
            position_id: ID da posição a ser fechada
            
        Returns:
            Dicionário com resultado da operação
        """
        if self.is_simulation:
            return self._close_simulation_position(position_id)
        
        try:
            # Obter informações da posição
            positions = self.mt5.positions_get(ticket=position_id)
            
            if not positions:
                logger.error(f"Posição {position_id} não encontrada")
                return {
                    "success": False,
                    "error": f"Posição {position_id} não encontrada"
                }
            
            position = positions[0]
            
            # Definir tipo de ordem para fechar a posição
            close_type = self.mt5.ORDER_TYPE_SELL if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY
            
            # Obter preço atual
            symbol = position.symbol
            last_tick = self.mt5.symbol_info_tick(symbol)
            if not last_tick:
                logger.error(f"Não foi possível obter o último preço para {symbol}")
                return {
                    "success": False,
                    "error": f"Não foi possível obter o último preço para {symbol}"
                }
            
            price = last_tick.bid if position.type == self.mt5.ORDER_TYPE_BUY else last_tick.ask
            
            # Preparar request para fechar posição
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": close_type,
                "position": position_id,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": "Position closed by WINFUT Trading Robot",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_FOK,
            }
            
            # Enviar request
            result = self.mt5.order_send(request)
            
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erro ao fechar posição: {result.retcode}, {result.comment}")
                return {
                    "success": False,
                    "error": f"Erro ao fechar posição: {result.retcode}, {result.comment}"
                }
            
            logger.info(f"Posição {position_id} fechada com sucesso. Ticket: {result.order}")
            return {
                "success": True,
                "position_id": position_id,
                "order_id": result.order,
                "message": "Posição fechada com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {str(e)}")
            return {
                "success": False,
                "error": f"Erro ao fechar posição: {str(e)}"
            }
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Fecha todas as posições abertas.
        
        Returns:
            Dicionário com resultado da operação
        """
        if self.is_simulation:
            return self._close_all_simulation_positions()
        
        try:
            # Obter todas as posições abertas
            positions = self.mt5.positions_get()
            
            if not positions:
                logger.info("Não há posições abertas para fechar")
                return {
                    "success": True,
                    "message": "Não há posições abertas para fechar",
                    "closed_positions": []
                }
            
            # Fechar cada posição
            closed_positions = []
            failed_positions = []
            
            for position in positions:
                position_id = position.ticket
                result = self.close_position(position_id)
                
                if result.get("success", False):
                    closed_positions.append(position_id)
                else:
                    failed_positions.append({
                        "position_id": position_id,
                        "error": result.get("error", "Erro desconhecido")
                    })
            
            logger.info(f"Fechadas {len(closed_positions)} de {len(positions)} posições")
            
            return {
                "success": len(failed_positions) == 0,
                "message": f"Fechadas {len(closed_positions)} de {len(positions)} posições",
                "closed_positions": closed_positions,
                "failed_positions": failed_positions
            }
            
        except Exception as e:
            logger.error(f"Erro ao fechar todas as posições: {str(e)}")
            return {
                "success": False,
                "error": f"Erro ao fechar todas as posições: {str(e)}",
                "closed_positions": []
            }
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Obtém lista de posições abertas.
        
        Returns:
            Lista de posições abertas
        """
        if self.is_simulation:
            return self.open_positions_sim
        
        try:
            # Obter posições abertas do MT5
            positions = self.mt5.positions_get()
            
            if not positions:
                logger.info("Não há posições abertas")
                return []
            
            # Converter posições para lista de dicionários
            positions_list = []
            
            for position in positions:
                # Calcular lucro e percentual
                profit = position.profit
                profit_pct = 0.0
                if position.price_open > 0:
                    profit_pct = (position.price_current - position.price_open) / position.price_open * 100
                    if position.type == self.mt5.ORDER_TYPE_SELL:
                        profit_pct = -profit_pct
                
                positions_list.append({
                    "id": position.ticket,
                    "symbol": position.symbol,
                    "type": "BUY" if position.type == self.mt5.ORDER_TYPE_BUY else "SELL",
                    "quantity": position.volume,
                    "open_price": position.price_open,
                    "current_price": position.price_current,
                    "stop_loss": position.sl,
                    "take_profit": position.tp,
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "open_time": datetime.fromtimestamp(position.time),
                    "comment": position.comment
                })
            
            return positions_list
            
        except Exception as e:
            logger.error(f"Erro ao obter posições abertas: {str(e)}")
            return self.open_positions_sim
    
    def place_order(self, order_type: str = "MARKET", side: str = "BUY", 
                     quantity: float = 1.0, price: Optional[float] = None,
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Coloca uma ordem de mercado ou limite.
        Método compatível com a interface do ProfitProAPI para uso pelo OrderManager.
        
        Args:
            order_type: Tipo de ordem ('MARKET' ou 'LIMIT')
            side: Direção da ordem ('BUY' ou 'SELL')
            quantity: Quantidade de contratos
            price: Preço limite (apenas para ordens LIMIT)
            stop_loss: Preço de stop loss (opcional)
            take_profit: Preço de take profit (opcional)
            
        Returns:
            Dicionário com resultado da operação
        """
        try:
            symbol = self.symbol
            
            if order_type == "MARKET":
                # Colocar ordem de mercado
                result = self.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            else:  # LIMIT
                # Verificar se o preço foi fornecido
                if price is None:
                    return {
                        "success": False,
                        "error": "Preço é obrigatório para ordens LIMIT"
                    }
                
                # Colocar ordem limite
                result = self.place_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
            # Formatar resposta para ser compatível com ProfitProAPI
            if "order_id" in result:
                return {
                    "success": True,
                    "order_id": result["order_id"],
                    "fill_price": result.get("price")
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Erro desconhecido ao colocar ordem")
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Obtém o status de uma ordem.
        Método compatível com a interface do ProfitProAPI para uso pelo OrderManager.
        
        Args:
            order_id: ID da ordem
            
        Returns:
            Dicionário com status da ordem
        """
        try:
            # Converter order_id para inteiro, pois o MT5 usa IDs numéricos
            mt5_order_id = int(order_id)
            
            # Obter ordens pendentes
            pending_orders = self._mt5.orders_get(ticket=mt5_order_id) if self._mt5 else None
            
            if pending_orders and len(pending_orders) > 0:
                # A ordem ainda está pendente
                return {
                    "status": "PENDING",
                    "order_id": order_id
                }
            
            # Verificar se a ordem foi executada (posição aberta)
            positions = self._mt5.positions_get(ticket=mt5_order_id) if self._mt5 else None
            
            if positions and len(positions) > 0:
                # A ordem foi executada e agora é uma posição aberta
                return {
                    "status": "FILLED",
                    "order_id": order_id,
                    "fill_price": positions[0].price_current
                }
            
            # Verificar histórico para ver se a ordem foi cancelada ou rejeitada
            history = self._mt5.history_orders_get(ticket=mt5_order_id) if self._mt5 else None
            
            if history and len(history) > 0:
                order = history[0]
                
                # Verificar o estado da ordem
                if order.state == self._mt5.ORDER_STATE_FILLED:
                    return {
                        "status": "FILLED",
                        "order_id": order_id,
                        "fill_price": order.price_current
                    }
                elif order.state == self._mt5.ORDER_STATE_CANCELED:
                    return {
                        "status": "CANCELLED",
                        "order_id": order_id
                    }
                elif order.state == self._mt5.ORDER_STATE_REJECTED:
                    return {
                        "status": "REJECTED",
                        "order_id": order_id
                    }
                else:
                    return {
                        "status": "UNKNOWN",
                        "order_id": order_id
                    }
            
            # Se estamos em modo simulado, retornamos dados simulados
            if not self._mt5:
                # 50% de chance de ser preenchida ou cancelada para simulação
                import random
                if random.random() > 0.5:
                    return {
                        "status": "FILLED",
                        "order_id": order_id,
                        "fill_price": 100000.0  # Preço simulado
                    }
                else:
                    return {
                        "status": "CANCELLED",
                        "order_id": order_id
                    }
            
            # Ordem não encontrada
            return {
                "status": "UNKNOWN",
                "order_id": order_id
            }
            
        except Exception as e:
            # Em caso de erro, retornar status desconhecido
            return {
                "status": "ERROR",
                "order_id": order_id,
                "error": str(e)
            }
            
    def modify_order(self, order_id: str, new_stop_loss: Optional[float] = None, 
                    new_take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Modifica uma ordem existente.
        Método compatível com a interface do ProfitProAPI para uso pelo OrderManager.
        
        Args:
            order_id: ID da ordem/posição
            new_stop_loss: Novo nível de stop loss (ou None para não modificar)
            new_take_profit: Novo nível de take profit (ou None para não modificar)
            
        Returns:
            Dicionário com resultado da modificação
        """
        try:
            # Converter order_id para inteiro, pois o MT5 usa IDs numéricos
            mt5_order_id = int(order_id)
            
            # Verificar se é uma posição aberta
            positions = self._mt5.positions_get(ticket=mt5_order_id) if self._mt5 else None
            
            if positions and len(positions) > 0:
                position = positions[0]
                
                # Preparar request para modificar posição
                request = {
                    "action": self._mt5.TRADE_ACTION_SLTP if self._mt5 else 0,
                    "symbol": position.symbol,
                    "position": mt5_order_id,
                    "sl": new_stop_loss if new_stop_loss is not None else position.sl,
                    "tp": new_take_profit if new_take_profit is not None else position.tp
                }
                
                # Enviar solicitação
                result = self._mt5.order_send(request) if self._mt5 else None
                
                if result and result.retcode == self._mt5.TRADE_RETCODE_DONE:
                    return {
                        "success": True,
                        "message": "Posição modificada com sucesso"
                    }
                else:
                    error_code = result.retcode if result else 0
                    error_msg = f"Falha ao modificar posição: {error_code}"
                    return {
                        "success": False,
                        "error": error_msg
                    }
            
            # Verificar se é uma ordem pendente
            orders = self._mt5.orders_get(ticket=mt5_order_id) if self._mt5 else None
            
            if orders and len(orders) > 0:
                order = orders[0]
                
                # Preparar request para modificar ordem
                request = {
                    "action": self._mt5.TRADE_ACTION_MODIFY if self._mt5 else 0,
                    "order": mt5_order_id,
                    "symbol": order.symbol,
                    "price": order.price_open,
                    "sl": new_stop_loss if new_stop_loss is not None else order.sl,
                    "tp": new_take_profit if new_take_profit is not None else order.tp,
                    "type_time": order.type_time,
                    "expiration": order.time_expiration
                }
                
                # Enviar solicitação
                result = self._mt5.order_send(request) if self._mt5 else None
                
                if result and result.retcode == self._mt5.TRADE_RETCODE_DONE:
                    return {
                        "success": True,
                        "message": "Ordem modificada com sucesso"
                    }
                else:
                    error_code = result.retcode if result else 0
                    error_msg = f"Falha ao modificar ordem: {error_code}"
                    return {
                        "success": False,
                        "error": error_msg
                    }
            
            # Se estamos em modo simulado, simulamos um sucesso
            if not self._mt5:
                return {
                    "success": True,
                    "message": "Ordem/posição modificada com sucesso (simulação)"
                }
            
            # Se não encontrou nem ordem nem posição
            return {
                "success": False,
                "error": "Ordem ou posição não encontrada"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def get_order_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Obtém histórico de ordens.
        
        Args:
            days: Número de dias para buscar histórico
            
        Returns:
            Lista de ordens do histórico
        """
        if self.is_simulation:
            return self.order_history_sim
        
        try:
            # Calcular intervalo de tempo
            now = datetime.now()
            from_date = now - timedelta(days=days)
            
            # Obter histórico de trades do MT5
            trades = self.mt5.history_deals_get(from_date, now)
            
            if not trades:
                logger.info(f"Nenhum trade encontrado nos últimos {days} dias")
                return []
            
            # Converter trades para lista de dicionários
            trades_list = []
            
            for trade in trades:
                # Determinar tipo de operação
                if trade.type == self.mt5.DEAL_TYPE_BUY:
                    type_str = "BUY"
                elif trade.type == self.mt5.DEAL_TYPE_SELL:
                    type_str = "SELL"
                else:
                    type_str = "OTHER"
                
                trades_list.append({
                    "id": trade.ticket,
                    "symbol": trade.symbol,
                    "type": type_str,
                    "quantity": trade.volume,
                    "price": trade.price,
                    "profit": trade.profit,
                    "time": datetime.fromtimestamp(trade.time),
                    "comment": trade.comment,
                    "order_id": trade.order,
                    "position_id": trade.position_id
                })
            
            # Ordenar por data, mais recente primeiro
            trades_list.sort(key=lambda x: x["time"], reverse=True)
            
            return trades_list
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico de ordens: {str(e)}")
            return self.order_history_sim
    
    # --- Métodos para simulação ---
    
    def _get_simulation_price_data(self, symbol: str, timeframe: str, period: str, limit: int) -> pd.DataFrame:
        """Gera dados de preço simulados para testes."""
        # Usar dados já gerados ou gerar novos
        if hasattr(self, 'sim_data') and self.sim_data is not None:
            # Filtrar por limite se necessário
            if len(self.sim_data) > limit:
                df = self.sim_data.tail(limit)
            else:
                df = self.sim_data.copy()
                
            # Garantir que datetime está no índice (e não como coluna)
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
                
            return df
        
        # Gerar novos dados
        df = self._generate_sample_data(symbol, timeframe, limit)
        
        # Garantir que datetime está no índice (e não como coluna)
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
            
        return df
    
    def _generate_sample_data(self, symbol: str, timeframe: str, num_bars: int, start_date=None) -> pd.DataFrame:
        """
        Gera dados OHLCV sintéticos para testes.
        
        Args:
            symbol: Símbolo para os dados
            timeframe: Timeframe dos dados
            num_bars: Número de barras a gerar
            start_date: Data inicial (opcional)
            
        Returns:
            DataFrame com dados OHLCV
        """
        # Definir data inicial
        if start_date is None:
            start_date = datetime.now() - timedelta(days=num_bars // 24)  # Estimar dias baseado em barras
        
        # Definir intervalo de tempo baseado no timeframe
        if timeframe == '1m':
            interval = timedelta(minutes=1)
        elif timeframe == '5m':
            interval = timedelta(minutes=5)
        elif timeframe == '15m':
            interval = timedelta(minutes=15)
        elif timeframe == '30m':
            interval = timedelta(minutes=30)
        elif timeframe == '1h':
            interval = timedelta(hours=1)
        elif timeframe == '1d':
            interval = timedelta(days=1)
        else:
            interval = timedelta(minutes=15)  # padrão para 15m
        
        # Gerar datas
        dates = [start_date + interval * i for i in range(num_bars)]
        
        # Símbolo específico para WINFUT
        if symbol and "WIN" in symbol:
            # Valores mais realistas para o mini índice
            base_price = 112000  # Valor base do IBOV em pontos
            volatility = 1000    # Volatilidade do IBOV em pontos
        else:
            # Valores genéricos para outros símbolos
            base_price = 100
            volatility = 2
        
        # Gerar preços
        data = []
        current_price = base_price
        
        for date in dates:
            # Verificar se é horário de negociação (9h às 18h em dias úteis)
            is_trading_hour = (0 <= date.weekday() <= 4) and (9 <= date.hour < 18)
            
            if not is_trading_hour:
                continue
            
            # Adicionar aleatoriedade com tendência
            change = np.random.normal(0, 1) * volatility
            
            # Adicionar alguma tendência
            if len(data) > 0:
                prev_close = data[-1]['close']
                # Tendência baseada no preço anterior
                trend = (prev_close - base_price) * 0.05
                change -= trend  # Força de reversão à média
            
            # Gerar OHLC
            open_price = current_price
            high_price = open_price + abs(change) + np.random.normal(0, 1) * volatility * 0.5
            low_price = open_price - abs(change) - np.random.normal(0, 1) * volatility * 0.5
            close_price = open_price + change
            
            # Garantir que high >= open, close e low <= open, close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume
            volume = int(np.random.exponential(1000))
            
            # Adicionar barra
            data.append({
                'datetime': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            # Atualizar preço atual
            current_price = close_price
        
        # Criar DataFrame
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        
        # Atualizar último preço
        if not df.empty:
            self.last_price = df['close'].iloc[-1]
        
        return df
    
    def _get_simulation_order_book(self, symbol: str) -> Dict[str, Any]:
        """Gera livro de ofertas simulado para testes."""
        # Obter último preço
        last_price = 0
        if hasattr(self, 'last_price') and self.last_price is not None:
            last_price = self.last_price
        elif len(self.sim_data) > 0:
            last_price = self.sim_data['close'].iloc[-1]
        else:
            last_price = 112000  # Valor padrão para WINFUT
        
        # Gerar bids e asks
        bids = []
        asks = []
        
        # Gerar 10 níveis para cada lado
        for i in range(10):
            bid_price = last_price - (i + 1) * 5
            ask_price = last_price + (i + 1) * 5
            
            # Volume aleatório
            bid_volume = int(np.random.exponential(100))
            ask_volume = int(np.random.exponential(100))
            
            bids.append({
                'price': bid_price,
                'volume': bid_volume
            })
            
            asks.append({
                'price': ask_price,
                'volume': ask_volume
            })
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_simulation_times_and_trades(self, symbol: str, limit: int) -> pd.DataFrame:
        """Gera dados de times & trades simulados para testes."""
        # Obter último preço
        last_price = 0
        if hasattr(self, 'last_price') and self.last_price is not None:
            last_price = self.last_price
        elif len(self.sim_data) > 0:
            last_price = self.sim_data['close'].iloc[-1]
        else:
            last_price = 112000  # Valor padrão para WINFUT
        
        # Gerar trades
        trades = []
        now = datetime.now()
        
        for i in range(limit):
            # Data do trade
            trade_time = now - timedelta(seconds=i * 5)  # Um trade a cada 5 segundos
            
            # Preço com pequena variação
            price = last_price + np.random.normal(0, 5)
            
            # Volume
            volume = int(np.random.exponential(10))
            
            # Lado (compra ou venda)
            side = "BUY" if np.random.random() > 0.5 else "SELL"
            
            trades.append({
                'datetime': trade_time,
                'price': price,
                'volume': volume,
                'side': side,
                'bid': price - 5,
                'ask': price + 5
            })
        
        # Criar DataFrame
        df = pd.DataFrame(trades)
        
        return df
    
    def _place_simulation_order(self, symbol: str, order_type: str, side: str, 
                               quantity: float, price: Optional[float], 
                               stop_loss: Optional[float], take_profit: Optional[float]) -> Dict[str, Any]:
        """Simula a colocação de uma ordem."""
        # Gerar ID de ordem
        order_id = self.order_id_counter
        self.order_id_counter += 1
        
        # Obter último preço
        last_price = 0
        if hasattr(self, 'last_price') and self.last_price is not None:
            last_price = self.last_price
        elif len(self.sim_data) > 0:
            last_price = self.sim_data['close'].iloc[-1]
        else:
            last_price = 112000  # Valor padrão para WINFUT
        
        # Para ordens de mercado, usar o último preço
        if order_type == "MARKET":
            exec_price = last_price
            
            # Simular execução imediata
            position = {
                "id": order_id,
                "symbol": symbol,
                "type": side,
                "quantity": quantity,
                "open_price": exec_price,
                "current_price": exec_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "profit": 0.0,
                "profit_pct": 0.0,
                "open_time": datetime.now(),
                "comment": f"Simulated {side} position"
            }
            
            # Adicionar à lista de posições abertas
            self.open_positions_sim.append(position)
            
            # Adicionar ao histórico de ordens
            self.order_history_sim.append({
                "id": order_id,
                "symbol": symbol,
                "type": f"{side} (MARKET)",
                "quantity": quantity,
                "price": exec_price,
                "profit": 0.0,
                "time": datetime.now(),
                "comment": "Simulated market order",
                "order_id": order_id,
                "position_id": order_id
            })
            
            # Atualizar saldo da conta
            multiplier = 0.20  # Valor do ponto do WINFUT
            margin = exec_price * quantity * multiplier
            self.account_info_sim["margin"] += margin
            self.account_info_sim["free_margin"] = self.account_info_sim["balance"] - self.account_info_sim["margin"]
            
            logger.info(f"Simulated {side} {order_type} order executed at {exec_price}")
            
            return {
                "success": True,
                "order_id": order_id,
                "position_id": order_id,
                "price": exec_price,
                "side": side,
                "quantity": quantity,
                "type": order_type
            }
        
        else:  # LIMIT order
            if price is None:
                return {
                    "success": False,
                    "error": "Preço é obrigatório para ordens limite"
                }
            
            # Adicionar ao histórico de ordens
            self.order_history_sim.append({
                "id": order_id,
                "symbol": symbol,
                "type": f"{side} (LIMIT)",
                "quantity": quantity,
                "price": price,
                "profit": 0.0,
                "time": datetime.now(),
                "comment": "Simulated limit order",
                "order_id": order_id,
                "position_id": None  # Ainda não executada
            })
            
            logger.info(f"Simulated {side} {order_type} order placed at {price}")
            
            return {
                "success": True,
                "order_id": order_id,
                "price": price,
                "side": side,
                "quantity": quantity,
                "type": order_type
            }
    
    def _close_simulation_position(self, position_id: int) -> Dict[str, Any]:
        """Simula o fechamento de uma posição."""
        # Procurar a posição pelo ID
        position_idx = None
        for i, position in enumerate(self.open_positions_sim):
            if position["id"] == position_id:
                position_idx = i
                break
        
        if position_idx is None:
            return {
                "success": False,
                "error": f"Posição {position_id} não encontrada"
            }
        
        # Obter informações da posição
        position = self.open_positions_sim[position_idx]
        
        # Simular execução ao último preço
        last_price = 0
        if hasattr(self, 'last_price') and self.last_price is not None:
            last_price = self.last_price
        elif len(self.sim_data) > 0:
            last_price = self.sim_data['close'].iloc[-1]
        else:
            last_price = 112000  # Valor padrão para WINFUT
        
        # Calcular lucro
        price_diff = last_price - position["open_price"]
        if position["type"] == "SELL":
            price_diff = -price_diff
        
        profit = price_diff * position["quantity"] * 0.20  # Valor do ponto do WINFUT
        
        # Adicionar ao histórico de ordens
        close_order_id = self.order_id_counter
        self.order_id_counter += 1
        
        self.order_history_sim.append({
            "id": close_order_id,
            "symbol": position["symbol"],
            "type": "SELL" if position["type"] == "BUY" else "BUY",  # Inverso da posição
            "quantity": position["quantity"],
            "price": last_price,
            "profit": profit,
            "time": datetime.now(),
            "comment": "Simulated position close",
            "order_id": close_order_id,
            "position_id": position_id
        })
        
        # Atualizar saldo da conta
        self.account_info_sim["balance"] += profit
        self.account_info_sim["equity"] = self.account_info_sim["balance"]
        self.account_info_sim["profit"] += profit
        self.account_info_sim["daily_pnl"] += profit
        
        # Liberar margem
        multiplier = 0.20  # Valor do ponto do WINFUT
        margin = position["open_price"] * position["quantity"] * multiplier
        self.account_info_sim["margin"] -= margin
        self.account_info_sim["free_margin"] = self.account_info_sim["balance"] - self.account_info_sim["margin"]
        
        # Remover a posição da lista
        self.open_positions_sim.pop(position_idx)
        
        logger.info(f"Simulated position {position_id} closed with profit: {profit}")
        
        return {
            "success": True,
            "position_id": position_id,
            "profit": profit,
            "message": f"Posição fechada com sucesso. Lucro: {profit}"
        }
    
    def _cancel_simulation_order(self, order_id: int) -> Dict[str, Any]:
        """Simula o cancelamento de uma ordem."""
        # Procurar a ordem pelo ID no histórico
        order_found = False
        for order in self.order_history_sim:
            if order["id"] == order_id and "LIMIT" in order["type"]:
                order_found = True
                order["comment"] = "Simulated order canceled"
                break
        
        if not order_found:
            return {
                "success": False,
                "error": f"Ordem {order_id} não encontrada ou não é uma ordem limite"
            }
        
        logger.info(f"Simulated order {order_id} canceled")
        
        return {
            "success": True,
            "order_id": order_id,
            "message": "Ordem cancelada com sucesso"
        }
    
    def _close_all_simulation_positions(self) -> Dict[str, Any]:
        """Simula o fechamento de todas as posições."""
        if not self.open_positions_sim:
            return {
                "success": True,
                "message": "Não há posições abertas para fechar",
                "closed_positions": []
            }
        
        # Fechar cada posição
        closed_positions = []
        for position in self.open_positions_sim[:]:  # Usar cópia para evitar problemas de iteração
            position_id = position["id"]
            result = self._close_simulation_position(position_id)
            
            if result.get("success", False):
                closed_positions.append(position_id)
        
        logger.info(f"Simulated closing all positions: {len(closed_positions)} positions closed")
        
        return {
            "success": True,
            "message": f"Fechadas {len(closed_positions)} posições",
            "closed_positions": closed_positions
        }


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Criar instância da API
    mt_api = MetaTraderAPI()
    
    # Conectar
    connected = mt_api.connect()
    print(f"Conectado: {connected}")
    
    if connected:
        # Obter dados de preço
        price_data = mt_api.get_price_data(timeframe="15m", limit=100)
        print(f"Dados de preço obtidos: {len(price_data)} barras")
        print(price_data.head())
        
        # Colocar ordem
        order_result = mt_api.place_market_order(side="BUY", quantity=1)
        print(f"Resultado da ordem: {order_result}")
        
        # Obter posições abertas
        positions = mt_api.get_open_positions()
        print(f"Posições abertas: {len(positions)}")
        for pos in positions:
            print(f"- {pos['type']} {pos['quantity']} @ {pos['open_price']}")
        
        # Desconectar
        mt_api.disconnect()
        print("Desconectado")