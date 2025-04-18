"""
Gerenciador de integração com a DLL do Profit Pro.
Este módulo fornece uma interface para usar as funções da DLL do Profit Pro
para obter dados de mercado e executar ordens.
"""
import logging
import os
import time
import platform
from pathlib import Path
from ctypes import *
from typing import Dict, List, Tuple, Any, Optional, Callable

# No Windows, importar wintypes e WinDLL
if platform.system() == 'Windows':
    from ctypes.wintypes import *
    from ctypes import WinDLL
    WINFUNCTYPE = CFUNCTYPE
else:
    # Em outros sistemas, criar dummy class
    class WinDLL:
        def __init__(self, *args, **kwargs):
            pass
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    WINFUNCTYPE = CFUNCTYPE

# Importar definições de tipos e funções da DLL
from profitTypes import *
from profit_dll import initializeDll

# Configuração de logging
logger = logging.getLogger(__name__)

# Códigos de erro
NL_OK = 0x00000000
NL_INTERNAL_ERROR = -2147483647
NL_NOT_INITIALIZED = NL_INTERNAL_ERROR + 1
NL_INVALID_ARGS = NL_NOT_INITIALIZED + 1
NL_WAITING_SERVER = NL_INVALID_ARGS + 1
NL_NO_LOGIN = NL_WAITING_SERVER + 1
NL_NO_LICENSE = NL_NO_LOGIN + 1

# Estados de conexão
CONNECTION_STATE_LOGIN = 0
CONNECTION_STATE_ROTEAMENTO = 1
CONNECTION_STATE_MARKET_DATA = 2
CONNECTION_STATE_MARKET_LOGIN = 3

class ProfitDLLManager:
    """
    Classe para gerenciar a integração com a DLL do Profit Pro.
    """
    
    def __init__(self, dll_path: str, activation_key: str):
        """
        Inicializa o gerenciador da DLL do Profit Pro.
        
        Args:
            dll_path: Caminho para o arquivo DLL
            activation_key: Chave de ativação do Profit Pro
        """
        self.dll_path = dll_path
        self.activation_key = activation_key
        self.dll = None
        
        # Estado da conexão
        self.connected = False
        self.market_connected = False
        self.broker_connected = False
        self.activated = False
        
        # Callbacks
        self._state_callback = None
        self._trade_callback = None
        self._account_callback = None
        
        self._initialize_dll()
        
    def _initialize_dll(self) -> bool:
        """
        Inicializa a DLL do Profit Pro.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        try:
            logger.info(f"Inicializando DLL do Profit Pro: {self.dll_path}")
            self.dll = initializeDll(self.dll_path)
            
            # Registrar callbacks
            self._register_callbacks()
            
            # Inicializar login
            ret = self.dll.DLLInitializeLogin(self.activation_key)
            if ret < NL_OK:
                logger.error(f"Falha ao inicializar login: {self._result_to_string(ret)}")
                return False
                
            logger.info("DLL do Profit Pro inicializada com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar DLL do Profit Pro: {str(e)}")
            self.dll = None
            return False
    
    def _register_callbacks(self) -> None:
        """
        Registra os callbacks da DLL.
        """
        # Callback de estado
        @WINFUNCTYPE(None, c_int32, c_int32)
        def state_callback(conn_type, result):
            if conn_type == CONNECTION_STATE_LOGIN:  # Notificações de login
                if result == 0:
                    self.connected = True
                    logger.info("Login: conectado")
                else:
                    self.connected = False
                    logger.error(f"Login: {result}")
                    
            elif conn_type == CONNECTION_STATE_ROTEAMENTO:  # Notificações de roteamento
                if result == 5:  # ROTEAMENTO_BROKER_CONNECTED
                    self.broker_connected = True
                    logger.info("Broker: Conectado")
                elif result > 2:
                    self.broker_connected = False
                    logger.warning("Broker: Sem conexão com corretora")
                else:
                    self.broker_connected = False
                    logger.warning(f"Broker: Sem conexão com servidores ({result})")
                    
            elif conn_type == CONNECTION_STATE_MARKET_DATA:  # Notificações de market data
                if result == 4:  # MARKET_CONNECTED
                    self.market_connected = True
                    logger.info("Market: Conectado")
                else:
                    self.market_connected = False
                    logger.warning(f"Market: {result}")
                    
            elif conn_type == CONNECTION_STATE_MARKET_LOGIN:  # Notificações de ativação
                if result == 0:  # CONNECTION_ACTIVATE_VALID
                    self.activated = True
                    logger.info("Ativação: OK")
                else:
                    self.activated = False
                    logger.error(f"Ativação: {result}")
                    
            # Verificar status geral da conexão
            if self.connected and self.market_connected and self.activated:
                logger.info("Serviços Profit Pro conectados")
            
        self._state_callback = state_callback
        self.dll.SetStateCallbackFunc(self._state_callback)
        
        # Callback de novas negociações
        @WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double, c_int, c_int, c_int, c_int, c_wchar)
        def trade_callback(asset_id, date, trade_number, price, vol, qtd, buy_agent, sell_agent, trade_type, is_edit):
            logger.debug(f"{asset_id.ticker} | Trade | {date}({trade_number}) {price}")
        
        self._trade_callback = trade_callback
        self.dll.SetNewTradeCallbackFunc(self._trade_callback)
        
        # Callback de contas
        @WINFUNCTYPE(None, c_int, c_wchar_p, c_wchar_p, c_wchar_p)
        def account_callback(broker_id, broker_name, account_id, holder_name):
            logger.info(f"Conta | {account_id} - {holder_name} | Corretora {broker_id} - {broker_name}")
        
        self._account_callback = account_callback
        self.dll.SetAccountCallbackFunc(self._account_callback)
    
    def get_quote(self, ticker: str, exchange: str = "B") -> Optional[Dict[str, Any]]:
        """
        Obtém a cotação atual de um ativo.
        
        Args:
            ticker: Código do ativo
            exchange: Código da bolsa (default: "B" para B3)
            
        Returns:
            Dict com informações da cotação ou None se não disponível
        """
        if not self.dll or not self.market_connected:
            logger.warning("DLL não inicializada ou sem conexão com market data")
            return None
            
        try:
            # Criando estrutura de asset
            asset_id = TAssetID()
            asset_id.ticker = c_wchar_p(ticker)
            asset_id.bolsa = c_wchar_p(exchange)
            asset_id.feed = c_int(0)  # 0 para Nelogica
            
            # TODO: Implementar chamada para obter cotação
            # Esta implementação depende da API exata do Profit Pro
            # Por enquanto, retornamos um dicionário simulado com a estrutura esperada
            
            logger.warning("Função get_quote ainda não implementada completamente")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter cotação para {ticker}: {str(e)}")
            return None
    
    def send_order(self, account_id: str, broker_id: int, ticker: str, 
                  exchange: str, order_side: int, order_type: int, 
                  quantity: int, price: float, stop_price: float = 0.0,
                  password: str = "") -> int:
        """
        Envia uma ordem para o mercado.
        
        Args:
            account_id: ID da conta
            broker_id: ID da corretora
            ticker: Código do ativo
            exchange: Código da bolsa (ex: "B" para B3)
            order_side: Lado da ordem (0 = compra, 1 = venda)
            order_type: Tipo da ordem (0 = limite, 1 = stop, 2 = mercado)
            quantity: Quantidade
            price: Preço limite (para ordens limite)
            stop_price: Preço de disparo (para ordens stop)
            password: Senha da conta (se necessário)
            
        Returns:
            ID da ordem ou valor negativo em caso de erro
        """
        if not self.dll or not self.broker_connected:
            logger.warning("DLL não inicializada ou sem conexão com broker")
            return -1
            
        try:
            # Criando estrutura para enviar ordem
            order = TConnectorSendOrder()
            order.Version = 0
            
            # Configurar conta
            order.AccountID.Version = 0
            order.AccountID.BrokerID = broker_id
            order.AccountID.AccountID = c_wchar_p(account_id)
            order.AccountID.SubAccountID = c_wchar_p("")
            order.AccountID.Reserved = 0
            
            # Configurar ativo
            order.AssetID.Version = 0
            order.AssetID.Ticker = c_wchar_p(ticker)
            order.AssetID.Exchange = c_wchar_p(exchange)
            order.AssetID.FeedType = 0  # Nelogica
            
            # Configurar detalhes da ordem
            order.Password = c_wchar_p(password)
            order.OrderType = order_type
            order.OrderSide = order_side
            order.Price = price
            order.StopPrice = stop_price
            order.Quantity = quantity
            
            # Enviar ordem
            order_id = self.dll.SendOrder(byref(order))
            
            if order_id < 0:
                logger.error(f"Erro ao enviar ordem: {self._result_to_string(order_id)}")
            else:
                logger.info(f"Ordem enviada com sucesso: ID {order_id}")
                
            return order_id
            
        except Exception as e:
            logger.error(f"Erro ao enviar ordem: {str(e)}")
            return -1
    
    def cancel_order(self, account_id: str, broker_id: int, 
                    order_id: int, cl_order_id: str = "",
                    password: str = "") -> bool:
        """
        Cancela uma ordem no mercado.
        
        Args:
            account_id: ID da conta
            broker_id: ID da corretora
            order_id: ID local da ordem
            cl_order_id: ID do cliente da ordem (opcional)
            password: Senha da conta (se necessário)
            
        Returns:
            True se o cancelamento foi bem-sucedido, False caso contrário
        """
        if not self.dll or not self.broker_connected:
            logger.warning("DLL não inicializada ou sem conexão com broker")
            return False
            
        try:
            # Criando estrutura para cancelar ordem
            cancel = TConnectorCancelOrder()
            cancel.Version = 0
            
            # Configurar conta
            cancel.AccountID.Version = 0
            cancel.AccountID.BrokerID = broker_id
            cancel.AccountID.AccountID = c_wchar_p(account_id)
            cancel.AccountID.SubAccountID = c_wchar_p("")
            cancel.AccountID.Reserved = 0
            
            # Configurar ordem
            cancel.OrderID.Version = 0
            cancel.OrderID.LocalOrderID = order_id
            cancel.OrderID.ClOrderID = c_wchar_p(cl_order_id)
            
            # Configurar senha
            cancel.Password = c_wchar_p(password)
            
            # Cancelar ordem
            ret = self.dll.SendCancelOrderV2(byref(cancel))
            
            if ret < NL_OK:
                logger.error(f"Erro ao cancelar ordem: {self._result_to_string(ret)}")
                return False
            else:
                logger.info(f"Ordem {order_id} cancelada com sucesso")
                return True
                
        except Exception as e:
            logger.error(f"Erro ao cancelar ordem: {str(e)}")
            return False
    
    def get_position(self, account_id: str, broker_id: int, 
                    ticker: str, exchange: str = "B") -> Optional[Dict[str, Any]]:
        """
        Obtém a posição atual de um ativo na conta.
        
        Args:
            account_id: ID da conta
            broker_id: ID da corretora
            ticker: Código do ativo
            exchange: Código da bolsa (ex: "B" para B3)
            
        Returns:
            Dict com informações da posição ou None se não disponível
        """
        if not self.dll or not self.broker_connected:
            logger.warning("DLL não inicializada ou sem conexão com broker")
            return None
            
        try:
            # Criando estrutura para consultar posição
            position = TConnectorTradingAccountPosition()
            position.Version = 0
            
            # Configurar conta
            position.AccountID.Version = 0
            position.AccountID.BrokerID = broker_id
            position.AccountID.AccountID = c_wchar_p(account_id)
            position.AccountID.SubAccountID = c_wchar_p("")
            position.AccountID.Reserved = 0
            
            # Configurar ativo
            position.AssetID.Version = 0
            position.AssetID.Ticker = c_wchar_p(ticker)
            position.AssetID.Exchange = c_wchar_p(exchange)
            position.AssetID.FeedType = 0  # Nelogica
            
            # Configurar tipo de posição
            position.PositionType = 1  # 1 = DayTrade, 2 = Consolidado
            
            # Obter posição
            ret = self.dll.GetPositionV2(byref(position))
            
            if ret < NL_OK:
                logger.error(f"Erro ao obter posição: {self._result_to_string(ret)}")
                return None
            
            # Criar dicionário com informações da posição
            result = {
                "open_quantity": position.OpenQuantity,
                "open_side": position.OpenSide,  # 0 = compra, 1 = venda
                "open_average_price": position.OpenAveragePrice,
                "daily_buy_quantity": position.DailyBuyQuantity,
                "daily_buy_average_price": position.DailyAverageBuyPrice,
                "daily_sell_quantity": position.DailySellQuantity,
                "daily_sell_average_price": position.DailyAverageSellPrice,
                "daily_available": position.DailyQuantityAvailable
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao obter posição para {ticker}: {str(e)}")
            return None
    
    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Obtém a lista de contas disponíveis.
        
        Returns:
            Lista de dicionários com informações das contas
        """
        if not self.dll or not self.connected:
            logger.warning("DLL não inicializada ou sem conexão")
            return []
            
        try:
            # Obter número de contas
            account_count = self.dll.GetAccountCount()
            if account_count <= 0:
                logger.warning("Nenhuma conta disponível")
                return []
                
            logger.info(f"Encontradas {account_count} contas")
            
            # Alocar buffer para as contas
            accounts = (TConnectorAccountIdentifierOut * account_count)()
            
            # Obter contas
            ret = self.dll.GetAccounts(0, 0, account_count, accounts)
            
            if ret < NL_OK:
                logger.error(f"Erro ao obter contas: {self._result_to_string(ret)}")
                return []
                
            # Processar contas
            result = []
            for i in range(account_count):
                account = accounts[i]
                account_info = {
                    "broker_id": account.BrokerID,
                    "account_id": "".join([c for c in account.AccountID[:account.AccountIDLength]]),
                    "sub_account_id": "".join([c for c in account.SubAccountID[:account.SubAccountIDLength]])
                }
                
                # Obter detalhes adicionais da conta
                account_details = TConnectorTradingAccountOut()
                account_details.Version = 0
                account_details.AccountID.Version = 0
                account_details.AccountID.BrokerID = account.BrokerID
                account_details.AccountID.AccountID = c_wchar_p(account_info["account_id"])
                account_details.AccountID.SubAccountID = c_wchar_p(account_info["sub_account_id"])
                account_details.AccountID.Reserved = 0
                
                if self.dll.GetAccountDetails(byref(account_details)) >= NL_OK:
                    # Alocar buffers para strings
                    broker_name = create_string_buffer(account_details.BrokerNameLength)
                    owner_name = create_string_buffer(account_details.OwnerNameLength)
                    sub_owner_name = create_string_buffer(account_details.SubOwnerNameLength)
                    
                    # Re-obter detalhes com buffers alocados
                    account_details.BrokerName = cast(broker_name, c_wchar_p)
                    account_details.OwnerName = cast(owner_name, c_wchar_p)
                    account_details.SubOwnerName = cast(sub_owner_name, c_wchar_p)
                    
                    if self.dll.GetAccountDetails(byref(account_details)) >= NL_OK:
                        account_info["broker_name"] = account_details.BrokerName
                        account_info["owner_name"] = account_details.OwnerName
                        account_info["sub_owner_name"] = account_details.SubOwnerName
                        account_info["is_sub_account"] = bool(account_details.AccountFlags & 1)
                        account_info["is_enabled"] = bool(account_details.AccountFlags & 2)
                
                result.append(account_info)
                
            return result
            
        except Exception as e:
            logger.error(f"Erro ao obter contas: {str(e)}")
            return []
    
    def is_connected(self) -> bool:
        """
        Verifica se está conectado ao Profit Pro.
        
        Returns:
            True se conectado, False caso contrário
        """
        return self.connected and self.market_connected and self.activated
    
    def disconnect(self) -> bool:
        """
        Desconecta do Profit Pro.
        
        Returns:
            True se desconectado com sucesso, False caso contrário
        """
        if not self.dll:
            return True
            
        try:
            # TODO: Implementar função de logout/finalização da DLL
            # Este método deve ser ajustado conforme a API exata do Profit Pro
            
            self.connected = False
            self.market_connected = False
            self.broker_connected = False
            self.activated = False
            
            logger.info("Desconectado do Profit Pro")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao desconectar do Profit Pro: {str(e)}")
            return False
    
    def _result_to_string(self, result: int) -> str:
        """
        Converte código de resultado em string.
        
        Args:
            result: Código de resultado
            
        Returns:
            String descritiva do resultado
        """
        if result == NL_INTERNAL_ERROR:
            return "Erro interno"
        elif result == NL_NOT_INITIALIZED:
            return "DLL não inicializada"
        elif result == NL_INVALID_ARGS:
            return "Argumentos inválidos"
        elif result == NL_WAITING_SERVER:
            return "Aguardando servidor"
        elif result == NL_NO_LOGIN:
            return "Nenhum login encontrado"
        elif result == NL_NO_LICENSE:
            return "Nenhuma licença encontrada"
        else:
            return f"Código de erro desconhecido: {result}"