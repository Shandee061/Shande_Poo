"""
Module for interacting with Profit Pro API to fetch market data and execute orders.
Supports both REST API and DLL-based interfaces, as well as simulation mode.
"""
import logging
import requests
import pandas as pd
import numpy as np
import datetime
import time
import os
import socket
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from config import PROFIT_API_URL, PROFIT_API_KEY, PROFIT_API_SECRET, SYMBOL, TIMEFRAME
from config import PROFIT_PRO_USE_DLL, PROFIT_PRO_HOST, PROFIT_PRO_PORT
from config import PROFIT_PRO_SIMULATION_MODE, PROFIT_PRO_DLL_PATH

# Import DLL Manager
from profit_dll_manager import ProfitDLLManager

logger = logging.getLogger(__name__)

class ProfitProAPI:
    """
    Class to interact with Profit Pro API for data fetching and order execution.
    Supports both REST API and DLL-based interfaces to Profit Pro.
    """
    
    def __init__(self, use_dll=PROFIT_PRO_USE_DLL, host=PROFIT_PRO_HOST, port=PROFIT_PRO_PORT, 
                 simulation_mode=PROFIT_PRO_SIMULATION_MODE, api_url=PROFIT_API_URL, 
                 api_key=PROFIT_API_KEY, api_secret=PROFIT_API_SECRET, symbol=SYMBOL,
                 timeframe=TIMEFRAME, dll_path=PROFIT_PRO_DLL_PATH, dll_version="5.0.3"):
        """
        Initialize the Profit Pro API connector.
        
        Args:
            use_dll (bool): Whether to use the DLL interface (True) or REST API (False)
            host (str): Hostname or IP for socket connection (usually localhost)
            port (int): Port number for socket connection
            simulation_mode (bool): Whether to use simulation mode for paper trading
            api_url (str): Base URL for REST API
            api_key (str): API key for REST API authentication
            api_secret (str): API secret for REST API authentication
            symbol (str): Trading symbol (e.g., "WINFUT", "WINM25")
            timeframe (str): Chart timeframe (e.g., "1m", "5m", "1h")
            dll_path (str): Path to the Profit Pro DLL file
            dll_version (str): Version of the Profit Pro DLL
        """
        # Common attributes
        self.symbol = symbol
        self.timeframe = timeframe
        self.connected = False
        
        # Simulation mode
        self.simulation_mode = simulation_mode
        logger.info(f"🧪 Profit Pro API initialized in {'SIMULATION' if simulation_mode else 'REAL'} mode")
        
        # REST API attributes
        self.api_url = api_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session() if not use_dll else None
        
        # DLL or socket interface attributes
        self.use_dll = use_dll
        self.host = host
        self.port = port
        self.socket = None
        self.dll_handle = None
        self.dll_path = dll_path
        self.dll_version = dll_version
        
        # Simulation mode attributes
        if self.simulation_mode:
            self._init_simulation_data()
        
        # Configure connection based on interface type
        if not self.use_dll and self.session:
            # Configure authentication headers for REST API
            self.session.headers.update({
                'X-API-Key': self.api_key,
                'X-API-Secret': self.api_secret,
                'Content-Type': 'application/json'
            })
        elif self.use_dll:
            # Will be initialized when connect() is called
            self.rtd_client = None
        
    def connect(self) -> bool:
        """
        Establishes connection with Profit Pro API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if self.use_dll:
            return self._connect_dll()
        else:
            return self._connect_rest_api()
            
    def _connect_rest_api(self) -> bool:
        """
        Connect to Profit API using REST interface.
        
        Returns:
            bool: Connection status
        """
        try:
            if not self.session:
                self.session = requests.Session()
                self.session.headers.update({
                    'X-API-Key': self.api_key,
                    'X-API-Secret': self.api_secret,
                    'Content-Type': 'application/json'
                })
                
            response = self.session.get(f"{self.api_url}/status")
            if response.status_code == 200:
                self.connected = True
                logger.info("Successfully connected to Profit Pro REST API")
                return True
            else:
                logger.error(f"Failed to connect to Profit Pro REST API: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Profit Pro REST API: {str(e)}")
            return False
            
    def _connect_dll(self) -> bool:
        """
        Connect to Profit Pro using DLL interface.
        
        Returns:
            bool: Connection status
        """
        try:
            logger.info(f"Attempting to connect to Profit Pro via DLL/Socket interface at {self.host}:{self.port}")
            
            # Try socket interface first
            if self.socket is None:
                try:
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.connect((self.host, self.port))
                    logger.info("Successfully connected to Profit Pro via socket")
                    self.connected = True
                    return True
                except Exception as sock_e:
                    logger.error(f"Failed to connect via socket: {str(sock_e)}")
                    self.socket = None
                    
                    # Try DLL interface if socket fails
                    try:
                        # Verify DLL path
                        if not self.dll_path:
                            logger.error("DLL path not configured. Please set dll_path in API constructor or PROFIT_PRO_DLL_PATH in config.")
                            return False
                            
                        # Initialize DLL connection using ProfitDLLManager
                        logger.info(f"Initializing DLL connection using {self.dll_path}")
                        
                        # Get activation key from environment or config
                        activation_key = os.environ.get("PROFIT_PRO_ACTIVATION_KEY", "")
                        if not activation_key:
                            logger.error("Activation key not found. Please set PROFIT_PRO_ACTIVATION_KEY in environment.")
                            return False
                        
                        # Create DLL manager
                        self.dll_handle = ProfitDLLManager(self.dll_path, activation_key)
                        
                        # Wait for connection to establish (max 10 seconds)
                        max_attempts = 20
                        for i in range(max_attempts):
                            if self.dll_handle.is_connected():
                                self.connected = True
                                logger.info("Successfully connected to Profit Pro via DLL")
                                return True
                            
                            time.sleep(0.5)
                            if i == max_attempts - 1:
                                logger.warning(f"DLL connection not established after {max_attempts/2} seconds")
                        
                        return self.dll_handle.is_connected()
                    except Exception as dll_e:
                        logger.error(f"Failed to connect via DLL: {str(dll_e)}")
                        return False
            else:
                # Already have a socket connection
                return self.connected
                
        except Exception as e:
            logger.error(f"Error in DLL/Socket connection: {str(e)}")
            return False
    
    def get_price_data(self, period: str = "1d", limit: int = 100) -> pd.DataFrame:
        """
        Fetches OHLCV (Open, High, Low, Close, Volume) data from Profit Pro.
        
        Args:
            period: Time period for the data (e.g., 1d for 1 day)
            limit: Number of candles to fetch
        
        Returns:
            pandas.DataFrame: OHLCV data with datetime index
        """
        if self.use_dll:
            return self._get_price_data_dll(period, limit)
        else:
            return self._get_price_data_rest(period, limit)
    
    def _get_price_data_rest(self, period: str = "1d", limit: int = 100) -> pd.DataFrame:
        """
        Fetches OHLCV data using REST API.
        """
        try:
            endpoint = f"{self.api_url}/market_data/{self.symbol}/{self.timeframe}"
            params = {
                "period": period,
                "limit": limit
            }
            
            if not self.session:
                logger.error("REST API session not initialized")
                return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                
            response = self.session.get(endpoint, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch price data: {response.status_code} - {response.text}")
                # Return empty dataframe with expected columns
                return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Convert numerical columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data via REST: {str(e)}")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            
    def _get_price_data_dll(self, period: str = "1d", limit: int = 100) -> pd.DataFrame:
        """
        Fetches OHLCV data using DLL/Socket interface.
        """
        try:
            logger.info(f"Fetching price data via DLL/Socket for {self.symbol} ({self.timeframe})")
            
            if self.socket:
                # If using socket connection, send request for price data
                request = {
                    "method": "get_price_data",
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "period": period,
                    "limit": limit
                }
                
                self.socket.sendall(json.dumps(request).encode('utf-8'))
                
                # Read response
                buffer = b""
                while b"\n" not in buffer:
                    data = self.socket.recv(4096)
                    if not data:
                        break
                    buffer += data
                
                response_data = json.loads(buffer.decode('utf-8').strip())
                
                if "error" in response_data:
                    logger.error(f"Error from socket: {response_data['error']}")
                    return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                
                # Process data
                if "data" in response_data and len(response_data["data"]) > 0:
                    df = pd.DataFrame(response_data["data"])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Convert numerical columns to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    return df
                else:
                    logger.warning("No data received from socket")
                    return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            
            elif self.dll_handle:
                # Using DLL to get price data
                try:
                    logger.info(f"Fetching price data via DLL for {self.symbol}")
                    
                    # Try to get quote data from the DLL
                    quote_data = self.dll_handle.get_quote(self.symbol)
                    
                    if not quote_data:
                        logger.warning(f"No quote data available for {self.symbol} via DLL")
                        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # TODO: This would need to be expanded to properly convert quote data to OHLCV format
                    # For now, this is a placeholder implementation
                    logger.warning("DLL quote data available but historical OHLCV not implemented yet")
                    return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                
                except Exception as dll_e:
                    logger.error(f"Error fetching price data via DLL: {str(dll_e)}")
                    return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            else:
                logger.error("No DLL/Socket connection available")
                return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                
        except Exception as e:
            logger.error(f"Error fetching price data via DLL/Socket: {str(e)}")
            return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_order_book(self) -> Dict[str, Any]:
        """
        Fetches current order book data from Profit Pro.
        
        Returns:
            dict: Order book data with bids and asks
        """
        if self.use_dll:
            return self._get_order_book_dll()
        else:
            return self._get_order_book_rest()
            
    def _get_order_book_rest(self) -> Dict[str, Any]:
        """
        Fetches order book data using REST API.
        """
        try:
            if not self.session:
                logger.error("REST API session not initialized")
                return {"bids": [], "asks": []}
                
            endpoint = f"{self.api_url}/order_book/{self.symbol}"
            response = self.session.get(endpoint)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch order book: {response.status_code} - {response.text}")
                return {"bids": [], "asks": []}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching order book via REST: {str(e)}")
            return {"bids": [], "asks": []}
            
    def _get_order_book_dll(self) -> Dict[str, Any]:
        """
        Fetches order book data using DLL/Socket interface.
        """
        try:
            logger.info(f"Fetching order book via DLL/Socket for {self.symbol}")
            
            if self.socket:
                # If using socket connection, send request for order book
                request = {
                    "method": "get_order_book",
                    "symbol": self.symbol
                }
                
                self.socket.sendall(json.dumps(request).encode('utf-8'))
                
                # Read response
                buffer = b""
                while b"\n" not in buffer:
                    data = self.socket.recv(4096)
                    if not data:
                        break
                    buffer += data
                
                response_data = json.loads(buffer.decode('utf-8').strip())
                
                if "error" in response_data:
                    logger.error(f"Error from socket: {response_data['error']}")
                    return {"bids": [], "asks": []}
                
                # Process data
                if "data" in response_data:
                    return response_data["data"]
                else:
                    logger.warning("No order book data received from socket")
                    return {"bids": [], "asks": []}
            
            elif self.dll_handle:
                # Using DLL to get order book
                try:
                    logger.info(f"Fetching order book via DLL for {self.symbol}")
                    
                    # TODO: Implement order book fetch using DLL
                    # This implementation depends on the exact API of the DLL
                    logger.warning("DLL order book implementation not complete")
                    return {"bids": [], "asks": []}
                    
                except Exception as dll_e:
                    logger.error(f"Error fetching order book via DLL: {str(dll_e)}")
                    return {"bids": [], "asks": []}
            else:
                logger.error("No DLL/Socket connection available")
                return {"bids": [], "asks": []}
                
        except Exception as e:
            logger.error(f"Error fetching order book via DLL/Socket: {str(e)}")
            return {"bids": [], "asks": []}
    
    def get_times_and_trades(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetches times and trades data from Profit Pro.
        
        Args:
            limit: Number of trades to fetch
        
        Returns:
            pandas.DataFrame: Times and trades data
        """
        try:
            endpoint = f"{self.api_url}/times_and_trades/{self.symbol}"
            params = {"limit": limit}
            response = self.session.get(endpoint, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch times and trades: {response.status_code} - {response.text}")
                return pd.DataFrame(columns=['datetime', 'price', 'quantity', 'side'])
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching times and trades: {str(e)}")
            return pd.DataFrame(columns=['datetime', 'price', 'quantity', 'side'])
    
    def _init_simulation_data(self):
        """
        Initialize data structures needed for simulation mode.
        """
        # Estruturas para armazenar informações simuladas
        self.sim_orders = {}  # Ordens simuladas
        self.sim_positions = []  # Posições abertas
        self.sim_order_history = []  # Histórico de ordens
        self.sim_account = {
            "balance": 100000.0,  # Saldo inicial
            "equity": 100000.0,
            "margin": 0.0,
            "free_margin": 100000.0,
            "margin_level": 100.0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0
        }
        self.sim_order_id_counter = 1000  # Contador para gerar IDs de ordens
        
        logger.info("🧪 Modo de simulação inicializado com saldo de R$ 100.000,00")
    
    def place_order(self, order_type: str, side: str, quantity: int, price: float = None, 
                   stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Places an order through Profit Pro API.
        
        Args:
            order_type: Type of order (MARKET, LIMIT, STOP)
            side: Order side (BUY, SELL)
            quantity: Number of contracts
            price: Price for limit orders
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            dict: Order response with order ID and status
        """
        
        # Check if we're in simulation mode
        if self.simulation_mode:
            return self._place_order_simulation(order_type, side, quantity, price, stop_loss, take_profit)
            
        # If not in simulation mode, proceed with actual API
        try:
            # Check if we're using DLL or REST API
            if self.use_dll and self.dll_handle and self.dll_handle.is_connected():
                return self._place_order_dll(order_type, side, quantity, price, stop_loss, take_profit)
            elif self.socket:
                return self._place_order_socket(order_type, side, quantity, price, stop_loss, take_profit)
            elif self.session:
                return self._place_order_rest(order_type, side, quantity, price, stop_loss, take_profit)
            else:
                logger.error("No connection method available to place order")
                return {"success": False, "error": "No connection available"}
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _place_order_rest(self, order_type: str, side: str, quantity: int, price: float = None, 
                       stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Places an order using REST API.
        """
        try:
            endpoint = f"{self.api_url}/order"
            
            payload = {
                "symbol": self.symbol,
                "order_type": order_type,
                "side": side,
                "quantity": quantity
            }
            
            if order_type == "LIMIT" and price is not None:
                payload["price"] = price
                
            if stop_loss is not None:
                payload["stop_loss"] = stop_loss
                
            if take_profit is not None:
                payload["take_profit"] = take_profit
                
            response = self.session.post(endpoint, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Failed to place order: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error placing order via REST: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _place_order_socket(self, order_type: str, side: str, quantity: int, price: float = None, 
                         stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Places an order using socket connection.
        """
        try:
            request = {
                "method": "place_order",
                "symbol": self.symbol,
                "order_type": order_type,
                "side": side,
                "quantity": quantity
            }
            
            if order_type in ["LIMIT", "STOP"] and price is not None:
                request["price"] = price
                
            if stop_loss is not None:
                request["stop_loss"] = stop_loss
                
            if take_profit is not None:
                request["take_profit"] = take_profit
                
            self.socket.sendall(json.dumps(request).encode('utf-8'))
            
            # Read response
            buffer = b""
            while b"\n" not in buffer:
                data = self.socket.recv(4096)
                if not data:
                    break
                buffer += data
            
            response_data = json.loads(buffer.decode('utf-8').strip())
            
            if "error" in response_data:
                logger.error(f"Failed to place order via socket: {response_data['error']}")
                return {"success": False, "error": response_data['error']}
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error placing order via socket: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _place_order_dll(self, order_type: str, side: str, quantity: int, price: float = None, 
                       stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Places an order using DLL interface.
        """
        try:
            logger.info(f"Placing order via DLL: {side} {order_type} for {quantity} contracts of {self.symbol}")
            
            if not self.dll_handle:
                logger.error("DLL handle not initialized")
                return {"success": False, "error": "DLL not initialized"}
                
            # Get available accounts
            accounts = self.dll_handle.get_accounts()
            if not accounts:
                logger.error("No trading accounts available")
                return {"success": False, "error": "No trading accounts available"}
                
            # Use the first account (in a real system, you might want to specify which account to use)
            account = accounts[0]
            account_id = account.get("account_id", "")
            broker_id = account.get("broker_id", 0)
            
            if not account_id:
                logger.error("Invalid account information")
                return {"success": False, "error": "Invalid account information"}
                
            # Convert order type from string to enum value expected by DLL
            dll_order_type = 0  # Default: Limit
            if order_type == "MARKET":
                dll_order_type = 2  # Market
            elif order_type == "STOP":
                dll_order_type = 1  # Stop
                
            # Convert side from string to enum value expected by DLL
            dll_side = 0 if side == "BUY" else 1  # 0 = Buy, 1 = Sell
            
            # Default price handling
            if price is None and order_type == "LIMIT":
                logger.error("Price must be specified for limit orders")
                return {"success": False, "error": "Price required for limit orders"}
                
            # Default to current price for market orders if price not specified
            if price is None and order_type == "MARKET":
                quote = self.dll_handle.get_quote(self.symbol)
                if quote:
                    price = quote.get("last", 0.0) or quote.get("bid", 0.0)
                else:
                    logger.error("No price data available for market order")
                    return {"success": False, "error": "No price data available"}
                    
            # Default stop price for stop orders
            stop_price = 0.0
            if order_type == "STOP" and price is not None:
                stop_price = price
                
            # Send the order
            result = self.dll_handle.send_order(
                account_id=account_id,
                broker_id=broker_id,
                ticker=self.symbol,
                exchange="B",  # B = B3 (Brazilian exchange)
                order_side=dll_side,
                order_type=dll_order_type,
                quantity=quantity,
                price=price or 0.0,
                stop_price=stop_price,
                password=""  # Password can be left empty if not required
            )
            
            if result < 0:
                error_msg = self.dll_handle._result_to_string(result)
                logger.error(f"Failed to place order via DLL: {error_msg}")
                return {"success": False, "error": error_msg}
                
            # If successful, result is the order ID
            return {
                "success": True,
                "order_id": result,
                "symbol": self.symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "status": "PENDING"  # Initial status is pending
            }
            
        except Exception as e:
            logger.error(f"Error placing order via DLL: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _place_order_simulation(self, order_type: str, side: str, quantity: int, price: float = None, 
                              stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Simulates placing an order without actually sending it to the broker.
        Used for paper trading or testing.
        
        Args:
            Same as place_order method
            
        Returns:
            dict: Simulated order response
        """
        logger.info(f"SIMULATION MODE: Placing {side} {order_type} order for {quantity} contracts")
        
        try:
            # Get current price data for simulation
            price_data = self.get_price_data(limit=1)
            
            if price_data.empty:
                return {
                    "success": False,
                    "error": "No price data available for simulation"
                }
                
            # Use latest close price for order simulation
            current_price = price_data["close"].iloc[-1]
            
            # Generate a simulated order ID
            order_id = f"SIM-{int(time.time())}-{np.random.randint(1000, 9999)}"
            
            # For market orders, simulate immediate fill
            if order_type == "MARKET":
                fill_price = current_price
                status = "FILLED"
            # For limit orders, check if price condition is met
            elif order_type == "LIMIT":
                if price is None:
                    return {
                        "success": False,
                        "error": "Price must be specified for limit orders"
                    }
                    
                # For buy limit, it fills if current price <= limit price
                # For sell limit, it fills if current price >= limit price
                if (side == "BUY" and current_price <= price) or (side == "SELL" and current_price >= price):
                    fill_price = price
                    status = "FILLED"
                else:
                    fill_price = None
                    status = "PENDING"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported order type in simulation: {order_type}"
                }
                
            # Create simulated response
            response = {
                "success": True,
                "order_id": order_id,
                "symbol": self.symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "status": status,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Add fill data if order was filled
            if status == "FILLED":
                response["fill_price"] = fill_price
                response["fill_time"] = datetime.datetime.now().isoformat()
                
            # Add stop loss and take profit if provided
            if stop_loss is not None:
                response["stop_loss"] = stop_loss
                
            if take_profit is not None:
                response["take_profit"] = take_profit
                
            logger.info(f"SIMULATION: Order result - {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error in order simulation: {str(e)}")
            return {"success": False, "error": f"Simulation error: {str(e)}"}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancels an existing order.
        
        Args:
            order_id: ID of the order to cancel
        
        Returns:
            dict: Cancellation response
        """
        # Check if we're in simulation mode
        if self.simulation_mode:
            return self._cancel_order_simulation(order_id)
            
        # If not in simulation mode, proceed with actual API
        try:
            # Check if we're using DLL or REST API
            if self.use_dll and self.dll_handle and self.dll_handle.is_connected():
                return self._cancel_order_dll(order_id)
            elif self.socket:
                return self._cancel_order_socket(order_id)
            elif self.session:
                return self._cancel_order_rest(order_id)
            else:
                logger.error("No connection method available to cancel order")
                return {"success": False, "error": "No connection available"}
                
        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _cancel_order_rest(self, order_id: str) -> Dict[str, Any]:
        """
        Cancels an order using REST API.
        """
        try:
            endpoint = f"{self.api_url}/order/{order_id}"
            response = self.session.delete(endpoint)
            
            if response.status_code != 200:
                logger.error(f"Failed to cancel order: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error canceling order via REST: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _cancel_order_socket(self, order_id: str) -> Dict[str, Any]:
        """
        Cancels an order using socket connection.
        """
        try:
            request = {
                "method": "cancel_order",
                "order_id": order_id
            }
            
            self.socket.sendall(json.dumps(request).encode('utf-8'))
            
            # Read response
            buffer = b""
            while b"\n" not in buffer:
                data = self.socket.recv(4096)
                if not data:
                    break
                buffer += data
            
            response_data = json.loads(buffer.decode('utf-8').strip())
            
            if "error" in response_data:
                logger.error(f"Failed to cancel order via socket: {response_data['error']}")
                return {"success": False, "error": response_data['error']}
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error canceling order via socket: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _cancel_order_dll(self, order_id: str) -> Dict[str, Any]:
        """
        Cancels an order using DLL interface.
        """
        try:
            logger.info(f"Canceling order via DLL: {order_id}")
            
            if not self.dll_handle:
                logger.error("DLL handle not initialized")
                return {"success": False, "error": "DLL not initialized"}
                
            # Get available accounts
            accounts = self.dll_handle.get_accounts()
            if not accounts:
                logger.error("No trading accounts available")
                return {"success": False, "error": "No trading accounts available"}
                
            # Use the first account (in a real system, you might want to specify which account to use)
            account = accounts[0]
            account_id = account.get("account_id", "")
            broker_id = account.get("broker_id", 0)
            
            if not account_id:
                logger.error("Invalid account information")
                return {"success": False, "error": "Invalid account information"}
                
            # Convert order_id to integer if it's a string number
            try:
                order_id_int = int(order_id)
            except ValueError:
                logger.error("Invalid order ID format for DLL cancellation")
                return {"success": False, "error": "Invalid order ID format"}
                
            # Send the cancel order request
            result = self.dll_handle.cancel_order(
                account_id=account_id,
                broker_id=broker_id,
                order_id=order_id_int,
                password=""  # Password can be left empty if not required
            )
            
            if not result:
                logger.error("Failed to cancel order via DLL")
                return {"success": False, "error": "Order cancellation failed"}
                
            # If successful
            return {
                "success": True,
                "order_id": order_id,
                "status": "CANCELLED",
                "timestamp": datetime.datetime.now().isoformat(),
                "message": "Order cancelled successfully"
            }
            
        except Exception as e:
            logger.error(f"Error canceling order via DLL: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _cancel_order_simulation(self, order_id: str) -> Dict[str, Any]:
        """
        Simulates canceling an order without actually sending it to the broker.
        Used for paper trading or testing.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            dict: Simulated cancellation response
        """
        logger.info(f"SIMULATION MODE: Cancelling order {order_id}")
        
        # Check if it's a simulated order (should start with SIM-)
        if not order_id.startswith("SIM-"):
            logger.warning(f"Attempted to cancel non-simulation order {order_id} in simulation mode")
            
        # In a real implementation, we would check against stored pending orders
        # For now, just return a successful cancellation
        return {
            "success": True,
            "order_id": order_id,
            "status": "CANCELLED",
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Order cancelled successfully (simulation mode)"
        }
    
    def modify_order(self, order_id: str, price: float = None, quantity: int = None,
                    stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Modifies an existing order.
        
        Args:
            order_id: ID of the order to modify
            price: New price for the order
            quantity: New quantity for the order
            stop_loss: New stop loss price
            take_profit: New take profit price
        
        Returns:
            dict: Modification response
        """
        # Check if we're in simulation mode
        if self.simulation_mode:
            return self._modify_order_simulation(order_id, price, quantity, stop_loss, take_profit)
            
        # If not in simulation mode, proceed with actual API call
        try:
            endpoint = f"{self.api_url}/order/{order_id}"
            
            payload = {}
            if price is not None:
                payload["price"] = price
            if quantity is not None:
                payload["quantity"] = quantity
            if stop_loss is not None:
                payload["stop_loss"] = stop_loss
            if take_profit is not None:
                payload["take_profit"] = take_profit
                
            response = self.session.put(endpoint, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Failed to modify order: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _modify_order_simulation(self, order_id: str, price: float = None, quantity: int = None,
                               stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Simulates modifying an order without actually sending it to the broker.
        Used for paper trading or testing.
        
        Args:
            order_id: ID of the order to modify
            price: New price for the order
            quantity: New quantity for the order
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            dict: Simulated modification response
        """
        logger.info(f"SIMULATION MODE: Modifying order {order_id}")
        
        # Check if it's a simulated order (should start with SIM-)
        if not order_id.startswith("SIM-"):
            logger.warning(f"Attempted to modify non-simulation order {order_id} in simulation mode")
        
        # Create modified order details
        modifications = {}
        if price is not None:
            modifications["price"] = price
        if quantity is not None:
            modifications["quantity"] = quantity
        if stop_loss is not None:
            modifications["stop_loss"] = stop_loss
        if take_profit is not None:
            modifications["take_profit"] = take_profit
        
        # In a real implementation, we would check against stored pending orders
        # For now, just return a successful modification
        return {
            "success": True,
            "order_id": order_id,
            "modifications": modifications,
            "status": "PENDING" if order_id.startswith("SIM-") else "UNKNOWN",
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Order modified successfully (simulation mode)"
        }
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Fetches current open positions.
        
        Returns:
            list: List of open positions
        """
        # Check if we're in simulation mode
        if self.simulation_mode:
            return self._get_open_positions_simulation()
            
        # If not in simulation mode, proceed with actual API
        try:
            # Check if we're using DLL or REST API
            if self.use_dll and self.dll_handle and self.dll_handle.is_connected():
                return self._get_open_positions_dll()
            elif self.socket:
                return self._get_open_positions_socket()
            elif self.session:
                return self._get_open_positions_rest()
            else:
                logger.error("No connection method available to fetch positions")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching open positions: {str(e)}")
            return []
            
    def _get_open_positions_rest(self) -> List[Dict[str, Any]]:
        """
        Fetches positions using REST API.
        """
        try:
            endpoint = f"{self.api_url}/positions"
            response = self.session.get(endpoint)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch open positions: {response.status_code} - {response.text}")
                return []
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching open positions via REST: {str(e)}")
            return []
            
    def _get_open_positions_socket(self) -> List[Dict[str, Any]]:
        """
        Fetches positions using socket connection.
        """
        try:
            request = {
                "method": "get_positions",
                "symbol": self.symbol  # Optional filter by symbol
            }
            
            self.socket.sendall(json.dumps(request).encode('utf-8'))
            
            # Read response
            buffer = b""
            while b"\n" not in buffer:
                data = self.socket.recv(4096)
                if not data:
                    break
                buffer += data
            
            response_data = json.loads(buffer.decode('utf-8').strip())
            
            if "error" in response_data:
                logger.error(f"Failed to fetch positions via socket: {response_data['error']}")
                return []
            
            if "data" in response_data:
                return response_data["data"]
            else:
                logger.warning("No position data received from socket")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching positions via socket: {str(e)}")
            return []
            
    def _get_open_positions_dll(self) -> List[Dict[str, Any]]:
        """
        Fetches positions using DLL interface.
        """
        try:
            logger.info(f"Fetching positions via DLL for {self.symbol}")
            
            if not self.dll_handle:
                logger.error("DLL handle not initialized")
                return []
                
            # Get available accounts
            accounts = self.dll_handle.get_accounts()
            if not accounts:
                logger.error("No trading accounts available")
                return []
                
            # Use the first account (in a real system, you might want to specify which account to use)
            account = accounts[0]
            account_id = account.get("account_id", "")
            broker_id = account.get("broker_id", 0)
            
            if not account_id:
                logger.error("Invalid account information")
                return []
                
            # Get position for the symbol
            position = self.dll_handle.get_position(
                account_id=account_id,
                broker_id=broker_id,
                ticker=self.symbol,
                exchange="B"  # B = B3 (Brazilian exchange)
            )
            
            if not position:
                logger.info(f"No open position found for {self.symbol}")
                return []
                
            # Format position data
            formatted_position = {
                "symbol": self.symbol,
                "side": "BUY" if position.get("open_quantity", 0) > 0 else "SELL",
                "quantity": abs(position.get("open_quantity", 0)),
                "entry_price": position.get("open_average_price", 0.0),
                "current_price": position.get("last_price", 0.0),
                "position_type": position.get("position_type", "DayTrade"),
                "pnl": position.get("unrealized_pnl", 0.0),
                "open_time": datetime.datetime.now().isoformat()  # This info may not be available from DLL
            }
            
            return [formatted_position] if formatted_position["quantity"] > 0 else []
            
        except Exception as e:
            logger.error(f"Error fetching positions via DLL: {str(e)}")
            return []
            
    def _get_open_positions_simulation(self) -> List[Dict[str, Any]]:
        """
        Simulates fetching open positions for paper trading or testing.
        
        Returns:
            list: Simulated list of open positions
        """
        logger.info("SIMULATION MODE: Fetching open positions")
        
        # In a full implementation, we would maintain a list of simulated positions
        # For now, return an empty list or sample positions for testing
        
        # Example of sample simulated position for testing
        price_data = self.get_price_data(limit=1)
        
        # If we have price data, generate a sample position
        if not price_data.empty:
            current_price = price_data['close'].iloc[-1]
            
            # For testing, create a single simulated position
            sample_position = {
                "position_id": f"SIM-POS-{int(time.time())}",
                "symbol": self.symbol,
                "side": "BUY" if np.random.random() > 0.5 else "SELL",
                "quantity": 1,
                "entry_price": current_price * 0.99 if np.random.random() > 0.5 else current_price * 1.01,
                "current_price": current_price,
                "pnl": np.random.uniform(-500, 500),
                "open_time": (datetime.datetime.now() - datetime.timedelta(hours=np.random.randint(1, 5))).isoformat(),
                "stop_loss": current_price * 0.95,
                "take_profit": current_price * 1.05
            }
            
            # Uncommenting this would return a sample position for testing
            # Only for testing UI, don't include in production
            # return [sample_position]
        
        # By default, return empty list of positions
        return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Fetches account information including balance.
        
        Returns:
            dict: Account information
        """
        # Check if we're in simulation mode
        if self.simulation_mode:
            return self._get_account_info_simulation()
            
        # If not in simulation mode, proceed with actual API
        try:
            # Check if we're using DLL or REST API
            if self.use_dll and self.dll_handle and self.dll_handle.is_connected():
                return self._get_account_info_dll()
            elif self.socket:
                return self._get_account_info_socket()
            elif self.session:
                return self._get_account_info_rest()
            else:
                logger.error("No connection method available to fetch account info")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching account info: {str(e)}")
            return {}
            
    def _get_account_info_rest(self) -> Dict[str, Any]:
        """
        Fetches account information using REST API.
        """
        try:
            endpoint = f"{self.api_url}/account"
            response = self.session.get(endpoint)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch account info: {response.status_code} - {response.text}")
                return {}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching account info via REST: {str(e)}")
            return {}
            
    def _get_account_info_socket(self) -> Dict[str, Any]:
        """
        Fetches account information using socket connection.
        """
        try:
            request = {
                "method": "get_account_info"
            }
            
            self.socket.sendall(json.dumps(request).encode('utf-8'))
            
            # Read response
            buffer = b""
            while b"\n" not in buffer:
                data = self.socket.recv(4096)
                if not data:
                    break
                buffer += data
            
            response_data = json.loads(buffer.decode('utf-8').strip())
            
            if "error" in response_data:
                logger.error(f"Failed to fetch account info via socket: {response_data['error']}")
                return {}
            
            if "data" in response_data:
                return response_data["data"]
            else:
                logger.warning("No account data received from socket")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching account info via socket: {str(e)}")
            return {}
            
    def _get_account_info_dll(self) -> Dict[str, Any]:
        """
        Fetches account information using DLL interface.
        """
        try:
            logger.info("Fetching account info via DLL")
            
            if not self.dll_handle:
                logger.error("DLL handle not initialized")
                return {}
                
            # Get available accounts
            accounts = self.dll_handle.get_accounts()
            if not accounts:
                logger.error("No accounts available via DLL")
                return {}
                
            # Use the first account information
            account = accounts[0]
            
            # Format the account information to match expected structure
            formatted_account = {
                "account_id": account.get("account_id", ""),
                "broker_id": account.get("broker_id", 0),
                "name": account.get("holder_name", ""),
                "broker_name": account.get("broker_name", ""),
                "currency": "BRL",  # Typically BRL for B3 (Brazilian exchange)
                "balance": 0.0,  # Not directly available, would need additional API calls
                "equity": 0.0,    # Not directly available, would need additional API calls
                "margin_used": 0.0,
                "free_margin": 0.0,
                "simulation": False,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return formatted_account
            
        except Exception as e:
            logger.error(f"Error fetching account info via DLL: {str(e)}")
            return {}
            
    def _get_account_info_simulation(self) -> Dict[str, Any]:
        """
        Simulates fetching account information for paper trading or testing.
        
        Returns:
            dict: Simulated account information
        """
        logger.info("SIMULATION MODE: Fetching account information")
        
        # Create a simulated account information
        # In a full implementation, would track changes as orders are placed
        return {
            "account_id": "SIM-ACCOUNT",
            "name": "Conta Simulada",
            "currency": "BRL",
            "balance": 100000.00,  # R$100.000,00 de saldo inicial
            "equity": 100000.00 + np.random.uniform(-2000, 2000),  # Patrimônio atual com variação simulada
            "margin_used": np.random.uniform(0, 5000),
            "free_margin": 95000.00 + np.random.uniform(-2000, 2000),
            "margin_level": 95.0 + np.random.uniform(-5, 5),
            "simulation": True,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Fetches status of a specific order.
        
        Args:
            order_id: ID of the order to get status for
            
        Returns:
            dict: Order status information
        """
        # Check if we're in simulation mode
        if self.simulation_mode:
            return self._get_order_status_simulation(order_id)
            
        # If not in simulation mode, proceed with actual API call
        try:
            endpoint = f"{self.api_url}/order/{order_id}"
            response = self.session.get(endpoint)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch order status: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching order status: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _get_order_status_simulation(self, order_id: str) -> Dict[str, Any]:
        """
        Simulates fetching order status for paper trading or testing.
        
        Args:
            order_id: ID of the order to get status for
            
        Returns:
            dict: Simulated order status
        """
        logger.info(f"SIMULATION MODE: Fetching status for order {order_id}")
        
        # Check if it's a simulated order
        if not order_id.startswith("SIM-"):
            logger.warning(f"Attempted to get status for non-simulation order {order_id} in simulation mode")
            return {
                "success": False, 
                "error": "Order not found in simulation mode"
            }
            
        # Create a simulated order status
        # In a real implementation, would track actual simulated orders
        statuses = ["PENDING", "FILLED", "CANCELLED", "REJECTED"]
        status = np.random.choice(statuses, p=[0.2, 0.6, 0.15, 0.05])  # Weighted probabilities
        
        # Get price for simulation
        price_data = self.get_price_data(limit=1)
        current_price = price_data['close'].iloc[-1] if not price_data.empty else 100000.0
        
        return {
            "success": True,
            "order_id": order_id,
            "status": status,
            "symbol": self.symbol,
            "side": "BUY" if "BUY" in order_id else "SELL",
            "quantity": 1,
            "fill_price": current_price if status == "FILLED" else None,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_order_history(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Fetches order history for the account.
        
        Args:
            start_date: Start date for order history (YYYY-MM-DD)
            end_date: End date for order history (YYYY-MM-DD)
        
        Returns:
            list: List of historical orders
        """
        # Check if we're in simulation mode
        if self.simulation_mode:
            return self._get_order_history_simulation(start_date, end_date)
            
        # If not in simulation mode, proceed with actual API call
        try:
            endpoint = f"{self.api_url}/order_history"
            
            params = {}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
                
            response = self.session.get(endpoint, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch order history: {response.status_code} - {response.text}")
                return []
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching order history: {str(e)}")
            return []
            
    def _get_order_history_simulation(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Simulates fetching order history for paper trading or testing.
        
        Args:
            start_date: Start date for order history (YYYY-MM-DD) 
            end_date: End date for order history (YYYY-MM-DD)
            
        Returns:
            list: Simulated list of historical orders
        """
        logger.info(f"SIMULATION MODE: Fetching order history from {start_date or 'beginning'} to {end_date or 'now'}")
        
        # Get price data for simulation
        price_data = self.get_price_data(limit=20)  # Get some data points for simulation
        
        if price_data.empty:
            logger.warning("No price data available for order history simulation")
            return []
            
        # Generate simulated order history
        simulated_orders = []
        
        # Convert start_date and end_date to datetime objects
        try:
            start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.datetime.now() - datetime.timedelta(days=30)
            end_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.datetime.now()
        except Exception as e:
            logger.error(f"Error parsing dates: {str(e)}")
            start_datetime = datetime.datetime.now() - datetime.timedelta(days=30)
            end_datetime = datetime.datetime.now()
            
        # Number of orders to simulate
        num_orders = np.random.randint(5, 15)
        
        # Get available prices
        prices = price_data['close'].values
        
        # Generate orders with timestamps between start and end dates
        for i in range(num_orders):
            # Generate a random timestamp between start and end
            time_range = (end_datetime - start_datetime).total_seconds()
            random_seconds = np.random.randint(0, int(time_range))
            timestamp = start_datetime + datetime.timedelta(seconds=random_seconds)
            
            # Get a random price 
            price = np.random.choice(prices)
            
            # Create order details
            order = {
                "order_id": f"SIM-HIST-{i}-{int(time.time())}",
                "symbol": self.symbol,
                "side": "BUY" if np.random.random() > 0.5 else "SELL",
                "order_type": "MARKET" if np.random.random() > 0.3 else "LIMIT",
                "quantity": np.random.randint(1, 5),
                "price": price,
                "status": np.random.choice(["FILLED", "CANCELLED", "REJECTED"], p=[0.8, 0.15, 0.05]),
                "timestamp": timestamp.isoformat(),
                "fill_price": price * (1 + np.random.uniform(-0.01, 0.01)) if np.random.random() > 0.2 else None,
                "pnl": np.random.uniform(-500, 800) if np.random.random() > 0.3 else None
            }
            
            simulated_orders.append(order)
            
        # Sort orders by timestamp
        simulated_orders.sort(key=lambda x: x["timestamp"])
        
        return simulated_orders
