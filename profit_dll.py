from ctypes import *
import sys
import platform
from typing import Any, Optional

# Importar WinDLL apenas se estivermos no Windows
if platform.system() == 'Windows':
    from ctypes.wintypes import *
    from ctypes import WinDLL
    WINFUNCTYPE = CFUNCTYPE
else:
    # Em outros sistemas operacionais, criar um substituto para WinDLL
    # que não faz nada, apenas para permitir que o código importe
    class WinDLL:
        def __init__(self, *args, **kwargs):
            pass
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    WINFUNCTYPE = CFUNCTYPE

from profitTypes import *

def initializeDll(path: str) -> Any:
    profit_dll = WinDLL(path)
    profit_dll.argtypes  = None

    profit_dll.SendSellOrder.restype = c_longlong
    profit_dll.SendBuyOrder.restype = c_longlong
    profit_dll.SendZeroPosition.restype = c_longlong
    profit_dll.GetAgentNameByID.restype = c_wchar_p
    profit_dll.GetAgentShortNameByID.restype = c_wchar_p
    profit_dll.GetPosition.restype = POINTER(c_int)
    profit_dll.SendMarketSellOrder.restype = c_int64
    profit_dll.SendMarketBuyOrder.restype = c_int64

    profit_dll.SendStopSellOrder.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_double, c_double, c_int]
    profit_dll.SendStopSellOrder.restype = c_longlong

    profit_dll.SendStopBuyOrder.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_double, c_double, c_int]
    profit_dll.SendStopBuyOrder.restype = c_longlong

    profit_dll.SendOrder.argtypes = [POINTER(TConnectorSendOrder)]
    profit_dll.SendOrder.restype = c_int64

    profit_dll.SendChangeOrderV2.argtypes = [POINTER(TConnectorChangeOrder)]
    profit_dll.SendChangeOrderV2.restype = c_int

    profit_dll.SendCancelOrderV2.argtypes = [POINTER(TConnectorCancelOrder)]
    profit_dll.SendCancelOrderV2.restype = c_int

    profit_dll.SendCancelOrdersV2.argtypes = [POINTER(TConnectorCancelOrders)]
    profit_dll.SendCancelOrdersV2.restype = c_int

    profit_dll.SendCancelAllOrdersV2.argtypes = [POINTER(TConnectorCancelAllOrders)]
    profit_dll.SendCancelAllOrdersV2.restype = c_int

    profit_dll.SendZeroPositionV2.argtypes = [POINTER(TConnectorZeroPosition)]
    profit_dll.SendZeroPositionV2.restype = c_int64

    profit_dll.GetAccountCount.argtypes = []
    profit_dll.GetAccountCount.restype = c_int

    profit_dll.GetAccounts.argtypes = [c_int, c_int, c_int, POINTER(TConnectorAccountIdentifierOut)]
    profit_dll.GetAccounts.restype = c_int

    profit_dll.GetAccountDetails.argtypes = [POINTER(TConnectorTradingAccountOut)]
    profit_dll.GetAccountDetails.restype = c_int

    profit_dll.GetSubAccountCount.argtypes = [POINTER(TConnectorAccountIdentifier)]
    profit_dll.GetSubAccountCount.restype = c_int

    profit_dll.GetSubAccounts.argtypes = [POINTER(TConnectorAccountIdentifier), c_int, c_int, c_int, POINTER(TConnectorAccountIdentifierOut)]
    profit_dll.GetSubAccounts.restype = c_int

    profit_dll.GetPositionV2.argtypes = [POINTER(TConnectorTradingAccountPosition)]
    profit_dll.GetPositionV2.restype = c_int

    profit_dll.GetOrderDetails.argtypes = [POINTER(TConnectorOrderOut)]
    profit_dll.GetOrderDetails.restype = c_int
    
    # Inicialização da DLL
    profit_dll.DLLInitializeLogin.argtypes = [c_wchar_p]
    profit_dll.DLLInitializeLogin.restype = c_int
    
    # Registro de callbacks
    STATECALLBACK = WINFUNCTYPE(None, c_int32, c_int32)
    profit_dll.SetStateCallbackFunc.argtypes = [STATECALLBACK]
    profit_dll.SetStateCallbackFunc.restype = c_int
    
    NEWTRADECALLBACK = WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double, c_int, c_int, c_int, c_int, c_wchar)
    profit_dll.SetNewTradeCallbackFunc.argtypes = [NEWTRADECALLBACK]
    profit_dll.SetNewTradeCallbackFunc.restype = c_int
    
    ACCOUNTCALLBACK = WINFUNCTYPE(None, c_int, c_wchar_p, c_wchar_p, c_wchar_p)
    profit_dll.SetAccountCallbackFunc.argtypes = [ACCOUNTCALLBACK]
    profit_dll.SetAccountCallbackFunc.restype = c_int

    return profit_dll