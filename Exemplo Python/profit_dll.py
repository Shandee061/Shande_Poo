from ctypes import *
from profitTypes import *

def initializeDll(path: str) -> WinDLL:
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

    return profit_dll