#Imports para execução da DLL
import ctypes
from getpass import getpass
from ctypes import *
import struct
from datetime import *

from profitTypes import *
from profit_dll import initializeDll

profit_dll = initializeDll("./ProfitDLL.dll")

# Error Codes
NL_OK                    = 0x00000000
NL_INTERNAL_ERROR        = -2147483647                   # Internal error
NL_NOT_INITIALIZED       = NL_INTERNAL_ERROR        + 1  # Not initialized
NL_INVALID_ARGS          = NL_NOT_INITIALIZED       + 1  # Invalid arguments
NL_WAITING_SERVER        = NL_INVALID_ARGS          + 1  # Aguardando dados do servidor
NL_NO_LOGIN              = NL_WAITING_SERVER        + 1  # Nenhum login encontrado
NL_NO_LICENSE            = NL_NO_LOGIN              + 1  # Nenhuma licença encontrada
NL_PASSWORD_HASH_SHA1    = NL_NO_LICENSE            + 1  # Senha não está em SHA1
NL_PASSWORD_HASH_MD5     = NL_PASSWORD_HASH_SHA1    + 1  # Senha não está em MD5
NL_OUT_OF_RANGE          = NL_PASSWORD_HASH_MD5     + 1  # Count do parâmetro maior que o tamanho do array
NL_MARKET_ONLY           = NL_OUT_OF_RANGE          + 1  # Não possui roteamento
NL_NO_POSITION           = NL_MARKET_ONLY           + 1  # Não possui posição
NL_NOT_FOUND             = NL_NO_POSITION           + 1  # Recurso não encontrado
NL_VERSION_NOT_SUPPORTED = NL_NOT_FOUND             + 1  # Versão do recurso não suportada
NL_OCO_NO_RULES          = NL_VERSION_NOT_SUPPORTED + 1  # OCO sem nenhuma regra
NL_EXCHANGE_UNKNOWN      = NL_OCO_NO_RULES          + 1  # Bolsa desconhecida
NL_NO_OCO_DEFINED        = NL_EXCHANGE_UNKNOWN      + 1  # Nenhuma OCO encontrada para a ordem
NL_INVALID_SERIE         = NL_NO_OCO_DEFINED        + 1  # (Level + Offset + Factor) inválido
NL_LICENSE_NOT_ALLOWED   = NL_INVALID_SERIE         + 1  # Recurso não liberado na licença
NL_NOT_HARD_LOGOUT       = NL_LICENSE_NOT_ALLOWED   + 1  # Retorna que não esta em HardLogout
NL_SERIE_NO_HISTORY      = NL_NOT_HARD_LOGOUT       + 1  # Série não tem histórico no servidor
NL_ASSET_NO_DATA         = NL_SERIE_NO_HISTORY      + 1  # Asset não tem o dados carregado
NL_SERIE_NO_DATA         = NL_ASSET_NO_DATA         + 1  # Série não tem dados (count = 0)
NL_HAS_STRATEGY_RUNNING  = NL_SERIE_NO_DATA         + 1  # Existe uma estratégia rodando
NL_SERIE_NO_MORE_HISTORY = NL_HAS_STRATEGY_RUNNING  + 1  # Não tem mais dados disponiveis para a serie
NL_SERIE_MAX_COUNT       = NL_SERIE_NO_MORE_HISTORY + 1  # Série esta no limite de dados possíveis
NL_DUPLICATE_RESOURCE    = NL_SERIE_MAX_COUNT       + 1  # Recurso duplicado
NL_UNSIGNED_CONTRACT     = NL_DUPLICATE_RESOURCE    + 1
NL_NO_PASSWORD           = NL_UNSIGNED_CONTRACT     + 1  # Nenhuma senha informada
NL_NO_USER               = NL_NO_PASSWORD           + 1  # Nenhum usuário informado no login
NL_FILE_ALREADY_EXISTS   = NL_NO_USER               + 1  # Arquivo já existe
NL_INVALID_TICKER        = NL_FILE_ALREADY_EXISTS   + 1
NL_NOT_MASTER_ACCOUNT    = NL_INVALID_TICKER        + 1  # Conta não é master

#Variaveis de Controle
bAtivo = False
bMarketConnected = False
bConnectado = False
bBrokerConnected = False

def NResultToString(nResult: int) -> str:
    if nResult == NL_INTERNAL_ERROR:
        return "NL_INTERNAL_ERROR"
    elif nResult == NL_NOT_INITIALIZED:
        return "NL_NOT_INITIALIZED"
    elif nResult == NL_INVALID_ARGS:
        return "NL_INVALID_ARGS"
    elif nResult == NL_WAITING_SERVER:
        return "NL_WAITING_SERVER"
    elif nResult == NL_NO_LOGIN:
        return "NL_NO_LOGIN"
    elif nResult == NL_NO_LICENSE:
        return "NL_NO_LICENSE"
    elif nResult == NL_PASSWORD_HASH_SHA1:
        return "NL_PASSWORD_HASH_SHA1"
    elif nResult == NL_PASSWORD_HASH_MD5:
        return "NL_PASSWORD_HASH_MD5"
    elif nResult == NL_OUT_OF_RANGE:
        return "NL_OUT_OF_RANGE"
    elif nResult == NL_MARKET_ONLY:
        return "NL_MARKET_ONLY"
    elif nResult == NL_NO_POSITION:
        return "NL_NO_POSITION"
    elif nResult == NL_NOT_FOUND:
        return "NL_NOT_FOUND"
    elif nResult == NL_VERSION_NOT_SUPPORTED:
        return "NL_VERSION_NOT_SUPPORTED"
    elif nResult == NL_OCO_NO_RULES:
        return "NL_OCO_NO_RULES"
    elif nResult == NL_EXCHANGE_UNKNOWN:
        return "NL_EXCHANGE_UNKNOWN"
    elif nResult == NL_NO_OCO_DEFINED:
        return "NL_NO_OCO_DEFINED"
    elif nResult == NL_INVALID_SERIE:
        return "NL_INVALID_SERIE"
    elif nResult == NL_LICENSE_NOT_ALLOWED:
        return "NL_LICENSE_NOT_ALLOWED"
    elif nResult == NL_NOT_HARD_LOGOUT:
        return "NL_NOT_HARD_LOGOUT"
    elif nResult == NL_SERIE_NO_HISTORY:
        return "NL_SERIE_NO_HISTORY"
    elif nResult == NL_ASSET_NO_DATA:
        return "NL_ASSET_NO_DATA"
    elif nResult == NL_SERIE_NO_DATA:
        return "NL_SERIE_NO_DATA"
    elif nResult == NL_HAS_STRATEGY_RUNNING:
        return "NL_HAS_STRATEGY_RUNNING"
    elif nResult == NL_SERIE_NO_MORE_HISTORY:
        return "NL_SERIE_NO_MORE_HISTORY"
    elif nResult == NL_SERIE_MAX_COUNT:
        return "NL_SERIE_MAX_COUNT"
    elif nResult == NL_DUPLICATE_RESOURCE:
        return "NL_DUPLICATE_RESOURCE"
    elif nResult == NL_UNSIGNED_CONTRACT:
        return "NL_UNSIGNED_CONTRACT"
    elif nResult == NL_NO_PASSWORD:
        return "NL_NO_PASSWORD"
    elif nResult == NL_NO_USER:
        return "NL_NO_USER"
    elif nResult == NL_FILE_ALREADY_EXISTS:
        return "NL_FILE_ALREADY_EXISTS"
    elif nResult == NL_INVALID_TICKER:
        return "NL_INVALID_TICKER"
    elif nResult == NL_NOT_MASTER_ACCOUNT:
        return "NL_NOT_MASTER_ACCOUNT"
    else:
        return str(nResult)

def evalDllReturn(function: str, ret: int) -> bool:
    if ret < NL_OK:
        print("{0}: {1}".format(
            function,
            NResultToString(ret)
        ))

        return False
    else:
        return True

def printOrder(title:str, orderId: TConnectorOrderIdentifier):
    order = TConnectorOrderOut(
        Version=0,
        OrderID=orderId
    )

    ret = profit_dll.GetOrderDetails(byref(order))
    if not evalDllReturn("GetOrderDetails-1", ret):
        return

    order.AssetID.Ticker   = ' ' * order.AssetID.TickerLength
    order.AssetID.Exchange = ' ' * order.AssetID.ExchangeLength
    order.TextMessage      = ' ' * order.TextMessageLength

    ret = profit_dll.GetOrderDetails(byref(order))
    if not evalDllReturn("GetOrderDetails-2", ret):
        return

    print('{0}: {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9}'.format(title,
        order.AssetID.Ticker,
        order.TradedQuantity,
        order.OrderSide,
        order.Price,
        order.AccountID.AccountID,
        order.AccountID.SubAccountID,
        order.OrderID.ClOrderID,
        order.OrderStatus,
        order.TextMessage
    ))

#BEGIN DEF
@WINFUNCTYPE(None, c_int32, c_int32)
def stateCallback(nType, nResult):
    global bAtivo
    global bMarketConnected
    global bConnectado

    nConnStateType = nType
    result = nResult

    if nConnStateType == 0: # notificacoes de login
        if result == 0:
            bConnectado = True
            print("Login: conectado")
        else :
            bConnectado = False
            print('Login: ' + str(result))
    elif nConnStateType == 1:
        if result == 5:
            # bBrokerConnected = True
            print("Broker: Conectado.")
        elif result > 2:
            # bBrokerConnected = False
            print("Broker: Sem conexão com corretora.")
        else:
            # bBrokerConnected = False
            print("Broker: Sem conexão com servidores (" + str(result) + ")")

    elif nConnStateType == 2:  # notificacoes de login no Market
        if result == 4:
            print("Market: Conectado" )
            bMarketConnected = True
        else:
            print("Market: " + str(result))
            bMarketConnected = False

    elif nConnStateType == 3: # notificacoes de login
        if result == 0:
            print("Ativação: OK")
            bAtivo = True
        else:
            print("Ativação: " + str(result))
            bAtivo = False

    if bMarketConnected and bAtivo and bConnectado:
        print("Serviços Conectados")

    return

@WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double, c_int, c_int, c_int, c_int)
def newHistoryCallback(assetId, date, tradeNumber, price, vol, qtd, buyAgent, sellAgent, tradeType):
    print(assetId.ticker + ' | Trade History | ' + date + ' (' + str(tradeNumber) + ') ' + str(price))
    return

@WINFUNCTYPE(None, TAssetID, c_int)
def progressCallBack(assetId, nProgress):
    print(assetId.ticker + ' | Progress | ' + str(nProgress))
    return

@WINFUNCTYPE(None, c_int, c_wchar_p, c_wchar_p, c_wchar_p)
def accountCallback(nCorretora, corretoraNomeCompleto, accountID, nomeTitular):
    print("Conta | " + accountID + ' - ' + nomeTitular + ' | Corretora ' + str(nCorretora) + ' - ' + corretoraNomeCompleto)
    return

@WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_int, c_int, c_double, POINTER(c_int), POINTER(c_int))
def priceBookCallback(assetId, nAction, nPosition, Side, nQtd, nCount, sPrice, pArraySell, pArrayBuy):
    if pArraySell is not None:
        print("todo - priceBookCallBack")
    return


@WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double, c_double, c_int, c_int, c_int, c_int, c_wchar)
def newTradeCallback(assetId, date, tradeNumber, price, vol, qtd, buyAgent, sellAgent, tradeType, bIsEdit):
    print(assetId.ticker + ' | Trade | ' + str(date) + '(' + str(tradeNumber) + ') ' + str(price))
    return


@WINFUNCTYPE(None, TAssetID, c_double, c_int, c_int)
def newTinyBookCallBack(assetId, price, qtd, side):
    if side == 0 :
        print(assetId.ticker + ' | TinyBook | Buy: ' + str(price) + ' ' + str(qtd))
    else :
        print(assetId.ticker + ' | TinyBook | Sell: ' + str(price) + ' ' + str(qtd))

    return


@WINFUNCTYPE(None, TAssetID, c_wchar_p, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double,
           c_double, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
def newDailyCallback(assetID, date, sOpen, sHigh, sLow, sClose, sVol, sAjuste, sMaxLimit, sMinLimit, sVolBuyer,
                     sVolSeller, nQtd, nNegocios, nContratosOpen, nQtdBuyer, nQtdSeller, nNegBuyer, nNegSeller):
    print(assetID.ticker + ' | DailySignal | ' + date + ' Open: ' + str(sOpen) + ' High: ' + str(sHigh) + ' Low: ' + str(sLow) + ' Close: ' + str(sClose))
    return

price_array_sell = []
price_array_buy = []

def descript_offer_array_v2(price_array):
    price_array_descripted = []
    n_qtd = price_array[0]
    n_tam = price_array[1]
    print(f"qtd: {n_qtd}, n_tam: {n_tam}")

    arr = cast(price_array, POINTER(c_char))
    frame = bytearray()
    for i in range(n_tam):
        c = arr[i]
        frame.append(c[0])

    start = 8
    for i in range(n_qtd):
        price = struct.unpack('d', frame[start:start + 8])[0]
        start += 8
        qtd = struct.unpack('q', frame[start:start+8])[0]
        start += 8
        agent = struct.unpack('i', frame[start:start+4])[0]
        start += 4
        offer_id = struct.unpack('q', frame[start:start+8])[0]
        start += 8
        date_length = struct.unpack('h', frame[start:start+2])[0]
        start += 2
        date = frame[start:start+date_length]
        start += date_length

        price_array_descripted.append([price, qtd, agent, offer_id, date])

    return price_array_descripted

@WINFUNCTYPE(None, TAssetID, c_int, c_int, c_int, c_int, c_int, c_longlong, c_double, c_int, c_int, c_int, c_int, c_int,
           c_wchar_p, POINTER(c_int), POINTER(c_int))
def offerBookCallbackV2(assetId, nAction, nPosition, Side, nQtd, nAgent, nOfferID, sPrice, bHasPrice,
                      bHasQtd, bHasDate, bHasOfferID, bHasAgent, date, pArraySell, pArrayBuy):
    global price_array_buy
    global price_array_sell

    if bool(pArraySell):
        price_array_sell = descript_offer_array_v2(pArraySell)

    if bool(pArrayBuy):
        price_array_buy = descript_offer_array_v2(pArrayBuy)

    if Side == 0:
        lst_book = price_array_buy
    else:
        lst_book = price_array_sell

    if lst_book and 0 <= nPosition <= len(lst_book):
        """
        atAdd = 0
        atEdit = 1
        atDelete = 2
        atDeleteFrom = 3
        atFullBook = 4
        """
        if nAction == 0:
            group = [sPrice, nQtd, nAgent]
            idx = len(lst_book)-nPosition
            lst_book.insert(idx, group)
        elif nAction == 1:
            group = lst_book[-nPosition - 1]
            group[1] = group[1] + nQtd
            group[2] = group[2] + nAgent
        elif nAction == 2:
            del lst_book[-nPosition - 1]
        elif nAction == 3:
            del lst_book[-nPosition - 1:]
    return


@WINFUNCTYPE(None, TAssetID, c_wchar_p, c_uint, c_double)
def changeCotationCallback(assetId, date, tradeNumber, sPrice):
    print("todo - changeCotationCallback")
    return

@WINFUNCTYPE(None, TAssetID, c_wchar_p)
def assetListCallback(assetId, strName):
    print ("assetListCallback Ticker=" + str(assetId.ticker) + " Name=" + str(strName))
    return

@WINFUNCTYPE(None, TAssetID, c_double, c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_uint, c_double)
def adjustHistoryCallbackV2(assetId, value, strType, strObserv, dtAjuste, dtDelib, dtPagamento, nFlags, dMult):
    print("todo - adjustHistoryCallbackV2")
    return

@WINFUNCTYPE(None, TAssetID, c_wchar_p, c_wchar_p, c_int, c_int, c_int, c_int, c_int, c_double, c_double, c_wchar_p, c_wchar_p)
def assetListInfoCallback(assetId, strName, strDescription, iMinOrdQtd, iMaxOrdQtd, iLote, iSecurityType, iSecuritySubType, dMinPriceInc, dContractMult, strValidDate, strISIN):
    print('TAssetListInfoCallback = Ticker: ' + str(assetId.ticker) +
          'Name: ' + str(strName) +
          'Descrição: ' + str(strDescription))
    return

@WINFUNCTYPE(None, TAssetID, c_wchar_p, c_wchar_p, c_int, c_int, c_int, c_int, c_int, c_double, c_double, c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p)
def assetListInfoCallbackV2(assetId, strName, strDescription, iMinOrdQtd, iMaxOrdQtd, iLote, iSecurityType, iSecuritySubType, dMinPriceInc, dContractMult, strValidDate, strISIN, strSetor, strSubSetor, strSegmento):
    print('TAssetListInfoCallbackV2 = Ticker: ' + str(assetId.ticker) +
          'Name: ' + str(strName) +
          'Descrição: ' + str(strDescription) +
          'Setor: ' + str(strSetor))
    return

@WINFUNCTYPE(None, TConnectorOrderIdentifier)
def orderCallback(orderId : TConnectorOrderIdentifier):
    printOrder("OrderCallback", orderId)

@WINFUNCTYPE(None, TConnectorAssetIdentifier)
def invalidAssetCallback(assetID : TConnectorAssetIdentifier):
    print("invalidAssetCallback: " + assetID.Ticker)

#END DEF

#EXEMPLOS
def SenSellOrder() :
    qtd = int(1)
    preco = float(100000)
    # precoStop = float(100000)
    nProfitID = profit_dll.SendSellOrder (c_wchar_p('CONTA'), c_wchar_p('BROKER'),
                                          c_wchar_p('PASS'),c_wchar_p('ATIVO'),
                                          c_wchar_p('BOLSA'),
                                          c_double(preco), c_int(qtd))

    print(str(nProfitID))

def wait_login():
    global bMarketConnected
    global bAtivo

    bWaiting = True
    while bWaiting:
        if bMarketConnected  :

            print("DLL Conected")

            bWaiting = False
    print('stop waiting')

def subscribeOffer():
    print("subscribe offer book")

    asset = input('Asset: ')
    bolsa = input('Bolsa: ')

    result = profit_dll.SubscribeOfferBook(c_wchar_p(asset), c_wchar_p(bolsa))
    print ("SubscribeOfferBook: " + str(result))

def subscribeTicker():
    asset = input('Asset: ')
    bolsa = input('Bolsa: ')

    result = profit_dll.SubscribeTicker(c_wchar_p(asset), c_wchar_p(bolsa))
    print ("SubscribeTicker: " + str(result))

def unsubscribeTicker():
    asset = input('Asset: ')
    bolsa = input('Bolsa: ')

    result = profit_dll.UnsubscribeTicker(c_wchar_p(asset), c_wchar_p(bolsa))
    print ("UnsubscribeTicker: " + str(result))

def printLastAdjusted():
    close = c_double()
    result = profit_dll.GetLastDailyClose(c_wchar_p("MGLU3"), c_wchar_p("B"), byref(close), 1)
    print(f'Last session close: {close}, result={str(result)}')

def printPosition():
    ticker = input('Asset: ')
    exchange = input('Bolsa: ')
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    positionType = int(input("Tipo da Posisão (1 - DayTrade, 2 - Consolidado): "))

    position = TConnectorTradingAccountPosition(
        Version=1,
        PositionType = positionType
    )
    position.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId
    )
    position.AssetID = TConnectorAssetIdentifier(
        Version=0,
        Ticker=ticker,
        Exchange=exchange,
        FeedType=0
    )

    ret = profit_dll.GetPositionV2(byref(position))

    if not evalDllReturn("GetPositionV2", ret):
        return

    print("Price: {0} | AvgSellPrice: {0} | AvgBuyPrice: {0} | SellQtd: {0} | BuyQtd: {0}".format(
        position.OpenAveragePrice,
        position.DailyAverageSellPrice,
        position.DailyAverageBuyPrice,
        position.DailySellQuantity,
        position.DailyBuyQuantity
    ))

def doZeroPosition():
    ticker = input('Asset: ')
    exchange = input('Bolsa: ')
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    rotPassword = getpass("Senha de Roteamento: ")

    positionType = int(input("Tipo da Posisão (1 - DayTrade, 2 - Consolidado): "))

    zeroRec = TConnectorZeroPosition(
        Version=1,
        PositionType = positionType,
        Password = rotPassword,
        Price = -1.0
    )
    zeroRec.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId
    )
    zeroRec.AssetID = TConnectorAssetIdentifier(
        Version=0,
        Ticker=ticker,
        Exchange=exchange,
        FeedType=0
    )

    ret = profit_dll.SendZeroPositionV2(byref(zeroRec))

    if not evalDllReturn("SendZeroPositionV2", ret):
        return
    
    print("ZeroOrderID: {0}".format(ret))

def dllStart():
    try:
        key = input("Chave de acesso: ")
        user = input("Usuário: ") # preencher com usuário da conta (email ou documento)
        password = getpass("Senha: ") # preencher com senha da conta

        bRoteamento = True

        if bRoteamento :
            result = profit_dll.DLLInitializeLogin(c_wchar_p(key), c_wchar_p(user), c_wchar_p(password), stateCallback, None, None, accountCallback,
                                              newTradeCallback, newDailyCallback, priceBookCallback,
                                              None, newHistoryCallback, progressCallBack, newTinyBookCallBack)
        else :
            result = profit_dll.DLLInitializeMarketLogin(c_wchar_p(key), c_wchar_p(user), c_wchar_p(password), stateCallback, newTradeCallback, newDailyCallback, priceBookCallback,
                                                 None, newHistoryCallback, progressCallBack, newTinyBookCallBack)

        profit_dll.SetAssetListCallback(assetListCallback)
        profit_dll.SetAdjustHistoryCallbackV2(adjustHistoryCallbackV2)
        profit_dll.SetAssetListInfoCallback(assetListInfoCallback)
        profit_dll.SetAssetListInfoCallbackV2(assetListInfoCallbackV2)
        profit_dll.SetOfferBookCallbackV2(offerBookCallbackV2)
        profit_dll.SetOrderCallback(orderCallback)
        profit_dll.SetInvalidTickerCallback(invalidAssetCallback)

        print('DLLInitialize: ' + str(result))
        wait_login()

    except Exception as e:
        print(str(e))

def dllEnd():
    result = profit_dll.DLLFinalize()

    print('DLLFinalize: ' + str(result))

# Funções de roteamento

gOrders = {}

def buyStopOrder():
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    rotPassword = getpass("Senha de Roteamento: ")

    ticker = input('Ativo: ')
    exchange = input('Bolsa: ')
    price = float(input('Preço: '))
    stopPrice = float(input('Preço Stop: '))
    amount = int(input('Quantidade: '))

    send_order = TConnectorSendOrder(
        Version = 0,
        Password = rotPassword,
        OrderType = TConnectorOrderType.Stop.value,
        OrderSide = TConnectorOrderSide.Buy.value,
        Price = price,
        StopPrice = stopPrice,
        Quantity = amount
    )
    send_order.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId,
        Reserved=0
    )
    send_order.AssetID = TConnectorAssetIdentifier(
        Version=0,
        Ticker=ticker,
        Exchange=exchange,
        FeedType=0
    )

    profitID = profit_dll.SendOrder(byref(send_order))

    if evalDllReturn("SendOrder", profitID):
        print("ProfitID: " + str(profitID))

def sellStopOrder():
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    rotPassword = getpass("Senha de Roteamento: ")

    ticker = input('Ativo: ')
    exchange = input('Bolsa: ')
    price = float(input('Preço: '))
    stopPrice = float(input('Preço Stop: '))
    amount = int(input('Quantidade: '))

    send_order = TConnectorSendOrder(
        Version = 0,
        Password = rotPassword,
        OrderType = TConnectorOrderType.Stop.value,
        OrderSide = TConnectorOrderSide.Sell.value,
        Price = price,
        StopPrice = stopPrice,
        Quantity = amount
    )
    send_order.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId,
        Reserved=0
    )
    send_order.AssetID = TConnectorAssetIdentifier(
        Version=0,
        Ticker=ticker,
        Exchange=exchange,
        FeedType=0
    )

    profitID = profit_dll.SendOrder(byref(send_order))

    if evalDllReturn("SendOrder", profitID):
        print("ProfitID: " + str(profitID))

def sendBuyMarketOrder():
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    rotPassword = getpass("Senha de Roteamento: ")

    ticker = input('Ativo: ')
    exchange = input('Bolsa: ')
    amount = int(input('Quantidade: '))

    send_order = TConnectorSendOrder(
        Version = 0,
        Password = rotPassword,
        OrderType = TConnectorOrderType.Market.value,
        OrderSide = TConnectorOrderSide.Buy.value,
        Price = -1,
        StopPrice = -1,
        Quantity = amount
    )
    send_order.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId,
        Reserved=0
    )
    send_order.AssetID = TConnectorAssetIdentifier(
        Version=0,
        Ticker=ticker,
        Exchange=exchange,
        FeedType=0
    )

    profitID = profit_dll.SendOrder(byref(send_order))

    if evalDllReturn("SendOrder", profitID):
        print("ProfitID: " + str(profitID))

def sendSellMarketOrder():
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    rotPassword = getpass("Senha de Roteamento: ")

    ticker = input('Ativo: ')
    exchange = input('Bolsa: ')
    amount = int(input('Quantidade: '))

    send_order = TConnectorSendOrder(
        Version = 0,
        Password = rotPassword,
        OrderType = TConnectorOrderType.Market.value,
        OrderSide = TConnectorOrderSide.Sell.value,
        Price = -1,
        StopPrice = -1,
        Quantity = amount
    )
    send_order.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId,
        Reserved=0
    )
    send_order.AssetID = TConnectorAssetIdentifier(
        Version=0,
        Ticker=ticker,
        Exchange=exchange,
        FeedType=0
    )

    profitID = profit_dll.SendOrder(byref(send_order))

    if evalDllReturn("SendOrder", profitID):
        print("ProfitID: " + str(profitID))

def getOrders():
    brokerId = input("Corretora: ")
    accountId = input("Conta: ")

    now = datetime.now()
    tomorrow = datetime.now() + timedelta(days=1)

    # retorno em historyCallback
    profit_dll.GetOrders(
        c_wchar_p(accountId),
        c_wchar_p(brokerId),
        c_wchar_p(now.strftime("%d/%m/%Y")),
        c_wchar_p(tomorrow.strftime("%d/%m/%Y")))

def getOrder():
    cl_ord_id = input('ClOrdID: ')
    profit_id = int(input('ProfitID: '))

    order_id = TConnectorOrderIdentifier(
        Version=0,
        LocalOrderID=profit_id,
        ClOrderID=cl_ord_id
    )

    printOrder("GetOrder", order_id)

def cancelOrder():
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    rotPassword = getpass("Senha de Roteamento: ")
    cl_ord_id = input('ClOrdID: ')

    cancel_order = TConnectorCancelOrder(
        Version=0,
        Password=rotPassword
    )
    cancel_order.OrderID = TConnectorOrderIdentifier(
        Version=0,
        LocalOrderID=-1,
        ClOrderID=cl_ord_id
    )
    cancel_order.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId,
        Reserved=0
    )

    ret = profit_dll.SendCancelOrderV2(byref(cancel_order))
    evalDllReturn('SendCancelOrderV2', ret)

def cancelAllOrders():
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    rotPassword = getpass("Senha de Roteamento: ")

    cancel_order = TConnectorCancelAllOrders(
        Version=0,
        Password=rotPassword
    )
    cancel_order.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId,
        Reserved=0
    )

    ret = profit_dll.SendCancelAllOrdersV2(byref(cancel_order))
    evalDllReturn('SendCancelAllOrdersV2', ret)

def changeOrder():
    brokerId = int(input("Corretora: "))
    accountId = input("Conta: ")
    subAccountId = input("SubConta: ")
    cl_ord_id = input('ClOrdID: ')
    rotPassword = getpass("Senha de Roteamento: ")

    price = float(input("Preço: "))
    amount = int(input('Quantidade: '))

    change_order = TConnectorChangeOrder(
        Version = 0,
        Password = rotPassword,
        Price = price,
        StopPrice = -1,
        Quantity = amount
    )
    change_order.AccountID = TConnectorAccountIdentifier(
        Version=0,
        BrokerID=brokerId,
        AccountID=accountId,
        SubAccountID=subAccountId,
        Reserved=0
    )
    change_order.OrderID = TConnectorOrderIdentifier(
        Version=0,
        LocalOrderID=-1,
        ClOrderID=cl_ord_id
    )

    profitID = profit_dll.SendChangeOrderV2(byref(change_order))
    evalDllReturn("SendChangeOrderV2", profitID)

def getAccountDetails(accountId : TConnectorAccountIdentifier) -> TConnectorTradingAccountOut:
    account = TConnectorTradingAccountOut(
        Version=0,
        AccountID=accountId
    )

    if (profit_dll.GetAccountDetails(byref(account)) != NL_OK):
        return None

    account.BrokerName = ' ' * account.BrokerNameLength
    account.OwnerName = ' ' * account.OwnerNameLength
    account.SubOwnerName = ' ' * account.SubOwnerNameLength

    if (profit_dll.GetAccountDetails(byref(account)) != NL_OK):
        return None

    return account

def getAccount():
    count = profit_dll.GetAccountCount()
    accountIDs = (TConnectorAccountIdentifierOut * count)()
    count = profit_dll.GetAccounts(0, 0, count, accountIDs)

    for i in range(count):
        accountID = TConnectorAccountIdentifier(
            Version=0,
            BrokerID=accountIDs[i].BrokerID,
            AccountID=accountIDs[i].AccountID
        )

        account = getAccountDetails(accountID)

        if account:
            print("GetAccount: {0} | {1} | {2}".format(
                accountID.BrokerID,
                accountID.AccountID,
                account.OwnerName
            ))

            subCount = profit_dll.GetSubAccountCount(accountID)
            if subCount > 0:
                subAccountIDs = (TConnectorAccountIdentifierOut * subCount)()
                subCount = profit_dll.GetSubAccounts(accountID, 0, 0, subCount, subAccountIDs)

                for j in range(subCount):
                    subAccountID = TConnectorAccountIdentifier(
                        Version=0,
                        BrokerID=subAccountIDs[j].BrokerID,
                        AccountID=subAccountIDs[j].AccountID,
                        SubAccountID=subAccountIDs[j].SubAccountID
                    )

                    subAccount = getAccountDetails(subAccountID)

                    if account:
                        print("GetAccount-Sub: {0} | {1}-{2} | {3}".format(
                            subAccountID.BrokerID,
                            subAccountID.AccountID,
                            subAccountID.SubAccountID,
                            subAccount.SubOwnerName
                        ))


if __name__ == '__main__':
    dllStart()

    strInput = ""
    while strInput != "exit":
        strInput = input('Insira o comando: ')
        if strInput == 'subscribe' :
            subscribeTicker()
        elif strInput == 'unsubscribe':
            unsubscribeTicker()
        elif strInput == 'offerbook':
            subscribeOffer()
        elif strInput == 'position':
            printPosition()
        elif strInput == 'zeroPosition':
            doZeroPosition()
        elif strInput == 'lastAdjusted':
            printLastAdjusted()
        elif strInput == 'buystop' :
            buyStopOrder()
        elif strInput == 'sellstop':
            sellStopOrder()
        elif strInput == 'changeOrder':
            changeOrder()
        elif strInput == 'cancelAllOrders':
            cancelAllOrders()
        elif strInput == 'getOrders':
            getOrders()
        elif strInput == 'getOrder':
            getOrder()
        elif strInput == 'cancelOrder':
            cancelOrder()
        elif strInput == 'getAccount':
            getAccount()
        elif strInput == 'sellAtMarket':
            sendSellMarketOrder()
        elif strInput == 'buyAtMarket':
            sendBuyMarketOrder()

    dllEnd()