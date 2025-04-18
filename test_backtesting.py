"""
Script para testar o backtesting com dados do Investing.com sem usar a interface Streamlit.
"""

import datetime
from investing_collector import InvestingCollector
from backtester import Backtester
from data_processor import DataProcessor
from models import ModelManager
from strategy import TradingStrategy
from risk_manager import RiskManager
from logger import setup_logger

# Configurar logger
logger = setup_logger("test_backtesting")

def run_test():
    """Executar teste de backtesting com dados do Investing.com"""
    logger.info("Iniciando teste de backtesting")
    
    try:
        # Inicializar componentes
        logger.info("Inicializando componentes")
        data_processor = DataProcessor()
        model_manager = ModelManager()
        risk_manager = RiskManager()
        strategy = TradingStrategy(model_manager, data_processor)
        backtester = Backtester(data_processor, model_manager, strategy, risk_manager)
        investing_collector = InvestingCollector()
        
        # Buscar dados do WINFUT
        logger.info("Buscando dados históricos do WINFUT")
        days = 30
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        historical_data = investing_collector.get_historical_data(
            symbol="WINFUT",
            interval="15m",
            start_date=start_date
        )
        
        if historical_data.empty:
            logger.error("Não foi possível obter dados do WINFUT")
            return False
        
        logger.info(f"Dados obtidos com sucesso: {len(historical_data)} períodos")
        logger.info(f"Período: {historical_data.index[0]} a {historical_data.index[-1]}")
        
        # Executar backtest
        logger.info("Executando backtest")
        results = backtester.run_backtest(historical_data)
        
        if "success" in results and results["success"] == False:
            logger.error(f"Falha no backtest: {results.get('error', 'Erro desconhecido')}")
            return False
        
        # Analisar resultados
        logger.info("Analisando resultados do backtest")
        analysis = backtester.analyze_results()
        
        # Mostrar resultados
        logger.info(f"Resultado do backtest:")
        logger.info(f"Total de operações: {analysis.get('total_trades', 0)}")
        logger.info(f"Operações vencedoras: {analysis.get('winning_trades', 0)}")
        logger.info(f"Operações perdedoras: {analysis.get('losing_trades', 0)}")
        logger.info(f"Taxa de acerto: {analysis.get('win_rate', 0):.2f}%")
        logger.info(f"Retorno total: {analysis.get('total_return', 0):.2f}%")
        logger.info(f"Retorno anualizado: {analysis.get('annualized_return', 0):.2f}%")
        logger.info(f"Drawdown máximo: {analysis.get('max_drawdown', 0):.2f}%")
        logger.info(f"Ratio Sharpe: {analysis.get('sharpe_ratio', 0):.2f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Erro durante o teste: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_test()
    if success:
        logger.info("Teste concluído com sucesso")
    else:
        logger.error("Teste falhou")