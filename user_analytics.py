"""
Módulo para análise e envio de dados de desempenho dos usuários.

Este módulo implementa funcionalidades para coletar, analisar e enviar
dados de desempenho dos usuários para um servidor central, permitindo
monitorar os resultados e implementar um modelo de cobrança baseado
em performance.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import hashlib
import uuid
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import base64
import hmac
from io import BytesIO

# Configurar logging
logger = logging.getLogger(__name__)

class UserPerformanceTracker:
    """Classe para monitorar e enviar dados de desempenho dos usuários"""
    
    def __init__(self, 
                api_endpoint: str = "https://api.winfutrobot.com.br/telemetry",
                user_id: Optional[str] = None,
                license_key: Optional[str] = None,
                api_key: Optional[str] = None,
                send_interval: int = 24, # Horas entre envios
                local_storage: bool = True,
                data_dir: str = "performance_data"):
        """
        Inicializa o rastreador de desempenho do usuário.
        
        Args:
            api_endpoint: URL do endpoint da API de telemetria
            user_id: ID do usuário (gerado automaticamente se não fornecido)
            license_key: Chave de licença do usuário
            api_key: Chave da API para autenticação
            send_interval: Intervalo em horas para envio dos dados
            local_storage: Armazenar dados localmente
            data_dir: Diretório para armazenamento local dos dados
        """
        self.api_endpoint = api_endpoint
        self.user_id = user_id or self._generate_user_id()
        self.license_key = license_key
        self.api_key = api_key
        self.send_interval = send_interval
        self.local_storage = local_storage
        self.data_dir = data_dir
        
        # Dados de desempenho
        self.daily_performance = {}
        self.unsent_data = []
        self.last_sent = None
        self.total_profit = 0.0
        self.commission_rate = 0.20  # 20% de comissão sobre lucro
        self.commission_due = 0.0
        
        # Criar diretório de dados se não existir
        if self.local_storage and not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Carregar dados existentes
        self._load_stored_data()
        
        logger.info(f"Tracker de desempenho inicializado para usuário: {self.user_id}")
    
    def _generate_user_id(self) -> str:
        """
        Gera um ID de usuário único baseado em características do sistema.
        
        Returns:
            ID de usuário gerado
        """
        # Coletar informações do sistema
        system_info = {}
        try:
            import platform
            system_info["os"] = platform.system()
            system_info["machine"] = platform.machine()
            system_info["processor"] = platform.processor()
        except:
            pass
        
        # Gerar hash a partir das informações do sistema e um UUID
        base_str = json.dumps(system_info) + str(uuid.uuid4())
        return hashlib.sha256(base_str.encode()).hexdigest()[:16]
    
    def _load_stored_data(self) -> None:
        """Carrega dados armazenados localmente."""
        if not self.local_storage:
            return
            
        try:
            # Carregar dados diários
            daily_file = os.path.join(self.data_dir, f"daily_{self.user_id}.json")
            if os.path.exists(daily_file):
                with open(daily_file, "r") as f:
                    self.daily_performance = json.load(f)
                    
            # Carregar dados não enviados
            unsent_file = os.path.join(self.data_dir, f"unsent_{self.user_id}.json")
            if os.path.exists(unsent_file):
                with open(unsent_file, "r") as f:
                    self.unsent_data = json.load(f)
                    
            # Carregar metadados
            meta_file = os.path.join(self.data_dir, f"meta_{self.user_id}.json")
            if os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    meta_data = json.load(f)
                    self.last_sent = meta_data.get("last_sent")
                    self.total_profit = meta_data.get("total_profit", 0.0)
                    self.commission_due = meta_data.get("commission_due", 0.0)
                    
            logger.info(f"Dados carregados: {len(self.daily_performance)} dias, {len(self.unsent_data)} registros não enviados")
        except Exception as e:
            logger.error(f"Erro ao carregar dados locais: {str(e)}")
    
    def _save_data(self) -> None:
        """Salva dados localmente."""
        if not self.local_storage:
            return
            
        try:
            # Salvar dados diários
            daily_file = os.path.join(self.data_dir, f"daily_{self.user_id}.json")
            with open(daily_file, "w") as f:
                json.dump(self.daily_performance, f)
                
            # Salvar dados não enviados
            unsent_file = os.path.join(self.data_dir, f"unsent_{self.user_id}.json")
            with open(unsent_file, "w") as f:
                json.dump(self.unsent_data, f)
                
            # Salvar metadados
            meta_file = os.path.join(self.data_dir, f"meta_{self.user_id}.json")
            meta_data = {
                "last_sent": self.last_sent,
                "total_profit": self.total_profit,
                "commission_due": self.commission_due
            }
            with open(meta_file, "w") as f:
                json.dump(meta_data, f)
        except Exception as e:
            logger.error(f"Erro ao salvar dados locais: {str(e)}")
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Registra um trade para análise de desempenho.
        
        Args:
            trade_data: Dicionário com dados do trade
        """
        try:
            # Extrair dados relevantes
            trade_id = trade_data.get("id", str(uuid.uuid4()))
            timestamp = trade_data.get("timestamp", datetime.now().isoformat())
            pnl = trade_data.get("pnl", 0.0)
            contract = trade_data.get("contract", "WINFUT")
            direction = trade_data.get("direction", "")
            
            # Calcular data do trade
            if isinstance(timestamp, str):
                trade_date = timestamp.split("T")[0]  # Formato ISO
            else:
                trade_date = timestamp.strftime("%Y-%m-%d")
                
            # Inicializar registro diário se necessário
            if trade_date not in self.daily_performance:
                self.daily_performance[trade_date] = {
                    "trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "profit": 0.0,
                    "loss": 0.0,
                    "net_pnl": 0.0,
                    "contracts": {}
                }
                
            # Inicializar registro do contrato se necessário
            if contract not in self.daily_performance[trade_date]["contracts"]:
                self.daily_performance[trade_date]["contracts"][contract] = {
                    "trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "profit": 0.0,
                    "loss": 0.0,
                    "net_pnl": 0.0
                }
                
            # Atualizar estatísticas diárias
            self.daily_performance[trade_date]["trades"] += 1
            self.daily_performance[trade_date]["contracts"][contract]["trades"] += 1
            
            if pnl > 0:
                self.daily_performance[trade_date]["winning_trades"] += 1
                self.daily_performance[trade_date]["profit"] += pnl
                self.daily_performance[trade_date]["contracts"][contract]["winning_trades"] += 1
                self.daily_performance[trade_date]["contracts"][contract]["profit"] += pnl
            else:
                self.daily_performance[trade_date]["losing_trades"] += 1
                self.daily_performance[trade_date]["loss"] += abs(pnl)
                self.daily_performance[trade_date]["contracts"][contract]["losing_trades"] += 1
                self.daily_performance[trade_date]["contracts"][contract]["loss"] += abs(pnl)
                
            self.daily_performance[trade_date]["net_pnl"] += pnl
            self.daily_performance[trade_date]["contracts"][contract]["net_pnl"] += pnl
            
            # Atualizar totais
            self.total_profit += pnl
            
            # Calcular comissão devida sobre lucro (apenas sobre lucro, não sobre prejuízo)
            if pnl > 0:
                self.commission_due += pnl * self.commission_rate
                
            # Adicionar à lista de dados não enviados
            self.unsent_data.append({
                "trade_id": trade_id,
                "timestamp": timestamp,
                "contract": contract,
                "direction": direction,
                "pnl": pnl,
                "user_id": self.user_id
            })
            
            # Salvar dados
            self._save_data()
            
            # Verificar se é hora de enviar dados
            self._check_send_data()
            
            logger.debug(f"Trade registrado: {trade_id}, PnL: {pnl}")
        except Exception as e:
            logger.error(f"Erro ao registrar trade: {str(e)}")
    
    def record_daily_summary(self, date: Optional[str] = None, 
                           summary_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Registra um resumo diário de desempenho.
        
        Args:
            date: Data no formato 'YYYY-MM-DD' (hoje se não especificado)
            summary_data: Dados de resumo diário (opcional)
        """
        try:
            # Usar hoje se data não for especificada
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
                
            # Usar dados existentes ou criar novo registro
            if date in self.daily_performance and summary_data is None:
                daily_data = self.daily_performance[date]
            else:
                daily_data = summary_data or {
                    "trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "profit": 0.0,
                    "loss": 0.0,
                    "net_pnl": 0.0,
                    "contracts": {}
                }
                
            # Atualizar estatísticas diárias
            self.daily_performance[date] = daily_data
            
            # Adicionar à lista de dados não enviados
            self.unsent_data.append({
                "date": date,
                "type": "daily_summary",
                "data": daily_data,
                "user_id": self.user_id
            })
            
            # Salvar dados
            self._save_data()
            
            # Verificar se é hora de enviar dados
            self._check_send_data()
            
            logger.debug(f"Resumo diário registrado para: {date}")
        except Exception as e:
            logger.error(f"Erro ao registrar resumo diário: {str(e)}")
    
    def _check_send_data(self) -> bool:
        """
        Verifica se é hora de enviar dados e envia se necessário.
        
        Returns:
            True se os dados foram enviados, False caso contrário
        """
        now = datetime.now()
        
        # Verificar se é hora de enviar
        if (self.last_sent is None or 
            (now - datetime.fromisoformat(self.last_sent)).total_seconds() / 3600 >= self.send_interval):
            return self.send_performance_data()
            
        return False
    
    def send_performance_data(self, force: bool = False) -> bool:
        """
        Envia dados de desempenho para o servidor.
        
        Args:
            force: Forçar envio mesmo se não for hora
            
        Returns:
            True se os dados foram enviados com sucesso, False caso contrário
        """
        # Verificar se há dados para enviar
        if not self.unsent_data and not force:
            logger.info("Sem dados para enviar")
            return False
            
        # Verificar se API está configurada
        if not self.api_key:
            logger.warning("API key não configurada, dados não enviados")
            return False
            
        try:
            # Preparar payload
            payload = {
                "user_id": self.user_id,
                "license_key": self.license_key,
                "timestamp": datetime.now().isoformat(),
                "data": self.unsent_data if self.unsent_data else [],
                "total_profit": self.total_profit,
                "commission_due": self.commission_due
            }
            
            # Adicionar resumo diário se force=True
            if force and not self.unsent_data:
                today = datetime.now().strftime("%Y-%m-%d")
                if today in self.daily_performance:
                    payload["data"].append({
                        "date": today,
                        "type": "daily_summary",
                        "data": self.daily_performance[today],
                        "user_id": self.user_id
                    })
            
            # Assinar payload
            signature = self._sign_payload(payload)
            
            # Enviar dados
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "X-Signature": signature
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            # Verificar resposta
            if response.status_code == 200:
                response_data = response.json()
                
                # Atualizar último envio
                self.last_sent = datetime.now().isoformat()
                
                # Limpar dados enviados
                self.unsent_data = []
                
                # Atualizar taxa de comissão se informada pelo servidor
                if "commission_rate" in response_data:
                    self.commission_rate = response_data["commission_rate"]
                    
                # Salvar dados atualizados
                self._save_data()
                
                logger.info(f"Dados enviados com sucesso: {len(payload['data'])} registros")
                return True
            else:
                logger.error(f"Erro ao enviar dados: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao enviar dados: {str(e)}")
            return False
    
    def _sign_payload(self, payload: Dict[str, Any]) -> str:
        """
        Assina o payload para autenticação.
        
        Args:
            payload: Dados a serem assinados
            
        Returns:
            Assinatura HMAC-SHA256 em base64
        """
        if not self.api_key:
            return ""
            
        try:
            # Converter payload para string JSON canônica (ordenada)
            payload_str = json.dumps(payload, sort_keys=True)
            
            # Calcular HMAC-SHA256
            hmac_obj = hmac.new(
                self.api_key.encode(),
                payload_str.encode(),
                hashlib.sha256
            )
            
            # Retornar assinatura em base64
            return base64.b64encode(hmac_obj.digest()).decode()
        except Exception as e:
            logger.error(f"Erro ao assinar payload: {str(e)}")
            return ""
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém o resumo diário de desempenho.
        
        Args:
            date: Data no formato 'YYYY-MM-DD' (hoje se não especificado)
            
        Returns:
            Dicionário com resumo diário
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        return self.daily_performance.get(date, {})
    
    def get_period_summary(self, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém o resumo de desempenho para um período.
        
        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD' (7 dias atrás se não especificado)
            end_date: Data final no formato 'YYYY-MM-DD' (hoje se não especificado)
            
        Returns:
            Dicionário com resumo do período
        """
        # Definir datas padrão
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
        # Inicializar resumo
        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "days": 0,
            "trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit": 0.0,
            "loss": 0.0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "daily_performance": {},
            "contracts": {}
        }
        
        # Calcular datas no período
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_datetime:
            date_str = current_date.strftime("%Y-%m-%d")
            
            if date_str in self.daily_performance:
                # Adicionar dados diários
                daily_data = self.daily_performance[date_str]
                summary["days"] += 1
                summary["trades"] += daily_data["trades"]
                summary["winning_trades"] += daily_data["winning_trades"]
                summary["losing_trades"] += daily_data["losing_trades"]
                summary["profit"] += daily_data["profit"]
                summary["loss"] += daily_data["loss"]
                summary["net_pnl"] += daily_data["net_pnl"]
                
                # Adicionar desempenho por contrato
                for contract, contract_data in daily_data.get("contracts", {}).items():
                    if contract not in summary["contracts"]:
                        summary["contracts"][contract] = {
                            "trades": 0,
                            "winning_trades": 0,
                            "losing_trades": 0,
                            "profit": 0.0,
                            "loss": 0.0,
                            "net_pnl": 0.0
                        }
                    
                    # Atualizar dados do contrato
                    summary["contracts"][contract]["trades"] += contract_data["trades"]
                    summary["contracts"][contract]["winning_trades"] += contract_data["winning_trades"]
                    summary["contracts"][contract]["losing_trades"] += contract_data["losing_trades"]
                    summary["contracts"][contract]["profit"] += contract_data["profit"]
                    summary["contracts"][contract]["loss"] += contract_data["loss"]
                    summary["contracts"][contract]["net_pnl"] += contract_data["net_pnl"]
                
                # Adicionar ao resumo diário
                summary["daily_performance"][date_str] = {
                    "net_pnl": daily_data["net_pnl"],
                    "trades": daily_data["trades"],
                    "win_rate": daily_data["winning_trades"] / daily_data["trades"] if daily_data["trades"] > 0 else 0.0
                }
            
            # Avançar para o próximo dia
            current_date += timedelta(days=1)
        
        # Calcular métricas adicionais
        if summary["trades"] > 0:
            summary["win_rate"] = summary["winning_trades"] / summary["trades"]
            
        if summary["loss"] > 0:
            summary["profit_factor"] = summary["profit"] / summary["loss"]
        
        return summary
    
    def get_commission_summary(self) -> Dict[str, Any]:
        """
        Obtém o resumo de comissões devidas.
        
        Returns:
            Dicionário com resumo de comissões
        """
        return {
            "total_profit": self.total_profit,
            "commission_rate": self.commission_rate,
            "commission_due": self.commission_due,
            "last_updated": datetime.now().isoformat()
        }
    
    def update_commission_rate(self, new_rate: float) -> None:
        """
        Atualiza a taxa de comissão.
        
        Args:
            new_rate: Nova taxa de comissão (0.0-1.0)
        """
        # Validar taxa
        if 0.0 <= new_rate <= 1.0:
            # Recalcular comissão devida
            if self.total_profit > 0:
                self.commission_due = self.total_profit * new_rate
                
            self.commission_rate = new_rate
            self._save_data()
            logger.info(f"Taxa de comissão atualizada para: {new_rate:.2%}")
        else:
            logger.error(f"Taxa de comissão inválida: {new_rate}")
    
    def generate_performance_report(self, 
                                  days: int = 30, 
                                  include_equity_curve: bool = True,
                                  format: str = "json") -> Dict[str, Any]:
        """
        Gera um relatório de desempenho completo.
        
        Args:
            days: Número de dias para incluir no relatório
            include_equity_curve: Incluir gráfico de curva de patrimônio
            format: Formato do relatório ('json' ou 'html')
            
        Returns:
            Dicionário com relatório de desempenho
        """
        # Calcular datas
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Obter resumo do período
        period_summary = self.get_period_summary(start_date, end_date)
        
        # Inicializar relatório
        report = {
            "user_id": self.user_id,
            "license_status": "Ativo" if self.license_key else "Não licenciado",
            "report_date": datetime.now().isoformat(),
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "days": period_summary["days"],
            },
            "performance": {
                "trades": period_summary["trades"],
                "winning_trades": period_summary["winning_trades"],
                "losing_trades": period_summary["losing_trades"],
                "win_rate": period_summary["win_rate"],
                "profit_factor": period_summary["profit_factor"],
                "net_pnl": period_summary["net_pnl"],
                "average_daily_pnl": period_summary["net_pnl"] / period_summary["days"] if period_summary["days"] > 0 else 0.0
            },
            "commission": self.get_commission_summary(),
            "contracts": period_summary["contracts"],
            "daily_performance": period_summary["daily_performance"]
        }
        
        # Adicionar curva de patrimônio
        if include_equity_curve:
            try:
                # Criar curva de patrimônio
                dates = sorted(period_summary["daily_performance"].keys())
                equity_values = []
                equity = 0.0
                
                for date in dates:
                    equity += period_summary["daily_performance"][date]["net_pnl"]
                    equity_values.append(equity)
                
                if dates and equity_values:
                    # Criar gráfico
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 5))
                    plt.plot(dates, equity_values, marker='o')
                    plt.title(f"Curva de Patrimônio - Últimos {days} dias")
                    plt.xlabel("Data")
                    plt.ylabel("Patrimônio Acumulado")
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    
                    # Salvar gráfico em buffer
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    
                    # Converter para base64
                    equity_curve_b64 = base64.b64encode(buffer.read()).decode()
                    report["equity_curve"] = equity_curve_b64
                    
                    plt.close()
            except Exception as e:
                logger.error(f"Erro ao gerar curva de patrimônio: {str(e)}")
        
        # Gerar relatório em HTML se solicitado
        if format.lower() == "html":
            try:
                # Gerar HTML com template simples
                html = f"""
                <html>
                <head>
                    <title>Relatório de Desempenho - {report["report_date"]}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .good {{ color: green; }}
                        .bad {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Relatório de Desempenho</h1>
                    <p>Usuário: {report["user_id"]} - Status: {report["license_status"]}</p>
                    <p>Período: {report["period"]["start_date"]} a {report["period"]["end_date"]} ({report["period"]["days"]} dias)</p>
                    
                    <h2>Desempenho Geral</h2>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                        </tr>
                        <tr>
                            <td>Trades Totais</td>
                            <td>{report["performance"]["trades"]}</td>
                        </tr>
                        <tr>
                            <td>Trades Vencedores</td>
                            <td class="good">{report["performance"]["winning_trades"]}</td>
                        </tr>
                        <tr>
                            <td>Trades Perdedores</td>
                            <td class="bad">{report["performance"]["losing_trades"]}</td>
                        </tr>
                        <tr>
                            <td>Taxa de Acerto</td>
                            <td>{report["performance"]["win_rate"]:.2%}</td>
                        </tr>
                        <tr>
                            <td>Fator de Lucro</td>
                            <td>{report["performance"]["profit_factor"]:.2f}</td>
                        </tr>
                        <tr>
                            <td>Resultado Líquido</td>
                            <td class="{'good' if report["performance"]["net_pnl"] > 0 else 'bad'}">
                                R$ {report["performance"]["net_pnl"]:.2f}
                            </td>
                        </tr>
                        <tr>
                            <td>Média Diária</td>
                            <td class="{'good' if report["performance"]["average_daily_pnl"] > 0 else 'bad'}">
                                R$ {report["performance"]["average_daily_pnl"]:.2f}
                            </td>
                        </tr>
                    </table>
                    
                    <h2>Comissão</h2>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                        </tr>
                        <tr>
                            <td>Lucro Total</td>
                            <td>R$ {report["commission"]["total_profit"]:.2f}</td>
                        </tr>
                        <tr>
                            <td>Taxa de Comissão</td>
                            <td>{report["commission"]["commission_rate"]:.2%}</td>
                        </tr>
                        <tr>
                            <td>Comissão Devida</td>
                            <td>R$ {report["commission"]["commission_due"]:.2f}</td>
                        </tr>
                    </table>
                """
                
                # Adicionar curva de patrimônio
                if "equity_curve" in report:
                    html += f"""
                    <h2>Curva de Patrimônio</h2>
                    <img src="data:image/png;base64,{report["equity_curve"]}" alt="Curva de Patrimônio" />
                    """
                
                # Adicionar desempenho por contrato
                if report["contracts"]:
                    html += """
                    <h2>Desempenho por Contrato</h2>
                    <table>
                        <tr>
                            <th>Contrato</th>
                            <th>Trades</th>
                            <th>Taxa de Acerto</th>
                            <th>Resultado</th>
                        </tr>
                    """
                    
                    for contract, data in report["contracts"].items():
                        win_rate = data["winning_trades"] / data["trades"] if data["trades"] > 0 else 0.0
                        html += f"""
                        <tr>
                            <td>{contract}</td>
                            <td>{data["trades"]}</td>
                            <td>{win_rate:.2%}</td>
                            <td class="{'good' if data["net_pnl"] > 0 else 'bad'}">
                                R$ {data["net_pnl"]:.2f}
                            </td>
                        </tr>
                        """
                    
                    html += "</table>"
                
                # Finalizar HTML
                html += """
                </body>
                </html>
                """
                
                report["html_report"] = html
            except Exception as e:
                logger.error(f"Erro ao gerar relatório HTML: {str(e)}")
                
        return report
    
    def save_report(self, filename: Optional[str] = None, format: str = "json") -> bool:
        """
        Salva um relatório de desempenho em arquivo.
        
        Args:
            filename: Nome do arquivo (gerado automaticamente se não especificado)
            format: Formato do relatório ('json' ou 'html')
            
        Returns:
            True se o relatório foi salvo com sucesso, False caso contrário
        """
        try:
            # Gerar relatório
            report = self.generate_performance_report(format=format)
            
            # Definir nome de arquivo padrão
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_report_{self.user_id}_{timestamp}.{format}"
                
            # Determinar caminho
            filepath = os.path.join(self.data_dir, filename)
            
            # Salvar relatório
            if format.lower() == "json":
                with open(filepath, "w") as f:
                    json.dump(report, f, indent=2)
            elif format.lower() == "html" and "html_report" in report:
                with open(filepath, "w") as f:
                    f.write(report["html_report"])
            else:
                logger.error(f"Formato inválido: {format}")
                return False
                
            logger.info(f"Relatório salvo em: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {str(e)}")
            return False


class UsageStatisticsCollector:
    """Classe para coletar estatísticas anônimas de uso do sistema"""
    
    def __init__(self, 
                api_endpoint: str = "https://api.winfutrobot.com.br/statistics",
                user_id: Optional[str] = None,
                send_interval: int = 24, # Horas entre envios
                data_dir: str = "usage_data"):
        """
        Inicializa o coletor de estatísticas.
        
        Args:
            api_endpoint: URL do endpoint da API de estatísticas
            user_id: ID do usuário (gerado automaticamente se não fornecido)
            send_interval: Intervalo em horas para envio dos dados
            data_dir: Diretório para armazenamento local dos dados
        """
        self.api_endpoint = api_endpoint
        self.user_id = user_id or self._generate_anonymous_id()
        self.send_interval = send_interval
        self.data_dir = data_dir
        
        # Dados de uso
        self.usage_data = {}
        self.session_start = datetime.now()
        self.last_sent = None
        self.feature_usage = {}
        
        # Criar diretório de dados se não existir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Carregar dados existentes
        self._load_stored_data()
        
        logger.info(f"Coletor de estatísticas inicializado: {self.user_id}")
    
    def _generate_anonymous_id(self) -> str:
        """
        Gera um ID anônimo para o usuário.
        
        Returns:
            ID anônimo gerado
        """
        # Gerar UUID v4 aleatório
        return f"anon_{uuid.uuid4().hex[:8]}"
    
    def _load_stored_data(self) -> None:
        """Carrega dados armazenados localmente."""
        try:
            usage_file = os.path.join(self.data_dir, f"usage_{self.user_id}.json")
            if os.path.exists(usage_file):
                with open(usage_file, "r") as f:
                    data = json.load(f)
                    self.usage_data = data.get("usage_data", {})
                    self.feature_usage = data.get("feature_usage", {})
                    self.last_sent = data.get("last_sent")
                    
            logger.debug(f"Dados de uso carregados para: {self.user_id}")
        except Exception as e:
            logger.error(f"Erro ao carregar dados de uso: {str(e)}")
    
    def _save_data(self) -> None:
        """Salva dados localmente."""
        try:
            usage_file = os.path.join(self.data_dir, f"usage_{self.user_id}.json")
            with open(usage_file, "w") as f:
                json.dump({
                    "usage_data": self.usage_data,
                    "feature_usage": self.feature_usage,
                    "last_sent": self.last_sent
                }, f)
        except Exception as e:
            logger.error(f"Erro ao salvar dados de uso: {str(e)}")
    
    def record_feature_usage(self, feature: str, count: int = 1) -> None:
        """
        Registra o uso de uma funcionalidade.
        
        Args:
            feature: Nome da funcionalidade
            count: Número de vezes usada
        """
        if feature not in self.feature_usage:
            self.feature_usage[feature] = 0
            
        self.feature_usage[feature] += count
        self._save_data()
    
    def record_session_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Registra métricas da sessão atual.
        
        Args:
            metrics: Dicionário com métricas da sessão
        """
        # Registrar dia
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.usage_data:
            self.usage_data[today] = {
                "sessions": 0,
                "total_duration": 0,
                "features": {},
                "metrics": {}
            }
            
        # Atualizar métricas
        for key, value in metrics.items():
            if key not in self.usage_data[today]["metrics"]:
                self.usage_data[today]["metrics"][key] = value
            else:
                # Para métricas numéricas, calcular média
                if isinstance(value, (int, float)) and isinstance(self.usage_data[today]["metrics"][key], (int, float)):
                    self.usage_data[today]["metrics"][key] = (self.usage_data[today]["metrics"][key] + value) / 2
                else:
                    self.usage_data[today]["metrics"][key] = value
                    
        # Atualizar features
        for feature, count in self.feature_usage.items():
            if feature not in self.usage_data[today]["features"]:
                self.usage_data[today]["features"][feature] = count
            else:
                self.usage_data[today]["features"][feature] += count
                
        # Atualizar sessão
        self.usage_data[today]["sessions"] += 1
        
        # Calcular duração da sessão
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60  # minutos
        self.usage_data[today]["total_duration"] += session_duration
        
        # Reiniciar contadores da sessão
        self.session_start = datetime.now()
        self.feature_usage = {}
        
        # Salvar dados
        self._save_data()
        
        # Verificar se é hora de enviar dados
        self._check_send_data()
    
    def _check_send_data(self) -> bool:
        """
        Verifica se é hora de enviar dados e envia se necessário.
        
        Returns:
            True se os dados foram enviados, False caso contrário
        """
        now = datetime.now()
        
        # Verificar se é hora de enviar
        if (self.last_sent is None or 
            (now - datetime.fromisoformat(self.last_sent)).total_seconds() / 3600 >= self.send_interval):
            return self.send_usage_data()
            
        return False
    
    def send_usage_data(self) -> bool:
        """
        Envia dados de uso para o servidor.
        
        Returns:
            True se os dados foram enviados com sucesso, False caso contrário
        """
        try:
            # Verificar se há dados para enviar
            if not self.usage_data:
                logger.info("Sem dados de uso para enviar")
                return False
                
            # Preparar payload
            payload = {
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "usage_data": self.usage_data,
                "platform_info": self._get_platform_info()
            }
            
            # Enviar dados
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            # Verificar resposta
            if response.status_code == 200:
                # Atualizar último envio
                self.last_sent = datetime.now().isoformat()
                
                # Limpar dados
                self.usage_data = {}
                
                # Salvar dados atualizados
                self._save_data()
                
                logger.info("Dados de uso enviados com sucesso")
                return True
            else:
                logger.error(f"Erro ao enviar dados de uso: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao enviar dados de uso: {str(e)}")
            return False
    
    def _get_platform_info(self) -> Dict[str, str]:
        """
        Obtém informações sobre a plataforma.
        
        Returns:
            Dicionário com informações da plataforma
        """
        info = {
            "platform": "unknown",
            "version": "unknown",
            "python_version": "unknown"
        }
        
        try:
            import platform
            import sys
            
            info["platform"] = platform.system()
            info["python_version"] = sys.version.split()[0]
            
            # Versão do aplicativo (se disponível)
            try:
                from config import VERSION
                info["version"] = VERSION
            except:
                pass
                
        except:
            pass
            
        return info


# Função para inicializar o sistema de analytics
def initialize_analytics(user_id: Optional[str] = None, 
                        license_key: Optional[str] = None,
                        api_key: Optional[str] = None,
                        api_endpoint: Optional[str] = None) -> Tuple[UserPerformanceTracker, UsageStatisticsCollector]:
    """
    Inicializa o sistema de análise de desempenho e estatísticas de uso.
    
    Args:
        user_id: ID do usuário (gerado automaticamente se não fornecido)
        license_key: Chave de licença do usuário
        api_key: Chave da API para autenticação
        api_endpoint: URL do endpoint da API de telemetria (opcional)
        
    Returns:
        Tupla com (tracker de desempenho, coletor de estatísticas)
    """
    try:
        # Inicializar tracker de desempenho
        performance_tracker = UserPerformanceTracker(
            user_id=user_id,
            license_key=license_key,
            api_key=api_key,
            api_endpoint=api_endpoint if api_endpoint else "https://api.winfutrobot.com.br/telemetry"
        )
        
        # Inicializar coletor de estatísticas
        usage_collector = UsageStatisticsCollector(
            user_id=user_id,
            api_endpoint="https://api.winfutrobot.com.br/statistics" if not api_endpoint else api_endpoint.replace("/telemetry", "/statistics")
        )
        
        # Registrar inicialização
        usage_collector.record_feature_usage("app_start")
        
        logger.info("Sistema de analytics inicializado")
        return performance_tracker, usage_collector
    except Exception as e:
        logger.error(f"Erro ao inicializar analytics: {str(e)}")
        # Retornar instâncias vazias em caso de erro
        return UserPerformanceTracker(), UsageStatisticsCollector()