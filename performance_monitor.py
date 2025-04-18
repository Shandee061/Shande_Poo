"""
Módulo para monitoramento contínuo de desempenho e adaptação autônoma.

Este módulo implementa sistemas para avaliar continuamente o desempenho do robô,
detectar condições adversas, e implementar mecanismos automáticos de circuit breaker
para proteger o capital em situações de mercado desfavoráveis.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import os
from enum import Enum
import matplotlib.pyplot as plt
import io
import base64

# Importar outros módulos necessários
from market_regime import MarketRegime, MarketRegimeDetector

logger = logging.getLogger(__name__)

class TradingStatus(Enum):
    """Enumeração dos possíveis estados de trading do robô"""
    ACTIVE = "active"                    # Trading normal ativo
    PAUSED = "paused"                    # Trading temporariamente pausado
    SUSPENDED = "suspended"              # Trading suspenso por circuit breaker
    MONITORING = "monitoring"            # Apenas monitorando sem operar
    LIMITED = "limited"                  # Operando com limitações (tamanho reduzido, etc)
    MANUAL_OVERRIDE = "manual_override"  # Controle manual ativado


class PerformanceMonitor:
    """Classe para monitorar desempenho e adaptar comportamento do robô"""
    
    def __init__(self, 
                 monitoring_interval: int = 60,        # Segundos entre verificações
                 warning_threshold: float = -2.0,      # % de drawdown para aviso
                 circuit_breaker_threshold: float = -5.0,  # % de drawdown para circuit breaker
                 daily_loss_limit: float = -3.0,       # % limite de perda diária
                 metrics_window: int = 20,             # Número de operações para métricas
                 auto_restart_time: int = 60,          # Minutos para reinício automático
                 enable_circuit_breaker: bool = True,  # Ativar circuit breaker
                 save_metrics: bool = True,            # Salvar métricas periodicamente
                 metrics_dir: str = "metrics"):
        """
        Inicializa o monitor de desempenho.
        
        Args:
            monitoring_interval: Intervalo em segundos entre verificações
            warning_threshold: Percentual de drawdown para aviso
            circuit_breaker_threshold: Percentual de drawdown para circuit breaker
            daily_loss_limit: Percentual de limite de perda diária
            metrics_window: Número de operações para cálculo de métricas
            auto_restart_time: Minutos para reinício automático após suspensão
            enable_circuit_breaker: Ativar circuit breaker automático
            save_metrics: Salvar métricas periodicamente
            metrics_dir: Diretório para salvar métricas
        """
        self.monitoring_interval = monitoring_interval
        self.warning_threshold = warning_threshold
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.daily_loss_limit = daily_loss_limit
        self.metrics_window = metrics_window
        self.auto_restart_time = auto_restart_time
        self.enable_circuit_breaker = enable_circuit_breaker
        self.save_metrics = save_metrics
        self.metrics_dir = metrics_dir
        
        # Inicializar estado
        self.trading_status = TradingStatus.ACTIVE
        self.last_check_time = datetime.now()
        self.suspension_time = None
        self.daily_stats = {}
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.initial_equity = 0.0
        self.daily_pnl = 0.0
        self.last_restart_check = datetime.now()
        
        # Histórico de operações
        self.trade_history = []
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        # Métricas de desempenho
        self.metrics = {}
        self.metrics_history = []
        self.notification_history = []
        
        # Callbacks personalizados
        self.status_change_callback = None
        self.circuit_breaker_callback = None
        self.warning_callback = None
        
        # Criar diretório para métricas se não existir
        if self.save_metrics:
            os.makedirs(metrics_dir, exist_ok=True)
    
    def register_callbacks(self, 
                          status_change: Optional[Callable] = None,
                          circuit_breaker: Optional[Callable] = None,
                          warning: Optional[Callable] = None):
        """
        Registra callbacks para eventos importantes.
        
        Args:
            status_change: Callback para mudanças de status
            circuit_breaker: Callback para ativação de circuit breaker
            warning: Callback para avisos de desempenho
        """
        self.status_change_callback = status_change
        self.circuit_breaker_callback = circuit_breaker
        self.warning_callback = warning
    
    def initialize(self, initial_equity: float) -> None:
        """
        Inicializa o monitor com equidade inicial.
        
        Args:
            initial_equity: Valor inicial da conta
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.trading_status = TradingStatus.ACTIVE
        self.reset_daily_stats()
        logger.info(f"Monitor de desempenho inicializado com equidade: {initial_equity:.2f}")
    
    def reset_daily_stats(self) -> None:
        """Reinicia estatísticas diárias."""
        today = datetime.now().date()
        self.daily_stats = {
            "date": today,
            "starting_equity": self.current_equity,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "loss": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "pnl": 0.0
        }
        self.daily_pnl = 0.0
    
    def update_equity(self, equity: float) -> None:
        """
        Atualiza o valor atual da equidade.
        
        Args:
            equity: Novo valor da equidade
        """
        prev_equity = self.current_equity
        self.current_equity = equity
        
        # Atualizar pico se necessário
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calcular drawdown atual
        if self.peak_equity > 0:
            self.current_drawdown = (equity - self.peak_equity) / self.peak_equity * 100
        else:
            self.current_drawdown = 0
        
        # Atualizar max drawdown se necessário
        if self.current_drawdown < self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        # Atualizar drawdown diário
        if self.daily_stats["starting_equity"] > 0:
            self.daily_stats["current_drawdown"] = (equity - self.daily_stats["starting_equity"]) / self.daily_stats["starting_equity"] * 100
            self.daily_pnl = self.daily_stats["current_drawdown"]
        
        # Atualizar max drawdown diário se necessário
        if self.daily_stats["current_drawdown"] < self.daily_stats["max_drawdown"]:
            self.daily_stats["max_drawdown"] = self.daily_stats["current_drawdown"]
        
        # Verificar circuit breaker
        self._check_circuit_breaker()
    
    def record_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Registra um resultado de operação.
        
        Args:
            trade_result: Dicionário com resultado da operação
        """
        # Registrar no histórico
        self.trade_history.append(trade_result)
        
        # Limitar tamanho do histórico
        if len(self.trade_history) > 1000:
            self.trade_history.pop(0)
        
        # Calcular estatísticas
        pnl = trade_result.get("pnl", 0)
        is_win = pnl > 0
        
        # Atualizar contadores
        if is_win:
            self.win_count += 1
            self.total_profit += pnl
            self.daily_stats["wins"] += 1
            self.daily_stats["profit"] += pnl
        else:
            self.loss_count += 1
            self.total_loss += abs(pnl)
            self.daily_stats["losses"] += 1
            self.daily_stats["loss"] += abs(pnl)
        
        self.daily_stats["trades"] += 1
        self.daily_stats["pnl"] += pnl
        
        # Atualizar equidade
        new_equity = self.current_equity + pnl
        self.update_equity(new_equity)
        
        # Atualizar métricas
        self._update_metrics()
        
        # Verificar limites após trade
        self._check_circuit_breaker()
    
    def check_status(self) -> TradingStatus:
        """
        Verifica o status atual e realiza checagens periódicas.
        
        Returns:
            Status atual de trading
        """
        now = datetime.now()
        
        # Verificar se é hora de uma checagem periódica
        time_diff = (now - self.last_check_time).total_seconds()
        
        if time_diff >= self.monitoring_interval:
            self.last_check_time = now
            
            # Verificar se é um novo dia
            if now.date() > self.daily_stats["date"]:
                logger.info("Novo dia detectado, reiniciando estatísticas diárias")
                self.reset_daily_stats()
            
            # Verificar se é hora de reiniciar após suspensão
            if self.trading_status == TradingStatus.SUSPENDED and self.suspension_time:
                time_since_suspension = (now - self.suspension_time).total_seconds() / 60
                
                if time_since_suspension >= self.auto_restart_time:
                    logger.info(f"Reiniciando trading após suspensão de {time_since_suspension:.1f} minutos")
                    self._change_status(TradingStatus.MONITORING)
                    self.add_notification("Trading reiniciado em modo de monitoramento após suspensão", "info")
            
            # Atualizar métricas periodicamente
            self._update_metrics()
            
            # Salvar métricas se configurado
            if self.save_metrics:
                self._save_current_metrics()
        
        return self.trading_status
    
    def add_notification(self, message: str, level: str = "info") -> None:
        """
        Adiciona uma notificação ao histórico.
        
        Args:
            message: Mensagem da notificação
            level: Nível da notificação (info, warning, error, success)
        """
        notification = {
            "timestamp": datetime.now(),
            "message": message,
            "level": level
        }
        
        self.notification_history.append(notification)
        
        # Limitar tamanho do histórico
        if len(self.notification_history) > 100:
            self.notification_history.pop(0)
        
        # Log da notificação
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)
    
    def pause_trading(self, reason: str = "manual") -> None:
        """
        Pausa o trading temporariamente.
        
        Args:
            reason: Motivo da pausa
        """
        if self.trading_status != TradingStatus.PAUSED:
            prev_status = self.trading_status
            self._change_status(TradingStatus.PAUSED)
            self.add_notification(f"Trading pausado: {reason} (status anterior: {prev_status.value})", "warning")
    
    def resume_trading(self) -> None:
        """Retoma o trading normal."""
        if self.trading_status != TradingStatus.ACTIVE:
            prev_status = self.trading_status
            self._change_status(TradingStatus.ACTIVE)
            self.add_notification(f"Trading retomado (status anterior: {prev_status.value})", "success")
    
    def manual_override(self, enabled: bool = True) -> None:
        """
        Ativa ou desativa o modo manual.
        
        Args:
            enabled: True para ativar override, False para desativar
        """
        if enabled and self.trading_status != TradingStatus.MANUAL_OVERRIDE:
            prev_status = self.trading_status
            self._change_status(TradingStatus.MANUAL_OVERRIDE)
            self.add_notification(f"Modo manual ativado (status anterior: {prev_status.value})", "warning")
        elif not enabled and self.trading_status == TradingStatus.MANUAL_OVERRIDE:
            self._change_status(TradingStatus.ACTIVE)
            self.add_notification("Modo manual desativado, retornando ao trading automático", "success")
    
    def force_restart(self) -> None:
        """Força reinício do trading (mesmo se estiver em circuit breaker)."""
        prev_status = self.trading_status
        self._change_status(TradingStatus.ACTIVE)
        self.add_notification(f"Trading reiniciado forçadamente (status anterior: {prev_status.value})", "warning")
    
    def limit_trading(self, reason: str = "performance") -> None:
        """
        Limita o trading (redução de tamanho, frequência, etc).
        
        Args:
            reason: Motivo da limitação
        """
        if self.trading_status != TradingStatus.LIMITED:
            prev_status = self.trading_status
            self._change_status(TradingStatus.LIMITED)
            self.add_notification(f"Trading limitado: {reason} (status anterior: {prev_status.value})", "warning")
    
    def _change_status(self, new_status: TradingStatus) -> None:
        """
        Altera o status de trading com notificação de callback.
        
        Args:
            new_status: Novo status de trading
        """
        old_status = self.trading_status
        self.trading_status = new_status
        
        # Registrar hora da suspensão se aplicável
        if new_status == TradingStatus.SUSPENDED:
            self.suspension_time = datetime.now()
        
        # Chamar callback de mudança de status se existir
        if self.status_change_callback and old_status != new_status:
            self.status_change_callback(old_status, new_status)
    
    def _check_circuit_breaker(self) -> None:
        """Verifica condições para ativação do circuit breaker."""
        if not self.enable_circuit_breaker:
            return
            
        # Não verificar se já está suspenso ou em override manual
        if self.trading_status in [TradingStatus.SUSPENDED, TradingStatus.MANUAL_OVERRIDE]:
            return
        
        triggered = False
        trigger_reason = ""
        
        # Verificar perda diária
        if self.daily_pnl <= self.daily_loss_limit:
            triggered = True
            trigger_reason = f"Limite de perda diária atingido: {self.daily_pnl:.2f}% (limite: {self.daily_loss_limit:.2f}%)"
        
        # Verificar drawdown atual
        elif self.current_drawdown <= self.circuit_breaker_threshold:
            triggered = True
            trigger_reason = f"Limite de drawdown atingido: {self.current_drawdown:.2f}% (limite: {self.circuit_breaker_threshold:.2f}%)"
        
        # Verificar sequência de perdas recentes
        elif len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-5:]
            losing_streak = all(trade.get("pnl", 0) < 0 for trade in recent_trades)
            
            if losing_streak:
                triggered = True
                trigger_reason = "Sequência de 5 perdas consecutivas detectada"
        
        # Ativar circuit breaker se necessário
        if triggered:
            self._change_status(TradingStatus.SUSPENDED)
            self.add_notification(f"Circuit breaker ativado: {trigger_reason}", "error")
            
            # Chamar callback específico
            if self.circuit_breaker_callback:
                self.circuit_breaker_callback(trigger_reason)
        
        # Verificar condição de aviso
        elif self.current_drawdown <= self.warning_threshold:
            warning_msg = f"Alerta de drawdown: {self.current_drawdown:.2f}% (limite de aviso: {self.warning_threshold:.2f}%)"
            self.add_notification(warning_msg, "warning")
            
            # Considerar limitar operações
            if self.trading_status == TradingStatus.ACTIVE:
                self._change_status(TradingStatus.LIMITED)
                
            # Chamar callback de aviso
            if self.warning_callback:
                self.warning_callback(warning_msg)
    
    def _update_metrics(self) -> None:
        """Atualiza métricas de desempenho."""
        # Usar apenas as operações mais recentes conforme configurado
        recent_trades = self.trade_history[-self.metrics_window:] if len(self.trade_history) >= self.metrics_window else self.trade_history
        
        if not recent_trades:
            return
        
        # Extrair valores
        pnls = [trade.get("pnl", 0) for trade in recent_trades]
        profits = [pnl for pnl in pnls if pnl > 0]
        losses = [abs(pnl) for pnl in pnls if pnl < 0]
        
        # Cálculos básicos
        win_count = len(profits)
        loss_count = len(losses)
        total_trades = win_count + loss_count
        
        # Métricas gerais
        self.metrics = {
            "timestamp": datetime.now(),
            "period": f"last_{len(recent_trades)}_trades",
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_count / total_trades if total_trades > 0 else 0,
            "total_profit": sum(profits),
            "total_loss": sum(losses),
            "net_pnl": sum(pnls),
            "avg_profit": sum(profits) / win_count if win_count > 0 else 0,
            "avg_loss": sum(losses) / loss_count if loss_count > 0 else 0,
            "largest_profit": max(profits) if profits else 0,
            "largest_loss": max(losses) if losses else 0,
            "profit_factor": sum(profits) / sum(losses) if sum(losses) > 0 else float('inf'),
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "daily_pnl": self.daily_pnl,
            "expectancy": ((win_count / total_trades * (sum(profits) / win_count)) - 
                          (loss_count / total_trades * (sum(losses) / loss_count))) if total_trades > 0 and win_count > 0 and loss_count > 0 else 0
        }
        
        # Calcular métricas avançadas se temos dados suficientes
        if len(pnls) >= 2:
            # Volatilidade de retornos
            returns_std = np.std(pnls)
            
            # Sharpe ratio (aproximado)
            avg_return = np.mean(pnls)
            sharpe = avg_return / returns_std if returns_std > 0 else 0
            
            # Sortino ratio (penaliza apenas variabilidade negativa)
            negative_returns = [pnl for pnl in pnls if pnl < 0]
            downside_dev = np.std(negative_returns) if negative_returns else 0
            sortino = avg_return / downside_dev if downside_dev > 0 else 0
            
            # Adicionar ao dicionário de métricas
            self.metrics.update({
                "volatility": returns_std,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino
            })
        
        # Calcular métricas de sequências
        current_streak = 0
        streak_type = None
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in recent_trades:
            pnl = trade.get("pnl", 0)
            if pnl > 0:  # Win
                if streak_type == "win":
                    current_streak += 1
                else:
                    streak_type = "win"
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            elif pnl < 0:  # Loss
                if streak_type == "loss":
                    current_streak += 1
                else:
                    streak_type = "loss"
                    current_streak = 1
                max_loss_streak = max(max_loss_streak, current_streak)
        
        # Adicionar métricas de sequência
        self.metrics.update({
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_streak_type": streak_type,
            "current_streak_length": current_streak
        })
        
        # Adicionar à série histórica
        self.metrics_history.append(self.metrics.copy())
        
        # Limitar tamanho do histórico
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
    
    def _save_current_metrics(self) -> None:
        """Salva métricas atuais em arquivo JSON."""
        if not self.save_metrics:
            return
            
        try:
            # Criar nome de arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
            
            # Preparar dados para serialização (converter datetime para string)
            metrics_copy = self.metrics.copy()
            metrics_copy["timestamp"] = metrics_copy["timestamp"].isoformat() if "timestamp" in metrics_copy else datetime.now().isoformat()
            
            # Salvar em arquivo
            with open(filename, 'w') as f:
                json.dump(metrics_copy, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {str(e)}")
    
    def get_notifications(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna as notificações mais recentes.
        
        Args:
            count: Número de notificações a retornar
            
        Returns:
            Lista com notificações mais recentes
        """
        return self.notification_history[-count:]
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo do status atual do monitor.
        
        Returns:
            Dicionário com resumo do status
        """
        return {
            "trading_status": self.trading_status.value,
            "current_equity": self.current_equity,
            "initial_equity": self.initial_equity,
            "peak_equity": self.peak_equity,
            "total_return": (self.current_equity / self.initial_equity - 1) * 100 if self.initial_equity > 0 else 0,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_stats["trades"],
            "daily_wins": self.daily_stats["wins"],
            "daily_losses": self.daily_stats["losses"],
            "daily_win_rate": self.daily_stats["wins"] / self.daily_stats["trades"] if self.daily_stats["trades"] > 0 else 0,
            "total_trades": len(self.trade_history),
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_count / (self.win_count + self.loss_count) if (self.win_count + self.loss_count) > 0 else 0,
            "profit_factor": self.total_profit / self.total_loss if self.total_loss > 0 else float('inf'),
            "circuit_breaker_enabled": self.enable_circuit_breaker,
            "suspension_time": self.suspension_time.isoformat() if self.suspension_time else None,
            "minutes_suspended": (datetime.now() - self.suspension_time).total_seconds() / 60 if self.suspension_time else 0
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas de desempenho atuais.
        
        Returns:
            Dicionário com métricas de desempenho
        """
        return self.metrics
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas detalhadas de operações.
        
        Returns:
            Dicionário com estatísticas de operações
        """
        if not self.trade_history:
            return {}
            
        # Extrair dados
        pnls = [trade.get("pnl", 0) for trade in self.trade_history]
        entry_times = [trade.get("entry_time", datetime.now()) for trade in self.trade_history]
        exit_times = [trade.get("exit_time", datetime.now()) for trade in self.trade_history]
        durations = [(exit - entry).total_seconds() / 60 for entry, exit in zip(entry_times, exit_times)]
        trade_types = [trade.get("type", "unknown") for trade in self.trade_history]
        
        # Calcular distribuição por tipo
        long_pnls = [pnl for pnl, type_ in zip(pnls, trade_types) if type_ == "long"]
        short_pnls = [pnl for pnl, type_ in zip(pnls, trade_types) if type_ == "short"]
        
        # Calcular estatísticas por duração
        short_duration = [pnl for pnl, dur in zip(pnls, durations) if dur < 15]
        medium_duration = [pnl for pnl, dur in zip(pnls, durations) if 15 <= dur < 60]
        long_duration = [pnl for pnl, dur in zip(pnls, durations) if dur >= 60]
        
        # Criar distribuição por hora do dia
        hour_distribution = {}
        for trade, entry_time in zip(self.trade_history, entry_times):
            hour = entry_time.hour
            if hour not in hour_distribution:
                hour_distribution[hour] = {"count": 0, "profit": 0, "loss": 0}
            
            hour_distribution[hour]["count"] += 1
            pnl = trade.get("pnl", 0)
            if pnl >= 0:
                hour_distribution[hour]["profit"] += pnl
            else:
                hour_distribution[hour]["loss"] += abs(pnl)
        
        return {
            "pnl_mean": np.mean(pnls),
            "pnl_median": np.median(pnls),
            "pnl_std": np.std(pnls),
            "avg_duration_minutes": np.mean(durations),
            "max_duration_minutes": max(durations),
            "min_duration_minutes": min(durations),
            "long_performance": {
                "count": len(long_pnls),
                "win_rate": len([p for p in long_pnls if p > 0]) / len(long_pnls) if long_pnls else 0,
                "avg_pnl": np.mean(long_pnls) if long_pnls else 0,
                "total_pnl": sum(long_pnls)
            },
            "short_performance": {
                "count": len(short_pnls),
                "win_rate": len([p for p in short_pnls if p > 0]) / len(short_pnls) if short_pnls else 0,
                "avg_pnl": np.mean(short_pnls) if short_pnls else 0,
                "total_pnl": sum(short_pnls)
            },
            "duration_performance": {
                "short": {
                    "count": len(short_duration),
                    "avg_pnl": np.mean(short_duration) if short_duration else 0,
                    "win_rate": len([p for p in short_duration if p > 0]) / len(short_duration) if short_duration else 0
                },
                "medium": {
                    "count": len(medium_duration),
                    "avg_pnl": np.mean(medium_duration) if medium_duration else 0,
                    "win_rate": len([p for p in medium_duration if p > 0]) / len(medium_duration) if medium_duration else 0
                },
                "long": {
                    "count": len(long_duration),
                    "avg_pnl": np.mean(long_duration) if long_duration else 0,
                    "win_rate": len([p for p in long_duration if p > 0]) / len(long_duration) if long_duration else 0
                }
            },
            "hour_distribution": hour_distribution
        }
    
    def get_equity_curve(self) -> str:
        """
        Gera e retorna um gráfico da curva de equidade em formato Base64.
        
        Returns:
            String em formato Base64 com imagem do gráfico
        """
        if not self.trade_history:
            return ""
            
        try:
            # Extrair dados para a curva
            dates = [trade.get("exit_time", datetime.now()) for trade in self.trade_history]
            pnls = [trade.get("pnl", 0) for trade in self.trade_history]
            
            # Calcular curva de equidade cumulativa
            equity = [self.initial_equity]
            for pnl in pnls:
                equity.append(equity[-1] + pnl)
            
            # Adicionar data inicial
            all_dates = [self.trade_history[0].get("entry_time", dates[0])] + dates
            
            # Criar figura
            plt.figure(figsize=(10, 6))
            plt.plot(all_dates, equity, 'b-')
            plt.title("Curva de Equidade")
            plt.xlabel("Data")
            plt.ylabel("Equidade")
            plt.grid(True)
            
            # Adicionar drawdown
            high_equity = pd.Series(equity).cummax()
            drawdown = [(eq / peak - 1) * 100 for eq, peak in zip(equity, high_equity)]
            
            # Criar segunda figura com drawdown
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plotar equidade
            ax1.plot(all_dates, equity, 'b-')
            ax1.set_title("Curva de Equidade e Drawdown")
            ax1.set_ylabel("Equidade")
            ax1.grid(True)
            
            # Plotar drawdown
            ax2.fill_between(all_dates, 0, drawdown, color='r', alpha=0.3)
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Data")
            ax2.grid(True)
            
            # Salvar em buffer
            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Converter para Base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            logger.error(f"Erro ao gerar curva de equidade: {str(e)}")
            return ""
    
    def save_performance_report(self, filename: Optional[str] = None) -> bool:
        """
        Salva um relatório completo de desempenho em JSON.
        
        Args:
            filename: Nome do arquivo para salvar (opcional)
            
        Returns:
            True se salvou com sucesso, False em caso de erro
        """
        try:
            # Criar nome default se não fornecido
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.metrics_dir, f"performance_report_{timestamp}.json")
            
            # Preparar dados do relatório
            report = {
                "status_summary": self.get_status_summary(),
                "performance_metrics": self.get_performance_metrics(),
                "trade_statistics": self.get_trade_statistics(),
                "recent_notifications": [
                    {
                        "timestamp": n["timestamp"].isoformat(),
                        "message": n["message"],
                        "level": n["level"]
                    } for n in self.get_notifications(20)
                ]
            }
            
            # Converter timestamps para strings
            for key in ["timestamp", "suspension_time"]:
                if key in report["status_summary"] and report["status_summary"][key] is not None:
                    if isinstance(report["status_summary"][key], datetime):
                        report["status_summary"][key] = report["status_summary"][key].isoformat()
            
            if "timestamp" in report["performance_metrics"] and report["performance_metrics"]["timestamp"] is not None:
                if isinstance(report["performance_metrics"]["timestamp"], datetime):
                    report["performance_metrics"]["timestamp"] = report["performance_metrics"]["timestamp"].isoformat()
            
            # Salvar em arquivo
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Relatório de desempenho salvo em {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar relatório de desempenho: {str(e)}")
            return False