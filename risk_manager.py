"""
Module for managing risk in the WINFUT trading robot.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import datetime

from config import RISK_MANAGEMENT

logger = logging.getLogger(__name__)

class RiskManager:
    """Class for managing trading risk"""
    
    def __init__(self):
        self.stop_loss_ticks = RISK_MANAGEMENT["stop_loss_ticks"]
        self.take_profit_ticks = RISK_MANAGEMENT["take_profit_ticks"]
        self.trailing_stop_ticks = RISK_MANAGEMENT["trailing_stop_ticks"]
        self.max_daily_loss = RISK_MANAGEMENT["max_daily_loss"]
        self.max_daily_trades = RISK_MANAGEMENT["max_daily_trades"]
        self.risk_per_trade = RISK_MANAGEMENT["risk_per_trade"]
        
        self.daily_trades = 0
        self.daily_pnl = 0
        self.trade_history = []
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.last_reset_date = datetime.date.today()
        
    def reset_daily_stats(self) -> None:
        """Resets daily trading statistics"""
        today = datetime.date.today()
        if today != self.last_reset_date:
            logger.info(f"Resetting daily stats from {self.last_reset_date} to {today}")
            self.daily_trades = 0
            self.daily_pnl = 0
            self.last_reset_date = today
    
    def can_place_trade(self, account_balance: float) -> Tuple[bool, str]:
        """
        Checks if a new trade can be placed based on risk management rules.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Tuple of (can_trade, reason)
        """
        self.reset_daily_stats()
        
        # Check if maximum daily trades exceeded
        if self.daily_trades >= self.max_daily_trades:
            return False, "max_daily_trades_exceeded"
        
        # Check if maximum daily loss exceeded
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "max_daily_loss_exceeded"
        
        # Check if account balance is sufficient
        min_balance_required = 1000  # Minimum balance required to trade
        if account_balance < min_balance_required:
            return False, "insufficient_account_balance"
        
        # Check current drawdown vs maximum allowed
        max_allowed_drawdown = 0.2  # 20% maximum allowed drawdown
        if account_balance > 0 and self.current_drawdown / account_balance > max_allowed_drawdown:
            return False, "maximum_drawdown_exceeded"
        
        return True, "trade_allowed"
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: Optional[float] = None) -> int:
        """
        Calculates optimal position size based on account balance and risk.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Optional override for risk percentage per trade
            
        Returns:
            Number of contracts to trade
        """
        if risk_per_trade is None:
            risk_per_trade = self.risk_per_trade
        
        # Maximum risk amount in currency
        risk_amount = account_balance * risk_per_trade
        
        # Value at risk per contract with stop loss
        value_at_risk_per_contract = 0.2 * self.stop_loss_ticks  # 0.2 is point value for WINFUT
        
        # Calculate number of contracts based on risk
        if value_at_risk_per_contract > 0:
            contracts = int(risk_amount / value_at_risk_per_contract)
        else:
            contracts = 1
        
        # Ensure minimum and maximum contract limits
        contracts = max(1, min(contracts, 10))  # Between 1 and 10 contracts
        
        logger.info(f"Position size calculation: balance={account_balance}, risk={risk_per_trade}, contracts={contracts}")
        return contracts
    
    def update_stats(self, trade_result: Dict[str, Any]) -> None:
        """
        Updates trading statistics after a trade is completed.
        
        Args:
            trade_result: Dictionary with trade result information
        """
        self.reset_daily_stats()
        
        # Update trade count
        self.daily_trades += 1
        
        # Update P&L
        pnl = trade_result.get("pnl", 0)
        self.daily_pnl += pnl
        
        # Update drawdown
        if pnl < 0:
            self.current_drawdown -= pnl
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            self.current_drawdown = max(0, self.current_drawdown - pnl)
        
        # Add to trade history
        self.trade_history.append({
            "timestamp": datetime.datetime.now(),
            "type": trade_result.get("type", "unknown"),
            "entry_price": trade_result.get("entry_price", 0),
            "exit_price": trade_result.get("exit_price", 0),
            "quantity": trade_result.get("quantity", 0),
            "pnl": pnl,
            "exit_reason": trade_result.get("exit_reason", "unknown")
        })
        
        logger.info(f"Trade stats updated: daily_trades={self.daily_trades}, daily_pnl={self.daily_pnl}, drawdown={self.current_drawdown}")
    
    def calculate_stop_loss(self, entry_price: float, position_type: int, 
                           atr_value: Optional[float] = None) -> float:
        """
        Calculates stop loss level based on entry price and position type.
        
        Args:
            entry_price: Entry price for the position
            position_type: Position type (1 for long, -1 for short)
            atr_value: Optional ATR value for dynamic stop loss
            
        Returns:
            Stop loss price
        """
        # Use ATR-based stop loss if provided
        if atr_value is not None:
            multiplier = 2.0  # 2x ATR for stop loss
            if position_type == 1:  # Long position
                return entry_price - (multiplier * atr_value)
            else:  # Short position
                return entry_price + (multiplier * atr_value)
        
        # Fixed tick-based stop loss
        tick_size = 0.2  # 0.2 points per tick for WINFUT
        stop_points = self.stop_loss_ticks * tick_size
        
        if position_type == 1:  # Long position
            return entry_price - stop_points
        else:  # Short position
            return entry_price + stop_points
    
    def calculate_take_profit(self, entry_price: float, position_type: int, 
                             atr_value: Optional[float] = None) -> float:
        """
        Calculates take profit level based on entry price and position type.
        
        Args:
            entry_price: Entry price for the position
            position_type: Position type (1 for long, -1 for short)
            atr_value: Optional ATR value for dynamic take profit
            
        Returns:
            Take profit price
        """
        # Use ATR-based take profit if provided
        if atr_value is not None:
            multiplier = 3.0  # 3x ATR for take profit (1.5:1 reward:risk ratio)
            if position_type == 1:  # Long position
                return entry_price + (multiplier * atr_value)
            else:  # Short position
                return entry_price - (multiplier * atr_value)
        
        # Fixed tick-based take profit
        tick_size = 0.2  # 0.2 points per tick for WINFUT
        take_profit_points = self.take_profit_ticks * tick_size
        
        if position_type == 1:  # Long position
            return entry_price + take_profit_points
        else:  # Short position
            return entry_price - take_profit_points
    
    def update_trailing_stop(self, position_type: int, entry_price: float, 
                            current_price: float, current_stop: float) -> float:
        """
        Updates trailing stop level based on price movement.
        
        Args:
            position_type: Position type (1 for long, -1 for short)
            entry_price: Entry price for the position
            current_price: Current market price
            current_stop: Current stop loss level
            
        Returns:
            Updated stop loss price
        """
        tick_size = 0.2  # 0.2 points per tick for WINFUT
        trail_points = self.trailing_stop_ticks * tick_size
        
        if position_type == 1:  # Long position
            # Calculate potential new stop loss
            potential_stop = current_price - trail_points
            # Only move stop loss up, never down
            return max(potential_stop, current_stop)
        else:  # Short position
            # Calculate potential new stop loss
            potential_stop = current_price + trail_points
            # Only move stop loss down, never up
            return min(potential_stop, current_stop)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generates risk management report.
        
        Returns:
            Dictionary with risk statistics
        """
        self.reset_daily_stats()
        
        # Calculate success rate
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade["pnl"] > 0)
        success_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average win and loss
        winning_pnl = [trade["pnl"] for trade in self.trade_history if trade["pnl"] > 0]
        losing_pnl = [trade["pnl"] for trade in self.trade_history if trade["pnl"] < 0]
        
        avg_win = sum(winning_pnl) / len(winning_pnl) if winning_pnl else 0
        avg_loss = sum(losing_pnl) / len(losing_pnl) if losing_pnl else 0
        
        # Calculate profit factor
        profit_factor = abs(sum(winning_pnl) / sum(losing_pnl)) if sum(losing_pnl) != 0 else float('inf')
        
        # Calculate expectancy
        expectancy = (success_rate * avg_win) - ((1 - success_rate) * abs(avg_loss)) if total_trades > 0 else 0
        
        return {
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "daily_pnl": self.daily_pnl,
            "max_daily_loss": self.max_daily_loss,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "success_rate": success_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "risk_per_trade": self.risk_per_trade
        }
