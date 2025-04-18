"""
Module for backtesting trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import matplotlib.pyplot as plt
import datetime

from config import BACKTEST_PARAMS, TECHNICAL_INDICATORS
from data_processor import DataProcessor
from strategy import TradingStrategy
from models import ModelManager
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

class Backtester:
    """Class for backtesting trading strategies"""
    
    def __init__(self, 
                 data_processor: DataProcessor, 
                 model_manager: ModelManager,
                 strategy: TradingStrategy,
                 risk_manager: RiskManager):
        self.data_processor = data_processor
        self.model_manager = model_manager
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = BACKTEST_PARAMS["initial_capital"]
        self.start_date = BACKTEST_PARAMS["start_date"]
        self.end_date = BACKTEST_PARAMS["end_date"]
        self.results = {}
        self.trade_history = []
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for backtesting.
        
        Args:
            df: Raw OHLCV data
            
        Returns:
            Processed DataFrame ready for backtesting
        """
        # Add technical indicators
        processed_df = self.data_processor.add_technical_indicators(df)
        
        # Add time features
        processed_df = self.data_processor.add_time_features(processed_df)
        
        # Filter data by date range
        if self.start_date and self.end_date:
            mask = (processed_df.index >= self.start_date) & (processed_df.index <= self.end_date)
            processed_df = processed_df.loc[mask]
        
        return processed_df
    
    def run_backtest(self, df: pd.DataFrame, window_size: int = 200) -> Dict[str, Any]:
        """
        Runs backtest on historical data.
        
        Args:
            df: OHLCV data for backtesting
            window_size: Size of rolling window for training
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        # Prepare data for backtesting
        processed_df = self.prepare_data(df)
        
        if processed_df.empty:
            logger.error("No data available for backtesting")
            return {"success": False, "error": "No data available for backtesting"}
        
        # Initialize backtest variables
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        stop_loss = None
        take_profit = None
        equity_curve = []
        trades = []
        current_bars_held = 0
        
        # Reserve some data points for initial training
        train_idx = window_size
        
        # Record initial capital
        equity_curve.append({
            "timestamp": processed_df.index[train_idx],
            "equity": capital,
            "position": position
        })
        
        # Loop through data points
        for i in range(train_idx, len(processed_df)):
            current_time = processed_df.index[i]
            current_price = processed_df['close'].iloc[i]
            
            # Training data is all data up to current point
            train_data = processed_df.iloc[:i]
            
            # Only retrain the model periodically (e.g., once a week)
            if i == train_idx or (i - train_idx) % 5 == 0:
                logger.info(f"Training models at {current_time}")
                
                # Process training data
                features, target = self.data_processor.process_data(train_data)
                
                if not features.empty and not target.empty:
                    # Train models
                    self.model_manager.train_models(features, target)
            
            # Current bar's data for signal generation
            current_data = processed_df.iloc[:i+1]
            
            # Update trailing stop if in a position
            if position != 0 and stop_loss is not None:
                if position == 1:  # Long position
                    new_stop = self.risk_manager.update_trailing_stop(
                        position, entry_price, current_price, stop_loss)
                    stop_loss = new_stop
                else:  # Short position
                    new_stop = self.risk_manager.update_trailing_stop(
                        position, entry_price, current_price, stop_loss)
                    stop_loss = new_stop
            
            # Check for stop loss or take profit
            if position != 0:
                current_bars_held += 1
                
                # Stop loss hit
                if (position == 1 and current_price <= stop_loss) or \
                   (position == -1 and current_price >= stop_loss):
                    # Calculate P&L
                    if position == 1:
                        pnl = (stop_loss - entry_price) * 0.2 * 5  # 0.2 points value for WINFUT
                    else:
                        pnl = (entry_price - stop_loss) * 0.2 * 5
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade = {
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": stop_loss,
                        "position": position,
                        "pnl": pnl,
                        "exit_reason": "stop_loss",
                        "bars_held": current_bars_held
                    }
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
                    current_bars_held = 0
                    
                    logger.info(f"Stop loss hit at {current_time}: {trade}")
                
                # Take profit hit
                elif (position == 1 and current_price >= take_profit) or \
                     (position == -1 and current_price <= take_profit):
                    # Calculate P&L
                    if position == 1:
                        pnl = (take_profit - entry_price) * 0.2 * 5
                    else:
                        pnl = (entry_price - take_profit) * 0.2 * 5
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade = {
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": take_profit,
                        "position": position,
                        "pnl": pnl,
                        "exit_reason": "take_profit",
                        "bars_held": current_bars_held
                    }
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
                    current_bars_held = 0
                    
                    logger.info(f"Take profit hit at {current_time}: {trade}")
                
                # Check for other exit signals
                elif current_bars_held >= 20:  # Time-based exit (20 bars)
                    # Calculate P&L
                    if position == 1:
                        pnl = (current_price - entry_price) * 0.2 * 5
                    else:
                        pnl = (entry_price - current_price) * 0.2 * 5
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade = {
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": position,
                        "pnl": pnl,
                        "exit_reason": "time_exit",
                        "bars_held": current_bars_held
                    }
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
                    current_bars_held = 0
                    
                    logger.info(f"Time-based exit at {current_time}: {trade}")
                
                # Technical indicator exit
                elif (position == 1 and current_data['macd'].iloc[-1] < current_data['macd_signal'].iloc[-1] and \
                     current_data['macd'].iloc[-2] > current_data['macd_signal'].iloc[-2]) or \
                     (position == -1 and current_data['macd'].iloc[-1] > current_data['macd_signal'].iloc[-1] and \
                     current_data['macd'].iloc[-2] < current_data['macd_signal'].iloc[-2]):
                    
                    # Calculate P&L
                    if position == 1:
                        pnl = (current_price - entry_price) * 0.2 * 5
                    else:
                        pnl = (entry_price - current_price) * 0.2 * 5
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade = {
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": position,
                        "pnl": pnl,
                        "exit_reason": "technical_exit",
                        "bars_held": current_bars_held
                    }
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    stop_loss = None
                    take_profit = None
                    current_bars_held = 0
                    
                    logger.info(f"Technical exit at {current_time}: {trade}")
            
            # Check for entry signals if not in a position
            if position == 0:
                # Generate signals using strategy
                signal_data = self.strategy.generate_trading_signals(current_data)
                signal = signal_data["signal"]
                
                if signal != 0:  # Valid entry signal
                    position = signal
                    entry_price = current_price
                    entry_time = current_time
                    
                    # Calculate stop loss and take profit
                    atr_value = current_data['atr'].iloc[-1]
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        entry_price, position, atr_value)
                    take_profit = self.risk_manager.calculate_take_profit(
                        entry_price, position, atr_value)
                    
                    logger.info(f"New position at {current_time}: {position}, entry_price: {entry_price}, stop_loss: {stop_loss}, take_profit: {take_profit}")
            
            # Record equity
            equity_curve.append({
                "timestamp": current_time,
                "equity": capital,
                "position": position
            })
        
        # Close any open position at the end of the backtest
        if position != 0:
            final_price = processed_df['close'].iloc[-1]
            
            # Calculate P&L
            if position == 1:
                pnl = (final_price - entry_price) * 0.2 * 5
            else:
                pnl = (entry_price - final_price) * 0.2 * 5
            
            # Update capital
            capital += pnl
            
            # Record trade
            trade = {
                "entry_time": entry_time,
                "exit_time": processed_df.index[-1],
                "entry_price": entry_price,
                "exit_price": final_price,
                "position": position,
                "pnl": pnl,
                "exit_reason": "backtest_end",
                "bars_held": current_bars_held
            }
            trades.append(trade)
            
            logger.info(f"Closing position at end of backtest: {trade}")
        
        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        annual_return = total_return * (252 / len(equity_df)) if len(equity_df) > 0 else 0
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_rfr = (1 + risk_free_rate) ** (1/252) - 1
        sharpe_ratio = np.sqrt(252) * (equity_df['returns'].mean() - daily_rfr) / equity_df['returns'].std() if len(equity_df['returns'].dropna()) > 0 else 0
        
        # Calculate win rate
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
        gross_loss = sum([t['pnl'] for t in trades if t['pnl'] < 0])
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Calculate average profits and losses
        avg_profit = gross_profit / len(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = gross_loss / len(losing_trades) if len(losing_trades) > 0 else 0
        avg_trade = sum([t['pnl'] for t in trades]) / len(trades) if len(trades) > 0 else 0
        
        # Calculate consecutive wins and losses
        win_loss_streak = []
        current_streak = 0
        
        for i, trade in enumerate(trades):
            if trade['pnl'] > 0:  # Winning trade
                if current_streak >= 0:  # Continuing a winning streak
                    current_streak += 1
                else:  # Starting a new winning streak
                    win_loss_streak.append(current_streak)  # Record previous losing streak
                    current_streak = 1
            else:  # Losing trade
                if current_streak <= 0:  # Continuing a losing streak
                    current_streak -= 1
                else:  # Starting a new losing streak
                    win_loss_streak.append(current_streak)  # Record previous winning streak
                    current_streak = -1
        
        # Add the final streak
        win_loss_streak.append(current_streak)
        
        # Find max consecutive wins and losses
        max_consecutive_wins = max([s for s in win_loss_streak if s > 0], default=0)
        max_consecutive_losses = abs(min([s for s in win_loss_streak if s < 0], default=0))
        
        # Calculate date range of the backtest
        start_date = processed_df.index[0].strftime("%Y-%m-%d") if len(processed_df.index) > 0 else "N/A"
        end_date = processed_df.index[-1].strftime("%Y-%m-%d") if len(processed_df.index) > 0 else "N/A"
        
        # Store results
        self.results = {
            "equity_curve": equity_df,
            "trades": trades,
            "initial_capital": self.initial_capital,
            "final_capital": capital,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "start_date": start_date,
            "end_date": end_date
        }
        
        logger.info(f"Backtest completed with {len(trades)} trades")
        logger.info(f"Performance: Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Drawdown: {max_drawdown:.2%}")
        
        # Save trade history
        self.trade_history = trades
        
        return self.results
    
    def plot_results(self) -> None:
        """Plots backtest results"""
        if not self.results:
            logger.error("No backtest results to plot")
            return
        
        equity_curve = self.results["equity_curve"]
        trades = self.results["trades"]
        
        plt.figure(figsize=(14, 10))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve.index, equity_curve['equity'], label='Equity')
        plt.title('Equity Curve')
        plt.ylabel('Capital')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(equity_curve.index, equity_curve['drawdown'], 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        
        plt.tight_layout()
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of backtest performance.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.results:
            return {"error": "No backtest results available"}
        
        return {
            "initial_capital": self.results["initial_capital"],
            "final_capital": self.results["final_capital"],
            "total_return": self.results["total_return"],
            "annual_return": self.results["annual_return"],
            "max_drawdown": self.results["max_drawdown"],
            "sharpe_ratio": self.results["sharpe_ratio"],
            "win_rate": self.results["win_rate"],
            "profit_factor": self.results["profit_factor"],
            "avg_trade": self.results["avg_trade"],
            "avg_profit": self.results.get("avg_profit", 0),
            "avg_loss": self.results.get("avg_loss", 0),
            "max_consecutive_wins": self.results.get("max_consecutive_wins", 0),
            "max_consecutive_losses": self.results.get("max_consecutive_losses", 0),
            "total_trades": self.results["total_trades"],
            "winning_trades": self.results.get("winning_trades", 0),
            "losing_trades": self.results.get("losing_trades", 0),
            "start_date": self.results.get("start_date", "N/A"),
            "end_date": self.results.get("end_date", "N/A")
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Provides detailed analysis of backtest results.
        
        Returns:
            Dictionary with detailed analysis
        """
        if not self.results or not self.trade_history:
            return {"error": "No backtest results available"}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Convert times to datetime if they're not already
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Calculate holding period
        trades_df['holding_period'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60  # minutes
        
        # Categorize trades
        trades_df['trade_type'] = trades_df['position'].map({1: 'long', -1: 'short'})
        trades_df['result'] = trades_df['pnl'].apply(lambda x: 'win' if x > 0 else 'loss')
        
        # Analysis by trade type
        long_trades = trades_df[trades_df['trade_type'] == 'long']
        short_trades = trades_df[trades_df['trade_type'] == 'short']
        
        long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
        # Analysis by exit reason
        exit_reason_stats = trades_df.groupby('exit_reason').agg({
            'pnl': ['count', 'mean', 'sum'],
            'result': lambda x: sum(x == 'win') / len(x) if len(x) > 0 else 0
        })
        
        # Analysis by time of day
        trades_df['hour'] = trades_df['entry_time'].dt.hour
        hourly_stats = trades_df.groupby('hour').agg({
            'pnl': ['count', 'mean', 'sum'],
            'result': lambda x: sum(x == 'win') / len(x) if len(x) > 0 else 0
        })
        
        # Analysis by day of week
        trades_df['day_of_week'] = trades_df['entry_time'].dt.dayofweek
        daily_stats = trades_df.groupby('day_of_week').agg({
            'pnl': ['count', 'mean', 'sum'],
            'result': lambda x: sum(x == 'win') / len(x) if len(x) > 0 else 0
        })
        
        # Consecutive wins and losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for i, row in trades_df.iterrows():
            if row['pnl'] > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        return {
            "total_trades": len(trades_df),
            "winning_trades": sum(trades_df['pnl'] > 0),
            "losing_trades": sum(trades_df['pnl'] <= 0),
            "win_rate": sum(trades_df['pnl'] > 0) / len(trades_df) if len(trades_df) > 0 else 0,
            "avg_winner": trades_df[trades_df['pnl'] > 0]['pnl'].mean() if any(trades_df['pnl'] > 0) else 0,
            "avg_loser": trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if any(trades_df['pnl'] <= 0) else 0,
            "largest_winner": trades_df['pnl'].max(),
            "largest_loser": trades_df['pnl'].min(),
            "avg_holding_period": trades_df['holding_period'].mean(),
            "max_holding_period": trades_df['holding_period'].max(),
            "min_holding_period": trades_df['holding_period'].min(),
            "long_trades": len(long_trades),
            "long_win_rate": long_win_rate,
            "short_trades": len(short_trades),
            "short_win_rate": short_win_rate,
            "exit_reason_stats": exit_reason_stats.to_dict(),
            "hourly_stats": hourly_stats.to_dict(),
            "daily_stats": daily_stats.to_dict(),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses
        }
