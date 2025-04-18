"""
Module for managing and executing orders through trading platforms.
Supports both Profit Pro and MetaTrader 5.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import datetime

from profit_api import ProfitProAPI
from metatrader_api import MetaTraderAPI
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

class OrderManager:
    """Class for managing trade execution and order tracking"""
    
    def __init__(self, trading_api: Union[ProfitProAPI, MetaTraderAPI], risk_manager: RiskManager):
        self.trading_api = trading_api
        self.risk_manager = risk_manager
        self.pending_orders = []
        self.open_positions = []
        self.order_history = []
        self.last_order_id = None
        
    def place_market_order(self, side: str, quantity: int, 
                          stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Places a market order.
        
        Args:
            side: Order side (BUY or SELL)
            quantity: Number of contracts
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            Dictionary with order result
        """
        try:
            logger.info(f"Placing {side} market order for {quantity} contracts")
            
            # Call Profit Pro API to place order
            result = self.trading_api.place_order(
                order_type="MARKET",
                side=side,
                quantity=quantity,
                price=None,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result.get("success", False):
                order_id = result.get("order_id")
                self.last_order_id = order_id
                
                logger.info(f"Market order placed successfully: {order_id}")
                
                # Add to order history
                order_record = {
                    "order_id": order_id,
                    "type": "MARKET",
                    "side": side,
                    "quantity": quantity,
                    "price": None,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "status": "FILLED",
                    "timestamp": datetime.datetime.now()
                }
                
                self.order_history.append(order_record)
                
                # Update open positions if order was filled
                if side == "BUY":
                    position_type = 1  # Long position
                else:
                    position_type = -1  # Short position
                    
                self.update_position(order_id, position_type, result.get("fill_price"), quantity, stop_loss, take_profit)
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "message": "Market order placed successfully"
                }
            else:
                logger.error(f"Failed to place market order: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error placing market order: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def place_limit_order(self, side: str, quantity: int, price: float,
                         stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """
        Places a limit order.
        
        Args:
            side: Order side (BUY or SELL)
            quantity: Number of contracts
            price: Limit price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            Dictionary with order result
        """
        try:
            logger.info(f"Placing {side} limit order for {quantity} contracts at {price}")
            
            # Call Profit Pro API to place order
            result = self.trading_api.place_order(
                order_type="LIMIT",
                side=side,
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result.get("success", False):
                order_id = result.get("order_id")
                self.last_order_id = order_id
                
                logger.info(f"Limit order placed successfully: {order_id}")
                
                # Add to pending orders
                order_record = {
                    "order_id": order_id,
                    "type": "LIMIT",
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "status": "PENDING",
                    "timestamp": datetime.datetime.now()
                }
                
                self.pending_orders.append(order_record)
                self.order_history.append(order_record)
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "message": "Limit order placed successfully"
                }
            else:
                logger.error(f"Failed to place limit order: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error placing limit order: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancels a pending order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dictionary with cancellation result
        """
        try:
            logger.info(f"Cancelling order: {order_id}")
            
            # Call Profit Pro API to cancel order
            result = self.trading_api.cancel_order(order_id)
            
            if result.get("success", False):
                logger.info(f"Order {order_id} cancelled successfully")
                
                # Update order status in history
                for order in self.order_history:
                    if order["order_id"] == order_id:
                        order["status"] = "CANCELLED"
                        break
                
                # Remove from pending orders
                self.pending_orders = [order for order in self.pending_orders if order["order_id"] != order_id]
                
                return {
                    "success": True,
                    "message": "Order cancelled successfully"
                }
            else:
                logger.error(f"Failed to cancel order {order_id}: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_position(self, order_id: str, position_type: int, entry_price: float, 
                       quantity: int, stop_loss: float = None, take_profit: float = None) -> None:
        """
        Updates positions after an order is filled.
        
        Args:
            order_id: Order ID
            position_type: Position type (1 for long, -1 for short)
            entry_price: Entry price for the position
            quantity: Position size
            stop_loss: Stop loss level
            take_profit: Take profit level
        """
        position = {
            "order_id": order_id,
            "position_type": position_type,
            "entry_price": entry_price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": datetime.datetime.now()
        }
        
        self.open_positions.append(position)
        logger.info(f"Position updated: {position}")
    
    def close_position(self, position_index: int, exit_price: float, exit_reason: str) -> Dict[str, Any]:
        """
        Closes an open position.
        
        Args:
            position_index: Index of the position in open_positions list
            exit_price: Exit price
            exit_reason: Reason for closing the position
            
        Returns:
            Dictionary with position closing result
        """
        try:
            if position_index < 0 or position_index >= len(self.open_positions):
                logger.error(f"Invalid position index: {position_index}")
                return {
                    "success": False,
                    "error": "Invalid position index"
                }
            
            position = self.open_positions[position_index]
            logger.info(f"Closing position: {position}")
            
            # Determine order side for closing
            if position["position_type"] == 1:  # Long position
                close_side = "SELL"
            else:  # Short position
                close_side = "BUY"
            
            # Place market order to close position
            result = self.trading_api.place_order(
                order_type="MARKET",
                side=close_side,
                quantity=position["quantity"]
            )
            
            if result.get("success", False):
                order_id = result.get("order_id")
                
                # Calculate P&L
                if position["position_type"] == 1:  # Long position
                    pnl = (exit_price - position["entry_price"]) * 0.2 * position["quantity"]
                else:  # Short position
                    pnl = (position["entry_price"] - exit_price) * 0.2 * position["quantity"]
                
                # Update trade history
                trade_result = {
                    "type": "long" if position["position_type"] == 1 else "short",
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "quantity": position["quantity"],
                    "entry_time": position["entry_time"],
                    "exit_time": datetime.datetime.now(),
                    "pnl": pnl,
                    "exit_reason": exit_reason
                }
                
                # Update risk manager stats
                self.risk_manager.update_stats(trade_result)
                
                # Remove position from open positions
                self.open_positions.pop(position_index)
                
                logger.info(f"Position closed successfully: {trade_result}")
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "trade_result": trade_result
                }
            else:
                logger.error(f"Failed to close position: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def close_all_positions(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Closes all open positions.
        
        Args:
            reason: Reason for closing all positions
            
        Returns:
            Dictionary with results
        """
        results = []
        position_indices = list(range(len(self.open_positions) - 1, -1, -1))
        
        for idx in position_indices:
            # Get current price
            price_data = self.trading_api.get_price_data(limit=1)
            if not price_data.empty:
                current_price = price_data['close'].iloc[-1]
                
                # Close the position
                result = self.close_position(idx, current_price, reason)
                results.append(result)
            else:
                logger.error("Failed to get current price for closing positions")
                results.append({
                    "success": False,
                    "error": "Failed to get current price"
                })
        
        return {
            "closed_positions": len(results),
            "results": results
        }
    
    def update_order_status(self) -> None:
        """Updates the status of pending orders"""
        for i, order in enumerate(self.pending_orders[:]):
            try:
                # Get order status from API
                order_info = self.trading_api.get_order_status(order["order_id"])
                
                if order_info.get("status") == "FILLED":
                    logger.info(f"Order {order['order_id']} has been filled")
                    
                    # Update order in history
                    for hist_order in self.order_history:
                        if hist_order["order_id"] == order["order_id"]:
                            hist_order["status"] = "FILLED"
                            break
                    
                    # Update positions
                    if order["side"] == "BUY":
                        position_type = 1  # Long position
                    else:
                        position_type = -1  # Short position
                        
                    self.update_position(
                        order["order_id"],
                        position_type,
                        order_info.get("fill_price", order["price"]),
                        order["quantity"],
                        order["stop_loss"],
                        order["take_profit"]
                    )
                    
                    # Remove from pending orders
                    self.pending_orders.pop(i)
                
                elif order_info.get("status") == "CANCELLED" or order_info.get("status") == "REJECTED":
                    logger.info(f"Order {order['order_id']} has been {order_info.get('status')}")
                    
                    # Update order in history
                    for hist_order in self.order_history:
                        if hist_order["order_id"] == order["order_id"]:
                            hist_order["status"] = order_info.get("status")
                            break
                    
                    # Remove from pending orders
                    self.pending_orders.pop(i)
                    
            except Exception as e:
                logger.error(f"Error updating order status for {order['order_id']}: {str(e)}")
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Returns list of current open positions.
        
        Returns:
            List of open positions
        """
        return self.open_positions.copy()
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Returns order history.
        
        Returns:
            List of orders
        """
        return self.order_history.copy()
    
    def has_open_positions(self) -> bool:
        """
        Checks if there are any open positions.
        
        Returns:
            Boolean indicating if there are open positions
        """
        return len(self.open_positions) > 0
    
    def modify_stop_loss(self, position_index: int, new_stop_loss: float) -> Dict[str, Any]:
        """
        Modifies stop loss for an open position.
        
        Args:
            position_index: Index of the position in open_positions list
            new_stop_loss: New stop loss level
            
        Returns:
            Dictionary with modification result
        """
        try:
            if position_index < 0 or position_index >= len(self.open_positions):
                logger.error(f"Invalid position index: {position_index}")
                return {
                    "success": False,
                    "error": "Invalid position index"
                }
            
            position = self.open_positions[position_index]
            
            # Call API to modify stop loss
            result = self.trading_api.modify_order(
                position["order_id"],
                stop_loss=new_stop_loss
            )
            
            if result.get("success", False):
                # Update position
                self.open_positions[position_index]["stop_loss"] = new_stop_loss
                
                logger.info(f"Modified stop loss for position {position_index} to {new_stop_loss}")
                
                return {
                    "success": True,
                    "message": "Stop loss modified successfully"
                }
            else:
                logger.error(f"Failed to modify stop loss: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error modifying stop loss: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def modify_take_profit(self, position_index: int, new_take_profit: float) -> Dict[str, Any]:
        """
        Modifies take profit for an open position.
        
        Args:
            position_index: Index of the position in open_positions list
            new_take_profit: New take profit level
            
        Returns:
            Dictionary with modification result
        """
        try:
            if position_index < 0 or position_index >= len(self.open_positions):
                logger.error(f"Invalid position index: {position_index}")
                return {
                    "success": False,
                    "error": "Invalid position index"
                }
            
            position = self.open_positions[position_index]
            
            # Call API to modify take profit
            result = self.trading_api.modify_order(
                position["order_id"],
                take_profit=new_take_profit
            )
            
            if result.get("success", False):
                # Update position
                self.open_positions[position_index]["take_profit"] = new_take_profit
                
                logger.info(f"Modified take profit for position {position_index} to {new_take_profit}")
                
                return {
                    "success": True,
                    "message": "Take profit modified successfully"
                }
            else:
                logger.error(f"Failed to modify take profit: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"Error modifying take profit: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
