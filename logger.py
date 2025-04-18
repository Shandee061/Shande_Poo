"""
Logger configuration for the WINFUT trading robot.
"""
import logging
import os
from datetime import datetime
import sys
from config import LOG_LEVEL, LOG_FILE

def setup_logger(name: str = "winfut_robot") -> logging.Logger:
    """
    Configure and return a logger.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Skip if logger is already configured
    if logger.handlers:
        return logger
    
    # Set log level
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create file handler
    try:
        file_handler = logging.FileHandler(LOG_FILE)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Set formatter for handler
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler: {str(e)}")
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("Logger initialized")
    return logger


def log_trade(logger: logging.Logger, trade_data: dict) -> None:
    """
    Log trade information.
    
    Args:
        logger: Logger object
        trade_data: Dictionary with trade information
    """
    side = trade_data.get("side", "unknown")
    quantity = trade_data.get("quantity", 0)
    price = trade_data.get("price", 0)
    pnl = trade_data.get("pnl", 0)
    
    if "entry" in trade_data:
        logger.info(f"TRADE ENTRY: {side} {quantity} @ {price}")
    elif "exit" in trade_data:
        logger.info(f"TRADE EXIT: {side} {quantity} @ {price}, P&L: {pnl}")
    else:
        logger.info(f"TRADE: {side} {quantity} @ {price}")


def log_signal(logger: logging.Logger, signal_data: dict) -> None:
    """
    Log signal information.
    
    Args:
        logger: Logger object
        signal_data: Dictionary with signal information
    """
    signal = signal_data.get("signal", 0)
    confidence = signal_data.get("confidence", 0)
    
    if signal == 1:
        logger.info(f"SIGNAL: BUY with confidence {confidence:.2f}")
    elif signal == -1:
        logger.info(f"SIGNAL: SELL with confidence {confidence:.2f}")
    elif signal == 2:
        logger.info(f"SIGNAL: EXIT LONG - {signal_data.get('metadata', {}).get('exit_reason', 'unknown')}")
    elif signal == -2:
        logger.info(f"SIGNAL: EXIT SHORT - {signal_data.get('metadata', {}).get('exit_reason', 'unknown')}")
    else:
        logger.info(f"SIGNAL: NEUTRAL with confidence {confidence:.2f}")


def log_error(logger: logging.Logger, error_message: str, exception: Exception = None) -> None:
    """
    Log error information.
    
    Args:
        logger: Logger object
        error_message: Error message
        exception: Optional exception object
    """
    if exception:
        logger.error(f"ERROR: {error_message} - {str(exception)}")
    else:
        logger.error(f"ERROR: {error_message}")


def clear_old_logs(log_file: str = LOG_FILE, days_to_keep: int = 7) -> None:
    """
    Remove log entries older than the specified number of days.
    
    Args:
        log_file: Path to log file
        days_to_keep: Number of days of logs to keep
    """
    if not os.path.exists(log_file):
        return
    
    try:
        # Get file modification time
        mod_time = os.path.getmtime(log_file)
        mod_date = datetime.fromtimestamp(mod_time)
        now = datetime.now()
        
        # Calculate age in days
        age_days = (now - mod_date).days
        
        # If log file is too old, truncate it
        if age_days > days_to_keep:
            with open(log_file, 'w') as f:
                f.write(f"Log file cleared on {now.strftime('%Y-%m-%d %H:%M:%S')} as it was {age_days} days old.\n")
    except Exception as e:
        print(f"Error clearing old logs: {str(e)}")


def rotate_logs(log_file: str = LOG_FILE, max_size_mb: int = 10) -> None:
    """
    Rotate log file if it exceeds the maximum size.
    
    Args:
        log_file: Path to log file
        max_size_mb: Maximum log file size in MB
    """
    if not os.path.exists(log_file):
        return
    
    try:
        # Get file size in MB
        file_size_mb = os.path.getsize(log_file) / (1024 * 1024)
        
        # If file is too big, rotate it
        if file_size_mb > max_size_mb:
            # Backup old log file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_file = f"{log_file}.{timestamp}"
            
            # Rename current log file
            os.rename(log_file, backup_file)
            
            # Create new empty log file
            with open(log_file, 'w') as f:
                f.write(f"Log file rotated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} after reaching {file_size_mb:.2f} MB.\n")
            
            # Delete old log files (keep only 5 most recent)
            log_dir = os.path.dirname(log_file) or "."
            log_basename = os.path.basename(log_file)
            
            backup_files = [f for f in os.listdir(log_dir) if f.startswith(log_basename + ".")]
            backup_files.sort(reverse=True)
            
            # Remove old backups
            for old_file in backup_files[5:]:
                try:
                    os.remove(os.path.join(log_dir, old_file))
                except Exception:
                    pass
    except Exception as e:
        print(f"Error rotating logs: {str(e)}")


# Initialize logging when module is imported
def init_logging():
    """Initialize logging with rotation and cleanup"""
    # Clear old logs
    clear_old_logs()
    
    # Rotate logs if needed
    rotate_logs()
    
    # Setup logger
    return setup_logger()


# Get main logger
logger = init_logging()
