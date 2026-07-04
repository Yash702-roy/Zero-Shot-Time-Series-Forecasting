import os
import logging

def setup_logger(log_file: str = "logs/pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up the global logger to log messages to both the console and a file.
    
    Args:
        log_file (str): Filepath to write logs to.
        level (int): Logging level.
        
    Returns:
        logging.Logger: The configured root logger.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers if any to avoid duplicated logs
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # 2. File Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logger initialized. Writing logs to {log_file}")
    return root_logger
