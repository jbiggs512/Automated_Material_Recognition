import logging
import os
from pathlib import Path

def setup_logging(level=logging.INFO, log_file=None):
    '''Sets up logging with console and optional file handlers.'''
    
    handlers = [logging.StreamHandler()]

    log_dir = Path(os.getenv("PROJECT_DIR")) / os.getenv("LOG_DIR")
    log_path = log_dir / log_file  # full path for log file

    if log_file:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w"))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,  # important for notebooks / re-runs
)