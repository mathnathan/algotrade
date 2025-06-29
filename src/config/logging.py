import logging
import sys
from pathlib import Path
from .settings import settings

def setup_logging():
    """Configure application logging."""
    
    # Create logs directory if it doesn't exist
    settings.log_file.parent.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('alpaca').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()