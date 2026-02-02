"""
Module de logging structuré pour AIPROD V33
"""
import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), '../../logs')
LOG_FILE = os.path.join(LOG_DIR, 'aiprod_v33.log')

os.makedirs(LOG_DIR, exist_ok=True)

# Configuration du logger
logger = logging.getLogger('AIPROD_V33')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
	'[%(asctime)s] %(levelname)s %(name)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Pour usage direct : from src.utils.monitoring import logger
