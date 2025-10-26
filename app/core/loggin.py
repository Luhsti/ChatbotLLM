# app/core/logging.py
import logging
import sys

def setup_logging():
    """
    Configura o logging para a aplicação.
    """
    # Define o formato das mensagens de log
    log_format = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    
    # Configura um manipulador que envia os logs para o terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format))
    
    # Configura o logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Evita adicionar manipuladores duplicados se a função for chamada mais de uma vez
    if not root_logger.handlers:
        root_logger.addHandler(stream_handler)
