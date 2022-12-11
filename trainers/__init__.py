from trainers.logger import BaseLogger 
from trainers.nae import NAETrainer, NAELogger

def get_logger(cfg, writer):
    logger_type = cfg['logger']
    if logger_type == 'nae':
        logger = NAELogger(writer)
    else:
        logger = BaseLogger(writer)
    return logger 
