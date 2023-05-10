import sys
from loguru import logger

def get_logger():
        logger.remove()
        logger.add('./app.log',
                   format="{level} | {time} | {message} | context: {extra[context]}",
                   colorize=True,
                   level='DEBUG'
                   )
        return logger.bind(context={})
