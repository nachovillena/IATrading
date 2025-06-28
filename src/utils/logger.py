import logging
from typing import Optional

class Logger:
    def __init__(self, name: str = "IATrading", level: int = logging.INFO, to_console: bool = True, to_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Evita handlers duplicados
        if not self.logger.handlers:
            if to_console:
                ch = logging.StreamHandler()
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
            if to_file:
                fh = logging.FileHandler(f"logs/{to_file}")
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

    def set_level(self, level: int):
        self.logger.setLevel(level)

    def add_file_handler(self, filepath: str):
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(filepath)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get(self):
        return self.logger

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)