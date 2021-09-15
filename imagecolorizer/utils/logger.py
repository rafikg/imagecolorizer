import logging
import os
import sys
from typing import Optional

import termcolor


class CustomLogger:
    def __init__(
        self, name: str, output_dir: Optional[str] = None, logfile: Optional[str] = None
    ):
        self.name = name
        self.output_dir = output_dir
        self.logfile = logfile

    def get_logger(self):
        """
        :get_logger: Build a logger with customer formatter and handlers.

        :return: Logger instance
        """
        handlers = self._get_handlers()
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for handler in handlers:
            logger.addHandler(handler)
        return logger

    @staticmethod
    def get_colored_formatter(color: str = "green"):
        """
        Get colored logging formatter
        :param color: message's color
        :type color: str
        :return: logging.Formatter object
        """
        log_format = logging.Formatter(
            termcolor.colored(
                "[%(asctime)s] - [%(levelname)s] -  %(name)s -"
                " (%(filename)s).%(funcName)s(%(lineno)d)",
                color,
            )
            + "- %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return log_format

    def _get_handlers(self):
        handlers_list = []
        formatter = CustomLogger.get_colored_formatter()
        stream_handler = CustomLogger.get_stream_handler(formatter)
        handlers_list.append(stream_handler)
        if self.output_dir is not None:
            file_handler = self._get_file_handler(formatter)
            handlers_list.append(file_handler)
        return handlers_list

    @staticmethod
    def get_stream_handler(formatter: logging.Formatter):
        """
        build a stream handler
        :param formatter: message formatter
        :type formatter: logging.Formatter
        :return: logging.StreamHandler object
        """
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        return stream_handler

    def _get_file_handler(self, formatter: logging.Formatter):
        """
        We record only the warning and higher level
        :param formatter: formatter
        :type formatter: logging.Formatter
        :return:
        """
        if self.output_dir is not None:
            file_handler = logging.FileHandler(
                os.path.join(self.output_dir, self.logfile)  # type: ignore
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            return file_handler
        else:
            raise ValueError("You should define the output dir")
