import logging
import sys


class LoggingUtility:
    def __init__(self, app=None, include_caller_info=True):
        self.app = app
        self.include_caller_info = include_caller_info
        self.logger = logging.getLogger(__name__)

        # Configure formatter based on caller info requirement
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        if self.include_caller_info:
            log_format = (
                "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
            )

        self.formatter = logging.Formatter(log_format)

        # Configure console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.DEBUG)
        self.console_handler.setFormatter(self.formatter)

        # Clear existing handlers
        if not self.logger.handlers:
            self.logger.addHandler(self.console_handler)

        self.logger.setLevel(logging.DEBUG)
        self.handler = self.console_handler
        self.level = logging.DEBUG

        if app is not None:
            self.init_app(app)

    def _get_log_args(self):
        """Helper to add stacklevel when caller info is enabled"""
        if self.include_caller_info:
            return {"stacklevel": 3} if sys.version_info >= (3, 8) else {}
        return {}

    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **{**self._get_log_args(), **kwargs})

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **{**self._get_log_args(), **kwargs})

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **{**self._get_log_args(), **kwargs})

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **{**self._get_log_args(), **kwargs})
        self.intercept_error_log(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **{**self._get_log_args(), **kwargs})
        self.intercept_critical_log(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self.logger.exception(message, *args, **kwargs)

    def intercept_error_log(self, message, *args, **kwargs):
        # Perform actions or send notifications for error logs
        print("Intercepted Error Log:")
        print(message % args)
        # Add your custom logic here

    def intercept_critical_log(self, message, *args, **kwargs):
        # Perform actions or send notifications for critical logs
        print("Intercepted Critical Log:")
        print(message % args)
