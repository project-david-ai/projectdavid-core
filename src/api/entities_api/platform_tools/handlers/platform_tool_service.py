import inspect
import threading
from typing import Any, Dict, Generator, Union

from entities_api.platform_tools.handlers.code_interpreter.code_execution_client import \
    StreamOutput
from entities_api.platform_tools.handlers.computer.shell_command_interface import \
    ShellCommandInterface
from entities_api.platform_tools.handlers.vector_store.vector_search_handler import \
    VectorSearchHandler
from entities_api.platform_tools.handlers.web.web_search_handler import \
    FirecrawlService

from src.api.entities_api.services.logging_service import LoggingUtility

logging_utility = LoggingUtility()


class PlatformToolService:
    function_handlers = {
        "code_interpreter": None,
        "web_search": None,
        "vector_store_search": None,
        "computer": None,
    }

    def __init__(self, base_url=None, api_key=None, assistant_id=None, thread_id=None):
        self._stream_output_handler = None
        self._web_search_handler = None
        self._vector_search_handler = None
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self._call_cache = {}
        self._cache_lock = threading.Lock()

    def _get_stream_output_handler(self) -> StreamOutput:
        """Lazy initialization of streaming handler"""
        if self._stream_output_handler is None:
            self._stream_output_handler = StreamOutput()
        return self._stream_output_handler

    def _get_web_search_handler(self):
        if self._web_search_handler is None:
            self._web_search_handler = FirecrawlService()
        return self._web_search_handler

    def _get_vector__search_handler(self):
        if self._vector_search_handler is None:
            self._vector_search_handler = VectorSearchHandler(assistant_id=self.assistant_id)
        return self._vector_search_handler

    def call_function(
        self, function_name: str, arguments: Dict[str, Any]
    ) -> Union[Dict, Generator[str, None, None]]:
        """
        Enhanced function executor with streaming support.
        Returns:
            - A generator for streaming outputs, or
            - A dict for static results.
        """
        if not isinstance(arguments, dict):
            logging_utility.error("Invalid arguments type: %s", type(arguments))
            return {"error": "Arguments must be a dictionary"}
        cache_key = None
        if function_name != "code_interpreter":
            try:
                cache_key = (function_name, frozenset(arguments.items()))
                with self._cache_lock:
                    if cache_key in self._call_cache:
                        return self._call_cache[cache_key]
            except TypeError as e:
                logging_utility.warning("Cache bypassed: %s", str(e))
        if self.function_handlers[function_name] is None:
            if function_name == "code_interpreter__":
                self.function_handlers[function_name] = (
                    self._get_stream_output_handler().stream_output
                )
            elif function_name == "web_search":
                self.function_handlers[function_name] = (
                    self._get_web_search_handler().search_orchestrator
                )
            elif function_name == "vector_store_search":
                self.function_handlers[function_name] = (
                    self._get_vector__search_handler().execute_search
                )
            elif function_name == "computer":
                shell_service = ShellCommandInterface(thread_id=self.thread_id, idle_timeout=5)
                self.function_handlers[function_name] = shell_service.run_commands
            else:
                return {"error": f"Unsupported function: {function_name}"}
        handler = self.function_handlers[function_name]
        try:
            result = handler(**arguments)
            if inspect.isgenerator(result):
                return result
            if cache_key is not None:
                with self._cache_lock:
                    self._call_cache[cache_key] = result
            return result
        except Exception as e:
            logging_utility.error("Execution failed: %s", str(e))
            return {"error": str(e)}

    def __del__(self):
        """Cleanup streaming resources"""
        if self._stream_output_handler:
            self._stream_output_handler.close()
