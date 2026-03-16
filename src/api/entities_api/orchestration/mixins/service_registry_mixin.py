"""
Runtime DI container – lazy-instantiates *internal* service classes and
keeps them in `self._services`.
"""

import inspect
import os
from functools import lru_cache

from dotenv import load_dotenv
from entities_api.cache.assistant_cache import AssistantCache
from entities_api.platform_tools.handlers.code_interpreter.code_execution_client import \
    StreamOutput
from entities_api.utils.conversation_truncator import ConversationTruncator
from projectdavid import Entity
from projectdavid.clients.actions_client import ActionsClient
from projectdavid.clients.assistants_client import AssistantsClient
from projectdavid.clients.files_client import FileClient
from projectdavid.clients.messages_client import MessagesClient
from projectdavid.clients.runs import RunsClient
from projectdavid.clients.threads_client import ThreadsClient
from projectdavid.clients.users_client import UsersClient
from projectdavid.clients.vectors import VectorStoreClient

from src.api.entities_api.orchestration.mixins.client_factory_mixin import \
    ClientFactoryMixin
from src.api.entities_api.services.logging_service import LoggingUtility

load_dotenv()
LOG = LoggingUtility()


class MissingParameterError(ValueError):
    pass


class ServiceRegistryMixin(ClientFactoryMixin):

    def _get_service(self, service_cls, *, custom_params=None):
        if not hasattr(self, "_services"):
            self._services = {}
        if service_cls not in self._services:
            try:
                obj = (
                    service_cls(*custom_params)
                    if custom_params
                    else service_cls(*self._resolve_init_parameters(service_cls))
                )
                self._services[service_cls] = obj
                LOG.debug("Instantiated %s", service_cls.__name__)
            except Exception as exc:
                LOG.error("Init failed for %s: %s", service_cls.__name__, exc, exc_info=True)
                raise
        return self._services[service_cls]

    def _invalidate_service_cache(self, service_cls):
        if hasattr(self, "_services") and service_cls in self._services:
            del self._services[service_cls]

    @lru_cache(maxsize=32)
    def _resolve_init_parameters(self, service_cls):
        sig = inspect.signature(service_cls.__init__)
        resolved = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(self, name):
                resolved.append(getattr(self, name))
            elif param.default is not inspect.Parameter.empty:
                resolved.append(param.default)
            else:
                raise MissingParameterError(f"{service_cls.__name__}: '{name}' not found")
        return tuple(resolved)

    @property
    def project_david_client(self) -> Entity:
        """
        Lazily-cached default Project-David SDK handle.

        Uses ADMIN_API_KEY + BASE_URL, exactly like the old BaseInference.
        """
        return self._get_project_david_client(
            api_key=os.getenv("ADMIN_API_KEY"),
            base_url=os.getenv("ASSISTANTS_BASE_URL"),
        )

    @property
    def conversation_truncator(self) -> ConversationTruncator:
        return self._get_service(ConversationTruncator)

    @property
    def user_client(self) -> UsersClient:
        return self._get_service(UsersClient)

    @property
    def assistant_service(self) -> AssistantsClient:
        return self._get_service(AssistantsClient)

    @property
    def thread_service(self) -> ThreadsClient:
        return self._get_service(ThreadsClient)

    @property
    def message_service(self) -> MessagesClient:
        return self._get_service(MessagesClient)

    @property
    def run_service(self) -> RunsClient:
        return self._get_service(RunsClient)

    @property
    def action_client(self) -> ActionsClient:
        return self._get_service(ActionsClient)

    @property
    def vector_store_service(self) -> VectorStoreClient:
        return self._get_service(VectorStoreClient)

    @property
    def files(self) -> FileClient:
        return self._get_service(FileClient)

    @property
    def assistant_cache(self) -> AssistantCache:
        return self._get_service(AssistantCache)

    @lru_cache(maxsize=128)
    def cached_user_details(self, user_id):
        """Thin wrapper around UsersClient to avoid redundant calls."""
        return self.user_client.get_user(user_id)

    @property
    def code_execution_client(self) -> StreamOutput:
        return self._get_service(StreamOutput)

    def setup_services(self):
        """Sub-classes may pre-register services here."""
        pass
