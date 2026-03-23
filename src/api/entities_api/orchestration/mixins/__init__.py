# src/api/entities_api/orchestration/mixins/__init__.py

from src.api.entities_api.orchestration.mixins.assistant_cache_mixin import AssistantCacheMixin

# Non-MRO mixins (kept, but placed after ordered stack)
from src.api.entities_api.orchestration.mixins.client_factory_mixin import ClientFactoryMixin
from src.api.entities_api.orchestration.mixins.code_interpreter_mixin import CodeInterpreterMixin
from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)
from src.api.entities_api.orchestration.mixins.context_mixin import ContextMixin
from src.api.entities_api.orchestration.mixins.delegation_mixin import DelegationMixin
from src.api.entities_api.orchestration.mixins.device_inventory_mixin import NetworkInventoryMixin
from src.api.entities_api.orchestration.mixins.file_search_mixin import FileSearchMixin
from src.api.entities_api.orchestration.mixins.json_utils_mixin import JsonUtilsMixin
from src.api.entities_api.orchestration.mixins.native_exec_mixin import NativeExecMixin
from src.api.entities_api.orchestration.mixins.platform_tool_handlers_mixin import (
    PlatformToolHandlersMixin,
)
from src.api.entities_api.orchestration.mixins.scratchpad_mixin import ScratchpadMixin
from src.api.entities_api.orchestration.mixins.service_registry_mixin import ServiceRegistryMixin
from src.api.entities_api.orchestration.mixins.shell_execution_mixin import ShellExecutionMixin
from src.api.entities_api.orchestration.mixins.streaming_mixin import StreamingMixin
from src.api.entities_api.orchestration.mixins.tool_routing_mixin import ToolRoutingMixin
from src.api.entities_api.orchestration.mixins.web_search_mixin import WebSearchMixin

__all__ = [
    # Ordered to match _ProviderMixins MRO
    "ServiceRegistryMixin",
    "AssistantCacheMixin",
    "JsonUtilsMixin",
    "NativeExecMixin",
    "ContextMixin",
    "DelegationMixin",
    "ToolRoutingMixin",
    "PlatformToolHandlersMixin",
    "ConsumerToolHandlersMixin",
    "CodeInterpreterMixin",
    "ShellExecutionMixin",
    "FileSearchMixin",
    "WebSearchMixin",
    "ScratchpadMixin",
    "NetworkInventoryMixin",
    # Additional exports
    "ClientFactoryMixin",
    "StreamingMixin",
]
