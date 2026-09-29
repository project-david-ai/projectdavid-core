from entities_api.platform_tools.definitions.scratch_pad.append_scratchpad import (
    append_scratchpad,
)
from entities_api.platform_tools.definitions.scratch_pad.read_scratchpad import (
    read_scratchpad,
)
from entities_api.platform_tools.definitions.web_search.perform_web_search import (
    perform_web_search,
)
from entities_api.platform_tools.definitions.web_search.read_web_page import (
    read_web_page,
)
from entities_api.platform_tools.definitions.web_search.scroll_web_page import (
    scroll_web_page,
)
from entities_api.platform_tools.definitions.web_search.search_web_page import (
    search_web_page,
)

# Assistant-level capability declaration.
#
# Platform web functions MUST NOT be persisted individually as custom
# function tools. The web_search capability is expanded by Core at runtime
# into perform_web_search/read_web_page/search_web_page/scroll_web_page.
# Research workers execute multi-step tool chains internally.
# One turn is insufficient for read -> search -> read -> verify -> append.
RESEARCH_WORKER_MAX_TURNS = 8


WORKER_TOOLS = [
    perform_web_search,
    read_web_page,
    search_web_page,
    scroll_web_page,
    read_scratchpad,
    append_scratchpad,
]
