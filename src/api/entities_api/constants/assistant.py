PLATFORM_TOOLS = [
    "code_interpreter",
    "web_search",
    "search_web_page",
    "scroll_web_page",
    "read_web_page",
    "perform_web_search",
    "vector_store_search",
    "computer",
    "delegate_research_task",
    "delegate_engineer_task",
    "read_scratchpad",
    "update_scratchpad",
    "append_scratchpad",
    "get_device_info",
    "search_inventory_by_group",
]


DEFAULT_MODEL = "llama3.1"


CODE_INTERPRETER_MESSAGE = "Return the tool output clearly and directly.If a file URL is present, include it as a user-friendly download link."
DEFAULT_REMINDER_MESSAGE = (
    "give the user the output from tool as advised in system message."
)

CODE_ANALYSIS_TOOL_MESSAGE = (
    "Advise the user that their file can be downloaded by clicking the link below."
)

BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Executes Python code in a sandbox environment and returns JSON output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Runs a semantic (embedding‑based) + keyword filter search over files that have been embedded into one or more vector stores.  Results come back as an array of metadata + score objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural‑language search text.",
                    },
                    "vector_store_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Vector store IDs to search.  Omit to use the assistant‑level binding supplied in `tools=[{"type":"file_search", "vector_store_ids":[…] }]`.',
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                        "description": "Maximum number of hits to return per store (maps to `top_k` in `search_vector_store`).",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional Qdrant payload filter object—same structure you already pass through to `search_vector_store()`.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs web searches with structured results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms using advanced operators",
                        "examples": [
                            "filetype:pdf cybersecurity report 2023",
                            "site:github.com AI framework",
                        ],
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer",
            "description": "This function acts as your personal computer—specifically a Linux computer with internet access. When you send a list of computer commands, it executes them in a recoverable computer session, streaming output continuously. It simulates a Linux terminal environment, allowing you to run commands as if you were using your personal Linux workstation. The thread ID is managed internally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of Linux computer commands to execute sequentially, as if you were typing directly into your personal computer's terminal.",
                    }
                },
                "required": ["commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_store_search",
            "description": "Qdrant-compatible semantic search with advanced filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": [
                            "basic_semantic",
                            "filtered",
                            "complex_filters",
                            "temporal",
                            "explainable",
                            "hybrid",
                        ],
                        "description": "Search methodology",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["chat", "documents", "memory"],
                        "description": "Data domain to search",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Qdrant-compatible filter syntax",
                        "examples": {
                            "temporal": {
                                "created_at": {"$gte": 1672531200, "$lte": 1704067200}
                            },
                            "boolean": {
                                "$or": [{"status": "active"}, {"priority": {"$gte": 7}}]
                            },
                        },
                    },
                    "score_boosts": {
                        "type": "object",
                        "description": "Field-specific score multipliers",
                        "examples": {"priority": 1.5, "relevance": 2.0},
                    },
                },
                "required": ["query", "search_type", "source_type"],
            },
        },
    },
]
BASE_ASSISTANT_INSTRUCTIONS = '🔹 **STRICT TOOL USAGE PROTOCOL**\nALL tool calls MUST follow EXACT structure:\n{\n  "name": "<tool_name>",\n  "arguments": {\n    "<param>": "<value>"\n  }\n}\n\n🔹 **FORMATTING FUNCTION CALLS**\n1. Do not format function calls\n2. Never wrap them in markdown backs ticks\n3. Call them in plain text or they will fail\n🔹 **CODE INTERPRETER**\n1. Always print output or script feedback\n2. For example:\n3. import math\n4. sqrt_144 = math.sqrt(144)\n\n5. print(sqrt_144)\n\nFILE GENERATION & INTERPRETER:\n• The sandbox has these external libraries available:\n  pandas, matplotlib, openpyxl, python-docx, seaborn, scikit-learn, and entities_common.\n• All assets generated should be rendered as .png by default unless otherwise specified.\n• When returning file links, present them as neat, clickable markdown links (e.g.,\n  [Example File](http://yourserver/v1/files/download?file_id=...)) to hide raw URLs.\n\n🔹 **VECTOR SEARCH COMMANDMENTS**\n1. Temporal filters use UNIX timestamps (numeric)\n2. Numeric ranges: $eq/$neq/$gte/$lte\n3. Boolean logic: $or/$and/$not\n4. Text matching: $match/$contains\n\n🔹 **SEARCH TYPE EXAMPLES**\n1. Basic Semantic Search:\n{\n  "name": "vector_store_search",\n  "arguments": {\n    "query": "Ransomware attack patterns",\n    "search_type": "basic_semantic",\n    "source_type": "chat"\n  }\n}\n\n2. Temporal Search:\n{\n  "name": "vector_store_search",\n  "arguments": {\n    "query": "Zero-day vulnerabilities",\n    "search_type": "temporal",\n    "source_type": "chat",\n    "filters": {\n      "created_at": {\n        "$gte": 1672531200,\n        "$lte": 1704067200\n      }\n    }\n  }\n}\n\n3. Complex Filter Search:\n{\n  "name": "vector_store_search",\n  "arguments": {\n    "query": "Critical security patches",\n    "search_type": "complex_filters",\n    "source_type": "chat",\n    "filters": {\n      "$or": [\n        {"priority": {"$gt": 7}},\n        {"category": "emergency"}\n      ]\n    }\n  }\n}\n\n4. Assistant-Centric Search:\n{\n  "name": "vector_store_search",\n  "arguments": {\n    "query": "Quantum-resistant key exchange",\n    "search_type": "complex_filters",\n    "source_type": "chat",\n    "filters": {\n      "$and": [\n        {"message_role": "assistant"},\n        {"created_at": {"$gte": 1700000000}}\n      ]\n    }\n  }\n}\n\n5. Hybrid Source Search:\n{\n  "name": "vector_store_search",\n  "arguments": {\n    "query": "NIST PQC standardization",\n    "search_type": "temporal",\n    "source_type": "both",\n    "filters": {\n      "$or": [\n        {"doc_type": "technical_spec"},\n        {"thread_id": "thread_*"}\n      ]\n    }\n  }\n}\n\n🔹 **WEB SEARCH RULES**\nOptimized Query Example:\n{\n  "name": "web_search",\n  "arguments": {\n    "query": "CRYSTALS-Kyber site:nist.gov filetype:pdf"\n  }\n}\n\n🔹 **QUERY OPTIMIZATION PROTOCOL**\n1. Auto-condense queries to 5-7 key terms\n2. Default temporal filter: last 12 months\n3. Prioritize chat sources 2:1 over documents\n\n🔹 **RESULT CURATION RULES**\n1. Hide results with similarity scores <0.65\n2. Convert UNIX timestamps to human-readable dates\n3. Suppress raw JSON unless explicitly requested\n\n🔹 **VALIDATION IMPERATIVES**\n1. Double-quotes ONLY for strings\n2. No trailing commas\n3. UNIX timestamps as NUMBERS (no quotes)\n4. Operators must start with $\n\n🔹 **TERMINATION CONDITIONS**\nABORT execution for:\n- Invalid timestamps (non-numeric/string)\n- Missing required params (query/search_type/source_type)\n- Unrecognized operators (e.g., gte instead of $gte)\n- Schema violations\n\n🔹 **ERROR HANDLING**\n- Invalid JSON → Abort and request correction\n- Unknown tool → Respond naturally\n- Missing parameters → Ask for clarification\n- Format errors → Fix before sending\n\n🔹 **OUTPUT FORMAT RULES**\n- NEVER use JSON backticks\n- ALWAYS use raw JSON syntax\n- Bold timestamps: **2025-03-01**\n- Example output:\n  {"name": "vector_store_search", "arguments": {\n    "query": "post-quantum migration",\n    "search_type": "basic_semantic",\n    "source_type": "chat"\n  }}\n\n🔹 **LATEX / MARKDOWN FORMATTING RULES:**\n- For mathematical expressions:\n  1. **Inline equations**: Wrap with single `$`\n     Example: `Einstein: $E = mc^2$` → Einstein: $E = mc^2$\n  2. **Display equations**: Wrap with double `$$`\n     Example:\n     $$F = ma$$\n\n- **Platform considerations**:\n  • On GitHub: Use `\\(...\\)` for inline and `\\[...\\]` for block equations.\n  • On MathJax-supported platforms: Use standard `$` and `$$` delimiters.\n\n- **Formatting requirements**:\n  1. Always include space between operators: `a + b` not `a+b`.\n  2. Use `\\mathbf{}` for vectors/matrices: `$\\mathbf{F} = m\\mathbf{a}$`.\n  3. Avoid code blocks unless explicitly requested.\n  4. Provide rendering notes when context is unclear.\n\n🔹 **ADDITIONAL INTERNAL USAGE AND REASONING PROTOCOL**\n1. Minimize Unnecessary Calls: Invoke external tools only when the request explicitly requires data beyond core knowledge (e.g., real-time updates or computations), to avoid needless conversational friction.\n2. Strict Protocol Adherence: Every tool call must follow the exact prescribed JSON structure, without embellishments, and only include necessary parameters.\n3. Judicious Reasoning First: In R1 (reasoning) mode, prioritize internal knowledge and reasoning; invoke external tools only if the request specifically demands updated or computed data.\n4. Butler-like Courtesy and Clarity: Maintain a refined, courteous, and efficient tone, reminiscent of a well-trained butler, ensuring interactions are respectful and precise.\n5. Error Prevention and Clarification: If ambiguity exists, ask for further clarification before invoking any external tool, ensuring accuracy and efficiency.\n6. Optimized Query and Invocation Practices: Auto-condense queries, use appropriate temporal filters, and adhere to all validation rules to prevent schema or format errors.\n7. Self-Validation and Internal Checks: Verify if a request falls within core knowledge before invoking tools to maintain a balance between internal reasoning and external tool usage.\n\nFailure to comply will result in system rejection.'
WEB_SEARCH_PRESENTATION_FOLLOW_UP_INSTRUCTIONS = "Presentation Requirements:\n1. Mobile-first layout\n2. Domain authority badges\n3. Preserved source URLs\n4. Hidden metadata annotations\nFormat Template:\n[Source](url)  \n![Favicon](favicon_url)  \nExcerpt...  \n---\n"
JSON_VALIDATION_PATTERN = (
    '\\{\\s*"name"\\s*:\\s*".+?"\\s*,\\s*"arguments"\\s*:\\s*\\{.*?\\}\\s*\\}'
)
