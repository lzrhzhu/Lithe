"""Optional capability bundles for lithe.

These are NOT part of the zero-I/O core engine; they are reusable, host-agnostic
capabilities a host opts into. The core never imports from here, so
``import lithe`` alone stays storage- and I/O-free — bundles are pulled in
only when a host needs them.
"""
from lithe.bundles.admin import (  # noqa: F401
    ToolPackage, check_packages, list_tool_packages_admin, list_tools_admin,
    package_meta, package_of, tool_categories,
)
from lithe.bundles.command import CommandRunner, register_command_tools  # noqa: F401
from lithe.bundles.download import (  # noqa: F401
    register_download_tools,
)
from lithe.bundles.documents import (  # noqa: F401
    register_document_tools,
)
from lithe.bundles.host import (  # noqa: F401
    AgentHost, DictToolAdapter, StoreSink, assemble_messages, undo_run,
)
from lithe.bundles.images import (  # noqa: F401
    register_image_tools,
)
from lithe.bundles.mcp import (  # noqa: F401
    MCPServerConfig, MCPManager, parse_servers,
)
from lithe.bundles.patch import (  # noqa: F401
    register_apply_patch_tool,
)
from lithe.bundles.providers import (  # noqa: F401
    PRESETS, apply_preset, get_preset, known_providers,
)
from lithe.bundles.sandbox import CodeRunner, register_code_tools  # noqa: F401
from lithe.bundles.skills import (  # noqa: F401
    PACKAGE_FILE, ROOT_PACKAGE, SKILL_MAIN, RemoteSkillSource, SkillLibrary,
    SkillPackages, register_skill_tool,
)
from lithe.bundles.store import (  # noqa: F401
    BlobStore, ConversationStore, JsonlRunStore, RunStore, StoredAction,
    StoredMessage, StoredRun,
)
from lithe.bundles.subagents import (  # noqa: F401
    SubagentEngine, SubagentRoster, SubagentSpec, make_delegate_tool,
    make_parallel_delegate_tool, register_delegate_tool, register_delegate_tools,
)
from lithe.bundles.todos import (  # noqa: F401
    JsonTodoStore, TodoStore, register_todo_tools, todos_block,
)
from lithe.bundles.workspace import Workspace, register_file_tools  # noqa: F401

__all__ = [
    "AgentHost", "BlobStore", "CodeRunner", "CommandRunner", "ConversationStore",
    "DictToolAdapter", "JsonTodoStore", "JsonlRunStore", "MCPManager",
    "MCPServerConfig", "PRESETS", "RunStore",
    "PACKAGE_FILE", "ROOT_PACKAGE", "SKILL_MAIN", "RemoteSkillSource",
    "SkillLibrary", "SkillPackages", "StoreSink", "StoredAction",
    "StoredMessage", "StoredRun",
    "SubagentEngine", "SubagentRoster", "SubagentSpec", "TodoStore",
    "ToolPackage", "Workspace",
    "apply_preset", "assemble_messages", "check_packages",
    "get_preset", "known_providers", "list_tool_packages_admin",
    "list_tools_admin", "make_delegate_tool", "make_parallel_delegate_tool",
    "package_meta", "package_of", "parse_servers", "register_apply_patch_tool",
    "register_code_tools", "register_command_tools", "register_download_tools",
    "register_delegate_tool", "register_delegate_tools", "register_document_tools",
    "register_file_tools",
    "register_image_tools", "register_skill_tool", "register_todo_tools",
    "tool_categories",
    "todos_block", "undo_run",
]
