"""Storage bundle: the persistence Protocol + a zero-database default backend.

Core lithe is storage-free; this optional bundle gives hosts a ready-made
conversation store. ``RunStore`` / ``ConversationStore`` / ``BlobStore`` are the
contracts; :class:`JsonlRunStore` implements all three with plain files + blob
spillover (no SQLite). Hosts wanting a DB implement the Protocols themselves.
"""
from lithe.bundles.store.jsonl import JsonlRunStore, revertible_actions  # noqa: F401
from lithe.bundles.store.protocol import (  # noqa: F401
    BlobStore, ConversationStore, RunStore, StoredAction, StoredMessage,
    StoredRun, action_to_row, message_to_row,
)

__all__ = [
    "BlobStore", "ConversationStore", "JsonlRunStore", "RunStore",
    "StoredAction", "StoredMessage", "StoredRun", "action_to_row",
    "message_to_row", "revertible_actions",
]
