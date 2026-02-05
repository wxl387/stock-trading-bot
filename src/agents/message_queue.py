"""
Message Queue Module

SQLite-based persistent message queue for agent communication.
Thread-safe with connection pooling support.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import AgentMessage, AgentRole, MessagePriority, MessageType

logger = logging.getLogger(__name__)

# Module-level singleton
_message_queue: Optional["MessageQueue"] = None
_lock = threading.Lock()


def get_message_queue(db_path: Optional[str] = None) -> "MessageQueue":
    """
    Get or create the singleton MessageQueue instance.

    Args:
        db_path: Path to SQLite database file

    Returns:
        MessageQueue singleton instance
    """
    global _message_queue
    with _lock:
        if _message_queue is None:
            _message_queue = MessageQueue(db_path)
        return _message_queue


class MessageQueue:
    """
    SQLite-based persistent message queue for agent communication.

    Features:
    - Thread-safe operations with connection pooling
    - Persistent message storage
    - Support for conversation threading
    - Query by recipient, sender, processed status
    """

    DEFAULT_DB_PATH = "data/agent_messages.db"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the message queue.

        Args:
            db_path: Path to SQLite database file. Defaults to data/agent_messages.db
        """
        if db_path is None:
            # Use default path relative to project root
            from config.settings import DATA_DIR
            db_path = str(DATA_DIR / "agent_messages.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.Lock()

        # Initialize database schema
        self._init_db()

        logger.info(f"MessageQueue initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection.

        Returns:
            SQLite connection for current thread
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for better concurrency
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT NOT NULL,
                    requires_response INTEGER NOT NULL,
                    parent_message_id TEXT,
                    processed INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_recipient
                ON messages(recipient, processed)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_sender
                ON messages(sender)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_parent
                ON messages(parent_message_id)
            """)

    def enqueue(self, message: AgentMessage) -> None:
        """
        Add a message to the queue.

        Args:
            message: The message to enqueue
        """
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    id, sender, recipient, message_type, subject, content,
                    priority, timestamp, context, requires_response,
                    parent_message_id, processed, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.sender.value,
                    message.recipient.value,
                    message.message_type.value,
                    message.subject,
                    message.content,
                    message.priority.value,
                    message.timestamp.isoformat(),
                    json.dumps(message.context),
                    1 if message.requires_response else 0,
                    message.parent_message_id,
                    1 if message.processed else 0,
                    json.dumps(message.metadata),
                ),
            )

        logger.debug(f"Enqueued message {message.id}: {message.subject}")

    def _row_to_message(self, row: sqlite3.Row) -> AgentMessage:
        """Convert database row to AgentMessage."""
        return AgentMessage(
            id=row["id"],
            sender=AgentRole(row["sender"]),
            recipient=AgentRole(row["recipient"]),
            message_type=MessageType(row["message_type"]),
            subject=row["subject"],
            content=row["content"],
            priority=MessagePriority(row["priority"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            context=json.loads(row["context"]),
            requires_response=bool(row["requires_response"]),
            parent_message_id=row["parent_message_id"],
            processed=bool(row["processed"]),
            metadata=json.loads(row["metadata"]),
        )

    def get_messages_for_recipient(
        self,
        recipient: AgentRole,
        processed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AgentMessage]:
        """
        Get messages for a specific recipient.

        Args:
            recipient: The agent role to get messages for
            processed: Filter by processed status (None = all)
            limit: Maximum number of messages to return

        Returns:
            List of messages ordered by priority (desc) and timestamp (asc)
        """
        conn = self._get_connection()

        if processed is None:
            cursor = conn.execute(
                """
                SELECT * FROM messages
                WHERE recipient = ?
                ORDER BY priority DESC, timestamp ASC
                LIMIT ?
                """,
                (recipient.value, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM messages
                WHERE recipient = ? AND processed = ?
                ORDER BY priority DESC, timestamp ASC
                LIMIT ?
                """,
                (recipient.value, 1 if processed else 0, limit),
            )

        return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_messages_from_sender(
        self,
        sender: AgentRole,
        limit: int = 100,
    ) -> List[AgentMessage]:
        """
        Get messages from a specific sender.

        Args:
            sender: The agent role that sent the messages
            limit: Maximum number of messages to return

        Returns:
            List of messages ordered by timestamp (desc)
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM messages
            WHERE sender = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (sender.value, limit),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_conversation(
        self,
        agent1: AgentRole,
        agent2: AgentRole,
        limit: int = 50,
    ) -> List[AgentMessage]:
        """
        Get conversation history between two agents.

        Args:
            agent1: First agent role
            agent2: Second agent role
            limit: Maximum number of messages to return

        Returns:
            List of messages in chronological order
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM messages
            WHERE (sender = ? AND recipient = ?)
               OR (sender = ? AND recipient = ?)
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (agent1.value, agent2.value, agent2.value, agent1.value, limit),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_thread(self, message_id: str) -> List[AgentMessage]:
        """
        Get a conversation thread starting from a message.

        Args:
            message_id: ID of the root message

        Returns:
            List of messages in the thread, chronologically ordered
        """
        conn = self._get_connection()

        # Get the root message and all replies
        cursor = conn.execute(
            """
            WITH RECURSIVE thread AS (
                SELECT * FROM messages WHERE id = ?
                UNION ALL
                SELECT m.* FROM messages m
                JOIN thread t ON m.parent_message_id = t.id
            )
            SELECT * FROM thread ORDER BY timestamp ASC
            """,
            (message_id,),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_message_by_id(self, message_id: str) -> Optional[AgentMessage]:
        """
        Get a specific message by ID.

        Args:
            message_id: The message ID

        Returns:
            The message, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM messages WHERE id = ?",
            (message_id,),
        )
        row = cursor.fetchone()
        return self._row_to_message(row) if row else None

    def mark_processed(self, message_id: str) -> None:
        """
        Mark a message as processed.

        Args:
            message_id: ID of the message to mark
        """
        with self._transaction() as conn:
            conn.execute(
                "UPDATE messages SET processed = 1 WHERE id = ?",
                (message_id,),
            )
        logger.debug(f"Marked message {message_id} as processed")

    def get_unprocessed_count(self, recipient: AgentRole) -> int:
        """
        Get count of unprocessed messages for a recipient.

        Args:
            recipient: The agent role

        Returns:
            Number of unprocessed messages
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE recipient = ? AND processed = 0",
            (recipient.value,),
        )
        return cursor.fetchone()[0]

    def get_recent_messages(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[AgentMessage]:
        """
        Get recent messages within a time window.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of messages

        Returns:
            List of recent messages
        """
        conn = self._get_connection()
        cutoff = datetime.now().isoformat()
        cursor = conn.execute(
            """
            SELECT * FROM messages
            WHERE timestamp > datetime(?, '-' || ? || ' hours')
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (cutoff, hours, limit),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def delete_old_messages(self, days: int = 30) -> int:
        """
        Delete messages older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted messages
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM messages
                WHERE timestamp < datetime('now', '-' || ? || ' days')
                """,
                (days,),
            )
            deleted = cursor.rowcount

        logger.info(f"Deleted {deleted} messages older than {days} days")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get message queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        conn = self._get_connection()

        # Total messages
        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

        # By sender
        by_sender = {}
        for row in conn.execute(
            "SELECT sender, COUNT(*) as count FROM messages GROUP BY sender"
        ):
            by_sender[row["sender"]] = row["count"]

        # By message type
        by_type = {}
        for row in conn.execute(
            "SELECT message_type, COUNT(*) as count FROM messages GROUP BY message_type"
        ):
            by_type[row["message_type"]] = row["count"]

        # Unprocessed counts
        unprocessed = {}
        for row in conn.execute(
            """
            SELECT recipient, COUNT(*) as count FROM messages
            WHERE processed = 0 GROUP BY recipient
            """
        ):
            unprocessed[row["recipient"]] = row["count"]

        return {
            "total_messages": total,
            "by_sender": by_sender,
            "by_type": by_type,
            "unprocessed": unprocessed,
        }

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
