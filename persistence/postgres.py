"""PostgreSQL persistence layer.

Implements the ChatRepository protocol: chat history, per-session contacts,
identity mapping, and cross-session long-term memory. Self-migrating.
"""
from __future__ import annotations

import logging
from datetime import datetime

import psycopg2

from core.interfaces import ContactInfo, Turn, UserProfile

logger = logging.getLogger(__name__)


class PostgresDatabase:
    """Persists chat turns, contacts, and user profiles to Postgres."""

    # ---------- schema ----------
    CREATE_CHAT_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL DEFAULT 'default',
            user_query TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    MIGRATE_SESSION_SQL = """
        ALTER TABLE chat_history
        ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT 'default';
    """
    CREATE_CHAT_SESSION_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history (session_id);
    """
    ADD_USER_ID_SQL = """
        ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS user_id TEXT;
    """
    CREATE_CHAT_USER_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history (user_id);
    """
    CREATE_CONTACT_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS user_contacts (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL UNIQUE,
            name TEXT,
            email TEXT,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    CREATE_PROFILE_SQL = """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            long_term_summary TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    CREATE_SESSION_MAP_SQL = """
        CREATE TABLE IF NOT EXISTS session_user_map (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """

    # ---------- queries ----------
    UPSERT_CONTACT_SQL = """
        INSERT INTO user_contacts (session_id, name, email, phone)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (session_id) DO UPDATE SET
            name  = COALESCE(EXCLUDED.name,  user_contacts.name),
            email = COALESCE(EXCLUDED.email, user_contacts.email),
            phone = COALESCE(EXCLUDED.phone, user_contacts.phone),
            updated_at = CURRENT_TIMESTAMP;
    """
    UPSERT_PROFILE_SQL = """
        INSERT INTO user_profiles (user_id, name, email, phone)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            name  = COALESCE(EXCLUDED.name,  user_profiles.name),
            email = COALESCE(EXCLUDED.email, user_profiles.email),
            phone = COALESCE(EXCLUDED.phone, user_profiles.phone),
            updated_at = CURRENT_TIMESTAMP;
    """
    UPSERT_SUMMARY_SQL = """
        INSERT INTO user_profiles (user_id, long_term_summary)
        VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            long_term_summary = EXCLUDED.long_term_summary,
            updated_at = CURRENT_TIMESTAMP;
    """
    BIND_SESSION_SQL = """
        INSERT INTO session_user_map (session_id, user_id) VALUES (%s, %s)
        ON CONFLICT (session_id) DO UPDATE SET user_id = EXCLUDED.user_id;
    """
    BACKFILL_SQL = ("UPDATE chat_history SET user_id = %s "
                    "WHERE session_id = %s AND user_id IS NULL;")
    FIND_USER_BY_CONTACT_SQL = """
        SELECT user_id FROM user_profiles
        WHERE (%s IS NOT NULL AND email = %s)
           OR (%s IS NOT NULL AND phone = %s)
        LIMIT 1;
    """

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL is missing.")
        self._database_url = database_url
        self._conn = None
        self._ensure_tables()

    # ---------- setup ----------

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._database_url)
        return self._conn

    def _ensure_tables(self) -> None:
        statements = [
            self.CREATE_CHAT_TABLE_SQL,
            self.MIGRATE_SESSION_SQL,
            self.CREATE_CHAT_SESSION_INDEX_SQL,
            self.ADD_USER_ID_SQL,
            self.CREATE_CHAT_USER_INDEX_SQL,
            self.CREATE_CONTACT_TABLE_SQL,
            self.CREATE_PROFILE_SQL,
            self.CREATE_SESSION_MAP_SQL,
        ]
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            conn.commit()
            logger.info("Database tables ready (chat_history, user_contacts, user_profiles, session_user_map)")
        except Exception as e:
            conn.rollback()
            logger.error("Failed to prepare tables: %s", e)
            raise

    # ---------- chat history ----------

    def save_turn(self, session_id: str, user_query: str, assistant_response: str,
                  user_id: str | None = None) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chat_history (session_id, user_id, user_query, assistant_response) "
                    "VALUES (%s, %s, %s, %s);",
                    (session_id, user_id, user_query, assistant_response),
                )
            conn.commit()
            return True
        except Exception as e:
            self._safe_rollback()
            logger.error("DB save failed: %s", e)
            return False

    def get_history(self, session_id: str, limit: int = 50) -> list[Turn]:
        """Newest `limit` turns of a session, returned in chronological order."""
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT user_query, assistant_response, created_at FROM chat_history "
                    "WHERE session_id = %s ORDER BY created_at DESC LIMIT %s;",
                    (session_id, limit),
                )
                rows = cursor.fetchall()
            return [Turn(user_query=r[0], assistant_response=r[1], session_id=session_id,
                         created_at=r[2]) for r in reversed(rows)]
        except Exception as e:
            logger.error("DB fetch failed: %s", e)
            return []

    def get_user_turns(self, user_id: str, limit: int = 12) -> list[Turn]:
        """Newest turns across every session of this user (chronological, with session ids)."""
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT session_id, user_query, assistant_response, created_at "
                    "FROM chat_history WHERE user_id = %s "
                    "ORDER BY created_at DESC LIMIT %s;", (user_id, limit))
                rows = cursor.fetchall()
            return [Turn(session_id=r[0], user_query=r[1], assistant_response=r[2],
                         created_at=r[3]) for r in reversed(rows)]
        except Exception as e:
            logger.error("User turns fetch failed: %s", e)
            return []

    def get_user_history(self, user_id: str, limit: int = 6,
                         exclude_session: str | None = None) -> list[tuple[str, str]]:
        """Most recent turns from ALL this user's PREVIOUS sessions (chronological)."""
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT user_query, assistant_response FROM chat_history "
                    "WHERE user_id = %s AND (%s IS NULL OR session_id <> %s) "
                    "ORDER BY created_at DESC LIMIT %s;",
                    (user_id, exclude_session, exclude_session, limit))
                rows = cursor.fetchall()
            return list(reversed(rows))
        except Exception as e:
            logger.error("User history fetch failed: %s", e)
            return []

    # ---------- per-session contacts ----------

    def upsert_contact(self, session_id: str, contact: ContactInfo) -> bool:
        """Insert or update. NULL fields never overwrite previously stored values."""
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(self.UPSERT_CONTACT_SQL,
                               (session_id, contact.name, contact.email, contact.phone))
            conn.commit()
            return True
        except Exception as e:
            self._safe_rollback()
            logger.error("Contact save failed: %s", e)
            return False

    def get_contact(self, session_id: str) -> ContactInfo | None:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT name, email, phone FROM user_contacts WHERE session_id = %s;",
                    (session_id,))
                row = cursor.fetchone()
            if not row:
                return None
            return ContactInfo(name=row[0], email=row[1], phone=row[2])
        except Exception as e:
            logger.error("Contact fetch failed: %s", e)
            return None

    def get_all_contacts(self) -> list[ContactInfo]:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute("SELECT name, email, phone FROM user_contacts ORDER BY updated_at DESC;")
                rows = cursor.fetchall()
            return [ContactInfo(name=r[0], email=r[1], phone=r[2]) for r in rows]
        except Exception as e:
            logger.error("Contacts fetch failed: %s", e)
            return []

    # ---------- identity ----------

    def find_user_by_contact(self, email: str | None = None,
                             phone: str | None = None) -> str | None:
        email = email.strip().lower() if email else None
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(self.FIND_USER_BY_CONTACT_SQL, (email, email, phone, phone))
                row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error("find_user_by_contact failed: %s", e)
            return None

    def upsert_profile(self, user_id: str, contact: ContactInfo) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(self.UPSERT_PROFILE_SQL,
                               (user_id, contact.name, contact.email, contact.phone))
            conn.commit()
            return True
        except Exception as e:
            self._safe_rollback()
            logger.error("Profile save failed: %s", e)
            return False

    def get_profile(self, user_id: str) -> UserProfile | None:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT user_id, name, email, phone, long_term_summary, created_at, updated_at "
                    "FROM user_profiles WHERE user_id = %s;", (user_id,))
                row = cursor.fetchone()
            if not row:
                return None
            return UserProfile(user_id=row[0], name=row[1], email=row[2], phone=row[3],
                               long_term_summary=row[4] or "",
                               created_at=str(row[5]), updated_at=str(row[6]))
        except Exception as e:
            logger.error("Profile fetch failed: %s", e)
            return None

    def get_all_profiles(self) -> list[UserProfile]:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT user_id, name, email, phone, long_term_summary, created_at, updated_at "
                    "FROM user_profiles ORDER BY updated_at DESC;")
                rows = cursor.fetchall()
            return [UserProfile(user_id=r[0], name=r[1], email=r[2], phone=r[3],
                                long_term_summary=r[4] or "",
                                created_at=str(r[5]), updated_at=str(r[6])) for r in rows]
        except Exception as e:
            logger.error("Profiles fetch failed: %s", e)
            return []

    # ---------- session <-> user binding ----------

    def bind_session(self, session_id: str, user_id: str) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(self.BIND_SESSION_SQL, (session_id, user_id))
            conn.commit()
            return True
        except Exception as e:
            self._safe_rollback()
            logger.error("Session bind failed: %s", e)
            return False

    def get_session_user(self, session_id: str) -> str | None:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_id FROM session_user_map WHERE session_id = %s;",
                               (session_id,))
                row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error("Session-user fetch failed: %s", e)
            return None

    def backfill_session_user(self, session_id: str, user_id: str) -> None:
        """Attach this session's earlier anonymous turns to the user once identity is known."""
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(self.BACKFILL_SQL, (user_id, session_id))
            conn.commit()
        except Exception as e:
            self._safe_rollback()
            logger.error("Backfill failed: %s", e)

    # ---------- long-term memory ----------

    def update_long_term_summary(self, user_id: str, summary: str) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(self.UPSERT_SUMMARY_SQL, (user_id, summary))
            conn.commit()
            return True
        except Exception as e:
            self._safe_rollback()
            logger.error("Summary save failed: %s", e)
            return False

    # ---------- lifecycle ----------

    def _safe_rollback(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.rollback()

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("DB connection closed")
