import uuid

import sqlite3


MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS chats (
        user_id TEXT NOT NULL PRIMARY KEY,
        session_id TEXT NOT NULL,
        guardrail_type TEXT NOT NULL
    )
    """
]


async def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect("bot_service.db")


async def init_db():
    connection = await get_db_connection()
    cursor = connection.cursor()
    for migration in MIGRATIONS:
        cursor.execute(migration)
    connection.commit()


async def create_session(user_id: int):
    connection = await get_db_connection()
    cursor = connection.cursor()
    session_id = uuid.uuid4().hex
    cursor.execute(
        f"""
        INSERT INTO chats (user_id, session_id, guardrail_type)
        VALUES ('{user_id}', '{session_id}', 'large_language_model')
        ON CONFLICT (user_id) DO UPDATE SET
            session_id = excluded.session_id,
            guardrail_type = excluded.guardrail_type
        """,
    )
    connection.commit()


async def change_guardrail(user_id: int, gr_type: str):
    connection = await get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        f"""
        UPDATE chats
        SET guardrail_type = '{gr_type}'
        WHERE user_id = '{user_id}'
        """,
    )
    connection.commit()
