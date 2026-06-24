from db.mock import fill_db, clear_db

import uuid
import logging
from contextlib import asynccontextmanager

import aiosqlite

SQLITE_FILE_PATH = "bot_service.db"
logger = logging.getLogger("database")

MIGRATIONS = [
    # Service functional
    """
    CREATE TABLE IF NOT EXISTS Chats (
        userId TEXT NOT NULL PRIMARY KEY,
        sessionId TEXT NOT NULL,
        guardrailType TEXT NOT NULL
    )
    """,
    # Mock data
    """
    CREATE TABLE IF NOT EXISTS Clients (
        clientId TEXT PRIMARY KEY NOT NULL,
        fullName TEXT NOT NULL,
        phoneNumber TEXT NOT NULL,
        email TEXT,
        address TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Pets (
        petId TEXT PRIMARY KEY NOT NULL,
        clientId TEXT NOT NULL,
        name TEXT,
        nice INTEGER NOT NULL,
        mof TEXT NOT NULL,
        description TEXT,
        FOREIGN KEY (clientId) REFERENCES Clients(clientId)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Stays (
        stayId TEXT PRIMARY KEY NOT NULL,
        clientId TEXT NOT NULL,
        petId TEXT NOT NULL,
        visit TEXT NOT NULL,
        leave TEXT NOT NULL,
        plan TEXT NOT NULL,
        FOREIGN KEY (clientId) REFERENCES Clients(clientId),
        FOREIGN KEY (petId) REFERENCES Pets(petId)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pets_clientId ON Pets(clientId);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_stays_clientId ON Stays(clientId);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_stays_petId ON Stays(petId);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_client_id ON Chats(userId);
    """,
]


@asynccontextmanager
async def connect() -> aiosqlite.Connection:
    try:
        async with aiosqlite.connect(SQLITE_FILE_PATH) as connection:
            logger.debug("Given connection")
            yield connection
            logger.debug("Closed connection")
    except Exception as e:
        logger.exception(f"Trouble getting the connection: {e}")
        raise


async def init_db(mocked=True) -> None:
    async with connect() as connection:
        if mocked:
            await clear_db(connection)
            logger.debug("DB cleared for mocks")

        cursor = await connection.cursor()
        for migration in MIGRATIONS:
            await cursor.execute(migration)
        await connection.commit()
        await cursor.close()

        if mocked:
            await fill_db(connection)


async def create_session(user_id: int) -> None:
    async with connect() as connection:
        cursor = await connection.cursor()
        session_id = uuid.uuid4().hex
        await cursor.execute(
            """
            INSERT INTO Chats (userId, sessionId, guardrailType)
            VALUES (?, ?, 'large_language_model')
            ON CONFLICT (userId) DO UPDATE SET
                sessionId = excluded.sessionId,
                guardrailType = excluded.guardrailType
            """,
            (str(user_id), session_id),
        )
        await connection.commit()
        await cursor.close()


async def change_guardrail(user_id: int, gr_type: str) -> None:
    async with connect() as connection:
        cursor = await connection.cursor()
        await cursor.execute(
            "UPDATE Chats SET guardrailType = ? WHERE userId = ?",
            (gr_type, str(user_id)),
        )
        await connection.commit()
        await cursor.close()


async def has_session(user_id: int) -> bool:
    async with connect() as connection:
        cursor = await connection.cursor()
        await cursor.execute(
            "SELECT 1 FROM Chats WHERE userId = ?",
            (str(user_id),),
        )
        res = await cursor.fetchone()
        await cursor.close()
    return res is not None


async def get_guardrail(user_id: int) -> str:
    async with connect() as connection:
        cursor = await connection.cursor()
        await cursor.execute(
            "SELECT guardrailType FROM Chats WHERE userId = ?",
            (str(user_id),),
        )
        res = list(await cursor.fetchall())
        await cursor.close()

    logger.debug(f"Got guardrail '{res}' for user '{user_id}'")
    return res[0][0] if res else "disable"
