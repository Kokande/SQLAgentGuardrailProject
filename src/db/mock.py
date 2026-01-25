import os
import uuid
from typing import Optional
from random import randint, choice

import dotenv
import random
import sqlite3
from faker import Faker
from pydantic import BaseModel
from datetime import timedelta


class MockConfig(BaseModel):
    client_num: int
    min_pets: Optional[int] = 1
    max_pets: int
    shortest_stay: Optional[int] = 1
    longest_stay: int
    min_stays: Optional[int] = 1
    max_stays: int


async def fill_db(connection: sqlite3.Connection) -> None:
    dotenv.load_dotenv("configs/mock.properties")
    mock_cfg = MockConfig(
        client_num=os.getenv("CLIENT_NUM"),
        min_pets=os.getenv("MIN_PETS", 1),
        max_pets=os.getenv("MAX_PETS", 1),
        shortest_stay=os.getenv("SHORTEST_STAY", 1),
        longest_stay=os.getenv("LONGEST_STAY"),
        min_stays=os.getenv("MIN_STAYS", 1),
        max_stays=os.getenv("MAX_STAYS"),
    )

    faker = Faker()

    await fake_clients(connection, faker, mock_cfg)
    await fake_pets(connection, faker, mock_cfg)


# Clients mock
async def fake_clients(
    connection: sqlite3.Connection, faker: Faker, mock_cfg: MockConfig
) -> None:
    cursor = connection.cursor()

    for _ in range(mock_cfg.client_num):
        cursor.execute(
            """
            INSERT INTO Clients (clientId, fullName, phoneNumber, email, address)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                faker.name(),
                faker.phone_number(),
                faker.email(),
                faker.address(),
            ),
        )
    connection.commit()
    cursor.close()


# Pets mock
async def fake_pets(
    connection: sqlite3.Connection, faker: Faker, mock_cfg: MockConfig
) -> None:
    cursor = connection.cursor()
    cursor.execute("""SELECT clientId FROM Clients""")
    for client in cursor.fetchall():
        pet_amount = randint(mock_cfg.min_pets, mock_cfg.max_pets)
        for _ in range(pet_amount):
            mof = choice(["male", "female"])
            cursor.execute(
                """
                    INSERT INTO Pets (petId, clientId, name, nice, mof, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    client[0],
                    faker.name_male() if mof == "male" else faker.name_female(),
                    randint(0, 1),
                    mof,
                    choice(
                        [
                            "Любит рыбу",
                            "Не любит браться на ручки",
                            "Хозяева часто отдыхают на море и оставляют у нас",
                            "Не любит других животных",
                            "Любит расчёску",
                            "Любит активные игры",
                        ]
                    ),
                ),
            )
    connection.commit()
    cursor.close()


# Stays mock
async def fake_stays(
    connection: sqlite3.Connection, faker: Faker, mock_cfg: MockConfig
) -> None:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM pets")
    pets = cursor.fetchall()
    stay_plans = ["Стандарт", "Премиум", "Люкс", "Оздоровительный"]

    stays = []
    for pet in pets:
        num_stays = random.randint(mock_cfg.min_stays, mock_cfg.max_stays)

        for _ in range(num_stays):
            check_in = faker.date_between(start_date="-1y", end_date="today")
            stay_duration = random.randint(1, mock_cfg.longest_stay)
            check_out = check_in + timedelta(days=stay_duration)

            stay = {
                "stayId": str(uuid.uuid4()),
                "clientId": pet[1],
                "petId": pet[0],
                "visit": check_in.isoformat(),
                "leave": check_out.isoformat(),
                "plan": random.choice(stay_plans),
            }
            stays.append(stay)

    # Insert stays
    cursor.executemany(
        """
        INSERT INTO Stays (stayId, clientId, petId, visit, leave, plan)
        VALUES (:stayId, :clientId, :petId, :visit, :leave, :plan)
    """,
        stays,
    )

    connection.commit()
    cursor.close()


async def clear_db(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()

    for idx in ["idx_pets_clientId", "idx_stays_clientId", "idx_stays_petId"]:
        cursor.execute(f"DROP INDEX IF EXISTS {idx}")

    for table in ["Stays", "Pets", "Clients"]:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    connection.commit()
    cursor.close()
