import os
import uuid
from random import randint, choice
from typing import ClassVar

import dotenv
import random
import aiosqlite
from faker import Faker
from datetime import timedelta


class MockConfig:
    client_num: ClassVar[int] = 100
    min_pets: ClassVar[int] = 1
    max_pets: ClassVar[int] = 3
    shortest_stay: ClassVar[int] = 1
    longest_stay: ClassVar[int] = 5
    min_stays: ClassVar[int] = 1
    max_stays: ClassVar[int] = 3

    @classmethod
    def load(cls):
        cls.client_num = int(os.getenv("CLIENT_NUM", cls.client_num))
        cls.min_pets = int(os.getenv("MIN_PETS", cls.min_pets))
        cls.max_pets = int(os.getenv("MAX_PETS", cls.max_pets))
        cls.shortest_stay = int(os.getenv("SHORTEST_STAY", cls.shortest_stay))
        cls.longest_stay = int(os.getenv("LONGEST_STAY", cls.longest_stay))
        cls.min_stays = int(os.getenv("MIN_STAYS", cls.min_stays))
        cls.max_stays = int(os.getenv("MAX_STAYS", cls.max_stays))

        cls.validate()

    @classmethod
    def validate(cls):
        if cls.client_num < 1:
            raise ValueError("Client num can not be less than one")
        if cls.min_pets < 0 or cls.shortest_stay < 0 or cls.min_stays < 0:
            raise ValueError(
                "Values min_pets, shortest_stay, min_stays can not be less than zero"
            )
        if (
            cls.min_pets > cls.max_pets
            or cls.shortest_stay > cls.longest_stay
            or cls.min_stays > cls.max_stays
        ):
            raise ValueError(
                "Min/shortest values should be less than or equal to max/longest values"
            )


###############################
#        Clients mock         #
###############################
async def fake_clients(connection: aiosqlite.Connection, faker: Faker) -> None:
    cursor = await connection.cursor()

    for _ in range(MockConfig.client_num):
        await cursor.execute(
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
    await connection.commit()
    await cursor.close()


###############################
#         Pets mock           #
###############################
async def fake_pets(connection: aiosqlite.Connection, faker: Faker) -> None:
    cursor = await connection.cursor()
    await cursor.execute("""SELECT clientId FROM Clients""")
    for client in await cursor.fetchall():
        pet_amount = randint(MockConfig.min_pets, MockConfig.max_pets)
        for _ in range(pet_amount):
            mof = choice(["male", "female"])
            await cursor.execute(
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
    await connection.commit()
    await cursor.close()


###############################
#         Stays mock          #
###############################
async def fake_stays(
    connection: aiosqlite.Connection,
    faker: Faker,
) -> None:
    cursor = await connection.cursor()
    await cursor.execute("SELECT * FROM pets")
    pets = await cursor.fetchall()
    stay_plans = ["Стандарт", "Премиум", "Люкс", "Оздоровительный"]

    stays = []
    for pet in pets:
        num_stays = random.randint(MockConfig.min_stays, MockConfig.max_stays)

        for _ in range(num_stays):
            check_in = faker.date_between(start_date="-1y", end_date="today")
            stay_duration = random.randint(
                MockConfig.shortest_stay, MockConfig.longest_stay
            )
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
    await cursor.executemany(
        """
        INSERT INTO Stays (stayId, clientId, petId, visit, leave, plan)
        VALUES (:stayId, :clientId, :petId, :visit, :leave, :plan)
    """,
        stays,
    )

    await connection.commit()
    await cursor.close()


###############################
#    Management Scripts       #
###############################
async def fill_db(connection: aiosqlite.Connection) -> None:
    dotenv.load_dotenv("configs/mock.properties")
    MockConfig.load()

    faker = Faker()

    await fake_clients(connection, faker)
    await fake_pets(connection, faker)
    await fake_stays(connection, faker)


async def clear_db(connection: aiosqlite.Connection) -> None:
    cursor = await connection.cursor()

    for idx in ["idx_pets_clientId", "idx_stays_clientId", "idx_stays_petId"]:
        await cursor.execute(f"DROP INDEX IF EXISTS {idx}")

    for table in ["Stays", "Pets", "Clients"]:
        await cursor.execute(f"DROP TABLE IF EXISTS {table}")

    await connection.commit()
    await cursor.close()
