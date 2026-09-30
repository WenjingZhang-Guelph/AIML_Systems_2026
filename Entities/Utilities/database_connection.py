########################################################################################################################
# database_connection.py
#
# Postgres connections for the processing scripts, with credentials read from
# Secrets/<name>.env rather than hardcoded — each dataset lives in its own
# container on its own port, so the same code has to reach three different
# databases.
#
# The class keeps a list of everything it hands out so close() can tear it all
# down in one go. Nothing here is a context manager and nothing closes itself, so
# a script that forgets to call close() leaks connections until the process exits
# — survivable for one-shot scripts, which is all this is used for.
#
# Please refer to the LICENSE and DISCLAIMER files for more information regarding the use and distribution of this code.
# By using this code, you agree to abide by the terms and conditions in those files.
#
# Author: Noah Subedar [https://github.com/noahsub]
########################################################################################################################

import os
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine
from dotenv import load_dotenv

from Managers.path_manager import get_project_root


class DatabaseConnection:
    host: str
    port: int
    username: str
    password: str
    connections: list[psycopg2.extensions.connection]
    cursors: list[psycopg2.extensions.cursor]
    engines: list[sqlalchemy.Engine]

    def __init__(self, database_name: str, env_file: str = None):
        """`database_name` is the Postgres database; `env_file` names the
        credentials file under Secrets/.

        The two are separate because they don't always match — MIMIC-IV's
        connection uses the database 'postgres' with credentials from
        'mimic-iv.env', for instance.

        Fails with a TypeError on int(None) if the env file is missing or lacks
        DB_PORT, which is a confusing way to learn you forgot to create it.
        """
        load_dotenv(get_project_root() / 'Secrets' / env_file)
        self.host = os.getenv("DB_HOST")
        self.port = int(os.getenv("DB_PORT"))
        self.username = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")
        self.database_name = database_name
        self.connections = []
        self.cursors = []
        self.engines = []

    def get_credentials(self):
        return self.username, self.password

    def get_connection(self) -> psycopg2.extensions.connection:
        """New psycopg2 connection, tracked so close() can clean it up."""
        user, password = self.get_credentials()

        connection = psycopg2.connect(
            dbname=self.database_name,
            user=user,
            password=password,
            host=self.host,
            port=self.port,
        )
        self.connections.append(connection)
        return connection

    def get_cursor(self, connection: psycopg2.extensions.connection) -> psycopg2.extensions.cursor:
        cursor = connection.cursor()
        self.cursors.append(cursor)
        return cursor

    def get_engine(self) -> sqlalchemy.Engine:
        """SQLAlchemy engine, for the pandas read_sql paths.

        Note the password goes into the URL unescaped — a password containing
        '@' or ':' will produce a malformed connection string.
        """
        user, password = self.get_credentials()
        connection_url = f"postgresql+psycopg2://{user}:{password}@{self.host}:{self.port}/{self.database_name}"
        engine = create_engine(connection_url)
        self.engines.append(engine)
        return engine

    def close(self):
        """Tear everything down, cursors first.

        Order matters — closing a connection out from under its cursors is what
        produces those "cursor already closed" errors during shutdown.
        """
        for cursor in self.cursors:
            cursor.close()
        self.cursors.clear()

        for connection in self.connections:
            connection.close()
        self.connections.clear()

        for engine in self.engines:
            engine.dispose()
        self.engines.clear()

    def __str__(self):
        # Prints the password in the clear. Debugging aid only — don't log this
        # anywhere that ends up in a shared file.
        return (
            f"{'HOST:':<16} {self.host}\n"
            f"{'PORT:':<16} {self.port}\n"
            f"{'USERNAME:':<16} {self.username}\n"
            f"{'PASSWORD:':<16} {self.password}\n"
            f"{'DATABASE:':<16} {self.database_name}"
        )