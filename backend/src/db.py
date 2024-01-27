import psycopg2
from psycopg2 import sql
import os
import logging

class Database:
    logger = logging.getLogger(__name__)
    conn = None

    def __init__(self):
        database_name = os.environ.get('DATABASE_NAME', 'postgres')
        database_user = os.environ.get('DATABASE_USER')
        database_password = os.environ.get('DATABASE_PASSWORD')
        database_host = os.environ.get('DATABASE_HOST')
        database_port = os.environ.get('DATABASE_PORT', '5432')


        db_config = {
            'dbname': database_name,
            'user': database_user,
            'password': database_password,
            'host': database_host,
            'port': database_port
        }
        print(f"configs -> {db_config}")
        try:
            print("trying to connect to database")
            self.conn = psycopg2.connect(**db_config)
            self.logger.info('Conectado ao banco de dados com sucesso!')

        except Exception as ex:
            self.logger.exception(f"Erro ao tentar conectar no banco {ex}")

    def save_data(self, essay, grade):
        try:
            cursor = self.conn.cursor()

            tabela = 'essays'
            colunas = ['essay', 'grade']
            valores = (essay, grade)

            query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *;").format(
                sql.Identifier(tabela),
                sql.SQL(', ').join(map(sql.Identifier, colunas)),
                sql.SQL(', ').join([sql.Placeholder()] * len(colunas))
            )

            cursor.execute(query, valores)

            id_inserido = cursor.fetchone()[0]
            self.logger.info(f'Registro inserido com ID: {id_inserido}')

            self.conn.commit()

            cursor.close()
            self.conn.close()
        except Exception as ex:
            self.logger.exception(f"Erro ao tentar salvar dados no banco: {ex}")