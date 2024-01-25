import psycopg2
from psycopg2 import sql
import os
import logging

logger = logging.getLogger(__name__)

def connect_db():
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

    try:
        conn = psycopg2.connect(**db_config)
        logger.info('Conectado ao banco de dados com sucesso!')
        return conn

    except Exception as ex:
        logger.exception(f"Erro ao tentar conectar no banco {ex}")
        return None

def save_data(essay, grade):
    try:
        conn = connect_db()
        cursor = conn.cursor()

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
        logger.info(f'Registro inserido com ID: {id_inserido}')

        conn.commit()

        cursor.close()
        conn.close()
    except Exception as ex:
        logger.exception(f"Erro ao tentar salvar dados no banco: {ex}")