from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .entities.entities import Essay, Base, Theme
import traceback

def close_session(session):
    session.close()


class DatabaseManager:
    def __init__(self):
        self.engine = self.init_connection()

    def init_connection(self):
        db_user = 'postgres'
        db_password = 'postgres'
        db_host = '172.17.0.1'
        db_port = '5432'
        db_name = 'text-grader'

        # Cria a string de conexão
        db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

        #db_url = 'postgresql://postgres:postgres@localhost:5432/text-grader?sslmode=disable'
        #todo colocar isso automático

        # Criar a engine de conexão
        engine = create_engine(db_url, pool_size=10, max_overflow=20)
        return engine

    def create_tables(self):
        session = self.create_session()
        Base.metadata.create_all(self.engine)
        session.close()

    def create_essays(self, essay, theme, origin, date=datetime.today()):
        try:
            print(1)
            session = self.create_session()
            print(2)
            essay = Essay().convert_object(essay, theme, origin, date)
            print(3)
            session.add(essay)
            print(4)
            session.commit()
            print(5)
            close_session(session)
            print("Created essay: {}".format(essay.title))
            return essay
        except Exception as e:
            error_message = traceback.format_exc()
            print('Error creating essays: ', error_message)


    def create_theme(self, name, date, context):
        session = self.create_session()
        theme = Theme().convert_object(name, date, context)

        session.add(theme)

        session.commit()
        close_session(session)

        print("Created: {}".format(name))
        return theme


    def get(self):
        session = self.create_session()
        print("=========== selecting =================")
        result = session.execute(text("SELECT * FROM text_data"))
        rows = result.fetchall()

        # Exibir os resultados
        for row in rows:
            print(row)

        close_session(session)


    def get_theme_by_name(self, name):
        try:
            session = self.create_session()
            theme = session.query(Theme).filter_by(name=name).first()
            return theme
        except Exception as e:
            print("Ocorreu um erro:", e)
        finally:
            session.close()


    def create_session(self):
        Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        return Session()
