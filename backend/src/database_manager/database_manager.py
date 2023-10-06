from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .entities.entities import Essay, Base, Theme
from datetime import datetime


def close_session(session):
    session.close()


class DatabaseManager:
    def __init__(self):
        self.engine = self.init_connection()

    def init_connection(self):
        print("-" * 30)
        db_url = 'postgresql://postgres:postgres@localhost:5432/text-grader'
        #todo colocar isso automático

        # Criar a engine de conexão
        engine = create_engine(db_url)
        return engine

    def create_tables(self):
        print("CREATE TABLES")
        session = self.create_session()
        Base.metadata.create_all(self.engine)
        session.close()

    def create_essays(self, essay, theme, date=datetime.today()):
        try:
            session = self.create_session()
            essay = Essay().convert_object(essay, theme, date)
            session.add(essay)
            session.commit()
            close_session(session)
            print("Created essay: {}".format(essay.title))
            return essay
        except Exception as e:
            error_message = traceback.format_exc()
            print('Error creating essays: ', error_message)

        session.commit()
        close_session(session)

    def create_theme(self, name, date, context):
        session = self.create_session()
        theme = Theme().convert_object(name, date, context)

        session.add(theme)

        session.commit()
        close_session(session)
        return theme


    def get(self):
        session = self.create_session()
        result = session.execute(text("SELECT * FROM text_data"))
        rows = result.fetchall()

        # Exibir os resultados
        for row in rows:
            print(row)

        close_session(session)

    def create_session(self):
        Session = sessionmaker(bind=self.engine)
        return Session()
