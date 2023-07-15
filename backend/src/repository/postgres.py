from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

db_name = 'text-grader'
db_user = 'postgres'
db_pass = 'postgres'
db_host = 'db'
db_port = '5432'


def init_connection():
    print("-"*30)
    db_url = 'postgresql://postgres:postgres@db/text-grader'

    # Criar a engine de conexão
    engine = create_engine(db_url)

    # Criar uma sessão
    Session = sessionmaker(bind=engine)
    session = Session()

    # Exemplo de consulta
    print("=========== selecting =================")
    result = session.execute(text("SELECT * FROM text_data"))
    rows = result.fetchall()

    # Exibir os resultados
    for row in rows:
        print(row)

    # Fechar a sessão
    session.close()