from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Definindo a conexão
engine = create_engine('postgresql://postgres:julie123@localhost:5432/postgres')

# Definindo a classe de modelo
Base = declarative_base()

class Redacao(Base):
    __tablename__ = 'redacoes'
    id = Column(Integer, primary_key=True)
    corpo = Column(String)
    nota = Column(Float)

# Criando a tabela (se ela não existir)
Base.metadata.create_all(engine)

# Inserindo dados
Session = sessionmaker(bind=engine)
session = Session()

redacao = Redacao(corpo='Exemplo de corpo da redação', nota=8.5)
session.add(redacao)
session.commit()

# Recuperando dados
redacoes = session.query(Redacao).all()
for redacao in redacoes:
    print(redacao.corpo, redacao.nota)
