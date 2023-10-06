from datetime import datetime

from sqlalchemy import Column, Enum, String, Float, Date, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import re

Base = declarative_base()


class Origin(Enum):
    UOL = 'UOL'

    def __len__(self):
        return 10


class Theme(Base):
    __tablename__ = 'themes'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    date = Column(Date)
    context = Column(String)
    essays = relationship('Essay', back_populates='theme')

    @classmethod
    def convert_object(cls, name, date, context):
        return cls(
            name=name,
            date=format_date(date),
            context=context
        )


class Essay(Base):
    __tablename__ = 'essays'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(String)
    grade = Column(Float)
    origin = Column(String(10))  # Adicionado default value
    analysis = Column(String)
    date = Column(Date)
    theme_id = Column(Integer, ForeignKey('themes.id'))
    theme = relationship('Theme', back_populates='essays')

    @classmethod
    def convert_object(cls, essay, theme, date=datetime.today()):
        if isinstance(date, str):
            date = format_date(date)

        return cls(
            title=essay["titulo"],
            grade=essay["nota"],
            content=essay["texto"],
            analysis=essay["analise"],
            origin=Origin.UOL,
            theme=theme,
            date=date
        )

    @classmethod
    def  convert_object_uol(cls, essay, theme, date=datetime.today()):
        if isinstance(date, str):
            date = format_date(date)
        print(21)
        return cls(
            title=essay["titulo"],
            #grade=essay["nota"],
            content=preprocess_content(essay["texto"]),
            #analysis=essay["analise"],
            #origin=UOL,
            #theme=theme,
            #date=date
        )


def format_date(date):
    date_format = '%Y-%m-%dT%H:%M'
    try:
        data_datetime = datetime.strptime(date, date_format)
    except:
        mes, ano = date.split('/')
        mes_number = meses[mes]
        data_com_dia = f"{ano}-{mes_number}-01"  # Adicionando o dia 01 à data
        data_datetime = datetime.strptime(data_com_dia, '%Y-%m-%d')

    return data_datetime

def preprocess_content(content):
    return re.sub(CLEANR, '', content)
