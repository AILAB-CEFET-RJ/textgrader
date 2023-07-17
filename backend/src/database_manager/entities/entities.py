from enum import Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Enum, String, Float, Date, Integer, ForeignKey
from datetime import datetime
from sqlalchemy.orm import relationship

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
        print("CONVERTING")
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


def format_date(date):
    date_format = '%Y-%m-%dT%H:%M'
    return datetime.strptime(date, date_format)