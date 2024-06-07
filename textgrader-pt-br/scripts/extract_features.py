import pandas as pd
from settings import OUTPUT_DF
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download("punkt")

def word_count(x):
    """
    Extrai a quantidade de palavras
    """
    lista = word_tokenize(x)
    return len(lista)

def word_count_unique(x):
    """
    Extrai a quantidade de palavras únicas
    """
    lista = set(word_tokenize(x))
    return len(lista)

def sentence_count(x):
    """
    Extrai a quantidade de sentenças
    """
    lista = sent_tokenize(x)
    return len(lista)


def generate_word_sentence_features(df: pd.DataFrame, column='texto') -> pd.DataFrame:
    """
    Extrai algumas features básicas dos textos

    Esse método extrai alguma features dos textos, como contagem de palavras, contagem de palavras únicas
    e contagem de sentenças

    Args:
        df: datafame que contém as redações
        column: nome da coluna que conterá os textos
    """

    df['word_count'] = df[column].apply(lambda x: word_count(x))
    df['unique_word_count'] = df[column].apply(lambda x: word_count_unique(x))
    df['sentence_count'] = df[column].apply(lambda x: sentence_count(x))

    return df


def separate_train_test(df: pd.DataFrame) -> tuple:
    """
    Separa os conjuntos de treino e de teste, estratificando pelo tema,
    de modo a garantir que todos os temas possuam representação semelhante
    no treino e no teste

    Args:
        df: dataframe que iremos separar em treino e teste
    """
    X_train, x_test = train_test_split(df, test_size=0.2, random_state=0, stratify=df['tema'])

    X_train = X_train.reset_index()
    x_test = x_test.reset_index()

    return X_train, x_test


df_geral = pd.read_parquet(f"{OUTPUT_DF}/df_geral.parquet")
df = generate_word_sentence_features(df_geral)
df_train,df_test = separate_train_test(df)

df_train.to_parquet(f"{OUTPUT_DF}/df_train.parquet")
df_test.to_parquet(f"{OUTPUT_DF}/df_test.parquet")