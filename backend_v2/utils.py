from nltk import word_tokenize, sent_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def generate_basic_features(df: pd.DataFrame) -> tuple:
    """
        Extrai features básicas e separa os datasets em treino e teste

        Extrai features básicas do conjunto de dados, como a quantidade de palavras, 
        quantidade de palavras únicas e a quantidade de sentenças, além disso separa o dataset 
        em treino e teste de forma estratificada, buscando manter proporções iguais no treino e no teste
        para textos de cada um dos cerca de 170 temas que possuimos

        Args: 
            df: dataframe de redações com os targets extraidos 
    """

    df = generate_word_sentence_features(df)

    return df



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

 
def generate_word_sentence_features(df : pd.DataFrame,column = 'texto') -> pd.DataFrame:
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


def fit_vectorizer(df_train: pd.DataFrame):
    """
    Método responsável por usar o conjunto de treino para treinar o vetorizador de textos 

    O método recebe um dataframe contendo as redações que foram selecionadas para o conjunto de treino
    e a partir dessas redações cria um vocabulário, o método então usa esse vocabulário
     para treinar um vetorizador do tipo TF-IDF 

    Args:
        df_train: conjunto de redações que foram selecionadas para treino
    """
    ## obtém a lista de textos do conjunto usado para treinar o vetorizador
    lista = list(df_train['texto'])

    ## cria o vetorizador que utiliza TF-IDF
    tv = TfidfVectorizer(max_features=1000)

    ## treina o vetorizador com o conjunto fornecido para treino
    vectorizer = tv.fit(lista)
    
    return vectorizer

def vectorize_data(df, model) -> pd.DataFrame:
    """
    Usa um vetorizador TF-IDF pré-treinado para obter a representação vetorial para o conjunto de dados

    Args:
        df: dataset que contém as redações
        model: vetorizador TF-IDF pré-treinado
    """
    lista = list(df['texto'])

    ## realiza efetivamente a vetorização, transformando em uma matriz esparsa
    X = model.transform(lista)

    # transforma a matriz esparsa em um dataframe organizado com as frequencias TF-IDF das palavras 
    df_vetorizado = pd.DataFrame(X.A, columns=model.get_feature_names_out())
    
    ## caso as redações contenham palavras que dão nome as features do datased
    ## (ex: texto, link, palavra) devemos removê-las

    colunas_a_remover = df.columns 

    df_vetorizado = df_vetorizado.drop(columns = colunas_a_remover, errors = 'ignore')

    df = pd.concat([df,df_vetorizado],axis = 1)
    df = df.drop(columns = ['texto'],errors = 'ignore')

    return df


def process_data(text):
    data = {'texto': [text]}
    df = pd.DataFrame(data)
    df_2 = generate_basic_features(df)

    lista = list(df_2['texto'])
    tv = TfidfVectorizer(max_features=1000)
    vectorizer = tv.fit(lista)

    X = vectorizer.transform(lista)
    df_vect = pd.DataFrame(X.A, columns=vectorizer.get_feature_names_out())
    df_vect = df_vect.drop(columns=df_2,errors = 'ignore')

    df = pd.concat([df_2,df_vect],axis = 1)
    df = df.drop(columns = ['texto'],errors = 'ignore')
