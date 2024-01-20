import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize, sent_tokenize
from .config import * 
from xgboost import XGBRegressor
import pickle


def get_competencias(coluna : pd.Series) -> str:
    """
    Essa função obtém o nome da competência contida na coluna

    Args:
        coluna: coluna que contem os dicionários da competência

    Essa função pega a coluna com os vários dicionarios, verifica se todos os dicionários 
    se referem à uma mesma competência e caso todos os dicionários se refiream à uma única competência, 
    retorna a string contendo o nome dessa competência
    """

    competencias = coluna.transform(lambda x:x['competencia']).unique()

    if(len(competencias) != 1):
        raise('Ou não há competencias, ou há mais de uma onde deveria ter apenas uma')
        
    label_competencia = competencias[0]

    return label_competencia

def process_all(df_entrada,coluna = 'nota'):
    """
    Extrai a partr da coluna competências, os nomes e notas das diferentes competências

    A coluna competências, é composta de dicionários contendo diferentes informações, entre elas, 
    os nomes das competências e as notas obtidas nas competências, essa função realiza preprocessamentos 
    extraindo as notas obitdas nas diferentes competências dos dicionários contidos nessas competências

    Args:
        df_entrada: dataframe a partir do qual vamos extrair as competências e as notas
        coluna: nome da chave nos dicionários que contem a nota da competência 
        (por algum motivo, no grande conjunto 1, essa chave é 'nota' como esperado, mas nos outros 
        dois conjuntos essa chave é 'motivo', possível bug no crawler)
    """
    
    ## 'expande' o conteudo contido na coluna competencias, em que cada registro é uma lista
    ## de dicionários, (cada dicionário associado a uma competência), 
    ## para varias colunas, cada uma associada a uma competência em que o registro 
    ## corresponde a um único dicionário (o dicionário conterá o nome da competencia a nota atribuida)
    ## e o motivo pelo qual a nota foi atribuida
    df_competencias = pd.DataFrame(df_entrada['competencias'].to_list())
    
    lista_competencias = df_competencias

    for item in lista_competencias:
      
        competencia = get_competencias(df_competencias[item])

        ## obtém a nota da competencia presente na coluna, a armazena numa coluna que informará a nota da competencia
        df_entrada[f'{competencia}_nota'] = df_competencias[item].transform(lambda x:x[coluna])
       
    return df_entrada


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



def separate_train_test(df: pd.DataFrame) -> tuple:
    """
    Separa os conjuntos de treino e de teste, estratificando pelo tema, 
    de modo a garantir que todos os temas possuam representação semelhante
    no treino e no teste

    Args: 
        df: dataframe que iremos separar em treino e teste
    """
    X_train, x_test = train_test_split(df, test_size=0.2,random_state=0, stratify = df['tema'])
    
    X_train = X_train.reset_index()
    x_test = x_test.reset_index()
    
    return X_train, x_test


def fit_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza um pipeline de treino e teste dentro de um conjunto de textos

    Recebe um único conjunto com textos de treino e de teste concatenados, considerando esse conjunto 
    separa de volta os textos em treino e teste, treina o modelo com os textos de treino, 
    e usa esse modelo para realizar previsões no conjunto de teste 

    Args: 
        df: conjunto de redações
    """
    
    df_train = df[df['group'] == 'train'].drop(columns = ['group'])
    df_test = df[df['group'] == 'test'].drop(columns = ['group'])
    
    id_train = df_train[ID_VARS]
    X_train = df_train.drop(columns = EXCLUDE_COLS,errors = 'ignore')
    y_train = df_train[TARGETS_1].astype(float)
    
    ## treina o modelo 
    xgb = XGBRegressor()
    fittado = xgb.fit(X_train, y_train)
    
    id_test = df_test[ID_VARS]
    X_test = df_test.drop(columns = EXCLUDE_COLS,errors = 'ignore')
    y_test = df_test[TARGETS_1].astype(float)
   
    PRED_COLS = [col + f'_pred' for col in TARGETS_1]
    
    preds = pd.DataFrame()
    preds[ID_VARS] = id_test
    preds[PRED_COLS] =  xgb.predict(X_test)
    preds[PRED_COLS] = preds[PRED_COLS].astype(int)

    return preds


def separate_sets(df:pd.DataFrame) -> tuple:
    """
    Separa o conjunto de dados de acordo com os 3 tipos de texto e seleciona colunas apropriadas

    Separa o conjunto de dados de acordo com os 3 tipos de texto, e depois de separar, deixa 
    apenas os targets que são apropriados para aquele conjunto de textos 

    Args: 
        df: Dataframe que contém os três conjuntos de textos
    """

    primeiro_df = df[df['conjunto'] == 1]
    primeiro_df = primeiro_df.drop(columns = TARGETS_2 + TARGETS_3)

    segundo_df = df[df['conjunto'] == 2]
    segundo_df = segundo_df.drop(columns = TARGETS_1 + TARGETS_3)

    terceiro_df = df[df['conjunto'] == 3]
    terceiro_df = terceiro_df.drop(columns = TARGETS_1 + TARGETS_2)
    
    return primeiro_df,segundo_df,terceiro_df



def prepare_report_table(df_real: pd.DataFrame,df_pred : pd.DataFrame) -> pd.DataFrame:
    """
    Junta o conjunto de teste e as previsões feitas

    Juntamos o conjunto de treino com o conjunto de teste, fazemos um melt para
    transformar cada conceito em uma linha, e, com isso ficar mais fácil para realizar a avaliaçao 
    do desempenho preditivo em etapas futuras

    Args: 
        df_real: dataframe com o conjunto de teste
        df_pred: dataframe com as previsões realizadas tendo em mente
        o conjunto de teste
    """
    
    df_real = df_real[ID_VARS + TARGETS_1]
    df = pd.merge(df_real,df_pred, on =[ 'index','tema','conjunto'], suffixes = ['_real','_pred'])
    df = df.melt(id_vars = ['index','tema','conjunto'])
    df['valor'] = df['variable'].transform(lambda x:x.split('_')[-1])
    df['conceito'] = df['variable'].transform(lambda x:x.split('_')[0])
    df = df.drop(columns = 'variable')
    df = df.set_index(['index','tema','conjunto','conceito','valor']).unstack('valor').reset_index()
    df.columns = ['index','tema','conjunto','conceito','nota','previsao']

    return df



def vectorize_data(df: pd.DataFrame,model: pickle) -> pd.DataFrame:
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
