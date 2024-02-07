import pandas as pd
import numpy as np
from .functions import *
from .config import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import pickle
from typing import Dict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import corpora
from gensim.models import LsiModel
import tensorflow_hub as hub
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
 
def get_text_jsons(dicio_jsons: dict) -> pd.DataFrame:
    """
        Junta todos os Jsons em um único arquivo, criando uma coluna para indicar o tema das redações

        Há uma pasta, a qual contém os Jsons obtidos através de um crawler que obtém 
        redações de cerca de 170 temas. Essa função pega cada json, abre o json como um dataset
        e marca o tema das redações numa coluna no dataset. Após fazer isso em todos os datasets, 
        a função junta todos os datasets em um único dataset contendo todas as redações. 

        Args: 
            dicio_jsons: dicionário em que as chaves são os nomes dos arquivos json, e os valores são
            métodos que acessam os arquivos jsons

    """
    lista_chaves = list(dicio_jsons.keys())

    lista_dfs = []

    for chave in lista_chaves:
        numero = chave.split('-')[1]
        df = dicio_jsons[chave]()
        df['tema'] = int(numero)
        lista_dfs.append(df)

    dfzao = pd.concat(lista_dfs)

    dfzao = dfzao.drop(columns = ['texto_comentado', 'cometarios', 'titulo', 'link'], errors = 'ignore')

    return dfzao


def preprocess_targets(df_total:pd.DataFrame) -> pd.DataFrame:
    """
        Extrai as notas que cada redação obteve em cada conceito

        Há três tipos de redação (possivelmente de acordo com a edição do enem), esses três tipos possuem 
        diferentes conceitos no qual estão sendo avaliadas (mais detalhes no apêndice). Nesse método, 
        separamos os três tipos, extraímos os targets de forma personalizada para da tipo de texto 
        (nos 3 tipos, os targets estão em 'estruturas de dados' diferentes)

        Args: 
            df_total: dataframe com todas as redações
    """

    ## marca os conjuntos
    df_total['conjunto'] = 2
    df_total['conjunto'] = np.where(df_total['tema'] <= 85, 1,df_total['conjunto'])
    df_total['conjunto'] = np.where(df_total['tema'] >= 137, 3,df_total['conjunto']) 

    ## separa e refina cada conjunto separadamente
    df_primeiro = df_total[df_total['conjunto'] == 1]
    df_geral = process_all(df_primeiro)
    df_geral = df_geral.drop(columns = ['nota','competencias','- Ruim_nota'],errors = 'ignore')


    ## o range dos targets 2 e 3 é de 0 a 10, com numeros quebrados, 
    ## multiplicamos por 100, esses targets e passamos todos os targets para valores inteiros,
    ## pois isso facilita o trabalho com kappa de cohen, mais a frente 

    df_geral[ALL_TARGETS] = df_geral[ALL_TARGETS].astype(float)
    
    return df_geral



def generate_basic_features(df: pd.DataFrame) -> tuple:
    """
        Extrai features básicas e separa os datasets em treino e teste

        Extrai features básicas do conjuntoke de dados, como a quantidade de palavras, 
        quantidade de palavras únicas e a quantidade de sentenças, além disso separa o dataset 
        em treino e teste de forma estratificada, buscando manter proporções iguais no treino e no teste
        para textos de cada um dos cerca de 170 temas que possuimos

        Args: 
            df: dataframe de redações com os targets extraidos 
    """

    df = generate_word_sentence_features(df)
    df_train,df_test = separate_train_test(df)

    return df_train, df_test

def create_versioned_dict(dfs_dict: Dict, create_folder=False) -> Dict:
    """Creates a dictionary in order to save files versioned in the correct pattern.

    Args:
        dfs_dict: Dictionary with datasets.
        run_date: reference date of the execution.

    Returns:
        dict: datasets and paths to be written

    """
  
    partitioned_dict = {}

    if create_folder:
        print('nao sei, taokey')
    else:
        for key, dataframe in dfs_dict.items():
            partitioned_dict[key] = dataframe

    return partitioned_dict



def fit_tf_idf(df_train: pd.DataFrame) -> pickle:
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

    dicio_tf_idf = {}

    for count in MAX_FEATURES:
        ## cria o vetorizador que utiliza TF-IDF
        tv = TfidfVectorizer(min_df = 10,max_features=count)

        ## treina o vetorizador com o conjunto fornecido para treino
        vectorizer_vez = tv.fit(lista)

        key = "TF_IDF_" + str(count)

        # prepara o dicionário para versionamento
        dicio_tf_idf[key] = vectorizer_vez
    
    return dicio_tf_idf



def fit_doc_2_vec(df_train: pd.DataFrame) -> pickle:
    lista_ensaios = list(df_train['texto'])
    essay_corpora = [word_tokenize(i) for i in lista_ensaios]

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(essay_corpora)]

    dicio_docs = {}

    for vector_size in MAX_FEATURES:
        model = Doc2Vec(documents, vector_size=vector_size)

        key = f'd2v_{vector_size}'
        dicio_docs[key] = model

    return dicio_docs


def fit_vectorizer(df_train):
    dicio_geral = {}


    dicio_tf_idf = fit_tf_idf(df_train)
    dicio_geral.update(dicio_tf_idf)

    dicio_doc_2_vec = fit_doc_2_vec(df_train)
    dicio_geral.update(dicio_doc_2_vec)

    try: 
        dicio_use = fit_use()
        dicio_geral.update(dicio_use)
    except:
        print('nao é possível trabalhar com o Universal sentence encoder')

    
    dicio_lsi = fit_lsi(df_train)
    dicio_geral.update(dicio_lsi)


    
    print(dicio_geral.keys())

    return dicio_geral


 
def fit_lsi(df_train):

    lista = list(df_train['texto'])

    texts = []
    for essay in lista:
        ## tokeniza o ensaio
        tokenized = word_tokenize(essay)
        ## adiciona a lista de palavras na lista de listas que representa o corpus
        texts.append(tokenized)

    ## constroi o dicionario a partir da lista de listas que representa o nosso corpus
    dictionary = corpora.Dictionary(texts)

    ## constroi a matriz termo-documento a partir da lista de listas que representa o nosso corpus
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]

    dicio_lsi = {}

    for vector_size in MAX_FEATURES:
        ## constroi o modelo LSI a partir da matriz termo-documento, do numero de tópicos e do dicionario
        lsamodel = LsiModel(doc_term_matrix,num_topics = vector_size, id2word = dictionary)
        key = f'LSI_{vector_size}'

        dicio_lsi[key] = lsamodel

    return dicio_lsi


 



def vectorize_all(df_train: pd.DataFrame,df_test: pd.DataFrame,model : pickle) -> tuple:
    """
        Método responsavel por vetorizar os conjuntos de treino e teste

        O método usa o vetorizador TD-IDF treinado anteriormente com o conjunto de treino 
        para vetorizar, tanto o conjunto de treino, quanto o conjunto de teste.
        Além de vetorizar esses conjuntos, o método também separa os três tipos de texto.
        Cabe pontuar que nessa exploração inicial, nosso pipeline somente está trabalhando com o primeiro tipo 
        de texto, mas estamos nos preparando para trabalhar com os outros tipos de texto 

        Args:
            df_train: conjunto de redações de treino
            df_test: conjunto de redações de teste
            model: vetorizador treinado anteriormente
    
    """

    df_train,df_train_list = vectorize_data(df_train,model)
    df_test,df_test_list = vectorize_data(df_test,model)

   

    return df_train_list,df_test_list


def fit_use():
    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/")

    dicio = {'USE':embedder}

    return dicio


def separate_all(df_treino,df_teste):

    primeiro_treino, segundo_treino, terceiro_treino = separate_sets(df_treino)
    primeiro_teste, segundo_teste, terceiro_teste = separate_sets(df_teste)
  
    
    return primeiro_treino,primeiro_teste


def general_fit(df_train_list: pd.DataFrame,df_test_list: pd.DataFrame,model_type) -> tuple:
    """
        Realiza o pipeline de treino e previsão tanto de forma geral, quanto de forma separada por tema

        O método realiza o pipeline de treino e previsão de duas formas: na primeira delas, para cada 
        tema, o método treina um modelo no conjunto de treino usando apenas as redações do tema
        e usa esse mesmo modelo para realizar previsoes no conjunto de teste, considerando apenas as
        redações do tema.    
        Na segunda delas o metodo treina um único modelo para todos os temas e depois utiliza esse 
        único modelo para prever todas as redações independente do tema

        Args:
            df_train: conjunto de redações de treino
            df_teste: conjunto de redaç~oes de teste
    
    """

    dict_pred = {}
 

    for key, value in df_train_list.items():
        df_train = df_train_list[key]()
        df_test = df_test_list[key]()

        new_key = f'{key}'

        df_train['group'] = 'train'
        df_test['group'] = 'test'
        df = pd.concat([df_train,df_test])
        df = df.drop(columns = 'texto',errors = 'ignore')
        pred = fit_predict_general(df,model_type)
      
        dict_pred[new_key] =  pred


    return dict_pred


def classification_fit(df_train_list: pd.DataFrame,df_test_list: pd.DataFrame):
    model = RandomForestClassifier()

    dict_resp = general_fit(df_train_list,df_test_list,model)

    return dict_resp


def regression_fit(df_train_list: pd.DataFrame,df_test_list: pd.DataFrame):
    model = RandomForestRegressor()

    dict_resp = general_fit(df_train_list,df_test_list,model)

    return dict_resp




def prepare_reports(df_real_lista :pd.DataFrame,
                    df_pred_geral_lista :pd.DataFrame,
                    df_pred_especifica_lista:pd.DataFrame) -> pd.DataFrame:
    """
    A partir das previsões com as duas metodologias usadas, prepara o report de resultados

    O método, pega as previsões feitas com cada modelo (tanto o geral, quanto o específico), 
    e junta com o dataset de teste, ,expandindo o dataframe fazendo um melt para transformar 
    cada conceito em uma linha separada, após isso, fazemos um groupby, para calcular, 
    para cada conceito, o kappa de cohen, com isso conseguimos ter uma visão, por conceito do 
    desempenho do modelo com o conjunto de treino geral e com o conjunto de treino específico

    Args:
        df_real: dataframe de teste
        df_pred_geral: dataframe com as previsões feitas usando um único modelo para todos 
        os temas
        df_pred_especifica: dataframe com as previsões feitas usando um modelo por tema
    """
    dicio = {}

    for key,value in df_real_lista.items():

        df_pred_geral = df_pred_geral_lista[key]()
        df_pred_especifica = df_pred_especifica_lista[key]()
        df_real = df_real_lista[key]()
        
        report_geral = prepare_report_table(df_real,df_pred_geral)
        report_especifica = prepare_report_table(df_real,df_pred_especifica)


        res = report_geral.groupby(['conceito']).apply(lambda x: cohen_kappa_score(x['nota'],x['previsao']))
        score_geral = pd.DataFrame(data = res).reset_index()
        score_geral.columns = ['conceito','score_geral']
    
        res = report_especifica.groupby(['conceito']).apply(lambda x: cohen_kappa_score(x['nota'],x['previsao']))
        score_especifica = pd.DataFrame(data = res).reset_index()
        score_especifica.columns = ['conceito','score_especifica']

        df_score = pd.merge(score_geral,score_especifica, on = ['conceito'])

        dicio[key] = df_score

    return dicio