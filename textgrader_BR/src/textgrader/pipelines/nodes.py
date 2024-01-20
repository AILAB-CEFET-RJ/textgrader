import pandas as pd
import numpy as np
from .functions import *
from .config import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import pickle

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
    df_primeiro_refined = process_all(df_primeiro)

    df_segundo = df_total[df_total['conjunto'] == 2]
    df_segundo_refined = process_all(df_segundo,coluna = 'motivo')

    df_terceiro = df_total[df_total['conjunto'] == 3]
    df_terceiro_refined = process_all(df_terceiro, coluna = 'motivo')

    ## junta os conjuntos e exclui algumas colunas inconvenientes
    df_geral = pd.concat([df_primeiro_refined,df_segundo_refined,df_terceiro_refined])
    df_geral = df_geral.drop(columns = ['nota','competencias','- Ruim_nota'],errors = 'ignore')


    ## o range dos targets 2 e 3 é de 0 a 10, com numeros quebrados, 
    ## multiplicamos por 100, esses targets e passamos todos os targets para valores inteiros,
    ## pois isso facilita o trabalho com kappa de cohen, mais a frente 

    df_geral[ALL_TARGETS] = df_geral[ALL_TARGETS].astype(float)
    df_geral[TARGETS_2] = df_geral[TARGETS_2] * 100
    df_geral[TARGETS_3] = df_geral[TARGETS_3] * 100

    return df_geral



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
    df_train,df_test = separate_train_test(df)

    return df_train, df_test


def fit_vectorizer(df_train: pd.DataFrame) -> pickle:
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
    tv = TfidfVectorizer(min_df = 10,max_features=1000)

    ## treina o vetorizador com o conjunto fornecido para treino
    vectorizer = tv.fit(lista)
    
    return vectorizer


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

    df_train = vectorize_data(df_train,model)
    df_test = vectorize_data(df_test,model)

    primeiro_treino, segundo_treino, terceiro_treino = separate_sets(df_train)
    primeiro_teste, segundo_teste, terceiro_teste = separate_sets(df_test)
    

    return primeiro_treino,primeiro_teste


def separate_all(df_treino,df_teste):

    primeiro_treino, segundo_treino, terceiro_treino = separate_sets(df_treino)
    primeiro_teste, segundo_teste, terceiro_teste = separate_sets(df_teste)
  
    
    return primeiro_treino,primeiro_teste


def fit_predict_both_ways(df_train: pd.DataFrame,df_test: pd.DataFrame) -> tuple:
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

    df_train['group'] = 'train'
    df_test['group'] = 'test'
    df = pd.concat([df_train,df_test])
    df = df.drop(columns = 'texto',errors = 'ignore')
    pred1 = fit_predict(df)
    pred2 = df.groupby(['tema']).apply(lambda x: fit_predict(x)).reset_index(drop = True)
  
    
    return pred1,pred2


def prepare_reports(df_real :pd.DataFrame,
                    df_pred_geral :pd.DataFrame,
                    df_pred_especifica:pd.DataFrame) -> pd.DataFrame:
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

    report_geral = prepare_report_table(df_real,df_pred_geral)
    report_especifica = prepare_report_table(df_real,df_pred_especifica)


    res = report_geral.groupby(['conceito']).apply(lambda x: cohen_kappa_score(x['nota'],x['previsao']))
    score_geral = pd.DataFrame(data = res).reset_index()
    score_geral.columns = ['conceito','score_geral']
 
    res = report_especifica.groupby(['conceito']).apply(lambda x: cohen_kappa_score(x['nota'],x['previsao']))
    score_especifica = pd.DataFrame(data = res).reset_index()
    score_especifica.columns = ['conceito','score_especifica']

    df_score = pd.merge(score_geral,score_especifica, on = ['conceito'])

    return df_score
 
 