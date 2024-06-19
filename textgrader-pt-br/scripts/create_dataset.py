import glob
from settings import *
import pandas as pd
import numpy as np
from settings import DF_PATH, ALL_TARGETS, COLUMNS_REPLACE


def get_competencias(coluna: pd.Series) -> str:
    """
    Essa função obtém o nome da competência contida na coluna

    Args:
        coluna: coluna que contem os dicionários da competência

    Essa função pega a coluna com os vários dicionarios, verifica se todos os dicionários
    se referem à uma mesma competência e caso todos os dicionários se refiream à uma única competência,
    retorna a string contendo o nome dessa competência
    """

    competencias = coluna.transform(lambda x: x['competencia']).unique()

    if (len(competencias) != 1):
        raise ('Ou não há competencias, ou há mais de uma onde deveria ter apenas uma')

    label_competencia = competencias[0]

    return label_competencia


def process_all(df_entrada, coluna='nota'):
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
        df_entrada[f'{COLUMNS_REPLACE[competencia]}'] = df_competencias[item].transform(lambda x: x[coluna])

    return df_entrada


def read_files():
    json_files = glob.glob(DATA_PATH)

    file_names = [file.split("/")[-1] for file in json_files]
    print(file_names)

    lista_dfs = []

    for chave in file_names:
        chave = chave.replace(".json", "")
        numero = chave.split('-')[1]

        # Read the JSON file into a dataframe
        df = pd.read_json(f'../jsons/{chave}.json')

        # Display the dataframe
        print(f"Running to theme {numero}")
        df['tema'] = int(numero)
        lista_dfs.append(df)

    df_total = pd.concat(lista_dfs)

    df_total = df_total.drop(columns=['texto_comentado', 'cometarios', 'titulo', 'link'], errors='ignore')
    return df_total


def extract(df_total):
    ## marca os conjuntos
    df_total['conjunto'] = 2
    df_total['conjunto'] = np.where(df_total['tema'] <= 85, 1, df_total['conjunto'])
    df_total['conjunto'] = np.where(df_total['tema'] >= 137, 3, df_total['conjunto'])

    ## separa e refina cada conjunto separadamente
    #df_primeiro = df_total[df_total['conjunto'] == 1]
    df_geral = process_all(df_total)
    df_geral = df_geral.drop(columns=['nota', 'competencias'], errors='ignore')

    ## o range dos targets 2 e 3 é de 0 a 10, com numeros quebrados,
    ## multiplicamos por 100, esses targets e passamos todos os targets para valores inteiros,
    ## pois isso facilita o trabalho com kappa de cohen, mais a frente

    df_geral[ALL_TARGETS] = df_geral[ALL_TARGETS].astype(float)
    return df_geral


df_total = read_files()

df_total = df_total.assign(
    c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0,
    c6=0.0, c7=0.0, c8=0.0, c9=0.0, c10=0.0,
    c11=0.0, c12=0.0, c13=0.0, c14=0.0, c15=0.0,
)

df_geral = extract(df_total)

df_geral.to_parquet(f'{OUTPUT_DF}/df_geral.parquet')
df_geral.to_csv(f'{OUTPUT_DF}/df_geral.csv')

print(len(df_geral))
print("CREATE DATASET: DONE!")

