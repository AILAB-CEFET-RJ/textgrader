import pandas as pd 
import numpy as np
from settings import DF_PATH, ALL_TARGETS

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

# Leitura do arquivo CSV
df_total = pd.read_csv(DF_PATH)

    # Restante do código...
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