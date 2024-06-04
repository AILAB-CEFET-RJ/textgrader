import glob
from settings import *
import pandas as pd

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
    print(df)
    df['tema'] = int(numero)
    lista_dfs.append(df)

df_total = pd.concat(lista_dfs)

df_total = df_total.drop(columns = ['texto_comentado', 'cometarios', 'titulo', 'link'], errors = 'ignore')

df_total.to_csv(f'{OUTPUT_DF}/dataframe.csv', index=False)