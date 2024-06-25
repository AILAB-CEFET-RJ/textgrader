import os
import pandas as pd
import json


def preprocess_texto(texto):
    t = str.replace(texto, " q ", "que")
    t = str.replace(t, " v ", "5")

    return t.lower()


def jsons_to_csv(json_dir, csv_file, parquet_file_path=None):
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    df = pd.DataFrame(columns=["texto","nota", "labels"])

    count = 0
    # Lê cada arquivo JSON e adiciona seus dados à lista
    for file in files:
        file_path = os.path.join(json_dir, file)
        if count >= 3000:
            break
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            for obj in json_data:
                preprocessed = preprocess_texto(obj["texto"])
                count +=1
                df = df._append({"texto": preprocessed, "nota": obj["nota"], "labels":int(obj["nota"])}, ignore_index=True)


    # Salva o DataFrame como um arquivo CSV
    df.to_csv(csv_file, index=False)
    if parquet_file_path:
        df.to_parquet(parquet_file_path, index=False)


# Diretório contendo os arquivos JSON
json_directory = '../../../textgrader-pt-br/jsons'
# Nome do arquivo CSV a ser salvo
csv_file_path = 'output-3k.csv'
parquet_file_path = 'output-parquet-3k.parquet'

# Chama a função para converter JSONs em CSV
jsons_to_csv(json_directory, csv_file_path, parquet_file_path)

print(f'DataFrame salvo como {csv_file_path}')
