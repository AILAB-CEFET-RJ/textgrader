import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split


def preprocess_texto(texto):
    t = str.replace(texto, " q ", "que")
    t = str.replace(t, " v ", "5")

    return t.lower()


def jsons_to_csv(json_dir, csv_file, parquet_file_path=None):
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    df = pd.DataFrame(columns=["texto","nota", "labels"])
    labels_old = {
        "20":1,
        "40":2,
        "50":3,
        "80":4,
        "100":5,
        "120":6,
        "150":7,
        "160":8,
        "200":9,
        "0":0
    }
    labels = {}
    label_count = 0
    count = 0
    # Lê cada arquivo JSON e adiciona seus dados à lista
    for file in files:
        file_path = os.path.join(json_dir, file)
        #if count >= 3000:
        #    break
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            for obj in json_data:
                preprocessed = preprocess_texto(obj["texto"])
                count +=1
                if count % 1000 == 0:
                    print(count)

                nota = 0
                try:
                    nota = int(obj["nota"])
                except Exception as e:
                    nota = int(float(obj["nota"]))


                ## em alguns temas a redação tem notas entre 0 e 10 apenas
                if nota > 0 and nota < 10:
                    nota = nota*100

                if nota not in labels.keys():
                    labels[nota] = label_count
                    label_count += 1
                n = labels[nota]
                df = df._append({"texto": preprocessed, "nota": nota, "labels": n}, ignore_index=True)

    with open('labels.json', 'w', encoding='utf-8') as arquivo:
        json.dump(labels, arquivo, indent=4)

    print(labels)
    # Divisão em treino e o restante (teste + validação)
    train_data, test_data = train_test_split(df, test_size=0.8, random_state=42)

    # Salva o DataFrame como um arquivo CSV
    train_data.to_csv(f"train_{csv_file}", index=False)
    test_data.to_csv(f"test_{csv_file}", index=False)

    if parquet_file_path:
        train_data.to_parquet(f"train_{parquet_file_path}", index=False)
        test_data.to_parquet(f"test_{parquet_file_path}", index=False)


# Diretório contendo os arquivos JSON
json_directory = '../../../textgrader-pt-br/jsons'
# Nome do arquivo CSV a ser salvo
csv_file_path = 'output.csv'
parquet_file_path = 'output-parquet.parquet'

# Chama a função para converter JSONs em CSV
jsons_to_csv(json_directory, csv_file_path, parquet_file_path)

print(f'Arquivos salvos!')
