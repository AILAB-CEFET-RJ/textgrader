import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split


def preprocess_texto(texto):
    t = str.replace(texto, " q ", "que")
    t = str.replace(t, " v ", "5")

    return t.lower()


def jsons_to_csv(json_dir, csv_file, conjunto, parquet_file_path=None):
    dir_fmt = f"{json_dir}/{conjunto}"
    files = [f for f in os.listdir(dir_fmt) if f.endswith('.json')]

    df = pd.DataFrame(columns=["texto", "nota", "labels"])
    labels = {}
    label_count = 0
    count = 0
    total_by_grades = {}
    for file in files:
        file_path = os.path.join(dir_fmt, file)
        #if count >= 3000:
        #    break
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            for obj in json_data:
                preprocessed = preprocess_texto(obj["texto"])
                #count += 1
                #if count % 1000 == 0:
                #    print(count)

                nota = 0
                try:
                    nota = int(obj["nota"])
                except Exception as e:
                    nota = int(float(obj["nota"]))

                if nota not in total_by_grades.keys():
                    total_by_grades[nota] = 1
                else:
                    total_by_grades[nota] += 1

                if nota not in labels.keys():
                    labels[nota] = label_count
                    label_count += 1

                n = labels[nota]
                df = df._append({"texto": preprocessed, "nota": nota, "labels": n}, ignore_index=True)

    with open(f'labels_{conjunto}.json', 'w', encoding='utf-8') as arquivo:
        json.dump(labels, arquivo, indent=4)

    with open(f'count_labels_{conjunto}.json', 'w', encoding='utf-8') as arquivo:
        json.dump(total_by_grades, arquivo, indent=4)

    print(labels)

    # Divisão em treino e o restante (teste + validação)
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(temp_data, random_state=42)

    # Salva o DataFrame como um arquivo CSV
    train_data.to_csv(f"train_{conjunto}_{csv_file}", index=False)
    labels_unicas = train_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de treino do {conjunto}: {labels_unicas}")

    test_data.to_csv(f"test_{conjunto}_{csv_file}", index=False)
    labels_unicas = test_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de test do {conjunto}: {labels_unicas}")

    val_data.to_csv(f"eval_{conjunto}_{csv_file}", index=False)
    labels_unicas = val_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de val do {conjunto}: {labels_unicas}")

    df.to_csv(f"df_{conjunto}_{csv_file}", index=False)
    labels_unicas = df["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos total do {conjunto}: {labels_unicas}")

    if parquet_file_path:
        train_data.to_parquet(f"train_{conjunto}_{parquet_file_path}", index=False)
        test_data.to_parquet(f"test_{conjunto}_{parquet_file_path}", index=False)
        val_data.to_parquet(f"eval_{conjunto}_{parquet_file_path}", index=False)
        df.to_parquet(f"df_{conjunto}_{parquet_file_path}", index=False)


# Diretório contendo os arquivos JSON
conjuntos = ["conjunto_1","conjunto_2","conjunto_3"]
json_directory = '../../../textgrader-pt-br/jsons'
# Nome do arquivo CSV a ser salvo
csv_file_path = 'output.csv'
parquet_file_path = 'output.parquet'

# Chama a função para converter JSONs em CSV
for c in conjuntos:
    jsons_to_csv(json_directory, csv_file_path, c, parquet_file_path)

print(f'Arquivos salvos!')
