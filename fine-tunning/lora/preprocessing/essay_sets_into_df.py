## script used to transform all jsons into dataframes divided by your own sets

import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(directory):
    json_data = {}
    if os.path.exists(directory) and os.path.isdir(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as json_file:
                            json_data[file_path] = json.load(json_file)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    else:
        print(f"Directory {directory} does not exist or is not a directory")
    return json_data


def transform_into_df(data, set_name):
    df = pd.DataFrame()
    themes = []
    label_count = 0
    labels = {}
    for file_content in data.values():
        #print(json.dumps(file_content, indent=4, ensure_ascii=False))
        for essay in file_content:
            if essay["tema"] not in themes:
                themes.append(essay["tema"])
                #print(essay["tema"])

            try:
                nota = int(essay["nota"])
            except Exception as e:
                nota = int(float(essay["nota"]))

            if nota not in labels.keys():
                labels[nota] = label_count
                label_count += 1

            n = labels[nota]

            selected_fields = {
                "id": essay["id"],
                "texto": essay["texto"],
                #"texto_comentado": essay["texto_comentado"],
                #"comentarios": essay["cometarios"],
                "nota": nota,
                "titulo": essay["titulo"],
                "tema": essay["tema"],
                #"link": essay["link"],
                "labels": n,
            }

            competencias = essay["competencias"]
            for c in competencias:
                c_name = c["competencia"].replace(" ", "_")
                selected_fields[f"nota_{c_name}"] = c["nota"]
                selected_fields[f"nivel_{c_name}"] = c["motivo"]

            new_df = pd.DataFrame([selected_fields])
            df = df._append(new_df)

    with open(f'{folder_name}/labels_{set_name}.json', 'w', encoding='utf-8') as arquivo:
        json.dump(labels, arquivo, indent=4)

    return df


def creating_train_test_divisor(df, conjunto):
    # Divisão em treino e o restante (teste + validação)
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(temp_data, random_state=42)

    # Salva o DataFrame como um arquivo CSV
    train_data.to_parquet(f"{folder_name}/train_{conjunto}.parquet", index=False)
    labels_unicas = train_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de treino do {conjunto}: {labels_unicas}")

    test_data.to_parquet(f"{folder_name}/test_{conjunto}.parquet", index=False)
    labels_unicas = test_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de test do {conjunto}: {labels_unicas}")

    val_data.to_parquet(f"{folder_name}/eval_{conjunto}.parquet", index=False)
    labels_unicas = val_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de val do {conjunto}: {labels_unicas}")

    df.to_parquet(f"{folder_name}/df_{conjunto}.parquet", index=False)
    labels_unicas = df["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos total do {conjunto}: {labels_unicas}")


folder_name = "data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Pasta '{folder_name}' criada.")
else:
    print(f"Pasta '{folder_name}' já existe.")

main_dir = "../../../textgrader-pt-br/jsons"
directories = ['conjunto_1', 'conjunto_2', 'conjunto_3']

for directory in directories:
    path_dir = os.path.join(main_dir, directory)
    json_files_data = get_data(path_dir)

    df = transform_into_df(json_files_data, directory)

    creating_train_test_divisor(df, directory)

    print(f"----------------> {directory} done! <----------------")
