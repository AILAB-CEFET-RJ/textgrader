## script used to transform all jsons into dataframes divided by your own sets

import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split

total_labels = {
    "conjunto_1": 0,
    "conjunto_2": 0,
    "conjunto_3": 0,
}
should_get_competency = False
folder_name = "data_multilabel" if should_get_competency else 'data_one_label'
should_compute_notas_as_intervals = True

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


def compute_notas_as_intervals(nota):
    if nota == 0:
        return 0
    elif 0 < nota <= 100:
        return 1
    elif 100 < nota <= 200:
        return 2
    elif 200 < nota <= 300:
        return 3
    elif 300 < nota <= 400:
        return 4
    elif 400 < nota <= 500:
        return 5
    elif 500 < nota <= 600:
        return 6
    elif 600 < nota <= 700:
        return 7
    elif 700 < nota <= 800:
        return 8
    elif 800 < nota <= 900:
        return 9
    elif 900 < nota <= 1000:
        return 10


def transform_into_df(data, set_name):
    df = pd.DataFrame()
    competencias_dict = {}

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

            # apenas as notas do conjunto 1 variam de 0 a 1000
            if should_compute_notas_as_intervals and set_name == "conjunto_1":
                nota = compute_notas_as_intervals(nota)

            if nota not in labels.keys():
                labels[nota] = label_count
                label_count += 1

            n = labels[nota]

            selected_fields = {
                #"id": essay["id"],
                "texto": essay["texto"],
                #"texto_comentado": essay["texto_comentado"],
                #"comentarios": essay["cometarios"],
                "nota": nota,
                #"titulo": essay["titulo"],
                #"tema": essay["tema"],
                #"link": essay["link"],
                "labels": n,
            }

            if should_get_competency:
                competencias = essay["competencias"]
                for c in competencias:
                    comp = c["competencia"]
                    c_name = comp.replace(" ", "_").replace(",", "").lower()
                    nota_column = f"nota_{c_name}"
                    if nota_column == "nota_domínio_da_modalidade_escrita_formal":
                        selected_fields["labels"] = str(c["nota"])

            new_df = pd.DataFrame([selected_fields])
            df = df._append(new_df)

    with open(f'{folder_name}/labels_{set_name}.json', 'w', encoding='utf-8') as arquivo:
        json.dump(labels, arquivo, indent=4)

    with open(f'{folder_name}/labels_competencias_{set_name}.json', 'w', encoding='utf-8') as arquivo:
        json.dump(competencias_dict, arquivo, indent=4, ensure_ascii=False)
    return df


def creating_train_test_divisor(df, conjunto):
    # Divisão em treino e o restante (teste + validação)
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(temp_data, random_state=42)
    suffixo = "nota_geral"

    # Salva o DataFrame como um arquivo CSV
    train_data.to_parquet(f"{folder_name}/train_{conjunto}_{suffixo}.parquet", index=False)
    labels_unicas = train_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de treino do {conjunto}: {labels_unicas}")

    test_data.to_parquet(f"{folder_name}/test_{conjunto}_{suffixo}.parquet", index=False)
    labels_unicas = test_data["nota"].nunique()

    print(f"Quantidade de labels unicas no conjuntos de test do {conjunto}: {labels_unicas}")

    val_data.to_parquet(f"{folder_name}/eval_{conjunto}_{suffixo}.parquet", index=False)
    labels_unicas = val_data["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de val do {conjunto}: {labels_unicas}")

    df.to_parquet(f"{folder_name}/df_{conjunto}_{suffixo}.parquet", index=False)
    df.to_csv(f"{folder_name}/df_{conjunto}_comp1.csv", index=False)
    labels_unicas = df["nota"].nunique()
    print(f"Quantidade de labels unicas no conjuntos total do {conjunto}: {labels_unicas}")
    total_labels[conjunto] = labels_unicas

if __name__ == '__main__':
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Pasta '{folder_name}' criada.")
    else:
        print(f"Pasta '{folder_name}' já existe.")

    main_dir = "../../textgrader-pt-br/jsons"
    directories = ['conjunto_1', 'conjunto_2', 'conjunto_3']

    for directory in directories:
        path_dir = os.path.join(main_dir, directory)
        json_files_data = get_data(path_dir)

        df = transform_into_df(json_files_data, directory)

        creating_train_test_divisor(df, directory)

        print(f"----------------> {directory} done! <----------------")

    with open(f'{folder_name}/total_label_count_interval.json', 'w', encoding='utf-8') as arquivo:
        json.dump(total_labels, arquivo, indent=4)

