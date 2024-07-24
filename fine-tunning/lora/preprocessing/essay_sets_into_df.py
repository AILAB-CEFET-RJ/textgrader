import os
import json

import pandas as pd


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
    for file_content in data.values():
        #print(json.dumps(file_content, indent=4, ensure_ascii=False))
        for essay in file_content:
            if essay["tema"] not in themes:
                themes.append(essay["tema"])
                print(essay["tema"])

            selected_fields = {
                "id": essay["id"],
                "texto": essay["texto"],
                #"texto_comentado": essay["texto_comentado"],
                #"comentarios": essay["cometarios"],
                "nota": essay["nota"],
                "titulo": essay["titulo"],
                "tema": essay["tema"],
                #"link": essay["link"]
            }

            competencias = essay["competencias"]
            for c in competencias:
                c_name = c["competencia"].replace(" ", "_")
                selected_fields[f"nota_{c_name}"] = c["nota"]
                selected_fields[f"nivel_{c_name}"] = c["motivo"]

            new_df = pd.DataFrame([selected_fields])
            df = df._append(new_df)

    df.to_csv(f"{set_name}.csv", index=False, encoding='utf-8')


main_dir = "../../../textgrader-pt-br/jsons"
directories = ['conjunto_1', 'conjunto_2', 'conjunto_3']
for directory in directories:
    path_dir = os.path.join(main_dir, directory)
    json_files_data = get_data(path_dir)

    transform_into_df(json_files_data, directory)

    print(f"----------------> {directory} done! <----------------")
