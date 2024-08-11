import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from essay_sets_into_df_nota_geral import get_data


import unicodedata
import re


def remove_acentos(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_string(input_str):
    # Remove acentos
    input_str = remove_acentos(input_str)
    # Substitui espaços por _
    input_str = input_str.replace(" ", "_")
    # Remove caracteres especiais (exceto _)
    input_str = re.sub(r"[^a-zA-Z0-9_]", "", input_str)
    return input_str.lower()


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Pasta '{path}' criada.")
    else:
        print(f"Pasta '{path}' já existe.")


def creating_dataframes(path, set_name, json_content, competency_name):
    df = pd.DataFrame()
    print("-"*50)
    print(f"CONJUNTO: {set_name} COMPETENCIA:{competency_name}")
    print("-" * 50)
    label_count = 1
    labels_mapping = {}
    for file_content in json_content.values():
        for essay in file_content:
            label_comp = 0
            for comp in essay["competencias"]:
                c = comp["competencia"]
                c_name = clean_string(c)

                if comp["nota"] not in labels_mapping:
                    labels_mapping[comp["nota"]] = label_count
                    label_count += 1

                if c_name == competency_name:
                    label_comp = comp["nota"]
                    break

            content = {
                "texto": essay["texto"],
                "labels": labels_mapping[label_comp],
            }

            new_df = pd.DataFrame([content])
            df = df._append(new_df)

    with open(f"data_competencias/{set_name}_labels.json", 'w', encoding='utf-8') as arquivo_json:
        json.dump(labels_mapping, arquivo_json, ensure_ascii=False, indent=4)

    return df


def creating_train_test_divisor(df, path, competency_name):
    # Divisão em treino e o restante (teste + validação)
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)

    test_data, val_data = train_test_split(temp_data, random_state=42)
    suffixo = competency_name

    # Salva o DataFrame como um arquivo CSV
    train_data.to_parquet(f"{path}/train_{suffixo}.parquet", index=False)
    train_data.to_csv(f"{path}/train_{suffixo}.csv", index=False)

    labels_unicas = train_data["labels"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de treino: {labels_unicas}")

    test_data.to_parquet(f"{path}/test_{suffixo}.parquet", index=False)
    test_data.to_csv(f"{path}/test_{suffixo}.csv", index=False)

    labels_unicas = test_data["labels"].nunique()

    print(f"Quantidade de labels unicas no conjuntos de test: {labels_unicas}")

    val_data.to_parquet(f"{path}/eval_{suffixo}.parquet", index=False)
    val_data.to_csv(f"{path}/eval_{suffixo}.csv", index=False)
    labels_unicas = val_data["labels"].nunique()
    print(f"Quantidade de labels unicas no conjuntos de val: {labels_unicas}")

    df.to_parquet(f"{path}/df_{suffixo}.parquet", index=False)
    df.to_csv(f"{path}/df_{suffixo}.csv", index=False)
    labels_unicas = df["labels"].nunique()
    print(f"Quantidade de labels unicas no conjuntos total: {labels_unicas}")


if __name__ == "__main__":
    folder_name = "data_competencias"
    create_folder(folder_name)
    main_dir = "../../textgrader-pt-br/jsons"
    directories = ["conjunto_1", "conjunto_2", "conjunto_3"]
    competencies = {
        "conjunto_1": [
            "dominio_da_modalidade_escrita_formal",
            "compreender_a_proposta_e_aplicar_conceitos_das_varias_areas_de_conhecimento_para_desenvolver_o_texto_dissertativoargumentativo_em_prosa",
            "selecionar_relacionar_organizar_e_interpretar_informacoes_em_defesa_de_um_ponto_de_vista",
            "conhecimento_dos_mecanismos_linguisticos_necessarios_para_a_construcao_da_argumentacao",
            "proposta_de_intervencao_com_respeito_aos_direitos_humanos",
        ],
        "conjunto_2": [
            "adequacao_ao_tema",
            "adequacao_e_leitura_critica_da_coletanea",
            "adequacao_ao_genero_textual",
            "adequacao_a_modalidade_padrao_da_lingua",
            "coesao_e_coerencia",
        ],
        "conjunto_3": [
            "conteudo",
            "estrutura_do_texto",
            "estrutura_de_ideias",
            "vocabulario",
            "gramatica_e_ortografia",
        ],
    }

    for set_name in competencies.keys():
        directory = os.path.join(folder_name, set_name)
        create_folder(directory)
        for competence_name in competencies[set_name]:
            json_data = get_data(os.path.join(main_dir, set_name))
            df = creating_dataframes(directory, set_name, json_data, competence_name)

            creating_train_test_divisor(df, directory, competence_name)
