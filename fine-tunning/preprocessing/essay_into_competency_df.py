import os

import pandas as pd
from sklearn.model_selection import train_test_split

from essay_sets_into_df_nota_geral import get_data

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Pasta '{path}' criada.")
    else:
        print(f"Pasta '{path}' já existe.")


def creating_dataframes(path, set_name, json_content, competency_name):
    df = pd.DataFrame()
    for file_content in json_content.values():
        for essay in file_content:
            label_comp = 0
            for comp in essay["competencias"]:
                c = comp["competencia"]
                c_name = c.replace(" ", "_").replace(",", "").lower()
                if c_name == competency_name:
                    label_comp = comp["nota"]
                    break

            content = {
                "texto": essay["texto"],
                "labels": label_comp,
            }

            new_df = pd.DataFrame([content])
            df = df._append(new_df)

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


if __name__ == '__main__':
    folder_name = "data_competencias"
    create_folder(folder_name)
    main_dir = "../../textgrader-pt-br/jsons"
    directories = ['conjunto_1']
    competency_name = "domínio_da_modalidade_escrita_formal"

    for set_name in directories:
        directory = os.path.join(folder_name, set_name)
        create_folder(directory)

        json_data = get_data(os.path.join(main_dir, set_name))
        df = creating_dataframes(directory, set_name, json_data, competency_name)

        creating_train_test_divisor(df, directory, competency_name)
