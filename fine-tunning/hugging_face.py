from huggingface_hub import login, upload_folder
import os
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


class HuggingFaceModel:
    def __init__(self):
        self.token = os.getenv("HF_TOKEN")
        login(token=self.token)

    def upload_model(self, path):
        upload_folder(
            repo_id="vansoares1/textgrader",
            folder_path=path,
            path_in_repo=path
        )
        print("> Models saved!")

    def upload_data(self, folder_path):
        upload_folder(
            repo_id="vansoares1/portuguese-essays",
            folder_path=folder_path,
            path_in_repo=folder_path,
            repo_type="dataset",
        )
        print("> Data saved!")

    def get_datasets(self, suffix):
        data_dir = "hf://datasets/vansoares1/portuguese-essays/preprocessing/data_competencias/conjunto_1"
        try:
            d = load_dataset(
                "parquet",
                data_files=f"{data_dir}/train_{suffix}.parquet",
                token=self.token,
                repo_id="vansoares1/portuguese-essays"
            )
            d_test = load_dataset(
                "parquet",
                data_files=f"{data_dir}/test_{suffix}.parquet",
                use_auth_token=self.token
            )
            d_eval = load_dataset(
                "parquet",
                data_files=f"{data_dir}/eval_{suffix}.parquet",
                use_auth_token=self.token
            )
        except Exception as e:
            print("USING CSV", e)
            d = load_dataset(
                "csv",
                data_files=f"{data_dir}/train_{suffix}.csv",
                use_auth_token=self.token
            )
            d_test = load_dataset(
                "csv",
                data_files=f"{data_dir}/test_{suffix}.csv",
                use_auth_token=self.token
            )
            d_eval = load_dataset(
                "csv",
                data_files=f"{data_dir}/eval_{suffix}.csv",
                use_auth_token=self.token
            )
        return d, d_test, d_eval


if __name__ == "__main__":
    path = os.getenv("HF_MODELS_PATH")
    model = HuggingFaceModel()
    folder_path = "preprocessing/data_competencias/"  # Substitua pelo caminho da sua pasta
    dataset_name = "vansoares1/portuguese-essays"  # Exemplo: "meu_usuario/meu_dataset"

    # Fazer o upload da pasta
    model.upload_data(folder_path)
