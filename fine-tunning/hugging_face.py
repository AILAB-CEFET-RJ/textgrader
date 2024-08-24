from huggingface_hub import login, upload_folder


class HuggingFaceModel:
    def __init__(self):
        login(token="seu_token_aqui")

    def upload_model(self, path):
        upload_folder(
            repo_id="vansoares1/textgrader",
            folder_path=path,
            path_in_repo="."
        )
        print("> Models saved!")
