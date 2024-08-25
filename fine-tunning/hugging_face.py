from huggingface_hub import login, upload_folder
import os

class HuggingFaceModel:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        login(token=token)

    def upload_model(self, path):
        upload_folder(
            repo_id="vansoares1/textgrader",
            folder_path=path,
            path_in_repo="."
        )
        print("> Models saved!")

if __name__ == "__main__":
    path = os.getenv("HF_MODELS_PATH")
    model = HuggingFaceModel()
    model.upload_model(path)