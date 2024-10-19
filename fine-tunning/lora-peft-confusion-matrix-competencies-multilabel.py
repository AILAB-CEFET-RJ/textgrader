import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
)
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup, AutoModel,
)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
from configs import Configs
from sklearn.metrics import cohen_kappa_score
import os
import traceback
from early_stopping import EarlyStopping
from hugging_face import HuggingFaceModel
from db import MongoDB

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def get_datasets(data_dir, suffix):
    d = load_dataset(
        "parquet",
        data_files=f"{data_dir}/train_{suffix}.parquet",
    )
    d_test = load_dataset(
        "parquet",
        data_files=f"{data_dir}/test_{suffix}.parquet",
    )
    d_eval = load_dataset(
        "parquet",
        data_files=f"{data_dir}/eval_{suffix}.parquet",
    )
    return d, d_test, d_eval


class MultiOutputModel(torch.nn.Module):
    def __init__(self, base_model):
        super(MultiOutputModel, self).__init__()
        self.base_model = base_model

        # Adicionando 5 classificadores para as 5 saídas
        self.classifier_1 = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)  # Exemplo de 3 classes
        self.classifier_2 = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)  # Exemplo de 4 classes
        self.classifier_3 = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)  # Exemplo de 2 classes
        self.classifier_4 = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)  # Exemplo de 5 classes
        self.classifier_5 = torch.nn.Linear(base_model.config.hidden_size, config.n_labels)  # Exemplo de 3 classes

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled_output vem da base do modelo (ex: BERT)

        # Previsões para cada uma das saídas
        output_1 = self.classifier_1(pooled_output)
        output_2 = self.classifier_2(pooled_output)
        output_3 = self.classifier_3(pooled_output)
        output_4 = self.classifier_4(pooled_output)
        output_5 = self.classifier_5(pooled_output)

        return output_1, output_2, output_3, output_4, output_5


def train_model(configs):
    start_time = time.time()
    configs.script_type = "confusion-matrix-competencies"

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=configs.lora_r,
        lora_alpha=configs.lora_alpha,
        lora_dropout=configs.lora_dropout,
        use_dora=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_name_or_path, padding=configs.padding_side
    )

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets, datasets_test, datasets_eval = get_datasets(configs.data_dir, configs.competence)

    metric = evaluate.load("accuracy")

    def tokenize(examples):
        outputs = tokenizer(examples["texto"], truncation=True, max_length=512)
        return outputs

    tokenize_datasets = datasets.map(
        tokenize,
        batched=True,
        remove_columns=["texto"],
    )

    tokenize_datasets_test = datasets_test.map(
        tokenize,
        batched=True,
        remove_columns=["texto"],
    )

    tokenize_datasets_eval = datasets_eval.map(
        tokenize,
        batched=True,
        remove_columns=["texto"],
    )

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        tokenize_datasets["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=configs.batch_size,
    )
    test_dataloader = DataLoader(
        tokenize_datasets_test["train"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=configs.batch_size,
    )
    eval_dataloader = DataLoader(
        tokenize_datasets_eval["train"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=configs.batch_size,
    )

    base_model = AutoModel.from_pretrained(configs.model_name_or_path, return_dict=True)
    model = MultiOutputModel(base_model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=configs.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * configs.num_epochs),
        num_training_steps=(len(train_dataloader) * configs.num_epochs),
    )

    torch.cuda.empty_cache()
    model.to(configs.device)

    all_predictions = [[] for _ in range(5)]  # Para armazenar previsões de todas as saídas
    all_references = [[] for _ in range(5)]
    labels_exception = None

    try:
        early_stopping = EarlyStopping(patience=configs.patience, verbose=True, configs=configs)
        for epoch in range(configs.num_epochs):
            model.train()
            train_losses = []
            valid_losses = []
            for step, batch in enumerate(train_dataloader):
                labels_exception = batch["labels"]
                batch.to(configs.device)
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = 0
                for i in range(5):
                    loss += torch.nn.CrossEntropyLoss()(outputs[i], batch[f"labels_{i+1}"])  # 5 saídas
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            torch.cuda.empty_cache()

            print("-" * 100)
            for step, batch in enumerate(tqdm(test_dataloader)):
                labels_exception = batch["labels"]
                batch.to(configs.device)
                with torch.no_grad():
                    outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = 0
                for i in range(5):
                    loss += torch.nn.CrossEntropyLoss()(outputs[i], batch[f"labels_{i+1}"])
                    predictions = outputs[i].argmax(dim=-1)
                    all_predictions[i].extend(predictions.cpu())
                    all_references[i].extend(batch[f"labels_{i+1}"].cpu())
                valid_losses.append(loss)
                metric.add_batch(
                    predictions=predictions,
                    references=batch[f"labels_{i+1}"],
                )

            test_metric = metric.compute()
            kappa = cohen_kappa_score(all_references[0], all_predictions[0])  # Usar uma das saídas
            valid_loss = np.mean([loss.detach().cpu().numpy() for loss in valid_losses])
            train_loss = np.mean([loss.detach().cpu().numpy() for loss in train_losses])
            configs.metrics[epoch] = {
                "test_metric": test_metric,
                "kappa": kappa,
                "train_loss": str(train_loss),
                "valid_loss": str(valid_loss)
            }

            print(f"Epoch [{epoch}/{configs.num_epochs}]  Loss: {valid_loss} | Cohen's Kappa: {kappa}")

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        filename = tb.filename
        lineno = tb.lineno
        line = tb.line
        error_message = f"An error occurred: {str(e)}\n"
        error_message += f"In file: {filename}, line {lineno}: {line}"
        config.except_message = error_message
        raise Exception(error_message)


if __name__ == '__main__':
    config = Configs()
    config.get_data_config()
    hf = HuggingFaceModel()
    db = MongoDB()

    sets = [1]
    for s in sets:
        config.set_conjunto(s)

        print(f"> USANDO O CONJUNTO {config.conjunto}")
        for comp in config.get_competencies_from_set():
            print("> TRAINING:", comp)
            config.competence = comp
            train_model(config)

            config_json = config.get_results_folder_path()
            hf.upload_model(config_json)
            print("> Results uploaded to HF!")

            db.save(config_json)
            print("> Results uploaded to MongoDB!")


        print("="*50)
        print(f"> CONJUNTO {s} DONE!!")
        print("=" * 50)

        import shutil

        # Caminho do diretório que você quer apagar
        path = f"results/confusion-matrix-competencies/{config.conjunto}/"

        if os.path.exists(path):
            shutil.rmtree(path)
        else:
            print("O diretório não existe.")
