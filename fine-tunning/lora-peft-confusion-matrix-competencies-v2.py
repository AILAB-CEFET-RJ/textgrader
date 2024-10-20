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
    get_linear_schedule_with_warmup, AutoModelForSequenceClassification, BitsAndBytesConfig,
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
import data_utils

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


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

    datasets, datasets_test, datasets_eval = data_utils.get_datasets(data_dir=configs.data_dir,
                                                                     suffix=configs.competence)

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

    model = AutoModelForSequenceClassification.from_pretrained(
        configs.model_name_or_path, return_dict=True, num_labels=configs.n_labels,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model)

    optimizer = AdamW(model.parameters(), lr=configs.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * configs.num_epochs),
        num_training_steps=(len(train_dataloader) * configs.num_epochs),
    )

    torch.cuda.empty_cache()
    model.to(configs.device)
    '''
    if torch.cuda.device_count() > 1:
        print(f"Utilizando {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    '''

    ## using evaluation data_one_label
    all_predictions = []
    all_references = []
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

                outputs = model(**batch)
                loss = outputs.loss
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
                    outputs = model(**batch)
                loss = outputs.loss
                valid_losses.append(loss)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                all_predictions.extend(predictions.cpu())
                all_references.extend(references.cpu())
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            test_metric = metric.compute()
            kappa = cohen_kappa_score(all_references, all_predictions)
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

    try:
        all_predictions = []
        all_references = []

        for step, batch in enumerate(tqdm(eval_dataloader)):
            labels_exception = batch["labels"]
            batch.to(configs.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            all_predictions.extend(predictions.cpu())
            all_references.extend(references.cpu())
            #print(f"predictions: {predictions} references: {references}")
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()
        kappa = cohen_kappa_score(all_references, all_predictions)
        print(f"Validation metric: {eval_metric}, Cohen's Kappa: {kappa}")
        configs.validation_metric = eval_metric
        config.cohen = kappa

    except Exception as e:
        print(f"Exception: {e} {e.args}")
        print(labels_exception)
        print("-" * 100)

    ## Calcular a matriz de confusão
    cm = confusion_matrix(all_references, all_predictions)
    cm_df = pd.DataFrame(cm)

    ## Saving log file
    elapsed_time = time.time() - start_time

    configs.processing_time = elapsed_time / 60
    configs.save_to_json(cm_df)
    print("finish!!")


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

            try:
                db.save(config_json)
                print("> Results uploaded to MongoDB!")

            except Exception as e:
                print("Error saving mongo: ", e)

        print("=" * 50)
        print(f"> CONJUNTO {s} DONE!!")
        print("=" * 50)

        import shutil

        # Caminho do diretório que você quer apagar
        path = f"results/confusion-matrix-competencies/{config.conjunto}/"

        if os.path.exists(path):
            shutil.rmtree(path)
        else:
            print("O diretório não existe.")
