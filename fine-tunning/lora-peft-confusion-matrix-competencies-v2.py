import time
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
    get_linear_schedule_with_warmup, AutoModelForSequenceClassification,
)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
from configs import Configs


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

    model = AutoModelForSequenceClassification.from_pretrained(
        configs.model_name_or_path, return_dict=True, num_labels=configs.n_labels
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

    model.to(configs.device)

    labels_exception = None
    try:
        for epoch in range(configs.num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                labels_exception = batch["labels"]
                batch.to(configs.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()

            print("-" * 100)
            for step, batch in enumerate(tqdm(test_dataloader)):
                labels_exception = batch["labels"]
                batch.to(configs.device)
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                #print(f"predictions: {predictions} references: {references}")
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            test_metric = metric.compute()
            print(f"epoch {epoch}:", test_metric)
            configs.metrics[epoch] = test_metric

    except Exception as e:
        print(f"Exception: {e}")
        print(labels_exception)
        print("-" * 100)

    ## using evaluation data_one_label
    all_predictions = []
    all_references = []

    try:
        for step, batch in enumerate(tqdm(eval_dataloader)):
            labels_exception = batch["labels"]
            batch.to(configs.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(references.cpu().numpy())
            #print(f"predictions: {predictions} references: {references}")
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"Validation metric: {eval_metric}")
        configs.validation_metric = eval_metric

    except Exception as e:
        print(f"Exception: {e}")
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

    for comp in config.get_competencies_from_set():
        print("> TRAINING:", comp)
        config.competence = comp
        train_model(config)
