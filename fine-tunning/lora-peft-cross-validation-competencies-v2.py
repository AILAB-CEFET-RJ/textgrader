import time
import traceback
import evaluate
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
)
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from sklearn.model_selection import KFold
from configs import Configs


def get_datasets_cv(data_dir, suffix):
    print(f"> READING FROM {data_dir}/train_{suffix}.parquet")
    print(f"> READING FROM {data_dir}/eval_{suffix}.parquet")
    d = load_dataset(
        "csv",
        data_files=f"{data_dir}/train_{suffix}.csv",
    )

    d_eval = load_dataset(
        "csv",
        data_files=f"{data_dir}/eval_{suffix}.csv",
    )

    return d, d_eval


if __name__ == '__main__':
    config = Configs()
    config.get_data_config()
    start_time = time.time()
    config.script_type = "cross-validation"

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_dora=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, padding=config.padding_side
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets, datasets_eval = get_datasets_cv(config.data_dir, config.sufix)


    # metric = evaluate.load("accuracy")


    def tokenize(examples):
        outputs = tokenizer(examples["texto"], truncation=True, max_length=512)
        return outputs


    tokenize_datasets = datasets.map(
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


    # Dividindo os dados em folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_count = 0

    labels_exception = None
    try:
        for train_idx, val_idx in kf.split(tokenize_datasets["train"]):
            fold_count += 1
            print(f"Training fold {fold_count}...")

            train_dataset = tokenize_datasets["train"].select(train_idx)
            val_dataset = tokenize_datasets["train"].select(val_idx)

            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=config.batch_size,
            )
            val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                collate_fn=collate_fn,
                batch_size=config.batch_size,
            )

            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, return_dict=True, num_labels=config.n_labels
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            print(model)

            optimizer = AdamW(model.parameters(), lr=config.lr)

            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0.06 * (len(train_dataloader) * config.num_epochs),
                num_training_steps=(len(train_dataloader) * config.num_epochs),
            )

            model.to(config.device)


            for epoch in range(config.num_epochs):
                model.train()
                for step, batch in enumerate(train_dataloader):
                    labels_exception = batch["labels"]
                    batch.to(config.device)
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                model.eval()
                metric = evaluate.load("accuracy")
                for step, batch in enumerate(tqdm(val_dataloader)):
                    labels_exception = batch["labels"]
                    batch.to(config.device)
                    with torch.no_grad():
                        outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    predictions, references = predictions, batch["labels"]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                val_metric = metric.compute()
                print(f"epoch {epoch}:", val_metric)
                config.metrics[f"fold_{fold_count}_epoch_{epoch}"] = val_metric

    except Exception as e:
        print(f"Exception: {e}")
        print(labels_exception)
        print(labels_exception.shape)
        traceback.print_exc()
        print("-" * 100)

    ## using evaluation data_one_label
    eval_dataloader = DataLoader(
        tokenize_datasets_eval["train"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
    )

    all_predictions = []
    all_references = []

    try:
        metric = load_metric("accuracy")
        for step, batch in enumerate(tqdm(eval_dataloader)):
            labels_exception = batch["labels"]
            batch.to(config.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(references.cpu().numpy())
            print(f"predictions: {predictions} references: {references}")
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"Validation metric: {eval_metric}")
        config.validation_metric = eval_metric
    except Exception as e:
        print(f"Exception: {e}")
        print(labels_exception)
        print("-" * 100)
        traceback.print_exc()


    ## Saving log file
    elapsed_time = time.time() - start_time

    config.processing_time = elapsed_time / 60
    config.save_to_json()
    print("finish!!")
