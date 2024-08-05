import logs
import time
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
import configs

obs = configs.get_data_config()

start_time = time.time()

peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    use_dora=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model, padding=configs.padding_side
)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

data_dir = "preprocessing/data_one_label"
datasets = load_dataset(
    "parquet", data_files=f"{data_dir}/train_conjunto_{configs.conjunto}.parquet"
)

datasets_eval = load_dataset(
    "parquet", data_files=f"{data_dir}/eval_conjunto_{configs.conjunto}.parquet"
)

# metric = evaluate.load("accuracy")


def tokenize(examples):
    outputs = tokenizer(examples["texto"], truncation=True, max_length=512)
    return outputs


tokenize_datasets = datasets.map(
    tokenize,
    batched=True,
    remove_columns=["texto", "nota"],
)

tokenize_datasets_eval = datasets_eval.map(
    tokenize,
    batched=True,
    remove_columns=["texto", "nota"],
)


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Dividindo os dados em folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {
    "batch_size": configs.batch_size,
    "model": configs.model_name_or_path,
    "epochs": configs.num_epochs,
    "metrics": {},
    "conjunto": configs.conjunto,
    "obs": obs,
    "padding_side": configs.padding_side,
    "train_size": len(tokenize_datasets["train"]["input_ids"]),
    "n_labels": configs.n_labels,
    "script_type": "cross_validation",
}

fold_count = 0

for train_idx, val_idx in kf.split(tokenize_datasets["train"]):
    fold_count += 1
    print(f"Training fold {fold_count}...")

    train_dataset = tokenize_datasets["train"].select(train_idx)
    val_dataset = tokenize_datasets["train"].select(val_idx)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=configs.batch_size,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=configs.batch_size,
    )

    model = AutoModelForCausalLM.from_pretrained(
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

    for epoch in range(configs.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch.to(configs.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        metric = load_metric("accuracy")
        for step, batch in enumerate(tqdm(val_dataloader)):
            batch.to(configs.device)
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
        results["metrics"][f"fold_{fold_count}_epoch_{epoch}"] = val_metric

## using evaluation data_one_label
eval_dataloader = DataLoader(
    tokenize_datasets_eval["train"],
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=configs.batch_size,
)

all_predictions = []
all_references = []

for step, batch in enumerate(tqdm(eval_dataloader)):
    batch.to(configs.device)
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
results["validation_metric"] = eval_metric

# Salvar resultados em um arquivo JSON
end_time = time.time()
elapsed_time = end_time - start_time
results["processing_time"] = elapsed_time / 60

logs.saving_results(results, "cv")
print("Finish!")
