import logs
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
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
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

tokenizer = AutoTokenizer.from_pretrained(configs.model_name_or_path, padding=configs.padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# datasets = load_dataset("glue", task)
datasets = load_dataset(
    "parquet", data_files=f"{configs.data_dir}/train_conjunto_{configs.conjunto}_interval.parquet"
)
datasets_test = load_dataset(
    "parquet", data_files=f"{configs.data_dir}/test_conjunto_{configs.conjunto}_interval.parquet"
)
datasets_eval = load_dataset(
    "parquet", data_files=f"{configs.data_dir}/eval_conjunto_{configs.conjunto}_interval.parquet"
)

metric = evaluate.load("accuracy")


def tokenize(examples):
    outputs = tokenizer(examples["texto"], truncation=True, max_length=512)
    return outputs


tokenize_datasets = datasets.map(
    tokenize,
    batched=True,
    remove_columns=["texto", "nota"],
)


tokenize_datasets_test = datasets_test.map(
    tokenize,
    batched=True,
    remove_columns=["texto", "nota"],
)

tokenize_datasets_eval = datasets_eval.map(
    tokenize,
    batched=True,
    remove_columns=["texto", "nota"],
)

# tokenize_datasets = tokenize_datasets.rename_column("label", "labels")


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
results = {
    "batch_size": configs.batch_size,
    "model": configs.model_name_or_path,
    "epochs": configs.num_epochs,
    "metrics": {},
    "conjunto": configs.conjunto,
    "obs": obs,
    "padding_side": configs.padding_side,
    "train_size": len(tokenize_datasets["train"]["input_ids"]),
    "test_size": len(tokenize_datasets_test["train"]["input_ids"]),
    "eval_size": len(tokenize_datasets_eval["train"]["input_ids"]),
    "n_labels": configs.n_labels,
}

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
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch.to(configs.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        print(f"predictions: {predictions} references: {references}")
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    test_metric = metric.compute()
    print(f"epoch {epoch}:", test_metric)
    results["metrics"][epoch] = test_metric

## using evaluation data_one_label
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

## Calcular a matriz de confusão
cm = confusion_matrix(all_references, all_predictions)
cm_df = pd.DataFrame(cm)

## Saving log file
end_time = time.time()
elapsed_time = end_time - start_time

results["processing_time"] = elapsed_time / 60

logs.saving_results(results, "cm", cm_df)

# model.save_pretrained()
print("finish!!")
