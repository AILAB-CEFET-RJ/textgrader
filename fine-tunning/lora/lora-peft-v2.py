#https://huggingface.co/spaces/PEFT/sequence-classification/blob/main/LoRA.ipynb
import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

#import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

## todo: - consigo dividir os dados em treino e teste?
## todo: - consigo juntar todas as redaçẽos para fazer um classificador de nota geral?
batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
peft_type = PeftType.LORA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
lr = 3e-4
padding_side = "right" ## todo: padding ser right ou left faz alguma diferença?

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

#datasets = load_dataset("glue", task)
datasets = load_dataset('parquet', data_files='preprocessing/output-parquet.parquet')
#metric = evaluate.load("glue", task)


def tokenize(examples):
    outputs = tokenizer(examples["texto"], examples["nota"], truncation=True, max_length=None)
    return outputs


tokenize_datasets = datasets.map(
    tokenize,
    batched=True,
    remove_columns=["texto", "nota"],
)

tokenize_datasets = tokenize_datasets.add_column(name="labels", column="labels")
tokenize_datasets = tokenize_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


train_dataloader = DataLoader(tokenize_datasets["train"], shuffle=True, collate_fn=collate_fn,  batch_size=batch_size)
#eval_dataloader = DataLoader(
#    tokenize_datasets["train"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
#)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print(model)

optimizer = AdamW(model.parameters(), lr=lr)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06*(len(train_dataloader)*num_epochs),
    num_training_steps=(len(train_dataloader)*num_epochs)
)

model.to(device)
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    #for step, batch in enumerate(tqdm(eval_dataloader)):
    #    batch.to(device)
    #    with torch.no_grad():
    #        outputs = model(**batch)
    #    predictions = outputs.logits.argmax(dim=-1)
    #    predictions, references = predictions, batch["labels"]
        #metric.add_batch(
        #    predictions=predictions,
        #    references=references,
        #)

    #eval_metric = metric.compute()
    print(f"epoch {epoch}:")#, eval_metric)

