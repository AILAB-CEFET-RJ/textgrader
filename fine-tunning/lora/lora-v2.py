from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

#https://www.philschmid.de/fine-tune-flan-t5-peft

# Load dataset from the hub
dataset = load_dataset('parquet', data_files='preprocessing/output-parquet.parquet')

print(f"Train dataset size: {len(dataset)}")

model_id = "google/flan-t5-small"

# Load tokenizer of FLAN-t5-XL
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("#"*50)
# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = dataset.map(
    lambda x: tokenizer(x["texto"], truncation=True), batched=True, remove_columns=["nota","texto"])
input_lenghts = [len(x) for x in tokenized_inputs["train"]["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = dataset.map(
    lambda x: tokenizer(x["nota"], truncation=True), batched=True, remove_columns=["texto","nota"])
target_lenghts = [len(x) for x in tokenized_targets["train"]["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")


def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["dê uma nota para a redação: " + item for item in sample["texto"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["nota"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("-"*50)
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["nota","texto"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# save datasets to disk for later easy loading
tokenized_dataset["train"].save_to_disk("data/train")
#tokenized_dataset["test"].save_to_disk("data/eval")


print("+"*50)
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM


# huggingface hub model id
model_id = "adalbertojunior/Llama-3-8B-Dolphin-Portuguese-v0.3" #"meta-llama/Meta-Llama-3-8B" #"philschmid/flan-t5-xxl-sharded-fp16"

# load model from the hub
model = AutoModelForCausalLM.from_pretrained(model_id)
print(model)

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


print("*"*50)
# trainable params: 18874368 || all params: 11154206720 || trainable%: 0.16921300163961817
from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


print("="*50)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
torch.cuda.empty_cache()

output_dir = "lora-output"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=False,
    per_device_train_batch_size=1,
    per_gpu_train_batch_size=1,
    learning_rate=1e-3,  # higher learning rate
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


print("#"*50)
print("Trainer train")
# train model
trainer.train()
print("#"*50)
# Save our LoRA model & tokenizer results
peft_model_id="results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)
