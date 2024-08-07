import torch
import json
from peft import (
    PeftType,
)
import sys

print("Definindo configs...")
batch_size = 5
model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# "neuralmind/bert-large-portuguese-cased"
# "FacebookAI/xlm-roberta-base"
# "google-bert/bert-base-multilingual-cased"
# "neuralmind/bert-large-portuguese-cased"
# #"roberta-large"
task = "mrpc"
peft_type = PeftType.LORA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> USING DEVICE {device}")

num_epochs = 10
lr = 3e-4
padding_side = "right"
lora_r = 8
lora_alpha =16
lora_dropout = 0.2
# n_labels = 33
#data_dir = "preprocessing/data_one_label"
data_dir = "preprocessing/data_multilabel"


conjunto = 1
n_labels = 10
#with open(f"{data_dir}/total_label_count_interval.json", "r") as arquivo:
#    conjuntos_labels = json.load(arquivo)
#n_labels = conjuntos_labels[f"conjunto_{conjunto}"]
#print(f"CONJUNTO {conjunto} TEM {n_labels} LABELS! ")


def get_data_config():
    if len(sys.argv) < 1:
        print("Uso: python meu_script.py  <observacao>")
        sys.exit()

    if len(sys.argv) > 1:
        obs = sys.argv[1]
    else:
        obs = ""

    return obs
