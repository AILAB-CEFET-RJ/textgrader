import os
from datetime import datetime

import torch
import json
from peft import (
    PeftType,
)
import sys


class Configs:
    def __init__(self):
        print("Definindo configs...")
        self.obs = None
        self.batch_size = 4
        self.model_name_or_path = "FacebookAI/xlm-roberta-base"
        # "neuralmind/bert-large-portuguese-cased"
        # "FacebookAI/xlm-roberta-base"
        # "google-bert/bert-base-multilingual-cased"
        # "neuralmind/bert-large-portuguese-cased"
        # #"roberta-large"

        self.task = "mrpc"
        self.peft_type = PeftType.LORA

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"> USING DEVICE {self.device}")

        self.num_epochs = 10
        self.lr = 3e-4
        self.padding_side = "right"
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.2
        # n_labels = 33
        #data_dir = "preprocessing/data_one_label"
        self.data_dir = "preprocessing/data_competencias/conjunto_1"

        self.conjunto = 1
        self.n_labels = 11
        self.sufix = "dominio_da_modalidade_escrita_formal"
        #with open(f"{data_dir}/total_label_count_interval.json", "r") as arquivo:
        #    conjuntos_labels = json.load(arquivo)
        #n_labels = conjuntos_labels[f"conjunto_{conjunto}"]
        #print(f"CONJUNTO {conjunto} TEM {n_labels} LABELS! ")

        self.date = datetime.now().strftime("%d-%m-%Y-%H-%M")
        self.competence = "dominio_da_modalidade_escrita_formal"
        self.processing_time = None
        self.validation_metric = None
        self.metrics = {}
        self.script_type = None

    def get_data_config(self):
        if len(sys.argv) < 1:
            print("Uso: python meu_script.py  <observacao>")
            sys.exit()

        if len(sys.argv) > 1:
            obs = sys.argv[1]
        else:
            obs = ""

        self.obs = obs

    def to_dict(self):
        return {
            "obs": self.obs,
            "batch_size": self.batch_size,
            "model_name_or_path": self.model_name_or_path,
            "task": self.task,
            "peft_type": str(self.peft_type),
            "device": str(self.device),
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "padding_side": self.padding_side,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "data_dir": self.data_dir,
            "conjunto": self.conjunto,
            "n_labels": self.n_labels,
            "sufix": self.sufix,
            "competence": self.competence,
            "date": self.date,
            "processing_time": self.processing_time,
            "validation_metric": self.validation_metric,
            "metrics": self.metrics,
            "script_type": self.script_type,
        }

    def save_to_json(self, confusion_matrix=None):
        os.makedirs(self.script_type, exist_ok=True)
        folder_path = (
            f"results/{self.date}-conjunto{self.conjunto}-{self.num_epochs}-epochs-{self.competence}"
        )
        os.makedirs(folder_path, exist_ok=True)

        if confusion_matrix is not None:
            confusion_matrix.to_csv(f"{folder_path}/confusion_matrix.csv", index=False)

        print(f"Results saved to {folder_path}/results.json")
        with open(f"{folder_path}/results.json", 'w') as json_file:
            json.dump(self.to_dict(), json_file, indent=4)