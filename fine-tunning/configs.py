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
        self.model_name_or_path = "xlm-roberta-base"
        # "neuralmind/bert-large-portuguese-cased"
        # "FacebookAI/xlm-roberta-base"
        # "google-bert/bert-base-multilingual-cased"
        # "neuralmind/bert-large-portuguese-cased"
        # #"roberta-large"

        self.task = "mrpc"
        self.peft_type = PeftType.LORA

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"> USING DEVICE {self.device}")

        self.num_epochs = 8
        self.lr = 3e-4
        self.padding_side = "left" #"right"
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        # n_labels = 33
        #data_dir = "preprocessing/data_one_label"

        self.conjunto = 2
        self.data_dir = f"preprocessing/data_competencias/conjunto_{self.conjunto}"
        self.checkpoints_dir = f"checkpoints/{self.conjunto}"

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
        self.cohen = None
        self.except_message = None
        self.patience = 5

    def set_conjunto(self, conjunto):
        self.conjunto = conjunto
        self.data_dir = f"preprocessing/data_competencias/conjunto_{conjunto}"

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
            "cohen": self.cohen,
            "except_message": self.except_message,
        }

    def save_to_json(self, confusion_matrix=None):
        folder_path = (
            f"results/{self.script_type}/{self.conjunto}/{self.date}/{self.competence}"
        )
        os.makedirs(folder_path, exist_ok=True)

        if confusion_matrix is not None:
            confusion_matrix.to_csv(f"{folder_path}/confusion_matrix.csv", index=False)

        print(f"Results saved to {folder_path}/results.json")
        with open(f"{folder_path}/results.json", 'w') as json_file:
            json.dump(self.to_dict(), json_file, indent=4)

    def get_competencies_from_set(self):
        if self.conjunto == 1:
            return [
                "dominio_da_modalidade_escrita_formal",
                "compreender_a_proposta_e_aplicar_conceitos_das_varias_areas_de_conhecimento_para_desenvolver_o_texto_dissertativoargumentativo_em_prosa",
                "selecionar_relacionar_organizar_e_interpretar_informacoes_em_defesa_de_um_ponto_de_vista",
                "conhecimento_dos_mecanismos_linguisticos_necessarios_para_a_construcao_da_argumentacao",
                "proposta_de_intervencao_com_respeito_aos_direitos_humanos"
            ]
        if self.conjunto == 2:
            return [
                "adequacao_ao_tema",
                "adequacao_e_leitura_critica_da_coletanea",
                "adequacao_ao_genero_textual",
                "adequacao_a_modalidade_padrao_da_lingua",
                "coesao_e_coerencia"
            ]

        return [
            "conteudo",
            "estrutura_do_texto",
            "estrutura_de_ideias",
            "vocabulario",
            "gramatica_e_ortografia"
        ]