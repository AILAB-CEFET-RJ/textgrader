FILES = results/*
COMMIT_MESSAGE = "Atualizando arquivos de resultados"
OBS ?= ""

.PHONY: run_scripts clean

# Definir os nomes dos scripts
SCRIPT1 = lora-peft-confusion-matrix-competencies-v2.py conjunto_1_competence
SCRIPT2 = script2.py
SCRIPT3 = script3.py

# Definir a regra padrão que será executada quando você rodar "make"
all: run_scripts

# Regra para rodar os scripts sequencialmente
run_scripts:
	@echo "Executando $(SCRIPT1)..."
	python $(SCRIPT1)
	@echo "Executando $(SCRIPT2)..."
	python $(SCRIPT2)
	@echo "Executando $(SCRIPT3)..."
	python $(SCRIPT3)

# Limpar arquivos temporários ou resultados
clean:
	@echo "Limpando arquivos temporários..."
	# Insira os comandos para limpar arquivos aqui, se necessário