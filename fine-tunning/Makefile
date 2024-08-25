FILES = results/*
COMMIT_MESSAGE = "Atualizando arquivos de resultados"
OBS ?= ""

.PHONY: conda clean

conda:
	@conda activate lora-2
# Limpar arquivos temporários ou resultados
clean:
	@echo "Limpando arquivos temporários..."
	rm -r nohup.out
