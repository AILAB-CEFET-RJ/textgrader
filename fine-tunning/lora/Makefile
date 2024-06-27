FILES = results/*
COMMIT_MESSAGE = "Atualizando arquivos de resultados"
OBS ?= ""

.PHONY: add commit push git-all run-conjunto2 run-conjunto1

add:
	git add $(FILES)

commit:
	git commit -m $(COMMIT_MESSAGE)

push:
	git push

git-all: add commit push

run-conjunto1:
	nohup python lora-peft-v2.py 1 $(OBS)
	sleep 10


run-conjunto2:
	nohup python lora-peft-v2.py 2 $(OBS)
	sleep 10

all: run-conjunto1 run-conjunto2 git-all