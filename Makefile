COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE := docker-compose

up:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up --build

down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down --remove-orphans

remove-logs:
	cd backend/src && sudo rm -r *_log.txt

execute-pipeline:
	cd textgrader-pt-br && python create_dataset.py

setup:
	pip install -r textgrader-pt-br/requirements.txt

run-all:
	cd textgrader-pt-br/scripts && \
	python create_dataset.py && \
	python extract_features.py && \
	python vectorize.py && \
	python fit_predict.py