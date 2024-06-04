COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE := docker-compose

up:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up --build

down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down --remove-orphans

remove-logs:
	cd backend/src && sudo rm -r *_log.txt