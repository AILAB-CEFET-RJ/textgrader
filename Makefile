COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE := docker-compose

up: docker-up
down: docker-down

docker-up:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up --build

docker-down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down

remove-logs:
	cd backend/src && sudo rm -r *_log.txt

create-network:
	docker network create app_network
