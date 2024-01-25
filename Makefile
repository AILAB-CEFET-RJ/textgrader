COMPOSE_FILE := docker-compose.yml

# Comandos
DOCKER_COMPOSE := docker-compose

# Alvos Padrão
up: docker-up
down: docker-down

# Inicializa os serviços do Docker Compose
docker-up:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up --build

# Derruba os serviços do Docker Compose
docker-down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down
