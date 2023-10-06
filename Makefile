start:
	docker compose up -d

stop:
	docker compose down

start-db:
	docker run -p 5432:5432 -v /tmp/database:/var/lib/postgresql/data -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=text-grader postgres:alpine