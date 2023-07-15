run:
	docker compose up -d

stop:
	docker compose down

update-requirements:
	pip3 freeze > backend/src/requirements.txt
