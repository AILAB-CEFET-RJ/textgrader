# text-grader frontend

### Instruções para rodar o text-grader

Para rodar a aplicação frontend localmente, temos duas opções:
1 - Rodar via docker composeÇ
    - Caso não tenha, instalar o docker-compose localmente;
    - Para rodar a aplicação rode make start (e caso queira parar a aplicação, rode make stop)
    - Acessar http://localhost:3000/essays/
        A partir daí será possível navegar pela aplicação, adicionando temas e redações

2 - Rodar instalando o django localmente:
    - instalar django local > https://docs.djangoproject.com/en/4.2/topics/install/
    - Rodar os seguintes comandos:
        - python manage.py migrate (para aplicar ao banco local as migrações das entidades de banco)
        - python manage.py runserver 0.0.0.0:3000 (para rodar a aplicação)
    - Acessar http://localhost:3000/essays/
        A partir daí será possível navegar pela aplicação, adicionando temas e redações


