from datetime import datetime 
import json
import os


def save(content):
    pasta_destino = 'essays'
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    agora = datetime.now()
    filename = f"{pasta_destino}/{agora.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    with open(filename, 'w', encoding='utf-8') as arquivo:
        json.dump(content, arquivo)

    print(f'Dados foram salvos em {filename}')