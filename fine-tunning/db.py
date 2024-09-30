from pymongo import MongoClient
import json


class MongoDB:
    def __init__(self):
        # Conectando ao MongoDB (substitua pela string de conexão adequada, se necessário)
        self.client = MongoClient('mongodb://localhost:27017/')

    def save(self, content):
        db = self.client['textgrader']
        collection = db['results']

        collection.insert_one(content)

        print("Documento salvo com sucesso!")
