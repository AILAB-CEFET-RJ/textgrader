from pymongo import MongoClient
import os


class MongoDB:
    def __init__(self):
        conn_string = os.getenv("MONGO_STR")
        self.client = MongoClient(conn_string)

    def save(self, content):
        db = self.client['textgrader']
        collection = db['results']

        collection.insert_one(content)

        print("Documento salvo com sucesso!")
