import numpy as np
import pandas as pd
import pickle
from flask import Flask, jsonify, request
from utils import process_data

app = Flask(__name__)

@app.route("/")
def primeiro_endpoint_get():
  return ("Tudo Funcionando Corretamente !", 200) 

@app.post("/model")
def segundo_endpoint():
    with open("xgboost_model.pkl","rb") as f:
        modelo_carregado = pickle.load(f)
    
    df = process_data(request.json['essay'])

    pred = modelo_carregado.predict(df)
    return (f"essay: {request.json} \n grades: {pred}", 200)

  

if __name__ == "__main__":
  debug = True # com essa opção como True, ao salvar, o "site" recarrega automaticamente.
  app.run(host='0.0.0.0', port=5000, debug=debug)