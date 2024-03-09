from flask import Flask, request
from functions import use_vectorizer, evaluate_redacao
app = Flask(__name__)

@app.route("/")
def primeiro_endpoint_get():
  return ("OK!", 200) 

@app.post("/model")
def segundo_endpoint():
    essay = request.json['essay']
    
    nota1,nota2,nota3,nota4,nota5 = evaluate_redacao(essay)

    obj = {
        "nota1": nota1,
        "nota2": nota2,
        "nota3": nota3,
        "nota4": nota4,
        "nota5": nota5
    }
    return (obj, 200)

  

if __name__ == "__main__":
  from support import use_vectorizer
  debug = True # com essa opção como True, ao salvar, o "site" recarrega automaticamente.
  app.run(host='0.0.0.0', port=5000, debug=debug)