import pandas as pd
import pickle
from support import use_vectorizer
import os
from datetime import datetime
import json


def use_vectorizer(df_train):
    vectorizer_vez = pickle.load(open('vectorizer.pkl','rb'))
    ## realiza efetivamente a vetorização, transformando em uma matriz esparsa
    X = vectorizer_vez.transform(df_train['texto'])
    
    # transforma a matriz esparsa em um dataframe organizado com as frequencias TF-IDF das palavras 
    df_vetorizado = pd.DataFrame(X.A, columns=vectorizer_vez.get_feature_names_out())

    return df_vetorizado

def evaluate_redacao(redacao):
   
    tupla = (redacao, )
    texto_df = pd.DataFrame(tupla,columns = ['texto'])
    modelo_salvo = pickle.load(open('model.pkl','rb'))
    result = modelo_salvo.predict(texto_df)

    notas = {}
    for i in range(1,6):
        key = f"nota_{i}"
        notas[key] = result[0][i-1]

    return notas



def persist_essay(essay, grades):
    if not os.path.exists('essays'):
        os.makedirs('essays')

    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S")
        
    obj = {"essay": essay, "grades": grades, "date": filename}
    with open (f'essays/{filename}.json', 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)