import pandas as pd
import pickle
from support import use_vectorizer


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

    nota1 = result[0][0]
    nota2 = result[0][1]
    nota3 = result[0][2]
    nota4 = result[0][3]
    nota5 = result[0][4]
        
    return nota1,nota2,nota3,nota4,nota5



 