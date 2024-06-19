import pandas as pd
from settings import OUTPUT_DF, TF_IDF_MAX_FEATURES
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords


# Baixe os recursos necessários
nltk.download('stopwords')

def vectorize_data(df, model_dict, id) -> pd.DataFrame:
    """
    Usa um vetorizador TF-IDF pré-treinado para obter a representação vetorial para o conjunto de dados

    Args:
        df: dataset que contém as redações
        model: vetorizador TF-IDF pré-treinado
    """
    lista = list(df['texto'])

    for key, value in model_dict.items():
        model = value

        ## realiza efetivamente a vetorização, transformando em uma matriz esparsa
        X = model.transform(lista)

        # transforma a matriz esparsa em um dataframe organizado com as frequencias TF-IDF das palavras
        df_vetorizado = pd.DataFrame(X.A, columns=model.get_feature_names_out())

        ## caso as redações contenham palavras que dão nome as features do datased
        ## (ex: texto, link, palavra) devemos removê-las

        colunas_a_remover = df.columns

        df_vetorizado = df_vetorizado.drop(columns=colunas_a_remover, errors='ignore')

        df = pd.concat([df, df_vetorizado], axis=1)
        df = df.drop(columns=['texto'], errors='ignore')

        print(key)
        #dicio[key] = df
        df.to_parquet(f"{OUTPUT_DF}/{key}_{id}.parquet")
    #retorno = dicio

    #return dicio
    print(f"Vectorized {id}!")


df_train = pd.read_parquet(f"{OUTPUT_DF}/df_train.parquet", engine="pyarrow")
df_test = pd.read_parquet(f"{OUTPUT_DF}/df_test.parquet")

lista = list(df_train['texto'])

dicio_tf_idf = {}

for count in TF_IDF_MAX_FEATURES:
    ## cria o vetorizador que utiliza TF-IDF
    stop_words = stopwords.words('portuguese')
    tv = TfidfVectorizer(min_df = 25,max_features=count)

    ## treina o vetorizador com o conjunto fornecido para treino
    vectorizer_vez = tv.fit(lista)

    key = "TF_IDF_" + str(count)

    # prepara o dicionário para versionamento
    dicio_tf_idf[key] = vectorizer_vez

vectorize_data(df_train,dicio_tf_idf, "train")
vectorize_data(df_test,dicio_tf_idf, "test")

print("VECTORIZE: done!")