import pandas as pd
from settings import ID_VARS, TARGETS_1,ALL_TARGETS


def prepare_report_table(df_real: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Junta o conjunto de teste e as previsões feitas

    Juntamos o conjunto de treino com o conjunto de teste, fazemos um melt para
    transformar cada conceito em uma linha, e, com isso ficar mais fácil para realizar a avaliaçao
    do desempenho preditivo em etapas futuras

    Args:
        df_real: dataframe com o conjunto de teste
        df_pred: dataframe com as previsões realizadas tendo em mente
        o conjunto de teste
    """

    df_real = df_real[ID_VARS + ALL_TARGETS]
    df = pd.merge(df_real, df_pred, on=['index', 'tema', 'conjunto'], suffixes=['_real', '_pred'])
    df = df.melt(id_vars=['index', 'tema', 'conjunto'])
    df['valor'] = df['variable'].transform(lambda x: x.split('_')[-1])
    df['conceito'] = df['variable'].transform(lambda x: x.split('_')[0])
    df = df.drop(columns='variable')
    df = df.set_index(['index', 'tema', 'conjunto', 'conceito', 'valor']).unstack('valor').reset_index()
    df.columns = ['index', 'tema', 'conjunto', 'conceito', 'nota', 'previsao']

    return df

