import pandas as pd
from settings import OUTPUT_DF, EXCLUDE_COLS, ID_VARS, TARGETS_1
from xgboost import XGBRegressor
from report import prepare_report_table
from sklearn.metrics import cohen_kappa_score

def fit_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza um pipeline de treino e teste dentro de um conjunto de textos

    Recebe um único conjunto com textos de treino e de teste concatenados, considerando esse conjunto
    separa de volta os textos em treino e teste, treina o modelo com os textos de treino,
    e usa esse modelo para realizar previsões no conjunto de teste

    Args:
        df: conjunto de redações
    """

    df_train = df[df['group'] == 'train'].drop(columns=['group'])
    df_test = df[df['group'] == 'test'].drop(columns=['group'])

    # id_train = df_train[ID_VARS]
    X_train = df_train.drop(columns=EXCLUDE_COLS, errors='ignore')
    y_train = df_train[TARGETS_1].astype(float)

    ## treina o modelo
    xgb = XGBRegressor()
    fittado = xgb.fit(X_train, y_train)

    id_test = df_test[ID_VARS]
    X_test = df_test.drop(columns=EXCLUDE_COLS, errors='ignore')
    y_test = df_test[TARGETS_1].astype(float)

    PRED_COLS = [col + f'_pred' for col in TARGETS_1]

    preds = pd.DataFrame()
    preds[ID_VARS] = id_test
    preds[PRED_COLS] = xgb.predict(X_test)
    preds[PRED_COLS] = preds[PRED_COLS].astype(int)

    return preds


dict_pred_1 = {}
dict_pred_2 = {}

df_train_list = {
    "TF_IDF_32": pd.read_parquet(f"{OUTPUT_DF}/TF_IDF_32_train.parquet"),
    "TF_IDF_64": pd.read_parquet(f"{OUTPUT_DF}/TF_IDF_64_train.parquet"),
}

df_test_list = {
    "TF_IDF_32": pd.read_parquet(f"{OUTPUT_DF}/TF_IDF_32_test.parquet"),
    "TF_IDF_64": pd.read_parquet(f"{OUTPUT_DF}/TF_IDF_64_test.parquet"),
}

for key, value in df_train_list.items():
    df_train = df_train_list[key]
    df_test = df_test_list[key]

    print(key)

    df_train['group'] = 'train'
    df_test['group'] = 'test'
    df = pd.concat([df_train, df_test])
    df = df.drop(columns='texto', errors='ignore')
    pred1 = fit_predict(df)
    pred2 = df.groupby(['tema']).apply(lambda x: fit_predict(x)).reset_index(drop=True)
    dict_pred_1[key] = pred1
    dict_pred_2[key] = pred2

#### CREATE REPORT

print("GENERATING REPORT...")
dicio = {}

for key, value in df_test_list.items():
    df_pred_geral = dict_pred_1[key]
    df_pred_especifica = dict_pred_2[key]
    df_real = df_test_list[key]

    report_geral = prepare_report_table(df_real, df_pred_geral)
    report_especifica = prepare_report_table(df_real, df_pred_especifica)

    res = report_geral.groupby(['conceito']).apply(lambda x: cohen_kappa_score(x['nota'], x['previsao']),  include_groups=False)
    score_geral = pd.DataFrame(data=res).reset_index()
    score_geral.columns = ['conceito', 'score_geral']

    res = report_especifica.groupby(['conceito']).apply(lambda x: cohen_kappa_score(x['nota'], x['previsao']),  include_groups=False)
    score_especifica = pd.DataFrame(data=res).reset_index()
    score_especifica.columns = ['conceito', 'score_especifica']

    df_score = pd.merge(score_geral, score_especifica, on=['conceito'])

    df_score.to_json(f"{OUTPUT_DF}/report_{key}.json", indent=2)
