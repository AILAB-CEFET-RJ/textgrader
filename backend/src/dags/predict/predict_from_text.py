import pandas as pd

from dags.utils import *
from dags.feature_engineering.word_embeddings import *

def predict_from_text(text: str = "testando 123 testando"):
    ESSAY_COLUMNS = 'essay_id essay_set essay new_text rater1_domain1 rater2_domain1 rater3_domain1 domain1_score rater1_domain2 rater2_domain2 domain2_score rater1_trait1 rater1_trait2 rater1_trait3 rater1_trait4 rater1_trait5 rater1_trait6 rater2_trait1 rater2_trait2 rater2_trait3 rater2_trait4 rater2_trait5 rater2_trait6 rater3_trait1 rater3_trait2 rater3_trait3 rater3_trait4 rater3_trait5 rater3_trait6'.split(' ')

    essays_dict = {}

    for column in ESSAY_COLUMNS:
        essays_dict[column] = []

    essays_dict['essay'].append(text)

    essays_dict['new_text'].append(text)

    essays_df = pd.DataFrame.from_dict(pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in essays_dict.items() ])))

    essays_df = essays_df[essays_df['new_text'] != "-"]

    essays_df = essays_df.rename(columns = {"essay_id":"text_id","essay_set":"text_set","essay":"text"})
    essays_df = pd.melt(essays_df,id_vars = ["text_id","text_set","text","new_text"], value_vars = ["domain1_score","domain2_score"])

    essays_df.columns = ['text_id', 'text_set', 'text', 'new_text', 'domain', 'score']

    essays_df["domain"] = essays_df["domain"].replace({"domain1_score":1,"domain2_score":2})

    essays_df['word_count'] = essays_df['new_text'].apply(lambda x: word_count(x))
    essays_df['sentence_count'] = essays_df['new_text'].apply(lambda x: sentence_count(x))
    essays_df['unique_word_count'] = essays_df['new_text'].apply(lambda x: word_count_unique(x))

    embedder = doc_2_vec_embedder(vector_size=512)

    essays_df = embedder.generate_embed_features(essays_df)

    model_directory = os.path.join(os.path.join('datalake','essay'),'model','set_1','domain_1')
    model_filename = 'lsi_40_topics_rf_model.pkl'

    model = get_model_from_pickle(model_directory,model_filename)

    X = essays_df[model.feature_names_in_]

    y_pred = model.predict(X)

    return y_pred[0] * 100
