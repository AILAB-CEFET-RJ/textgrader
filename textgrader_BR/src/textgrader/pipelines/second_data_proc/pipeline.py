from kedro.pipeline import Pipeline, node, pipeline

from textgrader.pipelines.config import *
from textgrader.pipelines.functions import *
from textgrader.pipelines.nodes import *



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
             node(
                func=get_text_jsons,
                inputs="texts_json",
                outputs="all_texts_parquet",
                name="get_text_jsons",
            ),


            node(
                func=preprocess_targets,
                inputs="all_texts_parquet",
                outputs="all_texts_with_targets",
                name="extract_targets",
            ),

            node(
                func=generate_basic_features,
                inputs="all_texts_with_targets",
                outputs= ['basic_train','basic_test'],
                name="generate_basic_features",
            ),

            node(
                func=fit_vectorizer,
                inputs="basic_train",
                outputs= 'vectorizer',
                name="fit_vectorizer",
            ),


            node(
                func = vectorize_all,
                inputs=["basic_train","basic_test",'vectorizer'],
                outputs= ['primeiro_treino','primeiro_teste'],
                name="vectorize_train",
            ),

                            
            node(
                func= fit_predict_both_ways,
                inputs= ['primeiro_treino','primeiro_teste'],
                outputs= ['primeira_pred_geral','primeira_pred_especifica'],
                name="fir_predict_both_ways",
            ),

            node(
                func= prepare_reports,
                inputs= ['primeiro_teste','primeira_pred_geral','primeira_pred_especifica'],
                outputs= 'final_scores_experiment',
                name="report_geral",
            ),

        ]
    )
