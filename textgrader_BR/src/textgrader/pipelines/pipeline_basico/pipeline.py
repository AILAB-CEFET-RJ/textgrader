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
                outputs= 'vectorizer_list',
                name="fit_vectorizer",
            ),


            node(
                func = vectorize_all,
                inputs=["basic_train","basic_test",'vectorizer_list'],
                outputs= ['vectorized_train_list','vectorized_test_list'],
                name="vectorize_train",
            ),

        ]
    )
