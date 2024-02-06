from kedro.pipeline import Pipeline, node, pipeline

from textgrader.pipelines.config import *
from textgrader.pipelines.functions import *
from textgrader.pipelines.nodes import *



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            

                            
            node(
                func= regression_fit,
                inputs= ['vectorized_train_list','vectorized_test_list'],
                outputs= 'reg_pred_geral_lista',
                name="fit_predict_reg",
            ),
 
        ]
    )
