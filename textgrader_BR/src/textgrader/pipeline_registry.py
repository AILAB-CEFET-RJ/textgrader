#Imports Gerais
from typing import Dict
from kedro.pipeline import Pipeline

#Imports das Pipelines do projeto

 
 
from .pipelines import second_data_proc as sdp
 

def register_pipelines() -> Dict[str, Pipeline]:
    
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
 
        'textgrader': sdp.create_pipeline()
    }

  
    return pipelines