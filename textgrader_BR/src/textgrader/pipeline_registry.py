#Imports Gerais
from typing import Dict
from kedro.pipeline import Pipeline

#Imports das Pipelines do projeto

 
 
from .pipelines import pipeline_basico as sdp
from .pipelines import regressao as rg
from .pipelines import classificacao as cls

def register_pipelines() -> Dict[str, Pipeline]:
    
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
 
       
        'textgrader': sdp.create_pipeline() + rg.create_pipeline() + cls.create_pipeline()
    }

  
    return pipelines