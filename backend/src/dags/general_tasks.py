from dags.experiments.experiments import *
from dags.model_training.doc_to_vec_trainer import *
from dags.preprocessing.general_preprocessing import *
from dags.preprocessing.spell_correct import *

import logging

from configs import configs

# from dags.feature_engineering.generate_features import *

logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


## tasks específicas de essays


def run_dag(task):
    print(f"RUN DAG - GENERAL TASKS = {task}")
    from dags.runner import run_task

    run_task(task)


def task_correct_essays():
    print("TASK CORRECT ESSAY")
    ec = essay_corrector()

    if config.BYPASS_CORRECTOR:
        ec.bypass_correction()
    else:
        ec.correct_texts()

    logger.info("finished to correct essays")


def task_generate_essay_datasets():
    print("TASK GENERATE ESSAY DATASETS")
    ep = essay_preprocessor()
    ep.generate_datasets()
    logger.info("finished to generate essay datasets")


## tasks especificas de short answer

def task_correct_short_answers():
    print("TASK CORRECT SHORT ANSWER")
    sac = short_answer_corrector()
    sac.correct_texts()


def task_generate_short_answer_datasets():
    print("TASK GENERATE SHORT ANSWER DATASETSS")
    sap = answer_preprocessor()
    sap.generate_datasets()


## tasks gerais 


def task_generate_features(selected_container, text_range):
    print("TASK GENERATE FEATURES")
    generate_features(selected_container, text_range)


def task_train_models(selected_container, text_range):
    print("TASK TRAIN MODELS")
    experiments_with_doc_2_vec(selected_container, text_range)
    experiments_with_universal_sentence_encoder(selected_container, text_range)
    experiments_with_lsi_topics(selected_container, text_range)


def task_evaluate_models(selected_container, text_range):
    print("TASK EVALUATE MODELS")
    evaluate_lsi_predictions(selected_container, text_range)
    evaluate_universal_sentence_encoder_predictions(selected_container, text_range)
    evaluate_doc_to_vec_predictions(selected_container, text_range)


def task_train_doc_2_vec():
    print("TASK TRAINS DOC 2 VEC")
    train_doc_to_vec()


def task_shared_tasks(selected_container, text_range):
    print("TASK SHARED TASKS")
    if config.RETRAIN_DOC_TO_VEC:
        task_train_doc_2_vec()

    task_generate_features(selected_container, text_range=text_range)
    task_train_models(selected_container, text_range=text_range)
    task_evaluate_models(selected_container, text_range=text_range)

def task_pipeline_essays():
    print("TASK PIPELINE ESSAYS")
    task_correct_essays()
    task_generate_essay_datasets()
    task_shared_tasks(selected_container=configs.ESSAY_CONTAINER, text_range=configs.ESSAY_TEXT_RANGE)


def task_pipeline_short_answer():
    print("TASK PIPELINE SHORT ANSWER")
    task_correct_short_answers()
    logger.info("finished to correct short answers")

    task_generate_short_answer_datasets()
    print("finished to generate short answers datasets")

    task_shared_tasks(selected_container=configs.SHORT_ANSWER_CONTAINER, text_range=configs.SHORT_ANSWER_TEXT_RANGE)
    print("finished to generate short answers datasets")
