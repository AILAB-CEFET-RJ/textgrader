---
sidebar_position: 1
---

# Overview

Let's check Textgrader **Backend Project Overview** in 5 minutes.

## Technologies

Textgrader API is a [FastAPI](https://fastapi.tiangolo.com/) project, so it uses FastAPI concepts that are avaiable with more details [here](https://fastapi.tiangolo.com/tutorial/first-steps/). Textgrader was originally a AI project, in this page we will cover just what is more relevant for Textgrader API use case.

Along with [FastAPI](https://fastapi.tiangolo.com/) this project also uses:

1. [Python](https://www.python.org/) (Python is a high-level, general-purpose programming language)
2. [FastAPI](https://fastapi.tiangolo.com/) (FastAPI is a modern, fast (high-performance), web framework for building APIs with Python)
3. [TensorFlow](https://www.tensorflow.org/) (TensorFlow is a free and open-source software library for machine learning and artificial intelligence.)
4. [Scikit-learn](https://scikit-learn.org/stable/index.html) (Scikit-learn is a free software machine learning library for the Python programming language)

## Running the Project Natively

Besides using [Docker](https://www.docker.com/), it's also possible to run the backend project natively, for this you will need installed on your machine:

1. [Python](https://www.python.org/) (Install 3.10 version)

With this installed you just need to run the commands bellow in ``backend`` folder:

```bash
python -m venv venv && . venv/bin/activate && pip install -U pip

pip install -U "jax[cpu]" tensorflow-cpu tensorflow_text tensorflow_hub scikit-learn nltk gensim pyarrow

pip install -U pandas fastapi "uvicorn[standard]" orjson openpyxl colorlog pyspellchecker coloredlogs

cd src && python requirements.py && python run.py task general_tasks task_correct_essays

python run.py task general_tasks task_pipeline_essays && uvicorn api:app --host 0.0.0.0 --reload"
```

## File Tree Structure

```txt title="/backend"
├── src                                     | Backend source directory
│   ├── README.md                           | Textgrader AI README.md
│   ├── api.py                              | Backend API code
│   ├── dags                                | Textgrader AI folder
│   │   ├── config.py                       | Textgrader AI configuration file
│   │   ├── escondo                         | Textgrader AI folder
│   │   │   ├── general.py                  | Textgrader AI code
│   │   │   ├── runner.py                   | Textgrader AI code
│   │   │   └── simple.py                   | Textgrader AI code
│   │   ├── experiments                     | Textgrader AI folder
│   │   │   ├── experiments.py              | Textgrader AI code
│   │   │   └── feature_engineering.py      | Textgrader AI code
│   │   ├── feature_engineering             | Textgrader AI folder
│   │   │   ├── generate_features.py        | Textgrader AI code
│   │   │   ├── latent_semantic_indexing.py | Textgrader AI code
│   │   │   └── word_embeddings.py          | Textgrader AI code
│   │   ├── general_tasks.py                | Textgrader AI code
│   │   ├── logging.py                      | Textgrader AI code
│   │   ├── model_training                  | Textgrader AI folder
│   │   │   ├── doc_to_vec_trainer.py       | Textgrader AI code
│   │   │   └── train_model.py              | Textgrader AI code
│   │   ├── predict                         | Textgrader AI folder
│   │   │   ├── predict.py                  | Textgrader AI code
│   │   │   └── predict_from_text.py        | Backend code to grade essays on real time
│   │   ├── preprocessing                   | Textgrader AI folder
│   │   │   ├── general_preprocessing.py    | Textgrader AI code
│   │   │   ├── spell_correct.py            | Textgrader AI code
│   │   │   └── word_features.py            | Textgrader AI code
│   │   ├── run.py                          | Textgrader AI code
│   │   ├── runner.py                       | Textgrader AI code
│   │   └── utils.py                        | Textgrader AI code
│   ├── requirements.py                     | Backend setup script
│   └── run.py                              | Textgrader AI code
└── vercel.json                             | Backend vercel configuration file
```
