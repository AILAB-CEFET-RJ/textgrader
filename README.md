# Textgrader

This repository contains TextGrader. In essence textgrader contains the various versions of a Essay and short answer evaluation system.

The system has 4 parts:

1) Preprocessing where we correct spelling change columns schema and do other minor preprocessing steps
2) Feature engineering, where we generate some basic features like word count and sentence count, and generate datasets embedding words 
with each one of the following 4 techniques: TF-IDF, WORD-2-VEC, USE, LSI.
3) Model training, where we train a random forest model using one of the following 3 approaches: Regression, Classification and Ordinal Classification. 
4) Model Evaluation, where we use the trained models to generate predictions and evaluate those predictions. 

# Requirements 
Our code uses python 3.9 and some libraries like scikit-learn and NLTK. To easily setup your environment run pip install requirements.txt

## Project elaborated by Ramon Grande Da Luz Bouças as 
## a parcial pre-requirement for obtaining a bachelor of science in computer science 
## at CEFET/RJ advisor Eduardo Bezerra Da Silva D.Sc
