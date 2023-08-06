import os, nltk, requests

import pandas as pd
import zipfile
from os import getcwd, path

from database_manager import database_manager as db_manager
from psycopg2.errors import UniqueViolation
from sqlalchemy.exc import IntegrityError
import traceback



def setup_ntlk():
    nltk.download('punkt')

    print('NLTK PUNKT setup finished')


ESSAY_PATH = 'datalake/essay/raw'
SHORT_ANSWER_PATH = 'datalake/short_answer/raw'

ESSAY_FILE = f"{ESSAY_PATH}/essays.xlsx"
SHORT_ANSWER_FILE = f"{SHORT_ANSWER_PATH}/short_answers.xlsx"


def create_datalake_dirs():
    os.makedirs(ESSAY_PATH, exist_ok=True)
    os.makedirs(SHORT_ANSWER_PATH, exist_ok=True)

def get_themes():
    ## getting themes
    theme_url ='https://raw.githubusercontent.com/vansoares/redacoes-crawler/main/redacoes_enem/redacoes_enem/results/temas-contexto.json'
    themes_json = requests.get(theme_url).json()
    for t in themes_json:
        try:
            if "context" in themes_json[t] and themes_json[t]["context"]  != "":
                title = themes_json[t]["title"]
                date = themes_json[t]["data"]
                context = themes_json[t]["context"]
                theme = dbManager.create_theme(title, date, context)

        except IntegrityError as e:
            assert isinstance(e.orig, UniqueViolation)  # proves the original exception
            print('Ja existe: {}'.format(themes_json[t]["title"]))
            continue

    print('-'*50)
    print('TEMAS SALVOS')
    print('-'*50)

def get_essays():
    CORPUS_ESSAYS = []
    url = 'https://raw.githubusercontent.com/vansoares/redacoes-crawler/main/redacoes_enem/redacoes_enem/results/tema-{}.json'
    for i in range(1, 170):
        theme_url = url.format(i)
        try:
            json = requests.get(theme_url).json()
            CORPUS_ESSAYS.append(json)

            for essay in json:
                theme = dbManager.get_theme_by_name(essay["tema"])
                essay = dbManager.create_essays(essay, theme, origin="VESTIBULAR_BRASIL_ESCOLA")

        except Exception as e:
            error_message = traceback.format_exc()
            print ('ERROR:', error_message)   

    print('-'*50)
    print('REDACOES SALVAS')
    print('-'*50)

def download_and_convert_brasil_escola_corpus_essay():

    get_themes()
    get_essays()
    
    #print(len(CORPUS_ESSAYS))

def download_and_convert_uol_corpus_essays():
    print("=================BAIXANDO REDAÇÕES=====================")
    url_base = 'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_{}.json'

    CORPUS_ESSAYS_JSONS = []
    for i in range (1, 11):
        index = str(i).zfill(2)
        url = url_base.format(index)
        json = requests.get(url).json()
        CORPUS_ESSAYS_JSONS.append(json)

    ESSAY_XLSX_COLUMNS = 'essay_id essay_set essay rater1_domain1 rater2_domain1 rater3_domain1 domain1_score rater1_domain2 rater2_domain2 domain2_score rater1_trait1 rater1_trait2 rater1_trait3 rater1_trait4 rater1_trait5 rater1_trait6 rater2_trait1 rater2_trait2 rater2_trait3 rater2_trait4 rater2_trait5 rater2_trait6 rater3_trait1 rater3_trait2 rater3_trait3 rater3_trait4 rater3_trait5 rater3_trait6'.split(
        ' ')

    corpus_essays_dict = {}

    for column in ESSAY_XLSX_COLUMNS:
        corpus_essays_dict[column] = []

    essay_id = 0

    for essay_set, corpus_essays_json in enumerate(CORPUS_ESSAYS_JSONS):
        try:

            corpus_essays = corpus_essays_json['redacoes']

            theme = dbManager.create_theme(corpus_essays_json["tema"], corpus_essays_json["data"],
                                        corpus_essays_json["contexto"])

            for essay in corpus_essays:
                print("Creating essay: {}".format(essay))
                essay_id += 1

                if len(essay['texto']) == 0: continue

                #corpus_essays_dict['essay_id'].append(essay_id)
                #corpus_essays_dict['essay_set'].append(1)
                #corpus_essays_dict['essay'].append(essay['texto'])
                #corpus_essays_dict['rater1_domain1'].append(essay['nota'] / 2)
                #corpus_essays_dict['rater2_domain1'].append(essay['nota'] / 2)
                #orpus_essays_dict['domain1_score'].append(essay['nota'])

                dbManager.create_essays(essay, theme, corpus_essays_json["data"])
                
        except IntegrityError as e:
            assert isinstance(e.orig, UniqueViolation)  # proves the original exception
            print('Ja existe: {}'.format(essay_set))
            continue 

    # https://stackoverflow.com/questions/61255750/convert-dictionary-of-dictionaries-using-its-key-as-index-in-pandas
    corpus_essay_dataframe = pd.DataFrame.from_dict(
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in corpus_essays_dict.items()])))

    corpus_essay_dataframe.to_excel(ESSAY_FILE, index=False)

if __name__ == '__main__':
    dbManager = db_manager.DatabaseManager()
    dbManager.create_tables()
    setup_ntlk()
    create_datalake_dirs()
    download_and_convert_uol_corpus_essays()
    #download_and_convert_brasil_escola_corpus_essay()