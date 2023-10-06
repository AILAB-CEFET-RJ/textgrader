import os, nltk, requests

import pandas as pd

from database_manager import database_manager as db_manager
import configs.configs as configs


def setup_ntlk():
    nltk.download('punkt')

    print('NLTK PUNKT setup finished')


ESSAY_PATH = 'datalake/essay/raw'
SHORT_ANSWER_PATH = 'datalake/short_answer/raw'

ESSAY_FILE = f"{ESSAY_PATH}/essays.xlsx"
SHORT_ANSWER_FILE = f"{SHORT_ANSWER_PATH}/short_answers.xlsx"


def create_datalake_dirs():
    os.makedirs(ESSAY_PATH, exist_ok=True, mode=0o777)
    os.makedirs(SHORT_ANSWER_PATH, exist_ok=True, mode=0o777)
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
                essay = dbManager.create_essays(essay, theme)

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

    CORPUS_ESSAYS_JSON_LINKS = [
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_01.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_02.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_03.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_04.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_05.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_06.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_07.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_08.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_09.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_10.json',
    ]

    CORPUS_ESSAYS_JSONS = [requests.get(corpus_essay_json_link).json() for corpus_essay_json_link in
                           CORPUS_ESSAYS_JSON_LINKS]

    corpus_essays_dict = {}

    for column in configs.ESSAY_XLSX_COLUMNS.split(' '):
        corpus_essays_dict[column] = []

    essay_id = 0

    for _, corpus_essays_json in enumerate(CORPUS_ESSAYS_JSONS):
        corpus_essays = corpus_essays_json['redacoes']

        theme = dbManager.create_theme(corpus_essays_json["tema"], corpus_essays_json["data"],
                                       corpus_essays_json["contexto"])

        for essay in corpus_essays:
            essay_id += 1

            if len(essay['texto']) == 0: continue

            corpus_essays_dict['essay_id'].append(essay_id)
            corpus_essays_dict['essay_set'].append(1)
            corpus_essays_dict['essay'].append(essay['texto'])
            corpus_essays_dict['rater1_domain1'].append(essay['nota'] / 2)
            corpus_essays_dict['rater2_domain1'].append(essay['nota'] / 2)
            corpus_essays_dict['domain1_score'].append(essay['nota'])

            dbManager.create_essays(essay, theme, corpus_essays_json["data"])

    # https://stackoverflow.com/questions/61255750/convert-dictionary-of-dictionaries-using-its-key-as-index-in-pandas
    corpus_essay_dataframe = pd.DataFrame.from_dict(
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in corpus_essays_dict.items()])))

    corpus_essay_dataframe.to_excel(ESSAY_FILE, index=False)

if __name__ == '__main__':
    dbManager = db_manager.DatabaseManager()
    dbManager.create_tables()
    #dbManager.get()
    setup_ntlk()
    create_datalake_dirs()
    #download_and_convert_uol_corpus_essays()
    download_and_convert_brasil_escola_corpus_essay()
