import os, nltk, requests

import pandas as pd

from urllib import request

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

def download_essays():
  ESSAY_LINK = 'https://zenodo.org/record/7641696/files/essays.xlsx?download=1'

  if os.path.isfile(ESSAY_FILE) == False:
    request.urlretrieve(ESSAY_LINK, ESSAY_FILE)

def download_short_answers():
  SHORT_ANSWER_LINK = 'https://zenodo.org/record/7641696/files/short_answers.xlsx?download=1'

  if os.path.isfile(SHORT_ANSWER_FILE) == False:
    request.urlretrieve(SHORT_ANSWER_LINK, SHORT_ANSWER_FILE)

def create_default_datalake():
  create_datalake_dirs()
  download_essays()
  download_short_answers()

def download_and_convert_uol_corpus_essays():
  if os.path.isfile(ESSAY_FILE) == True: return

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

  CORPUS_ESSAYS_JSONS = [requests.get(corpus_essay_json_link).json() for corpus_essay_json_link in CORPUS_ESSAYS_JSON_LINKS]

  ESSAY_XLSX_COLUMNS = 'essay_id essay_set essay rater1_domain1 rater2_domain1 rater3_domain1 domain1_score rater1_domain2 rater2_domain2 domain2_score rater1_trait1 rater1_trait2 rater1_trait3 rater1_trait4 rater1_trait5 rater1_trait6 rater2_trait1 rater2_trait2 rater2_trait3 rater2_trait4 rater2_trait5 rater2_trait6 rater3_trait1 rater3_trait2 rater3_trait3 rater3_trait4 rater3_trait5 rater3_trait6'.split(' ')

  corpus_essays_dict = {}

  for column in ESSAY_XLSX_COLUMNS:
    corpus_essays_dict[column] = []

  essay_id = 0

  for essay_set, corpus_essays_json in enumerate(CORPUS_ESSAYS_JSONS):
    corpus_essays = corpus_essays_json['redacoes']

    for essay in corpus_essays:
      essay_id += 1

      if len(essay['texto']) == 0: continue

      corpus_essays_dict['essay_id'].append(essay_id)
      corpus_essays_dict['essay_set'].append(1)
      corpus_essays_dict['essay'].append(essay['texto'])
      corpus_essays_dict['rater1_domain1'].append(essay['nota'] / 2)
      corpus_essays_dict['rater2_domain1'].append(essay['nota'] / 2)
      corpus_essays_dict['domain1_score'].append(essay['nota'])

  # https://stackoverflow.com/questions/61255750/convert-dictionary-of-dictionaries-using-its-key-as-index-in-pandas
  corpus_essay_dataframe = pd.DataFrame.from_dict(pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in corpus_essays_dict.items() ])))

  corpus_essay_dataframe.to_excel(ESSAY_FILE, index=False)

def create_corpus_datalake():
  create_datalake_dirs()
  download_and_convert_uol_corpus_essays()

if __name__ == '__main__':
  setup_ntlk()
  create_corpus_datalake()
