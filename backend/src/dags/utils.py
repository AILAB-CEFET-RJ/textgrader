import glob
import json
import os
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize


def word_count(x):
    lista = word_tokenize(x)
    return len(lista)


def word_count_unique(x):
    lista = set(word_tokenize(x))
    return len(lista)


def sentence_count(x):
    lista = sent_tokenize(x)
    return len(lista)


def generate_word_sentence_features(df, column='new_text'):
    df['word_count'] = df[column].apply(lambda x: word_count(x))
    df['unique_word_count'] = df[column].apply(lambda x: word_count_unique(x))
    df['sentence_count'] = df[column].apply(lambda x: sentence_count(x))

    return df


def save_parquet(df, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_parquet(os.path.join(directory, filename))


def save_as_pickle(model, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    pickle.dump(model, open(file_path, 'wb'))


def get_model_from_pickle(directory, filename):
    file_path = os.path.join(directory, filename)

    loaded_model = pickle.load(open(file_path, 'rb'))

    return loaded_model



def dict_to_json(dictionary, name):
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open(f'{name}', "w") as outfile:
        outfile.write(json_object)

