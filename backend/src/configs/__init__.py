import os
import logging

replacements = {'the the': 'the',
                'to to': 'to',
                'thats': "that\'s",
                'dont': "don\'t",
                'cause': 'because',
                'becouse': 'because',
                'cuz': 'because',
                'bc': 'because'}

ESSAY_CONTAINER = os.path.join('datalake', 'essay')
SHORT_ANSWER_CONTAINER = os.path.join('datalake', 'short_answer')
SHARED_CONTAINER = os.path.join('datalake', 'shared_container')

ESSAY_TEXT_RANGE = range(1, 2)
SHORT_ANSWER_TEXT_RANGE = range(1, 11)

CURRENT_TEXT_RANGE = None

RETRAIN_DOC_TO_VEC = True

ESSAY_ONLY = True

BYPASS_CORRECTOR = False

LOGLEVEL = logging.INFO

SETTINGS = {'essay': {'container': ESSAY_CONTAINER, 'range': list(range(1, 2))},
            'short_answer': {'container': SHORT_ANSWER_CONTAINER, 'range': list(range(1, 11))}
            }

SETTINGS_FILE = 'settings.json'

LSI_TOPIC_NUMBERS = [10, 20, 30, 40]

LSI_TOPIC_COLUMNS = ['text_set', '10_topics', '20_topics', '30_topics', '40_topics']
ESSAY_XLSX_COLUMNS = 'essay_id essay_set essay rater1_domain1 rater2_domain1 rater3_domain1 domain1_score rater1_domain2 rater2_domain2 domain2_score'

