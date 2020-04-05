from embeddings import Word2VecMuse, Bert, Glove
from categories import FR_CATEGORIES

LANGUAGES = {'English': 'en', 'French': 'fr', 'Arabic': 'ar'}
MODELS = ['fastText', 'Word2Vec', 'Bert', 'GLoVE']
EMBEDDING_MODEL = {'fasttext': Word2VecMuse, 'word2vec': Word2VecMuse, 'bert': Bert, 'glove': Glove}
