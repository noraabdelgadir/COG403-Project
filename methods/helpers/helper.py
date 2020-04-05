from helpers.embeddings import Word2VecMuse, Bert, Glove
from helpers.categories import *

LANGUAGES = {'English': 'en', 'French': 'fr', 'Arabic': 'ar'}
MODELS = ['fastText', 'Word2Vec', 'Bert', 'GLoVE']
EMBEDDING_MODEL = {
    'fasttext': Word2VecMuse, 
    'word2vec': Word2VecMuse, 
    'bert': Bert, 
    'glove': Glove
}

CATEGORY_TO_LIST = {
    'insect': INSECTS, 
    'shape': SHAPES, 
    'weather': WEATHER, 
    'food': FOOD, 
    'bird': BIRDS, 
    'mammal': MAMMALS, 
    'colour': COLOURS, 
    'flower': FLOWERS, 
    'country': COUNTRIES, 
    'vehicle': VEHICLES, 
    'sport': SPORTS
}

FR_CATEGORY_TO_LIST = {
    'insect': FR_INSECTS, 
    'shape': FR_SHAPES, 
    'weather': FR_WEATHER, 
    'food': FR_FOOD, 
    'bird': FR_BIRDS, 
    'mammal': FR_MAMMALS, 
    'colour': FR_COLOURS, 
    'flower': FR_FLOWERS, 
    'country': FR_COUNTRIES, 
    'vehicle': FR_VEHICLES, 
    'sport': FR_SPORTS
}

AR_CATEGORY_TO_LIST = {
    'insect': AR_INSECTS, 
    'shape': AR_SHAPES, 
    'weather': AR_WEATHER, 
    'food': AR_FOOD, 
    'bird': AR_BIRDS, 
    'mammal': AR_MAMMALS, 
    'colour': AR_COLOURS, 
    'flower': AR_FLOWERS, 
    'country': AR_COUNTRIES, 
    'vehicle': AR_VEHICLES
}
