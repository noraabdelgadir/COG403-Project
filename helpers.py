from embeddings import Word2VecMuse
LANGUAGES = {'English': 'en', 'French': 'fr', 'Arabic': 'ar'}
MODELS = ['fastText', 'Word2Vec']
EMBEDDING_MODEL = {'fasttext': Word2VecMuse, 'word2vec': Word2VecMuse}