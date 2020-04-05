from categories import CATEGORIES, CATEGORIES_LIST, FR_CATEGORIES_LIST, AR_CATEGORIES_LIST, FR_CATEGORIES, AR_CATEGORIES
from helpers import LANGUAGES, EMBEDDING_MODEL

def create_embedding_files(model, language, category, members):
    model = model.lower()
    path = "models/" + model + "/categories/" + language + "/" + category + ".txt"
    try:
        open(path)
    except FileNotFoundError:
        f = open(path, "x")
        dim = 300
        # get embeddings
        emb_path = "models/" + model + "/" + language + ".txt"
        if model == 'glove':
            emb_path = "models/glove/glove.6B.50d.txt"
            dim = 50
        embeddings = EMBEDDING_MODEL[model](members, emb_path).get_embeddings()

        f.write(str(len(embeddings)) + " " + str(dim) + "\n")
        for member, emb in embeddings.items():
            f.write(member + " " + (' '.join(map(str, emb))) + "\n")

languages = list(LANGUAGES.values())
models = ['fastText', 'Word2Vec', 'GLoVe']
for i, category_list in enumerate([CATEGORIES_LIST, FR_CATEGORIES_LIST , AR_CATEGORIES_LIST]):
    for model in models:
        for j, members in enumerate(category_list):
            create_embedding_files(model, languages[i], CATEGORIES[j], members)


for model in models:
    for i, members in enumerate([CATEGORIES, FR_CATEGORIES, AR_CATEGORIES]):
        create_embedding_files(model, languages[i], 'categories', members)
