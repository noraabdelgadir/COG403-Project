from helpers.categories import CATEGORIES, CATEGORIES_LIST, FR_CATEGORIES_LIST, AR_CATEGORIES_LIST, FR_CATEGORIES, AR_CATEGORIES
from helpers.helper import LANGUAGES, EMBEDDING_MODEL

# helper method for creating embedding files for a category in a language
def create_embedding_files(model, language, category, members):
    model = model.lower()

    # path of where to save the embedding file
    path = "models/" + model + "/categories/" + language + "/" + category + ".txt"
    try:
        open(path)
    except FileNotFoundError:
        f = open(path, "x")
        dim = 300 # dimension of the embedding

        # get embeddings
        emb_path = "models/" + model + "/" + language + ".txt"
        embeddings = EMBEDDING_MODEL[model](members, emb_path).get_embeddings()

        # first line of file is the number of embeddings and their dimension
        f.write(str(len(embeddings)) + " " + str(dim) + "\n")
        for member, emb in embeddings.items():
            f.write(member + " " + (' '.join(map(str, emb))) + "\n")


if __name__ == "__main__":
    languages = list(LANGUAGES.values())
    models = ['fastText', 'Word2Vec', 'GLoVE']

    # create embedding files for category members
    for i, category_list in enumerate([CATEGORIES_LIST, FR_CATEGORIES_LIST , AR_CATEGORIES_LIST]):
        for model in models:
            for j, members in enumerate(category_list):
                create_embedding_files(model, languages[i], CATEGORIES[j], members)

    # create embedding files for categories themselves e.g. English categories
    for model in models:
        for i, members in enumerate([CATEGORIES, FR_CATEGORIES, AR_CATEGORIES]):
            create_embedding_files(model, languages[i], 'categories', members)
