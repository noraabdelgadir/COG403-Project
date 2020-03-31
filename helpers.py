from embeddings import Word2VecMuse
from get_embeddings import get_embeddings
from members import get_members

LANGUAGES = {'English': 'en', 'French': 'fr', 'Arabic': 'ar'}
MODELS = ['fastText', 'Word2Vec']
EMBEDDING_MODEL = {'fasttext': Word2VecMuse, 'word2vec': Word2VecMuse}
CATEGORIES = ['vehicle', 'vegetable', 'tree', 'flower', 'colour', 'bird', 
        'device', 'organ', 'activity', 'carnivore', 'fruit', 'vegetable', 
        'fish', 'movement', 'communication', 'reptile', 'furniture', 'sport', 
        'clothing', 'city', 'country', 'tool'] 

def create_embedding_files(model, language, category):
    path = "models/" + model + "/categories/" + language + "_" + category + ".txt"
    try: 
        open(path)
    except FileNotFoundError:
        f = open(path, "x")
        members = get_members(category)
        f.write(str(len(members)) + " 300\n")
        # get embeddings
        emb_path = "models/" + model + "/" + language + ".txt"
        embeddings = EMBEDDING_MODEL[model](members, emb_path).get_embeddings()

        for member, emb in embeddings.items():
            f.write(member + " " + (' '.join(map(str, emb))) + "\n")
