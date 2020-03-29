import numpy as np
import io

# for fastText and Word2Vec embeddings
def get_embeddings(emb_path, words):
    word_emb = {}

    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            if word in words:
                word_emb[word] = vect
            if len(word_emb) == len(words):
                break

    return word_emb
