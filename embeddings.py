from bert_serving.client import BertClient
from get_members import get_isa, muse_check
import pandas as pd 
import numpy as np 
import io 

class Embedding(object):
    
    def __init__(self, word_list, model_path):
        self.word_list = word_list
        self.model_path = model_path
    
    def get_embeddings(self):
        raise NotImplementedError

class Glove(Embedding):

    def get_embeddings(self):
        embeddings_dict = {}
        with open(self.model_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                if word in self.word_list:
                    embeddings_dict[word] =vector
        vectors = list(embeddings_dict.values())
        return vectors 

class Bert(Embedding):
    # need to start bert-
    def get_embeddings(self):
        bc = BertClient()
        array = bc.encode(self.word_list)

        return array 

class Word2VecMuse(Embedding):

    def get_embeddings(self):
        word_emb = {}

        with io.open(self.model_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if word in self.word_list:
                    word_emb[word] = vect
                if len(word_emb) == len(self.word_list):
                    break

        return word_emb


if __name__ == "__main__":
    
    category = "vehicle"
    translation = "véhicule"

    word_list = muse_check(get_isa(category))
    word_list.append(translation)


    # example 
    path = "models/glove.6B/glove.6B.50d.txt"
    print(word_list)
    
    b = Glove(word_list, path)
    e = b.get_embeddings()
  