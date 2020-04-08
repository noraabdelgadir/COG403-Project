from bert_serving.client import BertClient
import pandas as pd 
import numpy as np 
import io 

class Embedding(object):
    
    # initialization function
    # word_list is the list of words the embeddings are to be retrived for
    # model_path is the path to where the embeddings are stored
    def __init__(self, word_list, model_path):
        self.word_list = word_list
        self.model_path = model_path
    
    # return embeddings based on the words and model
    def get_embeddings(self):
        raise NotImplementedError

class Glove(Embedding):
    # get embeddings from GLoVE
    def get_embeddings(self):
        embeddings_dict = {}
        with open(self.model_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                if word in self.word_list:
                    embeddings_dict[word] = vector
        return embeddings_dict

class Bert(Embedding):
    # get embeddings from BERT
    # make sure BERT is started
    def get_embeddings(self):
        bc = BertClient()
        array = bc.encode(self.word_list)
        
        return array 

class Word2VecMuse(Embedding):
    # get embeddings from Word2Vec and fastText
    def get_embeddings(self):
        word_emb = {}

        with io.open(self.model_path, 'r', encoding='utf-8', newline='\n') as f:
            next(f) # skip first line with number of embeddings and dimension
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if word in self.word_list:
                    word_emb[word] = vect
                if len(word_emb) == len(self.word_list):
                    break

        return word_emb

  