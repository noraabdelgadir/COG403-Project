import numpy as np 
np.random.seed(42)
from helpers.embeddings import Embedding, Bert, Glove, Word2VecMuse
from helpers.get_members import get_isa, muse_check
from helpers.helper import CATEGORY_TO_LIST
from keras.models import Sequential
from keras.layers import Dense 

from helpers.helper import *
from helpers.categories import CATEGORIES
from helpers.embeddings import Word2VecMuse

import pandas as pd  
import keras.backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
import inquirer

catmap = {'insect': 0, 'shape':1, 'weather': 2, 'food': 3, 'bird': 4, 'mammal': 5, 'colour': 6, 'flower': 7, 'country': 8, 'vehicle': 9, 'sport': 10} 

class NN:
    def __init__(self, language, categories, model):
        if not language in LANGUAGES:
            raise ValueError('Please enter a valid language from ' + str(LANGUAGES.keys()))
        self.language = LANGUAGES[language]
        if not categories:
            raise ValueError('Please enter at least one category')
        self.categories = categories
        if not model in MODELS:
            raise ValueError('Please enter a valid model from ' + str(MODELS))
        self.model = model.lower()
        self.category_map = {}
        self.members = {}
        self.translated_members = {}
        self.labels = []
        self.embeddings = []
        self.dim = 0
        self.data = pd.DataFrame()
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        if self.model == "bert":
            self.dim = 768
        else:
            self.dim = 300
        self.layer1 = 128
        self.layer2 = 64
        self.output = 11
        self.learning_rate = 2e-5
        self.activation1 = 'relu'
        self.activation2 = 'relu'

        print("MODEL: {} + layer1: {} + layer2: {} + LR: {}".format(self.model, self.layer1, self.layer2, self.learning_rate))
        # setting variables
        self.set_members()
        self.set_translated_members()
        self.set_labels_and_embeddings()
        print(self.labels)
        self.set_data_members()
        self.train_test_split(self.data)
        print('train size: ', self.train_set.shape)
        print('test size: ', self.test_set.shape)
        model = self.train()
        predictions = self.predict(model)
        self.evaluate(predictions)
    
    def set_members(self):
        for i in range(len(self.categories)):
            self.members[self.categories[i]] = CATEGORY_TO_LIST[self.categories[i]]
    
    def set_translated_members(self):
        cat_translation = EN_FR_CATEGORIES if self.language == 'fr' else EN_AR_CATEGORIES
        member_translation = FR_CATEGORY_TO_LIST if self.language == 'fr' else AR_CATEGORY_TO_LIST
        remove_categories = []
        for cat, members in self.members.items():
            if cat in member_translation:
                self.translated_members[cat_translation[cat]] = member_translation[cat]
            else:
                remove_categories.append(cat)

        for cat in remove_categories:
            del self.members[cat]

    def set_labels_and_embeddings(self):
        for cat in self.members:
            tr_cat = EN_FR_CATEGORIES[cat] if self.language == 'fr' else EN_AR_CATEGORIES[cat]
            en_path = "/Users/shydebnath/Documents/projects/COG403-Project/" + "models/" + self.model + "/categories/en/" + cat + ".txt"
            tr_path = "/Users/shydebnath/Documents/projects/COG403-Project/" + "models/" + self.model + "/categories/" + self.language + "/" + cat + ".txt"
            if self.model == 'glove':
                en_path = "/Users/shydebnath/Documents/projects/COG403-Project/models/glove/glove.6B/glove.6B.50d.txt"
                en_path = "/Users/shydebnath/Documents/projects/COG403-Project/models/glove/multilingual_embeddings/multilingual_embeddings.en"
                tr_path = "/Users/shydebnath/Documents/projects/COG403-Project/models/glove/glove.6B/glove.6B.50d.txt"
                tr_path = "/Users/shydebnath/Documents/projects/COG403-Project/models/glove/multilingual_embeddings/multilingual_embeddings." + self.language
            en_word_emb = EMBEDDING_MODEL[self.model](self.members[cat], en_path).get_embeddings()
            tr_word_emb = EMBEDDING_MODEL[self.model](self.translated_members[tr_cat], tr_path).get_embeddings()
            
            # add words to cat map dict to retrieve for y labels later
            for w in self.members[cat] + self.translated_members[tr_cat]:
                self.category_map[w] = cat 

            if self.model == 'bert':
                self.labels.extend(self.members[cat])
                self.labels.extend(self.translated_members[tr_cat])

                self.embeddings.extend(en_word_emb)
                self.embeddings.extend(tr_word_emb)
            else:
                not_found = list(set(self.translated_members[tr_cat]) - set(tr_word_emb.keys()))
                for member in not_found:
                    self.translated_members[tr_cat].remove(member)

                self.labels.extend(en_word_emb.keys())
                self.labels.extend(tr_word_emb.keys())
            
                self.embeddings.extend(en_word_emb.values())
                self.embeddings.extend(tr_word_emb.values())
    
    def set_data_members(self):
        self.data['class'] = [self.category_map[c] for c in self.labels]
        self.data['label'] = [catmap[self.category_map[c]] for c in self.labels]
        self.data = pd.concat([self.data, pd.DataFrame(data=self.embeddings)], axis=1)
        
      
    def train_test_split(self, df):

        train, test = train_test_split(df, test_size=0.2, shuffle=True)
        self.train_set = train
        self.test_set = test
     
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def train(self):
        X = self.train_set.iloc[:,2:]
        y = keras.utils.to_categorical(self.train_set['label'])
        optim = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model = Sequential()
        model.add(Dense(self.layer1, input_dim=self.dim, activation=self.activation1))
        model.add(Dense(self.layer2, activation=self.activation2))
        model.add(Dense(self.output, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy', 'categorical_accuracy', self.precision, self.recall])
        model.fit(X, y, epochs=250, batch_size=10, verbose=1)       

        return model 
    
    def get_class_performance(self, df):
        df[['label', 'class', 'pred_label']].to_excel(self.model+'_'+self.language+'.xlsx', index=False)
        classes, num_ex, precision, recall, o_acc= [], [], [], [], []
        for t in df['label'].unique():
            print(t)
            df_type = df[df['label']==t]
            classes.append(df_type['class'].unique())
            num_ex.append(len(df_type))
            #acc.append(accuracy_score(df_type['true_label'], df_type['pred_label']))
        
            tp = len(df[(df['label']==t) & (df['pred_label']==t)])
            fp = len(df[(df['label']!=t) & (df['pred_label']==t)])
            fn = len(df[(df['label']==t) & (df['pred_label']!=t)])
            tn = len(df[(df['label']!=t) & (df['pred_label']!=t)])
            print('tp: {} fp: {} fn: {} tn: {} dft: {}'.format(tp, fp, fn, tn, len(df_type)))
        
            o_acc.append((tp + tn)/(tp + tn + fp + fn))
            recall.append(tp/(tp+fn))
            
            if tp + fp == 0:
                precision.append('n/a')

            else:
                precision.append(tp/(tp+fp))
                
        return classes, num_ex, precision, recall, o_acc
    
    def predict(self, model):

        predictions = model.predict_classes(self.test_set.iloc[:, 2:])

        return predictions
    
    def evaluate(self, preds):
        self.test_set['pred_label'] = preds 

        classes, num_ex, precision, recall, o_acc = self.get_class_performance(self.test_set)

        df_results = pd.DataFrame()
        df_results['class'] = classes
        df_results['num ex'] = num_ex
        df_results['overall acc'] = o_acc
        df_results['precision'] = precision
        df_results['recall'] = recall
      
        print(df_results)

        tp = len(self.test_set[self.test_set['label'] == self.test_set['pred_label']])
        accuracy = tp/len(self.test_set)
        print("OVERALL ACCURACY: ", accuracy)

if __name__ == "__main__":

    languages = ['French', 'Arabic']
    # questions = [
    # inquirer.List('language',
    #             message="Which language do you want to test with English?",
    #             choices=languages,
    #         ),
    # inquirer.Checkbox('categories',
    #             message="Which categories do you want to include?",
    #             choices=CATEGORIES,
    #         ),
    # inquirer.List('model',
    #             message="Which word embedding model would you like to use?",
    #             choices=MODELS,
    #         ),
    # ]
    # answers = inquirer.prompt(questions)

    language = 'French'
    cats = ['insect', 'shape', 'weather', 'food', 'bird', 'mammal', 'colour', 'flower', 'country', 'vehicle', 'sport']
    model = 'Bert'
    # n = NN(answers['language'], answers['categories'], answers['model'])
    n = NN(language, cats, model)
