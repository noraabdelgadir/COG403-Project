import numpy as np 
np.random.seed(42)
from helpers.embeddings import Embedding, Bert, Glove, Word2VecMuse
from helpers.get_members import get_isa, muse_check
from helpers.helper import CATEGORY_TO_LIST
from keras.models import Sequential
from keras.layers import Dense 

import pandas as pd  
import keras.backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score
#from translate import get_translations
from sklearn.preprocessing import OneHotEncoder
import keras

MODEL = "word2vec"
#PATH = "/Users/shydebnath/Documents/glove.6B/glove.6B.50d.txt"
#PATH_TEST = "/Users/shydebnath/Documents/multilingual_embeddings/multilingual_embeddings.fr"
#PATH = "models/" + "word2vec" + "/categories/" + 'en' + "_" + category + ".txt"
#PATH_TEST = "models/" + "word2vec" + "/categories/" + 'fr' + "_" + category + ".txt"
if MODEL == "bert":
    DIM = 768
elif MODEL == "glove":
    DIM = 300
else:
    DIM = 300

print(MODEL, DIM)
# DATA 
categories = ['insect', 'shape', 'weather', 'food', 'bird', 'mammal', 'colour', 'flower', 'country',  'vehicle', 'sport',]
translation = ['insecte', 'forme', 'météo', 'nourriture', 'oiseau', 'mammifère', 'couleur', 'fleur', 'pays', 'véhicule', 'sport',]
#t = get_translations('fr', categories)
#print(t)

#insect 0
insect = ['fourmi','cafard', 'insecte','sauterelle', 'scarabée', 'nymphe','punaise']
# shape 1
shape = ['rectangle','ovale','rétrécissement','triangle','cube','étoile']
# weather 2
weather= ['atmosphère','sécheresse','vent', 'éléments','météorologie', 'météo', 'climat','saison','tornade','pluvieux','brouillard','vague','pluie','précipitation','dégel']
# food 3
food= ['maïs','salade','hamburger','dinde','saumon','gnocchi','nourrir','jambon','céréale','spaghetti','viande','poulet', 'volaille']
# bird 4
bird= ['coucou','perdrix','mallard','starling','cygne','faucon','albatros','toucan', 'rossignol','oie','vautour']
# mammal 5
mammal= ['tamarin','bovins', 'bétail','girafe','cerf','biche','baleine','babouin','fouine','léopard','mouton']
# colour 6
colour= ['orange','beige','couleur','cyan','rouge','mauve','verte', 'vert', 'violette', 'pourpre', 'violet', 'cobalt', 'magenta', 'azur', 'jaune', 'bleu', 'bleue']
# flower 7
flower = ['if', 'récolte', 'phytoplancton', 'iris', 'cyprès', 'algues']
# country 8
country= ['amérique', 'turquie', 'belgique', 'pays', 'irlande', 'portugal', 'égypte', 'mongolie', 'islande', 'corée', 'irak', 'lesotho', 'chili', 'brésil']
# vehicle 9
vehicle = ['voilier', 'vélo', 'bicyclette', 'navire', 'vaisseau', 'camionnette', 'fourgonnette', 'fourgon', 'camion', 'voiture', 'chariot', 'charrette', 'bateau', 'canot', 'traîneau', 'kayak']
# sport 10
sport= ['cheerleading', 'voile', 'squash', 'surf', 'biathlon', 'golf', 'aviron', 'lutte', 'volley', 'course', 'pêche', 'football', 'sumo']


translate = insect + shape + weather + food + bird + mammal + colour + flower + country + vehicle + sport
x_test_dict = {'insect': insect, 'shape': shape, 'weather': weather, 'food': food, 'bird': bird, 'mammal': mammal, 'colour': colour, 'flower': flower, 'country': country, 'vehicle': vehicle, 'sport':sport,} 
catmap = {'insect': 0, 'shape':1, 'weather': 2, 'food': 3, 'bird': 4, 'mammal': 5, 'colour': 6, 'flower': 7, 'country': 8, 'vehicle': 9, 'sport': 10} 
y_true_tr = [0]*len(insect) + [1]*len(shape) + [2]*len(weather) + [3]*len(food) + [4]*len(bird) + [5]*len(mammal) + [6]*len(colour) + [7]*len(flower) + [8]*len(country) + [9]*len(vehicle) + [10]*len(sport)
y_true_class = ['insect']*len(insect) + ['shape']*len(shape) + ['weather']*len(weather) + ['food']*len(food) + ['bird']*len(bird) + ['mammal']*len(mammal) + ['colour']*len(colour) + ['flower']*len(flower) + ['country']*len(country) + ['vehicle']*len(vehicle) + ['sport']*len(sport)

#  GET EMBEDDINGS AND RETURN IN DATAFRAME
def get_embeddings(type, word_list, path=None, lang=None, cat=None):
    if type == "bert":
        m = Bert(word_list, path)
        e = m.get_embeddings()
    elif type == "glove":
        path = "models/" + type + "/categories/" + lang + "_" + cat + ".txt"
        path = "/Users/shydebnath/Documents/multilingual_embeddings/multilingual_embeddings." + lang
        m = Glove(word_list, path)
        e = m.get_embeddings()
    else: 
        path = "models/" + type + "/categories/" + lang + "/" + cat + ".txt"
        m = Word2VecMuse(word_list, path)
        e = m.get_embeddings()
    
    return pd.DataFrame(e) 

# GET WORD LISTS FOR EACH CATEGORY 
df = pd.DataFrame()
cat = []
for c in categories:
    print(c)
    word_list = CATEGORY_TO_LIST[c]
    # word_list = list(set(muse_check(get_isa(c))))[:14]
    # if MODEL=='Glove':
    #     word_list = glove_check(word_list)
    print(word_list)
    #print(get_translations('fr', word_list))

    print(len(word_list))
    embs = get_embeddings(MODEL, word_list, lang='en', cat=c)
    
    if MODEL == "bert":
        df = df.append(embs)
        cat.append(embs.shape[0])
    else:
        df = df.append(embs.transpose()) 
        cat.append(embs.shape[1])
print(df)
print(cat)

# GET Y LABELS FOR WORDS
y_label = []
label = 0
for i in cat:
    for _ in range(i):
        y_label.append(label)
    label+=1
print(len(y_label))

# MODEL 
X = df 
y = keras.utils.to_categorical(y_label)
print(y)

model = Sequential()
model.add(Dense(150, input_dim=DIM, activation='softmax'))
model.add(Dense(50, activation='softmax'))
model.add(Dense(11, activation='sigmoid'))

# EVALUATION METRICS
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# def f1(y_true, y_pred):
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', precision, recall])

model.fit(X, y, epochs=250, batch_size=10, verbose=1) 

# GET PREDICTIONS
# get embeddings for translation words
df_xtest = pd.DataFrame()
true = []
trueclass = []
for key, frcat in zip(x_test_dict.keys(), translation):
    print(key, frcat)
    if MODEL == "bert" or MODEL == "glove":
        embs = get_embeddings(MODEL, x_test_dict[key],lang='fr', cat=frcat)
    else:
        embs = get_embeddings(MODEL, x_test_dict[key], lang='fr', cat=frcat).transpose()
    df_xtest = df_xtest.append(embs)
    for _ in range(embs.shape[0]):
        true.append(catmap[key])
        trueclass.append(key)
predictions = model.predict_classes(df_xtest)
print(len(predictions))
y_true_tr = true
y_true_class = trueclass
print(len(y_true_class), len(y_true_tr))
df_pred = pd.DataFrame()
df_pred['true_class'] = y_true_class
df_pred['true_label'] = y_true_tr
df_pred['pred_label'] = predictions

def get_class_performance(df):
    classes, num_ex, precision, recall, o_acc= [], [], [], [], []
    for t in df['true_label'].unique():
        print(t)
        df_type = df[df['true_label']==t]
        classes.append(df_type['true_class'].unique())
        num_ex.append(len(df_type))
        #acc.append(accuracy_score(df_type['true_label'], df_type['pred_label']))
    
        tp = len(df[(df['true_label']==t) & (df['pred_label']==t)])
        fp = len(df[(df['true_label']!=t) & (df['pred_label']==t)])
        fn = len(df[(df['true_label']==t) & (df['pred_label']!=t)])
        tn = len(df[(df['true_label']!=t) & (df['pred_label']!=t)])
        print('tp: {} fp: {} fn: {} tn: {} dft: {}'.format(tp, fp, fn, tn, len(df_type)))
    
        o_acc.append((tp + tn)/(tp + tn + fp + fn))
        recall.append(tp/(tp+fn))
        
        if tp + fp == 0:
            precision.append('n/a')

        else:
            precision.append(tp/(tp+fp))
            
    return classes, num_ex, precision, recall, o_acc

classes, num_ex, precision, recall, o_acc = get_class_performance(df_pred)

df_results = pd.DataFrame()
df_results['class'] = classes
df_results['num ex'] = num_ex
df_results['overall acc'] = o_acc
df_results['precision'] = precision
#df_results['precision'] = precision_score(y_true_tr, predictions, average=None)
df_results['recall'] = recall
#df_results['recall'] = recall_score(y_true_tr, predictions, average=None)
#df_results['accuracy'] = [accuracy_score(y_true_tr, predictions)] * 11
#df_results['accuracy'] = acc 


print(df_results)

tp = len(df_pred[df_pred['true_label'] == df_pred['pred_label']])
accuracy = tp/len(df_pred)
print("OVERALL ACCURACY: ", accuracy)