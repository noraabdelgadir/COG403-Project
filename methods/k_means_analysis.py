from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import inquirer
from k_means_clustering import KMeansClustering
from helpers.helper import MODELS, LANGUAGES
from helpers.categories import *

# calculate the accuracy based on the similarity of the expected
# dictionary and the actual dictionary
def analyze(expected, actual):
    category_cluster = {} # note which cluster each category dominates
    remaining_categories = list(set(expected.keys()))

    # count how many members of a category belong to a cluster
    cat_cluster_count = {}
    for cluster, categories in actual.items():
        for cat in categories:
            if cat not in cat_cluster_count:
                cat_cluster_count[cat] = {}
                cat_cluster_count[cat][cluster] = 0
            if cluster not in cat_cluster_count[cat]:
                cat_cluster_count[cat][cluster] = 0
            cat_cluster_count[cat][cluster] += len(list(categories[cat]))

    # if a category's members only exist in one cluster, assign that
    # cluster to that category
    for cat, cluster_count in cat_cluster_count.items():
        if len(cluster_count) == 1:
            cluster = list(cluster_count.keys())[0]
            category_cluster[cat] = cluster
            # remove category from remaining categories
            remaining_categories.remove(cat)
    
    # for the remaining categories, assign the cluster that has the 
    # maximum amount of its members
    for cat, cluster_count, in cat_cluster_count.items():
        if cat in remaining_categories:
            max_count = max(cluster_count.values())
            for cluster, count in cluster_count.items():
                if count == max_count:
                    category_cluster[cat] = cluster

    # calculate and return accuracy by seeing the difference in the 
    # number of expected members vs. the actual number
    accuracy = []
    for category, cluster in category_cluster.items():
        ac = actual[cluster][category]
        exp = expected[category]
        accuracy.append(len(ac)/len(exp) * 100)
    return accuracy

# combine two dictionaries by combining members of the same category
def combine(dict1, dict2):
    combined = {}
    for cat, mem in dict1.items():
        for catb, memb in dict2.items():
            if EN_FR_CATEGORIES[cat] ==  catb or \
                (cat in EN_AR_CATEGORIES and EN_AR_CATEGORIES[cat] == catb):
                combined[cat] = mem + memb

    return combined

if __name__ == "__main__":
    languages = list(LANGUAGES.keys()) 
    questions = [
    inquirer.List('model',
                message="Which model would you like to see the accuracy for?",
                choices=MODELS + ['All'],
            ),
    ]

    embedding_model = inquirer.prompt(questions)['model']

    model_performance_fr = [] # performance of French across models
    model_performance_ar = [] # performance of Arabic across models
    model_performance_en = [] # performance of English across models

    fasttext, word2vec, bert, glove = [], [], [], []

    ANALYSIS_MODELS = MODELS[:] # use all models
    if embedding_model != 'All': 
       ANALYSIS_MODELS = [embedding_model] # use a specific model

    # get accuracy across model(s)
    for model in ANALYSIS_MODELS:
        for language in languages:
            k = KMeansClustering(language, CATEGORIES, model)
            language = LANGUAGES[language]
            expected = combine(k.get_members(), k.get_translated_members())
            if language == 'en':
                expected = k.get_members()
            
            acc = analyze(expected, k.get_clustered_members())
        
            if language == 'fr':
                model_performance_fr.append(sum(acc)/len(acc))
            elif language == 'ar':
                model_performance_ar.append(sum(acc)/len(acc))
            else:
                model_performance_en.append(sum(acc)/len(acc))
            
            if model.lower() == 'fasttext':
                fasttext.append(acc)
            elif model.lower() == 'word2vec':
                word2vec.append(acc)
            elif model.lower() == 'bert':
                bert.append(acc)
            else:
                glove.append(acc)
    
    # visualize the accuracy across all models and languages
    if embedding_model == 'All':
        x = np.arange(len(MODELS))  # the label locations
        width = 0.2  # the width of the bars

        font = {'fontname':'Times New Roman'}

        colours = ['#700000', '#003070', '#5C0070']
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, model_performance_en, width, label='English alone', color='#700000')
        rects2 = ax.bar(x, model_performance_fr, width, label='English-French', color='#003070')
        rects3 = ax.bar(x + width, model_performance_ar, width, label='English-Arabic', color='#5C0070')

        ax.set_ylabel('Accuracy (%)', **font)
        ax.set_xlabel('Model', **font)
        ax.set_title('K-means clustering accuracy by model and language', **font)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, **font)
        ax.set_ylim([0,100])
        ax.legend(prop={'family': 'Times New Roman'})

        plt.show()

    # visualize the accuracy across one models, with all languages and categories
    else:
        model_to_accuarcy = {'fastText': fasttext, 'Word2Vec': word2vec, 'Bert': bert, 'GLoVE': glove}
        model = model_to_accuarcy[embedding_model]
        x = np.arange(len(CATEGORIES))  # the label locations
        width = 0.2  # the width of the bars

        font = {'fontname':'Times New Roman'}

        fig, ax = plt.subplots()
    
        rects1 = ax.bar(x - width, model[0], width, label='English alone', color='#700000')
        rects2 = ax.bar(x, model[1], width, label='English-French', color='#003070')
        rects3 = ax.bar(x + width, model[2] + [0], width, label='English-Arabic', color='#5C0070')

        ax.set_ylabel('Accuracy (%)', **font)
        ax.set_xlabel('Category', **font)
        ax.set_title('K-means Clustering accuracy per category for ' + str(embedding_model), **font)
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES, **font)
        ax.set_ylim([0,110])
        ax.legend(prop={'family': 'Times New Roman'})

        plt.show()

