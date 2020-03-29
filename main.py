import inquirer
from translate import get_translations
from members import get_members

MODE = 'category'

cats = ['vehicle', 'vegetable', 'tree', 'flower', 'colour', 'bird', 
        'device', 'organ', 'activity', 'carnivore', 'fruit', 'vegetable', 
        'fish', 'movement', 'communication', 'reptile', 'furniture', 'sport', 
        'clothing', 'city', 'country', 'tool'] 

languages = ['French', 'Arabic']

questions = [
    inquirer.List('language',
                message="Which language do you want to test with English?",
                choices=languages,
            ),
    inquirer.Checkbox('categories',
                message="Which categories do you want to include?",
                choices=cats,
            ),
    inquirer.List('model',
                message="Which word embedding model would you like to use?",
                choices=['fastText', 'Word2Vec'],
            ),
]

answers = inquirer.prompt(questions)
lang = answers["language"][:2].lower()
model = answers["model"]
members = {}
clustered_members = {}

for i in range(len(answers["categories"])):
    clustered_members['cluster' + str(i + 1)] = []
    cat = answers["categories"][i]
    members[cat] = get_members(cat)

translated_members = {} 
translated_categories = {}

for cat in members:
    c = list(get_translations(lang, [cat]).values())[0][0]
    translated_categories[cat] = c
    translated_members[c] = []
    for word in members[cat]:
        translation = list(get_translations(lang, [word]).values())
        if(not translation):
            # word can't be translated
            members[cat].remove(word)
        else:
            translated_members[c].extend(translation[0])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from get_embeddings import load_vec

for cat in members:
    path = "models/" + model + "/categories/en_" + cat + ".txt"
    try: 
        open(path)
    except FileNotFoundError:
        f = open(path, "x")
        # get embeddings
        embeddings = load_vec("models/" + model + "/en.txt", members[cat])
        f.write(str(len(members[cat])) + " 300\n")
        for member, emb in embeddings.items():
            f.write(member + " " + (' '.join(map(str, emb))) + "\n")

for cat in translated_members:
    path = "models/" + model + "/categories/" + lang + "_" + cat + ".txt"
    try: 
        open(path)
    except FileNotFoundError:
        f = open(path, "x")
        # get embeddings
        embeddings = load_vec("models/" + model + "/" + lang + ".txt", translated_members[cat])
        f.write(str(len(translated_members[cat])) + " 300\n")
        for member, emb in embeddings.items():
            f.write(member + " " + (' '.join(map(str, emb))) + "\n")

embeddings = []
labels = []

for cat in members:
    path = "models/" + model + "/categories/en_" + cat + ".txt"
    word_emb = load_vec(path, members[cat])
    labels.extend(word_emb.keys())
    embeddings.extend(word_emb.values())

for cat in translated_members:
    path = "models/" + model + "/categories/" + lang + "_" + cat + ".txt"
    if MODE == 'category':
        word_emb = load_vec(path, [cat])
        labels.extend(word_emb.keys())
        embeddings.extend(word_emb.values())
    else:
        word_emb = load_vec(path, translated_members[cat])
        labels.extend(word_emb.keys())
        embeddings.extend(word_emb.values())

from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True).fit(embeddings)

embeddings = pca.transform(embeddings)
x_coords = embeddings[:, 0]
y_coords = embeddings[:, 1]

plt.figure(figsize=(10, 8), dpi=80)
plt.scatter(x_coords, y_coords, marker='x')

plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
plt.title('Visualization of the multilingual word embedding space')

from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=len(answers["categories"]))
Kmean.fit(embeddings)

label_colours = ['red', 'blue', '#FFBAC2', '#91FB8C', '#8CB1FB', '#FB8CF6', 
                '#BF8CFB', '#FDB9C2', '#FDFBB9', '#B9EFFD', '#837399', '#997385',
                '#CFC7C3', '#C3CFC9', '#5C006A', '#406A00', '#6A3C00', '#6A003D',
                '#5D4252', '#425D49', '#B4A55B', '#5BB47A', '#B45B72']

for k, (label, x, y) in enumerate(zip(labels, x_coords, y_coords)):
    clustered_members['cluster' + str(Kmean.labels_.astype(int)[k] + 1)].append(label)
    color = label_colours[Kmean.labels_.astype(int)[k]]
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10,
                    color=color, weight='bold')

centres = Kmean.cluster_centers_
colours = ['c', 'g', 'y', 'm', 'k', '#AE5845', '#92C7AC', '#6D0240', '#F0B577', '#3A99AC', 
            '#FC0142', '#FC3E01', '#FCDA01', '#C3FC01', '#36FC01', '#01FC8E', '#01B7FC',
            '#0177FC', '#1F01FC', 'r', 'b', '#FFBAC2', '#91FB8C', '#8CB1FB', '#FB8CF6']
for i in range(len(centres)):
    plt.scatter(centres[i][0], centres[i][1], s=400, c=colours[i], marker='s')

plt.show()

# analysis

# MODE = 'category'
categorized_clusters = {}
if MODE == 'category':
    for cluster, membs in clustered_members.items():
        categorized_clusters[cluster] = {}
        for member in membs:
            for cat, values in members.items():
                if member in values:
                    if not translated_categories[cat] in categorized_clusters[cluster]:
                        categorized_clusters[cluster][translated_categories[cat]] = [member]
                    else:
                        categorized_clusters[cluster][translated_categories[cat]].append(member)
# MODE = 'members'
else:
    for cluster, membs in clustered_members.items():
        categorized_clusters[cluster] = {}
        for member in membs:
            categorized = False
            for cat, values in members.items():
                if member in values:
                    categorized = True
                    if not cat in categorized_clusters[cluster]:
                        categorized_clusters[cluster][cat] = [member]
                    else:
                        categorized_clusters[cluster][cat].append(member)
            
            if not categorized:
                for cat, values in translated_members.items():
                    if member in values:
                        cat = list(translated_categories.keys())[list(translated_categories.values()).index(cat)]
                        if not cat in categorized_clusters[cluster]:
                            categorized_clusters[cluster][cat] = [member]
                        else:
                            categorized_clusters[cluster][cat].append(member)

# print(categorized_clusters)
