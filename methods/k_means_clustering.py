from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from helpers.helper import *
from helpers.categories import CATEGORIES
from helpers.embeddings import Word2VecMuse
import inquirer

class KMeansClustering:
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
        self.members = {}
        self.clustered_members = {}
        self.translated_members = {}
        self.labels = []
        self.embeddings = []
        self.k_mean = None

        # setting variables
        self.set_members()
        self.set_translated_members()
        self.set_labels_and_embeddings()
        self.cluster()
    
    def set_members(self):
        for i in range(len(self.categories)):
            self.clustered_members['cluster' + str(i + 1)] = []
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
            en_path = "models/" + self.model + "/categories/en/" + cat + ".txt"
            tr_path = "models/" + self.model + "/categories/" + self.language + "/" + cat + ".txt"
            if self.model == 'glove':
                en_path = "models/glove/glove.6B.50d.txt"
                tr_path = "models/glove/glove.6B.50d.txt"
            
            en_word_emb = EMBEDDING_MODEL[self.model](self.members[cat], en_path).get_embeddings()
            tr_word_emb = EMBEDDING_MODEL[self.model](self.translated_members[tr_cat], tr_path).get_embeddings()
            
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

    def cluster(self):
        pca = PCA(n_components=2, whiten=True).fit(self.embeddings)
        self.embeddings = pca.transform(self.embeddings)

        self.k_mean = KMeans(n_clusters=len(self.categories)).fit(self.embeddings)

        for k, label in enumerate(self.labels):
            self.clustered_members['cluster' + str(self.k_mean.labels_.astype(int)[k] + 1)].append(label)

    def visualize(self):
        x_coords = self.embeddings[:, 0]
        y_coords = self.embeddings[:, 1]

        plt.figure(figsize=(10, 8), dpi=80)
        plt.scatter(x_coords, y_coords, marker='x')

        plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
        plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
        plt.title('Visualization of the multilingual word embedding space')

        label_colours = ['c', 'g', 'y', 'm', 'k', '#AE5845', '#92C7AC', '#6D0240', '#F0B577', '#3A99AC', 
                    '#FC0142', '#FC3E01', '#FCDA01', '#C3FC01', '#36FC01', '#01FC8E', '#01B7FC',
                    '#0177FC', '#1F01FC', 'r', 'b', '#FFBAC2', '#91FB8C', '#8CB1FB', '#FB8CF6']

        for k, (label, x, y) in enumerate(zip(self.labels, x_coords, y_coords)):
            color = label_colours[self.k_mean.labels_.astype(int)[k]]
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10,
                            color=color, weight='bold')

        centres = self.k_mean.cluster_centers_
        for i in range(len(centres)):
            plt.scatter(centres[i][0], centres[i][1], s=400, c='b', marker='o')

        plt.show()
    
    def get_members(self):
        return self.members
    
    def get_translated_members(self):
        return self.translated_members
    
    def get_clustered_members(self):
        categorized_clusters = {}
        for cluster, membs in self.clustered_members.items():
            categorized_clusters[cluster] = {}
            for member in membs:
                categorized = False
                for cat, values in self.members.items():
                    if member in values:
                        categorized = True
                        if not cat in categorized_clusters[cluster]:
                            categorized_clusters[cluster][cat] = [member]
                        else:
                            categorized_clusters[cluster][cat].append(member)

                if not categorized:
                    for cat, values in self.translated_members.items():
                        if member in values:
                            cat_translation = FR_EN_CATEGORIES if self.language == 'fr' else AR_EN_CATEGORIES
                            cat = cat_translation[cat]
                            if not cat in categorized_clusters[cluster]:
                                categorized_clusters[cluster][cat] = [member]
                            else:
                                categorized_clusters[cluster][cat].append(member)

        return categorized_clusters

if __name__ == "__main__":
    languages = ['French', 'Arabic']
    questions = [
    inquirer.List('language',
                message="Which language do you want to test with English?",
                choices=languages,
            ),
    inquirer.Checkbox('categories',
                message="Which categories do you want to include?",
                choices=CATEGORIES,
            ),
    inquirer.List('model',
                message="Which word embedding model would you like to use?",
                choices=MODELS,
            ),
    ]
    answers = inquirer.prompt(questions)

    k = KMeansClustering(answers['language'], answers['categories'], answers['model'])
    k.visualize()
