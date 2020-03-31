import inquirer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from members import get_members
from get_embeddings import get_embeddings
from sklearn.cluster import KMeans
from translate import get_translations
from helpers import LANGUAGES, MODELS, EMBEDDING_MODEL
from embeddings import Word2VecMuse

class KMeansClustering:
    def __init__(self, language, categories, model):
        if not language in LANGUAGES:
            raise ValueError('Please enter a valid language from ' + LANGUAGES.keys())
        self.language = LANGUAGES[language]
        if not categories:
            raise ValueError('Please enter at least one category')
        self.categories = categories
        if not model in MODELS:
            raise ValueError('Please enter a valid model from ' + MODELS)
        self.model = model.lower()
        self.members = {}
        self.clustered_members = {}
        self.translated_categories = {}
        self.labels = []
        self.embeddings = []
        self.k_mean = None

        # setting variables
        self.set_members()
        self.set_translated_categories()
        self.set_labels_and_embeddings()
        self.cluster()
    
    def set_members(self):
        for i in range(len(self.categories)):
            self.clustered_members['cluster' + str(i + 1)] = []
            self.members[self.categories[i]] = get_members(self.categories[i])
    
    def set_translated_categories(self):
        categories_to_remove = []
        for cat in self.categories:
            translated_cat = get_translations(self.language, [cat]).values()
            if not translated_cat:
                categories_to_remove.append(cat)
                del self.members[cat]
            else:
                self.translated_categories[cat] = list(translated_cat)[0][0]
        
        for cat in categories_to_remove:
            self.categories.remove(cat)
        
        if not self.categories:
            print(categories_to_remove + " doesn't/don't exist in translation language")
            exit(self)
                
    def set_labels_and_embeddings(self):
        for cat in self.members:
            path = "models/" + self.model + "/categories/en_" + cat + ".txt"
            word_emb = EMBEDDING_MODEL[self.model](self.members[cat], path).get_embeddings()
            self.labels.extend(word_emb.keys())
            self.embeddings.extend(word_emb.values())

        for tr_cat in self.translated_categories.values():
            path = "models/" + self.model + "/categories/" + self.language + "_" + tr_cat + ".txt"
            word_emb = EMBEDDING_MODEL[self.model]([tr_cat], path).get_embeddings()
            self.labels.extend(word_emb.keys())
            self.embeddings.extend(word_emb.values())
    
    def cluster(self):
        pca = PCA(n_components=2, whiten=True).fit(self.embeddings)
        self.embeddings = pca.transform(self.embeddings)

        self.k_mean = KMeans(n_clusters=len(self.categories)).fit(self.embeddings)

        for k, label in enumerate(self.labels):
            self.clustered_members['cluster' + str(self.k_mean.labels_.astype(int)[k] + 1)].append(label)

    def visualize(self):
        x_coords = self.embeddings[:, 0]
        y_coords = self.embeddings[:, 1]

        translated_cat_emb = {}
        for i in range(len(self.members.keys())):
            index = len(self.labels) - i - 1
            translated_cat_emb[self.labels[index]] = [x_coords[index], y_coords[index]]

        plt.figure(figsize=(10, 8), dpi=80)
        plt.scatter(x_coords, y_coords, marker='x')

        plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
        plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
        plt.title('Visualization of the multilingual word embedding space')

        label_colours = ['red', 'blue', '#FFBAC2', '#91FB8C', '#8CB1FB', '#FB8CF6', 
                        '#BF8CFB', '#FDB9C2', '#FDFBB9', '#B9EFFD', '#837399', '#997385',
                        '#CFC7C3', '#C3CFC9', '#5C006A', '#406A00', '#6A3C00', '#6A003D',
                        '#5D4252', '#425D49', '#B4A55B', '#5BB47A', '#B45B72']

        for k, (label, x, y) in enumerate(zip(self.labels, x_coords, y_coords)):
            color = label_colours[self.k_mean.labels_.astype(int)[k]]
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10,
                            color=color, weight='bold')

        centres = list(translated_cat_emb.values())
        colours = ['c', 'g', 'y', 'm', 'k', '#AE5845', '#92C7AC', '#6D0240', '#F0B577', '#3A99AC', 
                    '#FC0142', '#FC3E01', '#FCDA01', '#C3FC01', '#36FC01', '#01FC8E', '#01B7FC',
                    '#0177FC', '#1F01FC', 'r', 'b', '#FFBAC2', '#91FB8C', '#8CB1FB', '#FB8CF6']

        for i in range(len(centres)):
            plt.scatter(centres[i][0], centres[i][1], s=400, c=colours[i], marker='s')

        centres = self.k_mean.cluster_centers_
        for i in range(len(centres)):
            plt.scatter(centres[i][0], centres[i][1], s=400, c='k', marker='o')

        plt.show()
    
    def get_members(self):
        return self.members
    
    def get_clustered_members(self):
        categorized_clusters = {}
        for cluster, membs in self.clustered_members.items():
            categorized_clusters[cluster] = {}
            for member in membs:
                categorized = False
                for cat, values in self.members.items():
                    if member in values:
                        categorized = True
                        if not self.translated_categories[cat] in categorized_clusters[cluster]:
                            categorized_clusters[cluster][self.translated_categories[cat]] = [member]
                        else:
                            categorized_clusters[cluster][self.translated_categories[cat]].append(member)

                    # translation category name
                    if not categorized and self.translated_categories[cat] == member:
                        if not member in categorized_clusters[cluster]:
                            categorized_clusters[cluster][member] = [member]
                        else:
                            categorized_clusters[cluster][member].append(member)

        return categorized_clusters

if __name__ == "__main__":
    categories = ["vehicle", "vegetable"]
    lang = 'French'
    model = 'fastText'
    k = KMeansClustering(lang, categories, model)
    # k.visualize()
    print(k.get_clustered_members())
