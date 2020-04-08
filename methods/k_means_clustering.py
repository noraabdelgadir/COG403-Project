from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import inquirer
from helpers.helper import *
from helpers.categories import CATEGORIES
from helpers.embeddings import Word2VecMuse

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
        self.members = {} # categories and their members
        self.clustered_members = {} 
        self.translated_members = {} # translated categories and translated members
        self.labels = [] # name of members for the visualization
        self.embeddings = [] # embeddings of members for visualization
        self.k_mean = None # k-means clustering instance

        # setting variables
        self.set_members()
        self.set_translated_members()
        self.set_labels_and_embeddings()
        self.cluster()
    
    # create an empty list for each cluster
    # populate the self.members dictionary with categories and their members
    def set_members(self):
        for i in range(len(self.categories)):
            self.clustered_members['cluster' + str(i + 1)] = []
            self.members[self.categories[i]] = CATEGORY_TO_LIST[self.categories[i]][:]
    
    # populate the self.translated_members dictionary with categories and their members
    def set_translated_members(self):
        # only for non-English languages
        if not self.language == 'en':
            # use dictionary corresponding to the language
            cat_translation = EN_FR_CATEGORIES if self.language == 'fr' else EN_AR_CATEGORIES
            member_translation = FR_CATEGORY_TO_LIST if self.language == 'fr' else AR_CATEGORY_TO_LIST
            remove_categories = []
            # set each category and its members for the translated language
            for cat, members in self.members.items():
                if cat in member_translation:
                    self.translated_members[cat_translation[cat]] = member_translation[cat][:]
                else:
                    remove_categories.append(cat)

            # remove categories that don't exist in the translation language
            # e.g. sports doesn't exist in Arabic
            for cat in remove_categories:
                del self.members[cat]

    # use embedding model to get embeddings for category members
    # set labels for embeddings that exist in the model
    def set_labels_and_embeddings(self):
        if self.language == 'en':
              for cat in self.members:
                # path to the embeddings
                path = "models/" + self.model + "/categories/en/" + cat + ".txt"
                # get embeddings based on the embedding model
                word_emb = EMBEDDING_MODEL[self.model](self.members[cat], path).get_embeddings()
                
                if self.model == 'bert': # bert returns an array of embeddings
                    self.labels.extend(self.members[cat])
                    self.embeddings.extend(word_emb)
                else: # other models return dictionary of embeddings
                    self.labels.extend(word_emb.keys())
                    self.embeddings.extend(word_emb.values())

        else:
            for cat in self.members:
                # get embeddings for both English and translation language
                tr_cat = EN_FR_CATEGORIES[cat] if self.language == 'fr' else EN_AR_CATEGORIES[cat]
                en_path = "models/" + self.model + "/categories/en/" + cat + ".txt"
                tr_path = "models/" + self.model + "/categories/" + self.language + "/" + cat + ".txt"
                
                en_word_emb = EMBEDDING_MODEL[self.model](self.members[cat], en_path).get_embeddings()
                tr_word_emb = EMBEDDING_MODEL[self.model](self.translated_members[tr_cat], tr_path).get_embeddings()
                
                if self.model == 'bert': # bert returns an array of embeddings
                    self.labels.extend(self.members[cat])
                    self.labels.extend(self.translated_members[tr_cat])

                    self.embeddings.extend(en_word_emb)
                    self.embeddings.extend(tr_word_emb)
                else: # other models return dictionary of embeddings
                    self.labels.extend(en_word_emb.keys())
                    self.labels.extend(tr_word_emb.keys())
                
                    self.embeddings.extend(en_word_emb.values())
                    self.embeddings.extend(tr_word_emb.values())

    # using k-means clustering algorithm, cluster the words
    def cluster(self):
        pca = PCA(n_components=2, whiten=True).fit(self.embeddings)
        self.embeddings = pca.transform(self.embeddings)

        self.k_mean = KMeans(n_clusters=len(self.members)).fit(self.embeddings)

        for k, label in enumerate(self.labels):
            self.clustered_members['cluster' + str(self.k_mean.labels_.astype(int)[k] + 1)].append(label)

    # visualize the embeddings in 2D
    def visualize(self):
        x_coords = self.embeddings[:, 0]
        y_coords = self.embeddings[:, 1]

        plt.figure(figsize=(10, 8), dpi=80)
        plt.scatter(x_coords, y_coords, marker='x')

        plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
        plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
        plt.title('Visualization of the multilingual word embedding space')

        label_colours = ['#3E3C76', '#002C70', '#5C0070', '#700058', '#704100', 
                        '#087000',  '#700025', '#700700' ,'#00704C','#006670', 
                        '#707000']

        # label each colour according to its cluster
        for k, (label, x, y) in enumerate(zip(self.labels, x_coords, y_coords)):
            color = label_colours[self.k_mean.labels_.astype(int)[k]]
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=10,
                            color=color, weight='bold')

        # show the cluster centres
        centres = self.k_mean.cluster_centers_
        for i in range(len(centres)):
            plt.scatter(centres[i][0], centres[i][1], s=400, c='b', marker='o')

        plt.show()
    
    # members getter function
    def get_members(self):
        return self.members
    
    # translated_members getter function
    def get_translated_members(self):
        return self.translated_members
    
    # return the clusters with the category members in a list according to their category
    # e.g. {'cluster1': {'sport': ['basketball']}}
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
    
    # CLI interface for visualization
    languages = LANGUAGES.keys()
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
