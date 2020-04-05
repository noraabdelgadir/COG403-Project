import inquirer
from methods.k_means_clustering.k_means_clustering import KMeansClustering
from helpers import CATEGORIES, MODELS

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

if __name__ == "__main__":
    k = KMeansClustering(answers['language'], answers['categories'], answers['model'])
    k.visualize()
