original = {
    'vehicle': ['rocket', 'canoe', 'plane', 'vehicle', 'truck', 'sled', 'cart', 'car', 'tractor', 'sailboat', 'boat'], 
    'vegetable': ['lettuce', 'spinach', 'greens', 'plantain', 'squash', 'corn', 'asparagus', 'truffle', 'pumpkin', 'eggplant', 'fennel', 'celery', 'radish', 'cucumber', 'artichoke', 'vegetable', 'gumbo', 'beet', 'bean', 'mushroom'], 
    'tree': ['cassia', 'bonsai', 'tree', 'millettia', 'maria', 'oak', 'hazel']
}

clustered = {
    'cluster1': {
        'vegetable': ['corn', 'bean', 'greens', 'squash', 'vegetable', 'mushroom', 'pumpkin', 'lettuce', 'beet', 'cucumber', 'spinach', 'asparagus', 'plantain', 'celery', 'eggplant', 'gumbo', 'truffle', 'radish', 'artichoke', 'fennel']
    }, 
    'cluster2': {
        'vehicle': ['canoe'], 
        'tree': ['tree', 'maria', 'oak', 'hazel', 'bonsai', 'cassia', 'millettia', 'sapin'], 
        'vegetable': ['végétal']
    }, 
    'cluster3': {
        'vehicle': ['car', 'vehicle', 'boat', 'plane', 'rocket', 'truck', 'tractor', 'cart', 'sled', 'sailboat', 'véhicule']
    }
}

original = {
    'vehicle': ['rocket', 'plane', 'boat', 'truck', 'cart', 'vehicle', 'sled', 'sailboat', 'canoe', 'car', 'tractor'], 
    'tree': ['maria', 'hazel', 'bonsai', 'cassia', 'oak', 'tree', 'millettia'], 
    'flower': ['lily', 'bloomer', 'carnation', 'violet', 'iris', 'composite', 'silene', 'daisy', 'anemone', 'flower', 'hibiscus', 'dandelion', 'columbine', 'valerian', 'petunia'], 
    'colour': ['cyan', 'orange', 'magenta', 'white', 'beige', 'red', 'green', 'yellow', 'blue']
}

clustered = {
    'cluster1': {
        'sapin': ['maria', 'oak', 'hazel', 'sapin'], 
        'fleurs': ['iris', 'columbine', 'carnation', 'bloomer'], 
        'couleur': ['couleur']
    }, 
    'cluster2': {
        'fleurs': ['violet'], 
        'couleur': ['white', 'red', 'green', 'blue', 'yellow', 'orange', 'magenta', 'beige', 'cyan']
    }, 
    'cluster3': {
        'véhicule': ['car', 'vehicle', 'boat', 'plane', 'rocket', 'truck', 'canoe', 'tractor', 'cart', 'sled', 'sailboat', 'véhicule'], 
        'fleurs': ['composite']
    }, 
    'cluster4': {
        'sapin': ['tree', 'bonsai', 'cassia', 'millettia'], 
        'fleurs': ['flower', 'lily', 'daisy', 'anemone', 'hibiscus', 'dandelion', 'valerian', 'silene', 'petunia', 'fleurs']
    }
}


category_cluster = {}
for cluster, categories in clustered.items(): 
    if len(categories) == 1:
        category_cluster[list(categories.keys())[0]] = cluster

remaining_categories = list(set(original.keys() - set(category_cluster.keys())))
remaining_clusters = list(set(clustered.keys() - set(category_cluster.values())))

translation = {'vehicle': 'véhicule', 'tree': 'sapin', 'flower': 'fleurs', 'colour': 'couleur'}

cat_cluster_count = {}
for cluster, categories in clustered.items():
    if not len(categories) == 1 and cluster in remaining_clusters:
        for cat in categories:
            # only looking at remaining categories
            if cat in remaining_categories:
                if cat not in cat_cluster_count:
                    cat_cluster_count[cat] = {}
                    cat_cluster_count[cat][cluster] = 0
                if cluster not in cat_cluster_count[cat]:
                    cat_cluster_count[cat][cluster] = 0
                cat_cluster_count[cat][cluster] += len(list(categories[cat]))

for cat, cluster_count in cat_cluster_count.items():
    if len(cluster_count) == 1:
        cluster = list(cluster_count.keys())[0]
        category_cluster[cat] = [cluster]
        remaining_clusters.remove(cluster)
        remaining_categories.remove(cat)

for cat, cluster_count, in cat_cluster_count.items():
    if cat in remaining_categories:
        max_count = max(cluster_count.values())
        for cluster, count in cluster_count.items():
            if cluster in remaining_clusters and count == max_count:
                if not cat in category_cluster:
                    category_cluster[cat] = [cluster]
                else:
                    category_cluster[cat].append(cluster)

print(category_cluster)

for cat, clusters in category_cluster.items(): 
    print(cat, clusters)
    if len(clusters) > 1:
        # assign based on where the translation category is
        for cluster in clusters:
            # print(cluster)
            if translation[cat] in clustered[cluster][cat]:
                category_cluster[cat] = cluster
    else:
        category_cluster[cat] = clusters[0]

print(category_cluster)

for category, cluster in category_cluster.items():
    actual = clustered[cluster][category]
    expected = original[category]
    difference = list(set(expected) - set(actual))
    print(100 - len(difference)/len(expected) * 100)
