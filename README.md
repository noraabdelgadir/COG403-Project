# C0G403-Project

## Overview

Despite how bilingualism is common throughout the world, with around half of the world's population being able to speak at least two languages, there is still a lot that is not understood about language processing in the bilingual brain. This project is done to test whether or not bilinguals share a common semantic network between the two languages in their brain through testing whether two words in different languages are in the same semantic space.

## Methods and Materials

### Data

- Languages used are English, French, and Arabic. 
- Word embedding models used are fastText, Word2Vec, GLoVE, and BERT.
- Categories can be seen in [methods/helpers/categories.py](https://github.com/noraabdelgadir/COG403-Project/blob/master/methods/helpers/categories.py).
- Category members can be retrieved from [ConceptNet](http://conceptnet.io/)

### Dependencies 

Install the required modules by running:

```sh
pip install -r requirements.txt
```

#### MUSE Multilingual (fastText)

The categorized embeddings are saved under [models/fasttext/categories](https://github.com/noraabdelgadir/COG403-Project/tree/master/models/fasttext/categories).

#### Word2Vec

The categorized embeddings are saved under [models/word2vec/categories](https://github.com/noraabdelgadir/COG403-Project/tree/master/models/word2vec/categories).

#### GloVe

The categorized embeddings are saved under [models/glove/categories](https://github.com/noraabdelgadir/COG403-Project/tree/master/models/glove/categories).

#### Bert 

- download the multilingual model from: https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
- run the following to start the bert server:
```sh
bert-serving-start -model_dir /path/to/model/multi_cased_L-12_H-768_A-12  -num_worker=1
```
Should see the following when it is ready to recieve requests:
```
I:WORKER-0:[__i:gen:559]:ready and listening!
I:VENTILATOR:[__i:_ru:164]:all set, ready to serve request!
```

### Methods

#### K-means clustering

K-means clustering is a simple, unsupervised machine learning method that partitions data into clusters based on their distance to a cluster's centroid.

From the root of the project folder, run:

```sh
python3 methods/k_means_clustering.py
```

The CLI will prompt you for a language, categories, and a model from a list of choices to visualize.

To see the accuracy of the model visualized, run:

```sh
python3 methods/k_means_analysis.py
```

The CLI will prompt you for a model from a list of choices to visualize the accuracy for.

#### Neural network

From the root of the project folder, run:

```sh
python3 methods/ann.py
```
The following parameters can be modified in the network initialization:

```python
self.layer1 = 128
self.layer2 = 64
self.output = 11
self.learning_rate = 2e-5
self.activation1 = 'relu'
self.activation2 = 'relu'
```

Note: the neural network currently only works with English and French.

## Members

|  Nora Abdelgadir  | Shyamolima Debnath |
| :---------------: | :----------------: |
| [![noraabdelgadir]](https://github.com/noraabdelgadir) | [![shammied]](https://github.com/shammied) |

[noraabdelgadir]: https://avatars1.githubusercontent.com/u/35353626?s=60&v=3 
[shammied]: https://avatars0.githubusercontent.com/u/23609063?s=60&v=3
