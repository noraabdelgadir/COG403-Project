# C0G403-Project

## Overview

Despite how bilingualism is common throughout the world, with around half of the world's population being able to speak at least two languages, there is still a lot that is not understood about language processing in the bilingual brain. This project is done to test whether or not bilinguals share a common semantic network between the two languages in their brain through testing whether two words in different languages are in the same semantic space.

## Methods and Materials

### Data

### Preprocessing

#### MUSE Multilingual 
#### Word2Vec
#### GloVe
- download from: http://nlp.stanford.edu/data/glove.6B.zip
#### Bert 
- download the multilingual model from: https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
- run the following to start the bert server:
```sh
pip install bert-serving-server
pip install bert-serving-client
bert-serving-start -model_dir /path/to/model/multi_cased_L-12_H-768_A-12  -num_worker=1
```
Should see the following when it is ready to recieve requests:
```
I:WORKER-0:[__i:gen:559]:ready and listening!
I:VENTILATOR:[__i:_ru:164]:all set, ready to serve request!
```

### Models
- K-means clustering
- Neural network

## Members

|  Nora Abdelgadir  | Shyamolima Debnath |
| :---------------: | :----------------: |
| [![noraabdelgadir]](https://github.com/noraabdelgadir) | [![shammied]](https://github.com/shammied) |

[noraabdelgadir]: https://avatars1.githubusercontent.com/u/35353626?s=60&v=3 
[shammied]: https://avatars0.githubusercontent.com/u/23609063?s=60&v=3
