# Text Classification using PyTorch

This repository is an introduction to Deep Learning methods to a text classification task. It has been constructed on the [Quora insincere questions](https://www.kaggle.com/c/quora-insincere-questions-classification) Kaggle challenge :

<i>"On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers."</i>

A first [notebook](https://github.com/RBeaudet/Text-Classification-Using-PyTorch/blob/master/Didactic%20text%20classification%20notebook%20(1).ipynb) will walk you through <b>Torchtext</b>, a library designed to handle the preprocessing of text data, in order to perform the following :
- Load text data from a .csv file ;
- Tokenize sentences ;
- Build vocabulary ;
- Load pre-trained embedding vectors ;
- Create batch of data using dynamic padding.

A simple Text Classification model using <b>PyTorch</b> is then constructed with the following architecture :
- Embedding layer ;
- Recurrent layer (LSTM) ;
- Fully connected layer.

A second [notebook](https://github.com/RBeaudet/Text-Classification-Using-PyTorch/blob/master/Didactic%20text%20classification%20notebook%20(2).ipynb) introduces a more complex model which incorporates a <b>Self-Attention</b> method proposed in 2017 this [paper](https://github.com/RBeaudet/Text-Classification-Using-PyTorch/blob/master/A%20structured%20Self-Attentive%20sentence%20embedding.pdf). The model architecture presented in this notebook allows to :
- Decide wether to re-train the embedding layer or not ;
- Decide wether to use a LSTM or a GRU unit ;
- Decide the number of stacked recurrent layers ;
- Decide to use a bidirectional recurrent layer or not ;
- Incorporate an Attention mechanism.

For a modular approach, the [code](https://github.com/RBeaudet/Text-Classification-Using-PyTorch/tree/master/code) also comes in separate Python files :
- `data_utils.py` performs text preprocessing and data loading ;
- `model.py` defines the model architecture ;
- `solver.py` defines the training process ;
- `main.py` is a utilization example.

---

#### References :

Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio. "A Structured Self-attentive Sentence Embedding". 2017.


