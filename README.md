# Experiments with Neural Networks for Small and Large Scale Authorship Verification

Authors : **Marjan Hosseinia** and **Arjun Mukherjee** [\[pdf\]](https://arxiv.org/pdf/1803.06456.pdf)

We propose two models for a special case of authorship verification problem. The task is to investigate whether the two documents of a given pair are written by the same author. We consider the authorship verification problem for both small and large scale datasets. The underlying small-scale problem has two main challenges: First, the authors of the documents are unknown to us because no previous writing samples are available. Second, the two documents are short (a few hundred to a few thousand words) and may differ considerably in the genre and/or topic. To solve it we propose transformation encoder to transform one document of the pair into the other. This document transformation generates a loss which is used as a recognizable feature to verify if the authors of the pair are identical. For the large scale problem where various authors are engaged and more examples are available with larger length, a parallel recurrent neural network is proposed. It compares the language models of the two documents. We evaluate our methods on various types of datasets including Authorship Identification datasets of PAN competition, Amazon reviews, and machine learning articles. Experiments show that both methods achieve stable and competitive performance compared to the baselines.

### Prerequisits:
`Python 2.7, Theano, Keras 2.0`
### To see the list of argumnets:
 `python main.py -h` 
### To run  Cross Validation:
`python main.py`
 
### datasets:
* [PAN Authorship Verification datasets](http://pan.webis.de/data.html) PAN13,PAN14, PAN15
* Amazon reviews
* A subset of [MLPA-400](https://github.com/dainis-boumber/AA_CNN/wiki/MLPA-400-Dataset)
