# author: Marjan Hosseinia
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

SPLITTOR='$$$'


def load_embedding(embedding_file=''):
    #glove.6B.100d
    embeddings_index = {}
    f = open(embedding_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip()


def extract_text_lable(data_panda):
    all_texts_kn, all_texts_unkn, labels_all, id_all = [], [],[] ,[]
    for idx in range(data_panda.review.shape[0]):
        temp = data_panda.review[idx].split('$$$')
        id_doc = data_panda.id[idx]
        if len(temp) != 2:
            print(len(temp),data_panda.id[idx], data_panda.review[idx])
            quit()

        text_kn = BeautifulSoup(temp[0], "html.parser").get_text().encode('ascii','ignore')
        text_unkn = BeautifulSoup(temp[1], "html.parser").get_text().encode('ascii','ignore')
        text_kn, text_unkn = clean_str(text_kn), clean_str(text_unkn)
        all_texts_kn.append(text_kn)
        all_texts_unkn.append(text_unkn)
        labels_all.append(data_panda.sentiment[idx])
        id_all.append(id_doc)
    return all_texts_kn, all_texts_unkn, labels_all , id_all


def load_data(dataset_id='2015.cnn', isonefile=False, MAX_SEQUENCE_LENGTH = 1000):

    if not isonefile:
        data_train_ = pd.read_csv('train{}'.format(dataset_id), sep='\t')
        data_test_ = pd.read_csv('test{}'.format(dataset_id), sep='\t')
        print('size of original training and test files :')
        print (data_train_.shape, data_test_.shape)
        texts_kn, texts_unkn, labels_all , id_alls = extract_text_lable(data_train_)
        t0, t1, l1, id1 = extract_text_lable(data_test_)
        texts_kn = texts_kn + t0
        texts_unkn = texts_unkn + t1
        labels_all = labels_all + l1
        id_alls = id_alls + id1

    else:
        data_test_train_ = pd.read_csv('{}'.format(dataset_id), sep='\t')
        print (data_test_train_.shape)
        texts_kn, texts_unkn, labels_all, id_alls = extract_text_lable(data_test_train_)

    print('Total # of docs : %s' % len(texts_kn))
    assert len(texts_kn) == len(texts_unkn) == (len(id_alls)) == (len(labels_all))

    all_text = texts_kn + texts_unkn
    l = [len(t.split()) for t in all_text]
    print 'Mean of doc length: %.4f  max of doc length: %.4f' %(np.mean(l), np.max(l))

    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(all_text)
    l_sequences = tokenizer.texts_to_sequences(texts_kn)
    r_sequences = tokenizer.texts_to_sequences(texts_unkn)

    word_index = tokenizer.word_index
    print('Found %s unique tokens in all text' % len(word_index))

    l_data = pad_sequences(l_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    r_data = pad_sequences(r_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    assert l_data.shape == r_data.shape

    print('Shape of left/right-side tensor:', l_data.shape)
    return [l_data, r_data], labels_all, tokenizer.word_index, id_alls


def euclid_dist(inputs):

    output = K.sqrt(K.sum( K.square(inputs[0]-inputs[1]), axis=-1))
    output = K.expand_dims(output, 1)
    return output


def cosine_dist(inputs):
    x1 = K.l2_normalize(inputs[0], axis=-1)
    x2 = K.l2_normalize(inputs[1], axis=-1)
    output = K.sum(x1 * x2, axis=1, keepdims=True)
    return output


def dotm(inputs, axis=1):
    return K.sum(inputs[0] * inputs[1], axis=axis, keepdims=True)


def mean_of_l1(inputs):
    return K.mean(K.abs(inputs[0] - inputs[1]), axis=1, keepdims=True)


def sigmoid_kernel(inputs, lamda=None,c=1):
    if lamda is None:
        lamda = inputs[0].shape[1]

    output = K.tanh(lamda* dotm(inputs, axis=-1)+c)
    return output


def chi_squared(inputs, lamda=1):
    output = K.exp(lamda * K.sum(K.square(inputs[0]-inputs[1])/(inputs[0]-inputs[1]), axis=-1, keepdims=True))

    return output


def rbf_kernel(inputs, gamma=1):
    output = K.sum(K.square(inputs[0]-inputs[1]), axis=-1, keepdims=True)
    output = K.exp(-gamma*output)
    return output


def all_distances_moremetrics(inputs):
    euc = euclid_dist(inputs)
    cos = cosine_dist(inputs)
    rbf = rbf_kernel(inputs)
    #chi = chi_squared(inputs)
    sig = sigmoid_kernel(inputs)
    dt = dotm(inputs)
    mean = mean_of_l1(inputs)
    return K.concatenate([euc,cos,rbf,sig,  dt, mean],-1)


def out_shape_moremetrics(shape):
    #print shape
    return(None,6)











