# author Marjan Hosseinia
import numpy as np
from keras.models import Sequential
import os
from keras.layers import Dense, Input
from keras.layers import  Embedding, Merge, Dropout, SimpleRNN
from keras.models import Model
from keras import optimizers
import util
from keras import losses

os.environ['KERAS_BACKEND'] = 'theano'


def model(drop=0.3, hidden_units=64, word_index=None, embedding_index=None, EMBEDDING_DIM=8, MAX_SEQUENCE_LENGTH=1000):
    '''
    specifies NN architecture
    :param drop: dropout size
    :param hidden_units:
    :param word_index:
    :param embedding_index:
    :param EMBEDDING_DIM:
    :param MAX_SEQUENCE_LENGTH:
    :return: complete model and intermediate model
    '''

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    l_embedding_layer = Embedding(len(word_index) + 1,
                                  EMBEDDING_DIM,
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  weights=[embedding_matrix],
                                  trainable=True
                                  )
    l_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    l_embedded_sequences = l_embedding_layer(l_sequence_input)
    l_bi = SimpleRNN(hidden_units)(l_embedded_sequences)
    l_drop = Dropout(drop)(l_bi)
    l_model = Model(l_sequence_input, l_drop)

    # right CNN

    r_embedding_layer = Embedding(len(word_index) + 1,
                                  EMBEDDING_DIM,
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  weights=[embedding_matrix],
                                  trainable=True
                                  )
    r_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    r_embedded_sequences = r_embedding_layer(r_sequence_input)
    r_bi = SimpleRNN(hidden_units)(r_embedded_sequences)
    r_drop = Dropout(drop)(r_bi)
    r_model = Model(r_sequence_input, r_drop)

    merged = Merge([l_model, r_model], mode=util.all_distances_moremetrics, output_shape=util.out_shape_moremetrics,name='fusion')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(1, activation='sigmoid'))
    print (final_model.inputs)
    final_model.compile(loss=losses.binary_crossentropy,
                        optimizer=optimizers.rmsprop(),
                        metrics=['acc']

                        )
    layer_name = 'fusion'
    intermediate_layer_model = Model(inputs=final_model.input,
                                     outputs=final_model.get_layer(layer_name).output)

    return final_model , intermediate_layer_model


