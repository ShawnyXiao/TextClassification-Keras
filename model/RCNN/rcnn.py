# coding=utf-8

from keras import Input, Model
from keras import backend as K
from keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D


class RCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input_current = Input((self.maxlen,))
        input_left = Input((self.maxlen,))
        input_right = Input((self.maxlen,))

        embedder = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        embedding_current = embedder(input_current)
        embedding_left = embedder(input_left)
        embedding_right = embedder(input_right)

        x_left = SimpleRNN(128, return_sequences=True)(embedding_left)
        x_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedding_right)
        x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)
        x = Concatenate(axis=2)([x_left, embedding_current, x_right])

        x = Conv1D(64, kernel_size=1, activation='tanh')(x)
        x = GlobalMaxPooling1D()(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_current, input_left, input_right], outputs=output)
        return model
