import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D


class RCNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        super(RCNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.forward_rnn = SimpleRNN(128, return_sequences=True)
        self.backward_rnn = SimpleRNN(128, return_sequences=True, go_backwards=True)
        self.reverse = Lambda(lambda x: tf.reverse(x, axis=[1]))
        self.concatenate = Concatenate(axis=2)
        self.conv = Conv1D(64, kernel_size=1, activation='tanh')
        self.max_pooling = GlobalMaxPooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs) != 3:
            raise ValueError('The length of inputs of RCNN must be 3, but now is %d' % len(inputs))
        input_current = inputs[0]
        input_left = inputs[1]
        input_right = inputs[2]
        if len(input_current.get_shape()) != 2 or len(input_left.get_shape()) != 2 or len(input_right.get_shape()) != 2:
            raise ValueError('The rank of inputs of RCNN must be (2, 2, 2), but now is (%d, %d, %d)' % (len(input_current.get_shape()), len(input_left.get_shape()), len(input_right.get_shape())))
        if input_current.get_shape()[1] != self.maxlen or input_left.get_shape()[1] != self.maxlen or input_right.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of RCNN must be (%d, %d, %d), but now is (%d, %d, %d)' % (self.maxlen, self.maxlen, self.maxlen, input_current.get_shape()[1], input_left.get_shape()[1], input_right.get_shape()[1]))
        embedding_current = self.embedding(input_current)
        embedding_left = self.embedding(input_left)
        embedding_right = self.embedding(input_right)
        x_left = self.forward_rnn(embedding_left)
        x_right = self.backward_rnn(embedding_right)
        x_right = self.reverse(x_right)
        x = self.concatenate([x_left, embedding_current, x_right])
        x = self.conv(x)
        x = self.max_pooling(x)
        output = self.classifier(x)
        return output
