from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Concatenate, Conv1D, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D


class RCNNVariant(Model):
    """Variant of RCNN.

        Base on structure of RCNN, we do some improvement:
        1. Ignore the shift for left/right context.
        2. Use Bidirectional LSTM/GRU to encode context.
        3. Use Multi-CNN to represent the semantic vectors.
        4. Use ReLU instead of Tanh.
        5. Use both AveragePooling and MaxPooling.
    """

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 kernel_sizes=[1, 2, 3, 4, 5],
                 class_num=1,
                 last_activation='sigmoid'):
        super(RCNNVariant, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))
        self.concatenate = Concatenate()
        self.convs = []
        for kernel_size in self.kernel_sizes:
            conv = Conv1D(128, kernel_size, activation='relu')
            self.convs.append(conv)
        self.avg_pooling = GlobalAveragePooling1D()
        self.max_pooling = GlobalMaxPooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x_context = self.bi_rnn(embedding)
        x = self.concatenate([embedding, x_context])
        convs = []
        for i in range(len(self.kernel_sizes)):
            conv = self.convs[i](x)
            convs.append(conv)
        poolings = [self.avg_pooling(conv) for conv in convs] + [self.max_pooling(conv) for conv in convs]
        x = self.concatenate(poolings)
        output = self.classifier(x)
        return output
