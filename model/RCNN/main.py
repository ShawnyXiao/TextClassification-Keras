# coding=utf-8

import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.preprocessing import sequence

from rcnn import RCNN

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Prepare input for model...')
x_train_current = x_train
x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]])
x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)])
x_test_current = x_test
x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])
print('x_train_current shape:', x_train_current.shape)
print('x_train_left shape:', x_train_left.shape)
print('x_train_right shape:', x_train_right.shape)
print('x_test_current shape:', x_test_current.shape)
print('x_test_left shape:', x_test_left.shape)
print('x_test_right shape:', x_test_right.shape)

print('Build model...')
model = RCNN(maxlen, max_features, embedding_dims).get_model()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit([x_train_current, x_train_left, x_train_right], y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=([x_test_current, x_test_left, x_test_right], y_test))

print('Test...')
result = model.predict([x_test_current, x_test_left, x_test_right])
