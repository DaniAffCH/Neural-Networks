import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz', num_words=num_words, skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3)

np.load = np_load_old

maxlen = 500

x_train = pad_sequences(x_train, maxlen=maxlen, dtype='int32', padding='pre', truncating='pre', value=0.)
x_test = pad_sequences(x_test, maxlen=maxlen, dtype='int32', padding='pre', truncating='pre', value=0.)

from keras.layers import GRU

model = Sequential(layers=None, name=None)

model.add(Embedding(num_words, 50))

model.add(GRU(32, dropout=0.4, recurrent_dropout=0.2, return_sequences=True))

model.add(GRU(32, dropout=0.4, recurrent_dropout=0.2))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

model.fit(x=x_train, y=y_train, batch_size=512, epochs=5, verbose=1, callbacks=None, validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

model.evaluate(x=x_test, y=y_test, batch_size=None, verbose=1, sample_weight=None, steps=None)
