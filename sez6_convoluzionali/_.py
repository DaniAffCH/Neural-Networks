from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt

from keras import backend as k
gpus = k.tensorflow_backend._get_available_gpus()
print(gpus)

((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_trainDummy = to_categorical(y_train, num_classes=10)
y_testDummy = to_categorical(y_test, num_classes=10)

from keras.layers import Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential(layers=None, name=None)
#i filtri vengono ottimizzati dalla rete neurale
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation="relu", input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary(line_length=None, positions=None, print_fn=None)

model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

from keras.callbacks import EarlyStopping
# Se il modello non migliora dopo alcune epoche blocca l'addestramento
# se dopo 5 epoche il valore della validation loss non è migliorato di almeno 0.001 allora blocca l'addestramento
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

model.fit(x=x_train, y=y_trainDummy, batch_size=512, epochs=100, verbose=1, callbacks=[earlystop], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
model.evaluate(x=x_test, y=y_testDummy, batch_size=512, verbose=1, sample_weight=None, steps=None)
