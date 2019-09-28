import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import os
import struct
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import cv2

def load_mnist(path="/"):

    train_labels_path = os.path.join(path,"train-labels-idx1-ubyte")
    train_images_path = os.path.join(path,"train-images-idx3-ubyte")

    test_labels_path = os.path.join(path,"t10k-labels-idx1-ubyte")
    test_images_path = os.path.join(path,"t10k-images-idx3-ubyte")

    labels_path = [train_labels_path, test_labels_path]
    images_path = [train_images_path, test_images_path]

    labels = []
    images = []

    for path in zip(labels_path, images_path):

        with open(path[0],'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            lb = np.fromfile(lbpath, dtype=np.uint8)
            labels.append(lb)

        with open(path[1], 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images.append(np.fromfile(imgpath, dtype=np.uint8).reshape(len(lb), 784))

    return images[0], images[1], labels[0], labels[1]

#CLASSIFICAZIONE MULTIOUTPUT

#classificare cifre scritte a mano

#conversione da binario a np array, dimensioni di 28x28 px
x_train, x_test, y_train, y_test = load_mnist(path="/home/daniaffch/Scrivania/AI_t/sez2_BasiNeuralNetwork/MNIST_data")
x_train.shape
#visualizzazione immagine
plt.imshow(x_train[39].reshape([28,28]), cmap = "gray")
plt.axis("off")
y_train[39]

#normalizzazione da 0 a 1

x_train = x_train/255
x_test = x_test/255

x_test[1]
#l'output va da 0 a 9 (10 nodi di output)
# variabili di comodo (valori di 1 o 0)
from keras.utils import to_categorical
num_class = 10
y_train_dummy = to_categorical(y_train, num_classes=num_class)
y_test_dummy = to_categorical(y_test, num_classes=num_class)

model = Sequential()
layer1 = Dense(512, activation="relu", input_dim=x_train.shape[1], kernel_initializer='glorot_uniform')
layer2 = Dense(256, activation="relu")
layer3 = Dense(128, activation="relu")
outputLayer = Dense(num_class, activation="softmax")

model.add(layer1)
model.add(layer2)
model.add(layer3)
model.add(outputLayer)

model.summary()

from keras import optimizers
#implementazione momentum (il valore da inserire è gamma) e lr adattivo
sgd = optimizers.SGD(momentum=0.9, nesterov=True, decay=1e-6, lr=0.01)
#ADAM
adam = optimizers.adam()
#categorical_crossentropy per il multiclasse
model.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model.fit(x_train, y_train_dummy, epochs=15)
model.evaluate(x_test, y_test_dummy)

#visualizzazione errori

y_pred = model.predict_classes(x_test)
print(x_test)
for c in range(0, len(x_test)):
    if(y_test[c] != y_pred[c]):
        print("Numero %d classificato come %d" % (y_test[c], y_pred[c]))
        plt.imshow(x_train[c].reshape([28,28]), cmap = "gray")
        plt.axis("off")
        plt.show()

#src = "/home/daniaffch/Scrivania/AI_t/sez2_BasiNeuralNetwork/jpegmini_optimized/IMG_0683.jpg"
#image_file = Image.open(src)
#image_file = image_file.convert('1')
#image_file = image_file.rotate(-90)
#img = image_file.resize((28, 28))
#imgArray = np.array(img.getdata()).reshape(784)
#imgArray
#imgArray = np.array(img)
#imgArray = imgArray.reshape(1,784)
#print(model.predict_classes(imgArray))
