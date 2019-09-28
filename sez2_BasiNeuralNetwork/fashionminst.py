from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
from random import randint
from keras.layers import LeakyReLU

((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

#normalizzazione
x_train = x_train/255
x_test = x_test/255

#dummy
y_trainDummy = to_categorical(y_train, num_classes=10)
y_testDummy = to_categorical(y_test, num_classes=10)

plt.imshow(x_train[randint(0, 60000)].reshape([28,28]), cmap = "gray")
plt.axis("off")


def relu():
    global x_train, x_test, y_trainDummy, y_testDummy
    x_train.shape
    x_test.shape
    y_trainDummy.shape
    x_train = x_train.reshape(60000, 28*28)
    x_test = x_test.reshape(10000, 28*28)
    num_class = y_trainDummy.shape[1]
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

    model.compile(loss = "categorical_crossentropy", optimizer=optimizers.adam(), metrics=["accuracy"])
    model.fit(x_train, y_trainDummy, epochs=15)
    model.evaluate(x_test, y_testDummy)

    #visualizzazione errori
    y_pred = model.predict_classes(x_test)
    print(x_test)
    for c in range(0, len(x_test)):
        if(y_test[c] != y_pred[c]):
            print("Numero %d classificato come %d" % (y_test[c], y_pred[c]))
            plt.imshow(x_train[c].reshape([28,28]), cmap = "gray")
            plt.axis("off")
            plt.show()

def variant():
    global x_train, x_test, y_trainDummy, y_testDummy
    x_train.shape
    x_train = x_train.reshape(60000, 28*28)
    num_class = y_trainDummy.shape[1]
    model = Sequential()
    layer1 = Dense(512, input_dim=x_train.shape[1], kernel_initializer='glorot_uniform')
    layer2 = Dense(256)
    layer3 = Dense(128)
    outputLayer = Dense(num_class, activation="softmax")
    model.add(layer1)
    model.add(LeakyReLU(alpha=0.01))
    model.add(layer2)
    model.add(LeakyReLU(alpha=0.01))
    model.add(layer3)
    model.add(LeakyReLU(alpha=0.01))
    model.add(outputLayer)


relu()
