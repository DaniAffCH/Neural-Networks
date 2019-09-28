#Varianza = quanto il modello è sensibile alla casualità
#Varianza e Bias correlate

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
from random import randint
from keras.layers import LeakyReLU
from keras.regularizers import l1_l2
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

labels = ["T-shirt/top","Pantalone","Pullover","Vestito","Cappotto","Sandalo","Maglietta","Sneaker","Borsa","Stivaletto"]

#normalizzazione
x_train = x_train/255
x_test = x_test/255

#dummy
y_trainDummy = to_categorical(y_train, num_classes=10)
y_testDummy = to_categorical(y_test, num_classes=10)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
num_class = y_trainDummy.shape[1]


model = Sequential()
layer1 = Dense(512, activation="relu", input_dim=x_train.shape[1], kernel_initializer='glorot_uniform', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))
layer2 = Dense(256, activation="relu", kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))
layer3 = Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))
outputLayer = Dense(num_class, activation="softmax")
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.add(outputLayer)

model.summary()

model.compile(loss = "categorical_crossentropy", optimizer=optimizers.adam(), metrics=["accuracy"])
model.fit(x_train, y_trainDummy, epochs=100, batch_size=512)
evaluateTrain = model.evaluate(x_train, y_trainDummy)
evaluateTest = model.evaluate(x_test, y_testDummy)

print("Train accuracy = %.4f - Train loss = %.4f" % (evaluateTrain[1], evaluateTrain[0]))
print("Test accuracy = %.4f - Test loss = %.4f" % (evaluateTest[1], evaluateTest[0]))

url = "https://images-na.ssl-images-amazon.com/images/I/614YpfANNtL._UX679_.jpg"
img = io.imread(url)
plt.imshow(img)
#converto l'immagine
img_small = resize(img,(28,28))
img_gray = rgb2gray(img_small)

x = img_gray.reshape(1, 28*28)
#inverto i colori (sfondo nero)
x = 1. - x

plt.imshow(x.reshape(28,28), cmap="gray")

pred = model.predict_classes(x)
pred
