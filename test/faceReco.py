import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageDraw
import requests
from io import BytesIO

from skimage import io
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping

dataset = pd.read_json("/home/daniaffch/Scrivania/AI_t/test/face_detection.json", lines=True)

def findFace(imgChoosen):
    response = requests.get(dataset["content"][imgChoosen])
    img = Image.open(BytesIO(response.content))
    d = ImageDraw.Draw(img)

    for face, useless in enumerate(dataset["annotation"][imgChoosen]):
        pointTOPx = dataset["annotation"][imgChoosen][face]["points"][0]["x"]*dataset["annotation"][imgChoosen][face]["imageWidth"]
        pointTOPy = dataset["annotation"][imgChoosen][face]["points"][0]["y"]*dataset["annotation"][imgChoosen][face]["imageHeight"]
        pointBOTx = dataset["annotation"][imgChoosen][face]["points"][1]["x"]*dataset["annotation"][imgChoosen][face]["imageWidth"]
        pointBOTy = dataset["annotation"][imgChoosen][face]["points"][1]["y"]*dataset["annotation"][imgChoosen][face]["imageHeight"]

        pointTOP_LEFT = (pointTOPx, pointTOPy)
        pointTOP_RIGHT = (pointBOTx, pointTOPy)
        pointBOT_LEFT = (pointTOPx, pointBOTy)
        pointBOT_RIGHT = (pointBOTx, pointBOTy)

        line_color = (0, 255, 0)

        d.line([pointTOP_LEFT,pointTOP_RIGHT, pointBOT_RIGHT, pointBOT_LEFT, pointTOP_LEFT], fill=line_color, width=2)
    img.show()

x = np.asarray(dataset["content"].values)
y = np.array([])

for element, useless in enumerate(x):
        x[element] = rgb2gray(io.imread(x[element]))

for target in dataset["annotation"].values:
    del target[0]["notes"]
    del target[0]["label"]
    y = np.append(y, target[0])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

#REGRESSIONE
x_train[0].shape
y_train.shape

x_train

model = Sequential()
#Convoluzione 3D?????
convoluzionale1 = Conv2D(filters=64, kernel_size=2, padding='same', activation="relu", input_shape = (1321, 1433))
pool1 = MaxPooling2D(pool_size=2, strides=2)
antiOverfitting1 = Dropout(0.3)
convoluzionale2 = Conv2D(filters=32, kernel_size=2, padding='same', activation="relu")
pool2 = MaxPooling2D(pool_size=2, strides=2)
antiOverfitting2 = Dropout(0.3)
flattening = Flatten()
denso = Dense(256, activation="relu")
output = Dense(1, activation="linear")

model.add(convoluzionale1)
model.add(pool1)
model.add(antiOverfitting1)
model.add(convoluzionale2)
model.add(pool2)
model.add(antiOverfitting2)
model.add(flattening)
model.add(denso)
model.add(output)

model.summary()

model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
model.fit(x=x_train[0], y=y_train, batch_size=512, epochs=10, verbose=1, callbacks=[earlystop], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
