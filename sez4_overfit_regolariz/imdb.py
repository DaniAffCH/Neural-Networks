import matplotlib.pyplot as plt
import numpy as np

from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History
from keras.regularizers import l2
from keras import optimizers

from keras.datasets import imdb

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

(xTrain, yTrain), (xTest, yTest) = imdb.load_data(num_words=5000)

np.load = np_load_old

word_index = imdb.get_word_index()

#dati codificati per frequenza traslato di 3 caratteri in avanti

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = [reverse_word_index.get(i-3, "?") for i in xTrain[0]]
" ".join(decoded_review)

#preprocessing
def onehot_encoding(data, size):
    onehot = np.zeros((len(data), size))
    for i, d in enumerate(data):
        onehot[i,d] = 1
    return onehot

xTrain_oh = onehot_encoding(xTrain, 5000)
xTest_oh = onehot_encoding(xTest, 5000)

#creazione validation test

xTrain2 = xTrain[:-2500]
yTrain2 = yTrain[:-2500]

xVal = xTrain[-2500:]
yVal = yTrain[-2500:]



#creazione rete neurale

from keras.layers import Dropout

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(5000,), kernel_regularizer=l2(0.1)))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adamax", loss="binary_crossentropy", metrics=["accuracy"])
#oppure validation_split=0.1
model.fit(xTrain_oh, yTrain, epochs=15, batch_size=512, validation_data=(xVal, yVal))
model.evaluate(xTest_oh, yTest)

#overfitting!

#predict
frase = "Quentin Tarantino's 8th film 'Django Unchained' is one hell of a movie. A brutal, bloody, terrifying, hilarious and awe-inspiring western disguised as a buddy movie that is so great that if John Wayne and Sergio Leone were alive now, they would've approve of this movie. It's designed to shock you, polarize you, test you and maybe even surprise you. But let me clear on this: If you are not a fan of bloody violence and the running length of 165 minutes, see a shorter movie. But if you love to see what Tarantino can do with movies like this, then you're in for a treat. Set during slavery in 1858, the movie follows Django (Jamie Foxx), a slave who is found by a bounty hunter disguised as a dentist named Dr. King Schultz (The always reliable Christoph Waltz) who hires him as a bounty hunter and a free man to find the Brittle Brothers. After finding them and hunting them down at a plantation run by Big Daddy (a remarkable Don Johnson), they relax for the winter only for them to go on a mission to find and rescue Django's wife, Broomhilda (Kerry Washington) who is owned by Calvin J. Candie (Leonardo DiCaprio) a man who runs a plantation known as Candieland. He even has a renegade slave as a servant named Stephen (A nearly recognizable Samuel L. Jackson, hidden in makeup and some prosthetics), who will have a part to play in the last half of the movie. I think Quentin Tarantino has outdone himself once again. Being in the filmmaking game for 20 years now, you can't deny and even reject his style in what he is bringing to the screen (He also has a cameo in here as well). His dialogue is like reading a book that grabs you and makes you want to know what happens next. The look and scope of the film is magnificent, thanks to a brilliant Oscar-winning cinematographer Robert Richardson and the late production designer J. Michael Riva. The performances in this film are brilliant. Having won an Oscar for 'Ray', Jamie Foxx continues with his breathtaking performances that wows us. Here as Django, he is certainly fearless, baring his soul (and body) playing a man who is free from slavery, but can't be free by the rules and limitations of slavery. Christoph Waltz looks like he was born to be a part of Tarantino's entourage after his Oscar-winning performance for 'Inglorious Basterds'. Here, once again he brings humor and vulnerability to Dr. King Schultz. Never before have I ever seen an actor go that far and doesn't go over-the-top like Leonardo DiCaprio. As Calvin Candie, DiCaprio is certainly Oscar-worthy as a man who runs a tight ship by running a place where male slaves fight to the death and female slaves are being prostitutes and he seems to be the kind of guy to like even though he is a villain and he speaks Tarantino's dialogue like a pro. When he has a scene in which he reveals three dimples from a skull that belongs to his father, he is literally terrifying. Kerry Washington is superb as Broomhilda and Samuel L. Jackson is the real scene-stealer. The supporting cast is great from Walton Goggins, Jonah Hill, Michael Bacall, Michael Parks, James Remar, Robert Carradine to a small cameo by Franco Nero. 'Django Unchained' has a lot of things to say about slavery and how cruel it is. But at the same time, it provides the fact that if Tarantino rearranged history by shooting Adolf Hitler to a pulp while everything blows up at a movie theater, he can do it again by having a former slave whipping a man who used to beat him and his wife. Now, that's entertainment. This movie really is off the chain. It's not only one of the most captivating films of the year, it's one of the best films of the year. Go see it, it will be worth your time. Keep in mind though, there are characters, especially Django, Stephen, Candie and Schultz that uses the N-word numerous times in this movie. That seems relevant to the time period, don't ya think?"

#rimuovo la punteggiatura
from re import sub

review = sub(r'[^\w\s]','',frase)

review = review.lower()
review = review.split(" ")

reviewArray = []

for parola in review:
    if parola in word_index:
        index = word_index[parola]
        if index <= 5000:
            reviewArray.append(index+3)

x = onehot_encoding([reviewArray], 5000)

y = model.predict(x)
