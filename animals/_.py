from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras import regularizers

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('animals/training_set',
target_size = (64, 64),
batch_size = 4000,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('animals/test_set',
target_size = (64, 64),
batch_size = 4000,
class_mode = 'binary')

training_set[0][0].shape

model = Sequential()
regL1 = regularizers.l2(l=0.01)
model.add(Conv2D(filters = 64, kernel_size=(2,2), strides=(2, 2), input_shape=(64, 64, 3),padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', kernel_regularizer=regL1, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size=(2,2), strides=(2, 2), input_shape=(64, 64, 3),padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='glorot_normal', kernel_regularizer=regL1, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
