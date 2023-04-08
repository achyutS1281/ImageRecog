import numpy
import tensorflow
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils
seed = 21
from keras.datasets import cifar100
from PIL import Image
from numpy import asarray

# load the image and convert into
# numpy array

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
assert X_train.shape == (50000, 32, 32, 3)
assert X_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]
custom = keras.utils.custom_object_scope(keras.applications.MobileNetV2)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(class_num, activation='softmax'))
checkpoint = keras.callbacks.ModelCheckpoint("my-models/model", monitor='accuracy', verbose=0, save_best_only=True,
                                             save_weights_only=False, mode='auto', save_freq="epoch")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

numpy.random.seed(seed)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=2000, callbacks=[checkpoint])
scores = model.evaluate(X_test, y_test, verbose=0)
model.save('my-models/model')

print("Accuracy: %.2f%%" % (scores[1]*100))
