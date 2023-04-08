import numpy
import tensorflow
from PIL import Image
from imageio.core import urlopen
from numpy import array
from numpy.core.multiarray import asarray
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils

seed = 21
from keras.datasets import cifar10

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model = keras.models.load_model('my-models/model')
image = Image.open(urlopen(input("Paste the link of an image to analyze here: "))).resize((32, 32))
numpydata = asarray(image).reshape(-1, 32, 32, 3)
a = model.predict(numpydata)

print(a)
max_label = ""
for i in range(len(a[0])):

    if a[0][i] > 0.7:
        max_label = labels[i]
print(max_label)
print("this is an image of a", max_label)
