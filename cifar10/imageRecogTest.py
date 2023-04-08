import numpy
import tensorflow

from PIL import Image
from imageio.core import urlopen
from mpmath import im
from numpy import array
from numpy.core.multiarray import asarray
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils

seed = 21
size = int(input("size: "))
type=input("mode (file or url): ")

from keras.datasets import cifar10

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model = keras.models.load_model('my-models/model')
if type == 'file':
    image = Image.open(input("Input image file name: "))
elif type == 'url':
    image = Image.open(urlopen(input("Paste the link of an image to analyze here: ")))

image = image.resize((size, size))
image.save(fp='cache/tested.png')

numpydata = asarray(image).reshape(-1, 32, 32, 3)

a = model.predict(numpydata)


print(a)
max_label = ""
max_val = 0
for i in range(len(a[0])):

    if a[0][i] > 0.7 and a[0][i] > max_val:
        max_label = labels[i]
        max_val = a[0][i]
print(max_label)
print("this is an image of a", max_label)
