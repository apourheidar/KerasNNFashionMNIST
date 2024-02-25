from _ast import mod

import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
X_train_full.dtype


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] /255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

print(X_train.shape)
print(X_valid.shape)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(y_train[0])
print(class_names[y_train[0]])

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
print("Summary")
model.summary()
print("models")
model.layers
print("models 0-4")
print(model.layers[0].name)
model.layers[1].name
model.layers[2].name
model.layers[3].name


hidden=model.layers[1]
weights,bias=hidden.get_weights()
print("weights-bias")

print(weights.shape)
print(bias)

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))


import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

print("evaluation")
print(model.evaluate(X_test, y_test))

X_new = X_test[:3]
y_proba = model.predict(X_new)
print("Y probablities")
print(y_proba.round(2))

#y_pred = model.predict_classes(X_new)
#np.array(class_names)[y_pred]





