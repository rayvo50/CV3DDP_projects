# for the model implementations
import numpy as np
from model import Model, Dense, Activation, Dropout, Loss

# for loading the dataset only
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#for analysing results
import matplotlib.pyplot as plt


# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocess to fit model
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255
y_train = to_categorical(y_train, 10)
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255
y_test = to_categorical(y_test, 10)


# create the model
model = Model()
model.add(Dense(784, 64))
model.add(Activation("relu"))
model.add(Dense(64, 10))   
model.add(Activation("softmax"))

# set loss function
model.set_loss(Loss("cross_entropy"))

# print total number of parameters
print(f"Total params: {model.count_params()}")

# train model
history = model.fit(x_train, y_train, x_test, y_test, epochs=100, learning_rate=0.001, batch_size=32)

# plot loss and accurcy over epochs
_, ax = plt.subplots(ncols=2, figsize=(15,5))
ax[0].plot(history['loss'], label='Training Loss')
ax[0].plot(history['val_loss'], label='Test Loss')
ax[0].set_title('Loss Over Epochs')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].grid()
ax[1].plot(history['accuracy'], label='Training Accuracy')
ax[1].plot(history['val_accuracy'], label='Test Accuracy')
ax[1].set_title('Accuracy Over Epochs')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid()
plt.show()
