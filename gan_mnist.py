import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tqdm import tqdm_notebook


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

z_dim = 100

adam = Adam(lr=0.0002, beta_1=0.5)

g = Sequential()
g.add(Dense(256, input_dim=z_dim, 
                activation=LeakyReLU(alpha=0.3)))
g.add(Dense(512, activation=LeakyReLU(alpha=0.3)))
g.add(Dense(1024, activation=LeakyReLU(alpha=0.3)))
g.add(Dense(784, activation='sigmoid'))  
g.compile(loss='binary_crossentropy', 
            optimizer=adam, 
            metrics=['accuracy'])

d = Sequential()
d.add(Dense(1024, input_dim=784, 
                    activation=LeakyReLU(alpha=0.3)))
d.add(Dropout(0.3))
d.add(Dense(512, activation=LeakyReLU(alpha=0.3)))
d.add(Dropout(0.3))
d.add(Dense(256, activation=LeakyReLU(alpha=0.3)))
d.add(Dropout(0.3))
d.add(Dense(1, activation='sigmoid'))  
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

d.trainable = False
inputs = Input(shape=(z_dim, ))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


def plot_loss(losses):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]

    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_generated(n_ex=10, dim=(1, 10), figsize=(12, 2)):
    noise = np.random.normal(0, 1, size=(n_ex, z_dim))
    generated_images = g.predict(noise)
    generated_images = generated_images.reshape(n_ex, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

losses = {"D":[], "G":[]}

def train(epochs=1, plt_frq=1, BATCH_SIZE=128):
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)
    
    for e in tqdm_notebook(range(1, epochs+1)):
        if e == 1 or e%plt_frq == 0:
            print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batchCount):  
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            
    
            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))

            y = np.zeros(2*BATCH_SIZE)
            y[:BATCH_SIZE] = 0.9  

            d.trainable = True
            d_loss = d.train_on_batch(X, y)
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            d.trainable = False
            g_loss = gan.train_on_batch(noise, y2)

        losses["D"].append(d_loss)
        losses["G"].append(g_loss)

        if e == 1 or e%plt_frq == 0:
            plot_generated()
    plot_loss(losses)

train(epochs=100, plt_frq = 10, BATCH_SIZE = 256)