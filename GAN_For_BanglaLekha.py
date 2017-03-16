''' GAN For BanglaLekha-Isolated Dataset, Takes 10000 random image from the dataset and then generates sample according to them '''


from __future__ import print_function
import os, random
import numpy as np
import gzip
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input, merge
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
# from IPython import display
from keras.utils import np_utils
from tqdm import tqdm
import cv2
'''--------------------------------------------------------------------------------------------------------'''
''' Data set loading '''
def dataset_load(path):
    if path.endswith(".gz"):
        f=gzip.open(path,'rb')
    else:
        f=open(path,'rb')

    if sys.version_info<(3,):
        data=cPickle.load(f)
    else:
        data=cPickle.load(f,encoding="bytes")
    f.close()
    return data

data, dataLabel, dataMarking, imageFullName = dataset_load('./FullData.pkl.gz')
Max=0
print(imageFullName[0])
for i in range(len(dataLabel)):
    Max=max(Max,dataLabel[i])


''' This Portion is for Labeling and Dividing the dataset. Each sample Contains 1800 Images. Total 84 Samples '''
X_train = []
X_test = []
y_train=[]
y_test=[]

from collections import defaultdict
Dict=defaultdict(lambda:None)


for i in range(len(dataLabel)):
    if(Dict[dataLabel[i]] is None):
        Dict[dataLabel[i]]=1
    else:
        Dict[dataLabel[i]]=Dict[dataLabel[i]]+1

    if(Dict[dataLabel[i]]>1800):
        Value = data[i]
        NV = cv2.resize(Value, (28, 28))
        X_test.append(NV)
        y_test.append(dataLabel[i])
    else:
        Value=data[i]
        NV = cv2.resize(Value, (28, 28))
        X_train.append(NV)
        y_train.append(dataLabel[i])

batch_size = 128
nb_classes = 84
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (5, 5)

X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(X_train[0].shape)

'''--------------------------------------------------------------------------------------------------------'''

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

#
shp = X_train.shape[1:]
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)
'''---------------------------------------------------------------------------------------------------------'''
''' Generative model '''
nch = 200
g_input = Input(shape=[100])
H = Dense(nch * 14 * 14, init='glorot_normal')(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Reshape([nch, 14, 14])(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(nch / 2, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(nch / 4, 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()
'''---------------------------------------------------------------------------------------------------------'''
'''Discriminative model'''
# Build Discriminative model ...
d_input = Input(shape=shp)
H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()
'''---------------------------------------------------------------------------------------------------------'''


make_trainable(discriminator, False)

'''GAN model'''
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()


def plot_loss(losses):
    #        display.clear_output(wait=True)
    #        display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()


def plot_gen(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

'''---------------------------------------------------------------------------------------------------------'''

ntrain = 1000

trainidx = random.sample(range(0, X_train.shape[0]), ntrain) # Add Random Numbers
XT = X_train[trainidx, :, :, :] # Add Original Training samples, trainidx indicates the
                                # Number of samples that needs to be taken from original


'''Pre-train the discriminator network '''
noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100]) # Generate Random Values between 0 to 1,
                                                             # here size means the dimension of the array
generated_images = generator.predict(noise_gen) # Makes the generator predict about the noise
                                                # that is generated
'''---------------------------------------------------------------------------------------------------------'''
''' New Image and label creation'''
X = np.concatenate((XT, generated_images)) # Concatanate the original image and generated images
n = XT.shape[0]

y = np.zeros([2 * n, 2]) # For train labels
y[:n, 1] = 1 # Original Dataset / True label
y[n:, 0] = 1 # Fake Dataset / False label
#
make_trainable(discriminator, True)
discriminator.fit(X, y, nb_epoch=1, batch_size=128)
y_hat = discriminator.predict(X)

'''Accuracy of pre-trained discriminator'''
y_hat_idx = np.argmax(y_hat, axis=1) # Argmax for finding the index of the maximum value
y_idx = np.argmax(y, axis=1)
diff = y_idx - y_hat_idx
n_tot = y.shape[0]
n_rig = (diff == 0).sum()
acc = n_rig * 100.0 / n_tot
print ("Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot))

# set up loss storage vector
losses = {"d": [], "g": []} # Dictionary d for discriminator, g for generative


''' Main Training Loop'''
def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    print("Epoch Started - ",nb_epoch)
    for e in tqdm(range(nb_epoch)): #TQDM for a good verbose meter
        
        # Make generative images
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :] # Select Randorm 32 Images from X_train
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100]) # Generate random noise
        generated_images = generator.predict(noise_gen)


        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images)) # Concatanate original and generated images
        y = np.zeros([2 * BATCH_SIZE, 2]) # Labeling
        y[0:BATCH_SIZE, 1] = 1 # Labeling True
        y[BATCH_SIZE:, 0] = 1 #Labeling False

        # make_trainable(discriminator,True)
        d_loss = discriminator.train_on_batch(X, y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1 #Fake Label

        # make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)

        # Updates plots
        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()


# Train for 6000 epochs at original learning rates
train_for_n(nb_epoch=6000, plt_frq=500, BATCH_SIZE=32)


# Plot the final loss curves
plot_loss(losses)

# Plot some generated images from our GAN
plot_gen(25, (5, 5), (12, 12))


def plot_real(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    idx = np.random.randint(0, X_train.shape[0], n_ex)
    generated_images = X_train[idx, :, :, :]

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Plot real MNIST images for comparison
plot_real()