#########################
# Purpose: Model definitions and other utlities for MNIST, Fashion MNIST, 
########################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Layer, Add, Convolution2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.utils import np_utils
import tensorflow.keras.utils as np_utils
import global_vars as gv
import utils.ResNet as ResNet
from utils.fmnist import load_fmnist
import numpy as np
# np.random.seed(777)
from utils.dp import DPSequential


def data_mnist(one_hot=True):
    """
    Preprocess MNIST dataset
    """
    # the data, shuffled and split between train and test sets
    if gv.args.dataset == 'MNIST':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif gv.args.dataset == 'fMNIST':
        X_train, y_train = load_fmnist('/slstore/hanxiguo/data/', kind='train')
        X_test, y_test = load_fmnist('/slstore/hanxiguo/data/', kind='t10k')


    X_train = X_train.reshape(X_train.shape[0],
                              gv.IMAGE_ROWS,
                              gv.IMAGE_COLS,
                              gv.NUM_CHANNELS)

    X_test = X_test.reshape(X_test.shape[0],
                            gv.IMAGE_ROWS,
                            gv.IMAGE_COLS,
                            gv.NUM_CHANNELS)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    if gv.args.gar == 'siren' or gv.args.gar == 'fltrust':
        target_indices = np.random.choice(len(X_test), gv.args.root_size)
        Server_X = X_train[target_indices]
        Server_Y = y_train[target_indices]
        print("server dataset initialized..")
        print('server dataset shape:', Server_X.shape)
        X_train = np.delete(X_train, target_indices, axis=0)
        y_train = np.delete(y_train, target_indices, axis=0)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, gv.NUM_CLASSES).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, gv.NUM_CLASSES).astype(np.float32)
    if gv.args.gar == 'siren' or gv.args.gar == 'fltrust':
        return X_train, y_train, X_test, y_test, Server_X, Server_Y
    else:
        return X_train, y_train, X_test, y_test



def model_cifar():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))
    return model

def modelA():
    model = Sequential()
    # if gv.args.dp:
    #     # print("Use differential privacy model.")
    #     l2_norm_clip = 1.3
    #     noise_multiplier = 1.5
    #     model = DPSequential(l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier,
    #     layers=[
    #         Conv2D(64, (5, 5), padding='valid', input_shape=(gv.IMAGE_ROWS,
    #                                      gv.IMAGE_COLS,
    #                                      gv.NUM_CHANNELS)),
    #         Activation('relu'),
    #         Conv2D(64, (5, 5)),
    #         Activation('relu'),
    #         Dropout(0.25),
    #         Flatten(),
    #         Dense(128),
    #         Activation('relu'),
    #         Dropout(0.5),
    #         Dense(gv.NUM_CLASSES)
    #     ])
    #     return model
    model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(gv.IMAGE_ROWS,
                                         gv.IMAGE_COLS,
                                         gv.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(gv.NUM_CLASSES))
    return model


def modelB():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(gv.IMAGE_ROWS,
                                        gv.IMAGE_COLS,
                                        gv.NUM_CHANNELS)))
    model.add(Convolution2D(64, 8, 8,
                            subsample=(2, 2),
                            border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 6, 6,
                            subsample=(2, 2),
                            border_mode='valid'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 5, 5,
                            subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(gv.NUM_CLASSES))
    return model


def modelDP():
    model = Sequential()
    model.add(Convolution2D(16, 8, strides=2,
                            padding='same',
                            input_shape=(gv.IMAGE_ROWS,
                                         gv.IMAGE_COLS,
                                         gv.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, 1))

    model.add(Conv2D(32, 4, strides=2, padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPool2D(2, 1))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(gv.NUM_CLASSES))
    return model


def modelD():
    model = Sequential()

    model.add(Flatten(input_shape=(gv.IMAGE_ROWS,
                                   gv.IMAGE_COLS,
                                   gv.NUM_CHANNELS)))

    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(gv.NUM_CLASSES))
    return model

def modelE():
    model = Sequential()

    model.add(Flatten(input_shape=(gv.IMAGE_ROWS,
                                   gv.IMAGE_COLS,
                                   gv.NUM_CHANNELS)))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(gv.NUM_CLASSES))

    return model

def modelF():
    model = Sequential()

    model.add(Conv2D(32, (5, 5),
                            padding='valid',
                            input_shape=(gv.IMAGE_ROWS,
                                         gv.IMAGE_COLS,
                                         gv.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(gv.NUM_CLASSES))

    return model

def modelG():
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))

    return model

def model_LR():
    model = Sequential()

    model.add(Flatten(input_shape=(gv.IMAGE_ROWS,
                                   gv.IMAGE_COLS,
                                   gv.NUM_CHANNELS)))

    model.add(Dense(gv.NUM_CLASSES))

    return model

def model_resnet18():
    # model = ResNet.resnet18(num_classes=gv.NUM_CLASSES)
    # model.build(input_shape=(None, gv.IMAGE_COLS, gv.IMAGE_ROWS, gv.NUM_CHANNELS))
    model = ResNet.ResnetBuilder()
    model = model.build_resnet18()
    return model


def model_mnist(type):
    """
    Defines MNIST model using Keras sequential model
    """

    models = [model_cifar, modelA, modelB, modelDP, modelD, modelE, modelF, modelG, model_LR, model_resnet18]

    return models[type]()


def data_gen_mnist(X_train):
    datagen = ImageDataGenerator()

    datagen.fit(X_train)
    return datagen


def load_model(model_path, type=0):

    try:
        with open(model_path+'.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
            print('Loaded using json')
    except IOError:
        model = model_mnist(type=type)

    model.load_weights(model_path)
    return model
