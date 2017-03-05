from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop

from keras import backend as K
K.set_image_dim_ordering('th')

# Look if constants can become arguments (object attributes or function arguments)
nb_classes = 7
img_rows, img_cols = 48, 48
img_channels = 1

def basic():
    '''
    Basic model inspired from a Cifar10 tutorial by fchollet :
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
    Transformed to take grayscale picture as input
    '''
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms,
                  metrics=['accuracy'])
    return model
