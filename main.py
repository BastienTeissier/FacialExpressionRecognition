from models.vgg import basic, vgg13, vgg16
from preprocess import fer2013, fer2013_light, ck
from result import confusion_matrix, historic

from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
nb_classes = 7
nb_epoch = 100
save_weights = True
load_weights = False

# input image dimensions
img_rows, img_cols = 48, 48
# FER2013 is grayscale
img_channels = 1

def train_without_augmentation():
    model = vgg13()
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()
    if load_weights:
        model.load_weights('model_vgg_13_59.h5')
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_validation, Y_validation),
              shuffle=True)
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    evaluates = model.evaluate(X_test, Y_test)
    historic(history)
    confusion_matrix(predictions, Y_test)
    if save_weights:
        model.save_weights('model_vgg_13_eq.h5')

def train_without_augmentation_ck():
    model = vgg13()
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = ck()
    if load_weights:
        model.load_weights('model_vgg_13_eq.h5')
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_validation, Y_validation),
              shuffle=True)
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    evaluates = model.evaluate(X_test, Y_test)
    historic(history)
    confusion_matrix(predictions, Y_test)
    if save_weights:
        model.save_weights('model_vgg_13_ck.h5')

def train_with_augmentation():
    datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30.,
            horizontal_flip=True)

    model = vgg13()
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()

    if load_weights:
        model.load_weights('model_vgg_13_aug.h5')

    history = model.fit_generator(datagen.flow(X_train, Y_train,
                batch_size=batch_size), samples_per_epoch=50000, nb_epoch=nb_epoch,
                validation_data=(X_validation, Y_validation))
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)

    historic(history)

    confusion_matrix(predictions, Y_test)

    if save_weights:
        model.save_weights('model_vgg_13_aug.h5')

if __name__ == '__main__':
    train_without_augmentation_ck()
