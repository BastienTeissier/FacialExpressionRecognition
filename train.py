from models.vgg import basic, vgg13, vgg16
from models.vgg_face import VGGFace
from preprocess import fer2013, fer2013_light, ck
from result import confusion_matrix, historic

from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
nb_classes = 7
nb_epoch = 50
save_weights = True
load_weights = False

# input image dimensions
img_rows, img_cols = 48, 48
# FER2013 is grayscale
img_channels = 1

def train_without_augmentation():
    model = vgg16(lr=0.0001, dropout_in=0.25, dropout_out=0.5)
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()
    if load_weights:
        model.load_weights('model_vgg_16_eq.h5')
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_validation, Y_validation),
              shuffle=True)
    if save_weights:
        model.save_weights('vgg16_fer2013_np.h5')
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    evaluates = model.evaluate(X_test, Y_test)
    historic(history)
    confusion_matrix(predictions, Y_test)

def train_without_augmentation_basic():
    model = basic()
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()
    if load_weights:
        model.load_weights('model_vgg_16_eq.h5')
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_validation, Y_validation),
              shuffle=True)
    if save_weights:
        model.save_weights('basic_fer2013.h5')
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    evaluates = model.evaluate(X_test, Y_test)
    print(history)
    historic(history)
    confusion_matrix(predictions, Y_test)


def train_without_augmentation_ck():
    model = vgg16()
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = ck()
    if load_weights:
        model.load_weights('model_vgg_16_ck.h5')
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
        model.save_weights('model_vgg_16_ck.h5')

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

def train_with_augmentation_ck():
    datagen = ImageDataGenerator(
            rotation_range=10.,
            horizontal_flip=True)

    model = vgg16(lr=0.00005)
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = ck()

    if load_weights:
        model.load_weights('model_vgg_16_eq.h5')

    history = model.fit_generator(datagen.flow(X_train, Y_train,
                batch_size=batch_size), samples_per_epoch=2000, nb_epoch=nb_epoch,
                validation_data=(X_validation, Y_validation))
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)

    historic(history)

    confusion_matrix(predictions, Y_test)

    if save_weights:
        model.save_weights('model_vgg_16_ft_ck.h5')

def test(model=None):
    if model==None:
        model = basic()
        model.load_weights('basic_fer2013.h5')
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = ck()
    scores = model.evaluate(X_train, Y_train)
    print(scores)
    predictions = model.predict(X_train, batch_size=batch_size, verbose=1)
    confusion_matrix(predictions, Y_train)
    scores = model.evaluate(X_validation, Y_validation)
    print(scores)
    predictions = model.predict(X_validation, batch_size=batch_size, verbose=1)
    confusion_matrix(predictions, Y_validation)
    scores = model.evaluate(X_test, Y_test)
    print(scores)
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    confusion_matrix(predictions, Y_test)

    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()
    scores = model.evaluate(X_train, Y_train)
    print(scores)
    predictions = model.predict(X_train, batch_size=batch_size, verbose=1)
    confusion_matrix(predictions, Y_train)
    scores = model.evaluate(X_validation, Y_validation)
    print(scores)
    predictions = model.predict(X_validation, batch_size=batch_size, verbose=1)
    confusion_matrix(predictions, Y_validation)
    scores = model.evaluate(X_test, Y_test)
    print(scores)
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    confusion_matrix(predictions, Y_test)

if __name__ == '__main__':
    train_without_augmentation()
    #test()
    '''
    model = VGGFace(trainable=False)
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = ck(ch=3)
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_validation, Y_validation),
              shuffle=True)
    scores = model.evaluate(X_test, Y_test)
    print(scores)
    predictions = model.predict(X_test, batch_size=128, verbose=1)
    confusion_matrix(predictions, Y_test)
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = ck(ch=3)
    scores = model.evaluate(X_train, Y_train)
    print(scores)
    predictions = model.predict(X_train, batch_size=128, verbose=1)
    confusion_matrix(predictions, Y_train)
    '''
