from preprocess import fer2013
from models.vgg import basic
from train import train_without_augmentation

if __name__ == '__main__':
    '''Load & preprocess data data'''
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()
    '''Define model'''
    model = basic()
    '''Train model'''
    train_without_augmentation()
