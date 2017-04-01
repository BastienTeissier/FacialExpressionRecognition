from preprocess import *
from models.vgg import basic
from train import

if __name__ == '__main__':
    '''Load data'''
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()
    '''Preprocess data'''
    gaussian_train = gaussianFilter(X_train)
    X_train_preprocessed = median_substraction(gaussian_train[0][0])
    gaussian_validation = gaussianFilter(X_validation)
    X_validation_preprocessed = median_substraction(gaussian_validation[0][0])
    gaussian_test = gaussianFilter(X_test)
    X_test_preprocessed = median_substraction(gaussian_test[0][0])
    '''Define model'''
    model = basic()
    '''Train model'''
