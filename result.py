import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

nb_classes = 7

def confusion_matrix(predictions, labels):
    '''
    Displau the confusion matrix matrix when given the NN predictions and the labels
    '''
    mat = np.zeros((nb_classes, nb_classes))
    count = np.zeros(nb_classes)
    for i in range(len(labels)):
        mat[np.argmax(labels[i])]+=predictions[i]
        count[np.argmax(labels[i])]+=1
    for i in range(nb_classes):
        mat[i]/=count[i]
    sns.heatmap(mat, annot=True)
    plt.show()

def historic(history):
    '''
    Plot historic values of the keras fit process
    '''
    keys = history.history.keys()
    for key in keys:
        plt.plot(history.history[key])
        plt.title(key)
        plt.show()

emotions = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

if __name__=='__main__':
    for k,v in emotions.items():
        print('{0}->{1}'.format(k,v))
