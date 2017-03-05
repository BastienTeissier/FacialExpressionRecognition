import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

nb_classes = 7

def confusion_matrix(predictions, labels):
    mat = np.zeros((nb_classes, nb_classes))
    count = np.zeros(nb_classes)
    for i in range(len(labels)):
        mat[np.argmax(labels[i])]+=predictions[i]
        count[np.argmax(labels[i])]+=1
    for i in range(nb_classes):
        mat[i]/=count[i]
    sns.heatmap(mat, annot=True)
    plt.show()
