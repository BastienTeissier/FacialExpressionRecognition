import pandas as pd
import numpy as np

def load_sets(path):
    '''
    Load the fer2013 datasets from the csv file with the form (emotion, pixels, Usage)
    Datas are save in 3 separate files according to their use :
    - train_set.npy
    - validation_set.npy
    - test_set.npy
    Each file containing a list of tuples (data, label)
    '''
    df = pd.read_csv(path)
    train_set = []
    test_set = []
    validation_set = []
    for k in range(len(df)):
        # CSVs contain only string, we split it in an array
        if k%1000 == 0:
            print(k)
        pixels = df.loc[k]['pixels'].split(' ')

        img = np.zeros((48,48))
        # Reconstruct the image/matrix from the pixels array
        for i in range(len(pixels)):
            img[i//48][i%48] = int(pixels[i])
        if df.loc[k]['Usage'] == 'PrivateTest':
            test_set.append((np.array(img), df.loc[k]['emotion']))
        elif df.loc[k]['Usage'] == 'PublicTest':
            validation_set.append((np.array(img), df.loc[k]['emotion']))
        else:
            train_set.append((np.array(img), df.loc[k]['emotion']))
    np.save('train_set.npy', train_set)
    np.save('test_set.npy', test_set)
    np.save('validation_set.npy', validation_set)

if __name__ == '__main__':
    path = 'fer2013.csv'
    load_sets(path)
