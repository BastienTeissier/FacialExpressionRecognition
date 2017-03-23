import cv2

import numpy as np

def process_image(img):
    x,y = img.shape
    if x>y:
        r = 48.0/x
        dim = (48, int(r * y))
        res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    else:
        r = 48.0/y
        dim = (int(r * x), 48)
        res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    if len(res)<48:
        print('larger')
        k,n = 0, len(res)
        temp = []
        while k<48:
            if k<n:
                temp.append(res[k])
            else:
                temp.append(np.zeros((48,)))
            k+=1
        res = np.array(temp)
    else:
        temp_res = np.zeros((48,48))
        for i in range(48):
            k,n = 0,len(res[i])
            temp = []
            while k<48:
                if k<n:
                    temp.append(res[i][k])
                else:
                    temp.append(0)
                k+=1
            temp_res[i] = np.array(temp)
        res = temp_res
    return res
