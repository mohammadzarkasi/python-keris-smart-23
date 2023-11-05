import cv2
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

folders = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash',
]

list_path = []

for f in folders:
    path1 = 'dataset/' + f
    print(path1)
    counter = 0
    while True:
        counter += 1
        path2 = path1 + '/' + f + str(counter) + '.jpg'
        # print(path2)
        if os.path.exists(path2) == False:
            # print(path2, 'does not exist')
            break
        list_path.append(path2)

print('jumlah file:', len(list_path))
scaler = preprocessing.MinMaxScaler()

def read_img(path1):
    img1 = cv2.imread(path1,0)
    img2 = img1.flatten()
    img2 = img2 / 255

    print(img1)
    print(type(img1))
    print(img2)
    print(type(img2))
    print(img2.shape)

    # plt.imshow(img, cmap='gray')
    cv2.imshow('image',img1)
    cv2.waitKey(0)


read_img(list_path[0])

