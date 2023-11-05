import cv2
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MyModel (nn.Module):
    def __init__(self, in_features=30, h1=24, h2=12, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
  
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    def tes(self):
        print('ini tes')

folders = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash',
]

# for i,v in enumerate(folders):
#     print(i,v)

list_path = []

for kelas,f in enumerate(folders):
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
        list_path.append([path2, kelas])

print('jumlah file:', len(list_path))

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
    # cv2.imshow('image',img1)
    # cv2.waitKey(0)


my_data = []
def read_img2(path1, kelas):
    img1 = cv2.imread(path1,0)
    img2 = img1.flatten()
    img2 = img2 / 255

    img2 = np.append(img2, [kelas], axis=0)

    return img2

def read_img3(path1, kelas):
    img1 = cv2.imread(path1,0)
    img2 = img1.flatten()
    img2 = img2 / 255

    return img2, kelas

# read_img(list_path[0][0])

torch.manual_seed(41)
mymodel = MyModel(in_features=196608, out_features=6)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.01)

# tes1, kelas1 = read_img3(list_path[0][0], list_path[0][1])

# print(tes1)
def trainMyData(data1, kelas1):
    data1 = torch.FloatTensor(np.array([data1]) )
    kelas1 = torch.LongTensor(np.array([kelas1]) )
    # print(tes1)
    # print(kelas1)

    y_pred1 = mymodel.forward(data1)
    # print('predict:',y_pred1, kelas1)
    loss = criterion(y_pred1, kelas1)
    # print('loss:',loss, kelas1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

counter = 0
for i in range(3):
    list_path2 = random.sample(list_path, len(list_path)*8//10)
    print('path2: ', len(list_path2))
    for p in list_path2:
        counter += 1
        # if counter > 10:
        #     break
        data, kelas = read_img3(p[0],p[1])
        loss = trainMyData(data, kelas)

        if counter % 100 == 0:
            print(loss, kelas, p[0])


torch.save(mymodel.state_dict(), 'my_model.pt')


# print(my_data)

# r1 = random.sample(list_path)
# print(r1, len(list_path))
# r1 = random.choice(list_path)
# print(r1, len(list_path))
# r1 = random.choice(list_path)
# print(r1, len(list_path))
# r1 = random.choice(list_path)
# print(r1, len(list_path))