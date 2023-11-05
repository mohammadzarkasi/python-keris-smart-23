import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import cv2
import numpy as np

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


mymodel = MyModel(in_features=196608, out_features=6)
mymodel.load_state_dict(torch.load('my_model.pt'))
print(mymodel.eval())

def read_img3(path1):
    img1 = cv2.imread(path1,0)
    img2 = img1.flatten()
    img2 = img2 / 255

    return img2

criterion = nn.CrossEntropyLoss()

def testMyData(data1, kelas1):
    data1 = torch.FloatTensor(np.array([data1]) )
    kelas1 = torch.LongTensor(np.array([kelas1]) )

    y_pred1 = mymodel.forward(data1)
    loss = criterion(y_pred1, kelas1)
    
    return y_pred1, loss

s = random.choice(list_path)
print(s)
data1 = read_img3(s[0])
kelas1 = s[1]

with torch.no_grad():
    pred, loss = testMyData(data1, kelas1)
    print(pred)
    print(loss)
    pred2 = pred.tolist()
    # print('pred2',pred2)
    
    mv = -999999
    mi=-1
    for i,v in enumerate(pred2[0]):
        # print(i,v)
        if v > mv:
            mi = i
            mv = v
    print('max i:', mi, 'max v:', mv)