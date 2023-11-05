import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel (nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
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


torch.manual_seed(41)
mymodel = MyModel()

# model2 = MyModel()
# model2.load_state_dict(torch.load('augmented_compostnet.pt'))
# for param in model2.parameters():
#     print(param)