import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
  
        self.layer1 = nn.Conv1d(in_channels = 13, out_channels = 32, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.layer3 = nn.Conv1d(in_channels = 64, out_channels = 256, kernel_size = 3, padding = 1)
        self.layer4 = nn.Conv1d(in_channels = 256, out_channels = 12, kernel_size = 3, padding = 1)
        self.layer5 = nn.Linear(in_features=256, out_features=14)
        self.relu = torch.nn.ReLU()
        
             


    def forward(self, x):
      x = x.permute(0,2,1)
      x = self.layer1(x)
      x = self.relu(x)
      x = self.layer2(x)
      x = self.relu(x)
      x = self.layer3(x)
      x = self.relu(x)
      emma = self.layer4(x)
      x = torch.mean(x, dim=2)
      x = self.layer5(x)
      emma = emma.permute(0, 2, 1)
      return emma, x