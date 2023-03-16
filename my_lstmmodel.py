import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()  
        self.layer1 = nn.LSTM(input_size = 13, hidden_size = 256, num_layers = 3, bidirectional = True, batch_first = True)
        self.layer2 = nn.Linear(in_features = 512, out_features = 12)

    def forward(self, x, hn = None, cn = None):
      if hn is None or cn is None:
        hn = torch.zeros(6, x.size(0), 256)
        cn = torch.zeros(6, x.size(0), 256)
        x,_ = self.layer1(x,(hn,cn))
        x = self.layer2(x)
      return x