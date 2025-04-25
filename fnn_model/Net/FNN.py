import torch.nn as nn
import torch.nn.functional as F

class FNN_1(nn.Module):
    def __init__(
        self,
        n_input,
        n_hidden1,
        n_output
        ):
        super(FNN_1, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.predict = nn.Linear(n_hidden1, n_output)

    def forward(self, data):
        out = self.hidden1(data.feature)
        out = F.relu(out)
        out = self.predict(out)
        return out

class FNN_2(nn.Module):
    def __init__(
        self,
        n_input,
        n_hidden1,
        n_hidden2,
        n_output
        ):
        super(FNN_2, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.predict = nn.Linear(n_hidden2, n_output)

    def forward(self, data):
        out = self.hidden1(data.feature)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.predict(out)
        return out
