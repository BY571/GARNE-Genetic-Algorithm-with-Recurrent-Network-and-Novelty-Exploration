import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(layers):
    """
    Xavier initialization
    """
    for layer in layers:
        try:
            # for regular
            nn.init.uniform_(layer.weight, a=1, b=10)
        except:
            # for LSTM
            nn.init.uniform_(layer.weight_ih_l0, a=1, b=10)
            nn.init.uniform_(layer.weight_hh_l0, a=1, b=10)

class Model_FF(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_type):
        super(Model_FF, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.all_layer = [self.fc1, self.fc2, self.fc3]
        weights_init(self.all_layer)
    

    def forward(self, x, hidden_treshold):
        #x = torch.abs(x)
        x = x.view((1, self.state_size))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x)), hidden_treshold


class Model_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_type):
        super(Model_LSTM, self).__init__()
        self.action_type = action_type
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.all_layer = [self.fc1, self.lstm, self.fc2]
        weights_init(self.all_layer)


    def forward(self, x, hidden):
        x = x.view((1,self.state_size))        
        x = torch.relu(self.fc1(x))
        x = x.view(-1, 1, self.hidden_size)
        x, hidden_out = self.lstm(x, hidden)
        x = torch.relu(x)
        #if self.action_type == gym.spaces.box.Box
        action = self.fc2(x).squeeze(0)
        return action, hidden_out