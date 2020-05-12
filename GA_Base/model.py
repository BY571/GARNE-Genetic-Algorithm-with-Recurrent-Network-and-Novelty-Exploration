import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import gym

def weights_init(layers):
    """
    Xavier initialization
    """
    for layer in layers:
        try:
            # for regular
            nn.init.uniform_(layer.weight, a=-1, b=1) #2,25
        except:
            # for LSTM
            nn.init.uniform_(layer.weight_ih_l0, a=-1, b=1)
            nn.init.uniform_(layer.weight_hh_l0, a=-1, b=1)

class Model_FF(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_type):
        super(Model_FF, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        if action_type == gym.spaces.box.Box:
    	    self.action_type = "conti" 
        else:
            self.action_type = "discr"
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.all_layer = [self.fc1, self.fc2, self.fc3]
        #weights_init(self.all_layer)
    

    def forward(self, x, hidden_threshold):
        if self.action_type == "conti":
            x = x.view((1, self.state_size))
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            dist = Normal(x, scale=torch.FloatTensor([0.1]))
            action = dist.sample()
            action = torch.clamp(action, min=-1, max=1)
            return action.detach().numpy(), hidden_threshold
            
        else:
            x = x.view((1, self.state_size))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x), dim=-1)
            dist = Categorical(x)
            action = dist.sample()
            return action.detach().numpy(), hidden_threshold


class Model_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_type):
        super(Model_LSTM, self).__init__()
        self.action_type = action_type
        self.hidden_size = hidden_size
        self.state_size = state_size
        if action_type == gym.spaces.box.Box:
    	    self.action_type = "conti" 
        else:
            self.action_type = "discr"
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.all_layer = [self.fc1, self.lstm, self.fc2]
        weights_init(self.all_layer)


    def forward(self, x, hidden):
        if self.action_type == "conti":
            x = x.view((1,self.state_size))        
            x = torch.tanh(self.fc1(x))
            x = x.view(-1, 1, self.hidden_size)
            x, hidden_out = self.lstm(x, hidden)
            x = torch.tanh(self.fc2(x)).squeeze(0)
            dist = Normal(x, scale=torch.FloatTensor([0.1]))
            action = dist.sample()
            action = torch.clamp(action, min=-1, max=1)
            return action.detach().numpy(), hidden_out

        else:
            x = x.view((1,self.state_size))        
            x = torch.relu(self.fc1(x))
            x = x.view(-1, 1, self.hidden_size)
            x, hidden_out = self.lstm(x, hidden)
            x = torch.relu(x)
            x = F.softmax(self.fc2(x).squeeze(0),dim=-1)
            dist = Categorical(x)
            action = dist.sample()
            return action.detach().numpy(), hidden_out
    
    
class Model_CNN1D(nn.Module):
    def __init__(self, window, action_space, hidden_size):
        super(Model_CNN1D, self).__init__()
        self.hidden_size = hidden_size # threshold because of the option lstm 
        self.window_size = window
        assert self.window_size == 35, "Assertion error! observation_size/window length has to be 35!"
        self.action_space = action_space
        self.cv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.cv2 = nn.Conv1d(1, 1, kernel_size=2)

        self.fc1 = nn.Linear(34, self.action_space)
        self.activation = nn.LeakyReLU()
    def forward(self, x, hidden_threshold):
        x = x.reshape((1,1,35))
        x = self.activation(self.cv1(x))
        x = self.activation(self.cv2(x))
        flattened = x.flatten()
        return self.fc1(flattened), hidden_threshold