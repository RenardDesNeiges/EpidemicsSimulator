
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

# Named tuple for replay buffer storage
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent():
        
    def __init__(self,  env, 
                        model, 
                        criterion = nn.HuberLoss(),
                        lr = 5e-4, 
                        epsilon = 0.5, 
                        gamma = 0.99,
                        buffer_size = 10000, 
                        batch_size = 64):

        self.env = env
        
        model_params = {
            'in_dim':len(env.observation_space.sample().flatten()),
            'out_dim':env.action_space.n,
        }
        self.model = model(**model_params)
        self.targetModel = model(**model_params)
        
        self.criterion = criterion
                
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.memory = ReplayMemory(buffer_size)
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
    
    def load_model(self, savepath):
        torch.save(self.model.state_dict(),savepath)
        self.model.load_state_dict(torch.load(savepath))
    
    def save_model(self, savepath):
        torch.save(self.model.state_dict(),savepath)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return np.double(0)
        
        # Sample memory
        transitions = self.memory.sample(self.batch_size)

        # Convert Batch(Transitions) -> Transition(Batch)
        batch = Transition(*zip(*transitions))
        
        action_batch = torch.tensor([e for e in batch.action])
        state_batch = torch.cat(batch.state,0)
        next_states_batch = torch.cat(batch.next_state,0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(S, a) with the Q-value network
        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute max_ap Q(Sp) with the stable target network
        next_state_values = self.targetModel(next_states_batch).max(1)[0].detach().unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return np.double(loss)
    
    def act(self, obs):
        x = torch.Tensor(obs)
        
        epsilon = self.epsilon
        sample = random.random()
        
        action_distribution = self.model(x)
        Q = float(action_distribution.detach().max())
        if sample > epsilon:
            with torch.no_grad():
                return np.argmax(
                    np.exp(action_distribution.numpy())
                    /np.sum(np.exp(action_distribution.numpy())
                    )), Q
        else:
            return self.env.action_space.sample(), Q