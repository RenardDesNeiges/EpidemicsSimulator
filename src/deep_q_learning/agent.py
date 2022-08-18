
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
                        loss = nn.HuberLoss(),
                        lr = 5e-4, 
                        epsilon = 0.3, 
                        gamma = 0.99,
                        buffer_size = 10000, 
                        batch_size = 64):

        self.env = env
        
        self.model = model(in_dim=len(env.observation_space.sample().flatten()),
                           out_dim=env.action_space.sample().shape[0])
        self.targetModel = model()
        
        self.loss = loss
                
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.n_actions = 9
        
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

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        batch_next_states = [s for s in batch.next_state if s is not None]
        if len(batch_next_states)==0:
            return np.double(0)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state,0)
        action_batch = torch.cat(batch.action,0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(S, a) with the Q-value network
        state_action_values = self.model(state_batch).gather(1, action_batch.to(torch.int64))

        # Compute max_ap Q(Sp) with the stable target network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.targetModel(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return np.double(loss)
    
    def act(self, obs):
        x = torch.Tensor(obs)
        
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                action_distribution = self.model(x)
                return torch.round(action_distribution.sigmoid())
        else:
            return torch.Tensor(self.env.action_space.sample()).unsqueeze(0)