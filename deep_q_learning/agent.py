"""Implementation of the agent classes and associated RL algorithms.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from epidemic_env.env import Env
# Named tuple for replay buffer storage
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent(ABC):
    """Implements acting and learning. (Abstract class, for implementations see DQNAgent and NaiveAgent).

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    @abstractmethod
    def __init__(self,  env, *args, **kwargs):
        """
        Args:
            env (_type_): the simulation environment
        """
        
    @abstractmethod
    def load_model(self, savepath:str):
        """Loads weights from a file.

        Args:
            savepath (str): path at which weights are saved.
        """
        
    @abstractmethod
    def save_model(self, savepath:str):
        """Saves weights to a specified path

        Args:
            savepath (str): the path
        """
        
    @abstractmethod
    def optimize_model(self)->float:
        """Perform one optimization step.

        Returns:
            float: the loss
        """
    
    @abstractmethod
    def reset():
        """Resets the agent's inner state
        """
        
    @abstractmethod 
    def act(self, obs:torch.Tensor)->Tuple[int, float]:
        """Selects an action based on an observation.

        Args:
            obs (torch.Tensor): an observation

        Returns:
            Tuple[int, float]: the selected action (as an int) and associated Q/V-value as a float
        """

class DQNAgent(Agent):
    """Implements acting and learning using deep Q-learning (see [the DQN paper](https://arxiv.org/pdf/1312.5602.pdf)).  
    
    Q-learning aims to optmizes an agent's policy by maximizing the Bellman equation:
    $$
    Q^*(s,a) = \mathbb{E}_{s' \sim \mathcal{E}} [r+\gamma \max_{a'}Q^*(s',a')|s,a]
    $$
    
    In the case of Deep Q-Learning this is performed by minizing a sequence of loss functions \(L_i(\\theta_i)\) which change at each iteration \(i\) of the algorithm:
    $$
    L_i(\\theta_i) = \mathbb{E}_{s', a \sim \\rho(\cdot)} [ g(y_i - Q(s,a;\\theta_i)) ]
    $$
    where \(g : \mathbb{R}^n\\rightarrow\mathbb{R}\) is some loss function, usually [L2](https://en.wikipedia.org/wiki/Mean_squared_error) or [Huber-loss](https://en.wikipedia.org/wiki/Huber_loss) and \(y_i= \mathbb{E}_{s' \sim \\rho(\cdot)} [r + \gamma \max_{a'} Q(s',a';\\theta_{i-1}) ]\) is the *target*. The policy is updated with stochastic gradient descent where stochastic gardients are sampled from the following full gradient computation:
    $$
    \\nabla L_i(\\theta_i) = \mathbb{E}_{s' \sim \epsilon} [ g'(r + \gamma \max_{a'} Q(s',a';\\theta_{i-1})-  Q(s,a;\\theta_i)) \cdot \\nabla_{\\theta_i} Q(s,a;\\theta_i)) ]
    $$
    
    
    """

    def __init__(self,  env:Env,
                 model:torch.nn,
                 criterion=nn.HuberLoss(),
                 lr:float=5e-4,
                 epsilon:float=0.5,
                 gamma:float=0.99,
                 buffer_size:int=10000,
                 batch_size:int=64)->None:
        """

        Args:
            env (_type_): the simulation environment.
            model (_type_): the torch module to use for learning
            criterion (nn._Loss, optional): the loss function. Defaults to nn.HuberLoss().
            lr (float, optional): DQN's learning rate. Defaults to 5e-4.
            epsilon (float, optional): the exploration ratio epsilon (see above). Defaults to 0.5.
            gamma (float, optional): the gamma term (see above). Defaults to 0.99.
            buffer_size (int, optional): the replay buffer size. Defaults to 10000.
            batch_size (int, optional): the size of each training batch. Defaults to 64.
        """

        self.env = env

        model_params = {
            'in_dim': len(env.observation_space.sample().flatten()),
            'out_dim': env.action_space.n,
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
        torch.save(self.model.state_dict(), savepath)
        self.model.load_state_dict(torch.load(savepath))

    def save_model(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return np.double(0)

        # Sample memory
        transitions = self.memory.sample(self.batch_size)

        # Convert Batch(Transitions) -> Transition(Batch)
        batch = Transition(*zip(*transitions))

        action_batch = torch.tensor([e for e in batch.action])
        state_batch = torch.cat(batch.state, 0)
        next_states_batch = torch.cat(batch.next_state, 0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(S, a) with the Q-value network
        state_action_values = self.model(
            state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute max_ap Q(Sp) with the stable target network
        next_state_values = self.targetModel(next_states_batch).max(1)[
            0].detach().unsqueeze(1)
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        # Compute the loss
        loss = self.criterion(state_action_values,
                              expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return np.double(loss)
    
    def reset():
        pass # Not stateful

    def act(self, obs:np.ndarray):
        x = torch.Tensor(obs)

        epsilon = self.epsilon
        sample = random.random()

        Q_est = self.model(x)
        Q = float(Q_est.detach().max())
        if sample > epsilon:
            with torch.no_grad():
                return np.argmax(Q_est), Q
        else:
            return self.env.action_space.sample(), Q


class NaiveAgent(Agent):

    def __init__(self,  env:Env,
                 threshold:int=20000,
                 confine_time:int=4,):
        """Naive Agent implementation. Gives a baseline to compare reinforcement learning agents against. 
        The naive policy is the following:
        ```pseudocode
        If number of infected people > THRESHOLD
            confine the entire country for CONFINEMENT_TIME weeks
        ```

        Args:
            env (_type_): the simulation environment.
            threshold (int, optional): The infected threshold, upon which confiment must start. Defaults to 20000.
            confine_time (int, optional): The confinement time. Defaults to 4.
        """

        self.env = env
        self.threshold = threshold
        self.confine_time = confine_time
        self.timer = 0
    def load_model(self, savepath):
        pass

    def save_model(self, savepath):
        pass

    def optimize_model(self):
        #This is agent is born stupid and stays stupid
        return 0
    
    def reset(self,):
        self.timer = 0

    def act(self, obs):
        if self.timer > 0:
            self.timer -=1
            return 1, 0
        if obs > self.threshold:
            self.timer = self.confine_time
            return 1, 0
        return 0,0
    
    
class FactoredDQNAgent(Agent):
    """Implements acting and learning using factored deep Q learning.  
    
    Q-learning aims to optmizes an agent's policy by maximizing the Bellman equation:
    $$
    Q^*(s,a) = \mathbb{E}_{s' \sim \mathcal{E}} [r+\gamma \max_{a'}Q^*(s',a')|s,a] 
    $$
    
    In the case of Deep Q-Learning this is performed by minizing a sequence of loss functions \(L_i(\\theta_i)\) which change at each iteration \(i\) of the algorithm:
    $$
    L_i(\\theta_i) = \mathbb{E}_{s', a \sim \\rho(\cdot)} [ g(y_i - Q(s,a;\\theta_i)) ]
    $$
    where \(g : \mathbb{R}^n\\rightarrow\mathbb{R}\) is some loss function, usually [L2](https://en.wikipedia.org/wiki/Mean_squared_error) or [Huber-loss](https://en.wikipedia.org/wiki/Huber_loss) and \(y_i= \mathbb{E}_{s' \sim \\rho(\cdot)} [r + \gamma \max_{a'} Q(s',a';\\theta_{i-1}) ]\) is the *target*. The policy is updated with stochastic gradient descent where stochastic gardients are sampled from the following full gradient computation:
    $$
    \\nabla L_i(\\theta_i) = \mathbb{E}_{s' \sim \epsilon} [ g'(r + \gamma \max_{a'} Q(s',a';\\theta_{i-1})-  Q(s,a;\\theta_i)) \cdot \\nabla_{\\theta_i} Q(s,a;\\theta_i)) ]
    $$
    
    The method aims to improve performance in a setting where each action \(a\) is made of \(m\) independant subactions \(\\tilde{a}_j,~j\in[m]\). Here the idea is to compute a \(Q\) function for every action combination by summing up \(Q\)-terms associated with each action by summing up said term:
    $$
    Q(s,[a_1,a_2,...,a_m]) = \sum_{j\in[m]} Q_j(s,a_j)
    $$
    
    The factored DQN expects a multi-binary action space
    """

    def __init__(self,  env:Env,
                 model:torch.nn.Module,
                 criterion=nn.HuberLoss(),
                 lr:float=5e-4,
                 epsilon:float=0.5,
                 gamma:float=0.99,
                 buffer_size:int=10000,
                 batch_size:int=64):
        """

        Args:
            env (_type_): the simulation environment.
            model (_type_): the torch module to use for learning
            criterion (nn._Loss, optional): the loss function. Defaults to nn.HuberLoss().
            lr (float, optional): DQN's learning rate. Defaults to 5e-4.
            epsilon (float, optional): the exploration ratio epsilon (see above). Defaults to 0.5.
            gamma (float, optional): the gamma term (see above). Defaults to 0.99.
            buffer_size (int, optional): the replay buffer size. Defaults to 10000.
            batch_size (int, optional): the size of each training batch. Defaults to 64.
        """

        self.env = env

        model_params = {
            'in_dim': len(env.observation_space.sample().flatten()),
            'out_dim': 2*env.dyn.ACTION_CARDINALITY, # we want twice as many neurons as there are binary actions
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
        torch.save(self.model.state_dict(), savepath)
        self.model.load_state_dict(torch.load(savepath))

    def save_model(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return np.double(0)

        # Sample memory
        transitions = self.memory.sample(self.batch_size)

        # Convert Batch(Transitions) -> Transition(Batch)
        batch = Transition(*zip(*transitions))

        action_batch = torch.cat([torch.LongTensor(e) for e in batch.action],axis=0)
        state_batch = torch.cat(batch.state, 0)
        next_states_batch = torch.cat(batch.next_state, 0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(S, a) with the Q-value network
        _,q_est,_ = self.model(state_batch)
        state_action_values = torch.sum(q_est.gather(1,action_batch.unsqueeze(1)),axis=2)

        # Compute max_ap Q(Sp) with the stable target network
        _,q_target,_ = self.targetModel(next_states_batch)
        next_state_values = torch.sum(q_target.max(1).values.unsqueeze(1),axis=2)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) \
                                        + reward_batch.unsqueeze(1)

        # Compute the loss
        loss = self.criterion(state_action_values,
                              expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return np.double(loss)
    
    def reset():
        pass # Not stateful

    def act(self, obs):
        x = torch.Tensor(obs)

        act,_,Q_est = self.model(x)
        Q = float(Q_est.detach())
        if random.random() > self.epsilon:
            return act, Q
        else:
            return np.array([self.env.action_space.sample()]), Q