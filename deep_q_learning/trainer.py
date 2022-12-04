"""Defines the abstract Trainer class and it's associated methods: train, run and evaluate.
"""
from epidemic_env.env import Env
from deep_q_learning.agent import Agent
from torch.utils.tensorboard import SummaryWriter

from abc import ABC, abstractmethod
from typing import List, Dict, Any

LOG_FOLDER = 'runs/'
SAVE_FOLDER = 'models/'


class Trainer(ABC):
    """Trains a reinforcement learning agent in the simulated environment (this is an abstract class, for implementations see the `country_wide` and `distribued` modules).
    """
    
    @staticmethod
    @abstractmethod
    def evaluate(params: Dict[str,Any], weights_path:str, eval_iterations:int = 50, verbose:bool=False)->List[Dict[str,Any]]:
        """Loads pre-saved weights and runs an evaluation run of the model.

        Args:
            params (Dict): the run parameter dictionary
            weights_path (str): path to the pre-saved weights file
            eval_iterations (int, optional): How many eval episodes to run? Defaults to 50.
            verbose (bool, optional): if true, prints detailled output to stdout. Defaults to False.

        Returns:
            List(Dict(str:Any)): a list of dictionaries conaining the performance for each eval episode
            
            Each dict contains the following elements:
            'R_hist': R_hist,           <-- Reward history
            'obs_hist': obs_hist,       <-- Observation history
            'info_hist': info_hist,     <-- Simulation parameter history
            'loss_hist': loss_hist,     <-- Loss history
            'Q_hist': Q_hist,           <-- Q-value estimation history
        """
        
    @staticmethod
    @abstractmethod
    def train(env, agent:Agent, writer:SummaryWriter, params:Dict[str,Any])-> None:
        """Trains the model
        
        Args:
            agent (Agent): the agent class
            env (Env): the environment class
            writer (SummaryWriter): the tensorboard summary writer
            episode (int): the episode number (at time of eval)
            params (Dict): the run parameter dictionary
            iterations (int): the number of eval iterations to perform
        """

    @staticmethod
    @abstractmethod
    def run(params:Dict[str,Any])->float:
        """Runs training according to parameters from the parameters dictionary.

        Args:
            params (Dict): the parameters dictionary

        Returns:
            float: the average reward
        """
