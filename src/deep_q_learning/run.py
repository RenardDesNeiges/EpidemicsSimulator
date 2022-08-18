from epidemic_env.epidemic_env import EpidemicEnv
from deep_q_learning.agent import Agent
from deep_q_learning.model import DQN
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

DEFAULT_CONFIG = 'config/switzerland.yaml'
NUM_EPISODES = 1000
TARGET_UPDATE = 5
REWARD_SAMPLE = 2



class Trainer():
    def train():
        pass

    @staticmethod
    def run (env_path = DEFAULT_CONFIG):
        
        LOG_PATH = 'runs/log' + datetime.today().strftime('%m_%d.%H_%M_%S')
        
        env = EpidemicEnv(env_path)
        model = DQN
        params = None
        
        agent = Agent(env, model, params)
        
        writer = SummaryWriter(log_dir=LOG_PATH)
        

        cumulative_reward = 0
        cumulative_loss = 0

        for episode in range(NUM_EPISODES):
            # Initialize the environment and state
            S, _, _, _ = env.reset()
                
            finished = False
            while not finished:
                a = agent.act(S) # Choose a from S using the DQN policy net
            
                
            
                # Take action A, observe R, S0
                Sp, R, finished, _ = env.step(a) 
                
                # Store the transition in memory
                agent.memory.push(S, a, Sp, R)
                cumulative_reward += R
                S = Sp
                
                # Update the target network, copying all weights and biases in DQN
                if episode % TARGET_UPDATE == 0:
                    agent.targetModel.load_state_dict(agent.model.state_dict())

                # Perform one step of the optimization (on the policy network)
                loss = agent.optimize_model()
                cumulative_loss += loss
                if finished:
                    break
                
            if  episode%REWARD_SAMPLE == REWARD_SAMPLE-1:
                writer.add_scalar('Train/Reward', cumulative_reward/REWARD_SAMPLE, episode)
                writer.add_scalar('Train/Loss', cumulative_loss/REWARD_SAMPLE, episode)
                print("episode {}, avg reward = {}".format(episode, cumulative_reward/REWARD_SAMPLE))
                cumulative_reward = 0
                cumulative_loss = 0
        