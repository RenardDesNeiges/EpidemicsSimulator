"""Centralized implementation of DQN, implements training a single agent to handle epidemic mitigation across the country.
"""

from deep_q_learning.trainer import Trainer, SAVE_FOLDER, LOG_FOLDER
from epidemic_env.env import CountryWideEnv
from epidemic_env.visualize import Visualize
from deep_q_learning.agent import Agent, DQNAgent, NaiveAgent
import deep_q_learning.model as models

import torch

from typing import List, Dict, Any
from torch.utils.tensorboard import SummaryWriter

import numpy as np

class CountryWideTrainer(Trainer):
    """Deep-Q-learning training of a single agent for the entire country.
    """

    @staticmethod
    def _render_log(writer, hist, episode):
        print(f'Eval at episode {episode} Logging image')
        policy_img = Visualize.render_episode_country(hist).transpose(2, 0, 1)
        cities_img = Visualize.render_episode_city(hist).transpose(2, 0, 1)
        writer.add_image(f'Episode/PolicyView', policy_img, episode)
        writer.add_image(f'Episode/CityView', cities_img, episode)

    @staticmethod
    def _log_init(info, obs):
        info_hist = [info]
        obs_hist = [obs]
        rew_hist = []
        loss_hist = []
        Q_hist = []
        return rew_hist, info_hist, obs_hist, loss_hist, Q_hist

    def _log_hist(rew, info, obs, loss, distrib, rew_hist, info_hist, obs_hist, loss_hist, Q_hist):
        rew_hist.append(rew.detach().numpy()[0, 0])
        obs_hist.append(obs)
        info_hist.append(info)
        loss_hist.append(loss)
        Q_hist.append(distrib)

    @staticmethod
    def _tb_log(writer, episode, params, R_hist, info_hist, obs_hist,
               info, obs_next, loss_hist, agent, Q_hist):

        obs_hist.append(obs_next)
        info_hist.append(info)
        if episode % params['reward_sample_rate'] == params['reward_sample_rate']-1:
            writer.add_scalar('Alg/Avg_Reward',
                              np.mean(R_hist), episode)

            writer.add_scalar('Alg/Avg_Loss',
                              np.mean(loss_hist), episode)
            writer.add_scalar('Alg/Avg_Q_values',
                              np.mean(Q_hist), episode)
            writer.add_scalar('Alg/epsilon',
                              agent.epsilon, episode)
            writer.add_scalar('RewardShaping/dead_cost',
                              np.mean([e['dead_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/conf_cost',
                              np.mean([e['conf_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/ann_cost',
                              np.mean([e['ann_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/vacc_cost',
                              np.mean([e['vacc_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/hosp_cost',
                              np.mean([e['hosp_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/conf_dead_ratio',
                              np.mean([e['conf_cost'] for e in info_hist[:-1]])/np.mean([e['dead_cost'] for e in info_hist[:-1]]), episode)

    def _tb_eval_log(writer, episode, params, R_hist, info_hist):
        """ Reward metric computation """
        avg_R = np.mean([np.mean(e) for e in R_hist])
        var_R = np.var([np.mean(e) for e in R_hist])

        """ System specific metric computation """
        avg_Death = np.mean([e[-1]['parameters'][0]['dead']
                            for e in info_hist])
        var_Death = np.var([e[-1]['parameters'][0]['dead'] for e in info_hist])

        avg_PI = np.mean([np.max([_e['parameters'][0]['infected']
                         for _e in e]) for e in info_hist])
        var_PI = np.var([np.max([_e['parameters'][0]['infected']
                        for _e in e]) for e in info_hist])

        avg_Recovered = np.mean(
            [e[-1]['parameters'][0]['recovered'] for e in info_hist])
        var_Recovered = np.var(
            [e[-1]['parameters'][0]['recovered'] for e in info_hist])

        avg_Confined = np.mean(
            [np.sum([int(e['action']['confinement']) for e in i[:-1]])*7 for i in info_hist])
        var_Confined = np.var(
            [np.sum([int(e['action']['confinement']) for e in i[:-1]])*7 for i in info_hist])

        avg_Isolated = np.mean(
            [np.sum([int(e['action']['isolation']) for e in i[:-1]])*7 for i in info_hist])
        var_Isolated = np.var(
            [np.sum([int(e['action']['isolation']) for e in i[:-1]])*7 for i in info_hist])

        avg_Hospital = np.mean(
            [np.sum([int(e['action']['hospital']) for e in i[:-1]])*7 for i in info_hist])
        var_Hospital = np.var(
            [np.sum([int(e['action']['hospital']) for e in i[:-1]])*7 for i in info_hist])

        avg_fVac = np.mean([np.sum([int(e['action']['vaccinate'])
                           for e in i[:-1]])*7 for i in info_hist])
        var_fVac = np.var([np.sum([int(e['action']['vaccinate'])
                          for e in i[:-1]])*7 for i in info_hist])

        print(f'    Eval finished with E[R]={avg_R} and Var[R]={var_R}')

        """ Write the data to the logger """
        if episode % params['reward_sample_rate'] == params['reward_sample_rate']-1:
            writer.add_scalar('EvalAlg/Avg_Reward',
                              avg_R, episode)
            writer.add_scalar('EvalAlg/Var_Reward',
                              var_R, episode)

            writer.add_scalar('EvalSystem/Avg_Deaths',
                              avg_Death, episode)
            writer.add_scalar('EvalSystem/Var_Deaths',
                              var_Death, episode)

            writer.add_scalar('EvalSystem/Avg_PeakInfection',
                              avg_PI, episode)
            writer.add_scalar('EvalSystem/Var_PeakInfection',
                              var_PI, episode)

            writer.add_scalar('EvalSystem/Avg_Recovered',
                              avg_Recovered, episode)
            writer.add_scalar('EvalSystem/Var_Recovered',
                              var_Recovered, episode)

            writer.add_scalar('EvalSystem/Avg_ConfinedDays',
                              avg_Confined, episode)
            writer.add_scalar('EvalSystem/Var_ConfinedDays',
                              var_Confined, episode)

            writer.add_scalar('EvalSystem/Avg_IsolationDays',
                              avg_Isolated, episode)
            writer.add_scalar('EvalSystem/Var_IsolationDays',
                              var_Isolated, episode)

            writer.add_scalar('EvalSystem/Avg_AdditionalHospitalDays',
                              avg_Hospital, episode)
            writer.add_scalar('EvalSystem/Var_AdditionalHospitalDays',
                              var_Hospital, episode)

            writer.add_scalar('EvalSystem/Avg_FreeVaccinationDays',
                              avg_fVac, episode)
            writer.add_scalar('EvalSystem/Var_FreeVaccinationDays',
                              var_fVac, episode)
        return avg_R

    @staticmethod
    def train(env, agent, writer, params):
        """Trains the model
        
        Args:
            agent (Agent): the agent class
            env (CountryWideEnv): the environment class
            writer (SummaryWriter): the tensorboard summary writer
            episode (int): the episode number (at time of eval)
            params (Dict): the run parameter dictionary
            iterations (int): the number of eval iterations to perform
        """

        max_reward = -1000

        for episode in range(params['num_episodes']):

            if episode % params['target_update_rate'] == 0:
                agent.targetModel.load_state_dict(agent.model.state_dict())

            agent.epsilon = max(
                params['epsilon'] - params['epsilon'] *
                episode/params['epsilon_decrease'],
                params['epsilon_floor']
            )

            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, loss_hist, Q_hist = CountryWideTrainer._log_init(
                obs, info)

            while not finished:
                action, est_Q = agent.act(obs)
                obs_next, R, finished, info = env.step(action)
                agent.memory.push(obs, action, obs_next, R)

                loss = agent.optimize_model()
                CountryWideTrainer._log_hist(
                    R, info, obs_next, loss, est_Q, R_hist,
                    info_hist, obs_hist, loss_hist, Q_hist)

                obs = obs_next
                if finished:
                    break

            print("episode {}, avg reward = {}, epsilon = {}".format(
                episode, np.mean(R_hist), agent.epsilon))

            if params['log']:
                CountryWideTrainer._tb_log(  # We log at each time step
                    writer, episode, params, R_hist, info_hist, obs_hist,
                    info, obs_next, loss_hist, agent, Q_hist)

                # evaluation runs are performed with epsilon = 0
                if episode % params['eval_rate'] == 0 or (episode == params['num_episodes']-1):

                    last_reward = CountryWideTrainer._eval_model(
                        agent, env, writer, episode, params, params['eval_samples'])
                    if last_reward > max_reward:
                        max_reward = last_reward
                        print(
                            f"    New maximum reward (E[R]={last_reward}, saving weights!")
                        # save the last model
                        torch.save(agent.model, SAVE_FOLDER +
                                   params['run_name'] + '.pkl')

        return None

    @staticmethod
    def _eval_model(agent, env, writer, episode, params, iterations):
        """Evaluates the model as trainig is performed.
        
        Args:
            agent (Agent): the agent class
            env (CountryWideEnv): the environment class
            writer (SummaryWriter): the tensorboard summary writer
            episode (int): the episode number (at time of eval)
            params (Dict): the run parameter dictionary
            iterations (int): the eval iterations to perform
        """
        agent.epsilon = 0

        R_container = []
        Info_container = []
        for _it in range(iterations):

            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, _, Q_hist = CountryWideTrainer._log_init(
                obs, info)
            while not finished:
                action, est_Q = agent.act(obs)
                obs_next, R, finished, info = env.step(action)
                agent.memory.push(obs, action, obs_next, R)
                CountryWideTrainer._log_hist(
                    R, info, obs_next, 0, est_Q,
                    R_hist, info_hist, obs_hist, [], Q_hist)
                obs = obs_next
                if finished:
                    break

            R_container.append(R_hist)
            Info_container.append(info_hist)

            if _it == 0:
                CountryWideTrainer._render_log(writer, info_hist, episode)

        return CountryWideTrainer._tb_eval_log(  # We log at each time step
            writer, episode, params, R_container, Info_container)

    @staticmethod
    def run(params: Dict[str,Any]):
        """Runs training according to parameters from the parameters dictionary.

        Args:
            params (Dict): the parameters dictionary

        Returns:
            float: the average reward
        """

        if not params['log']:
            print(
                'WARNING LOGGING IS NOT ENABLED, NO TB LOGS OF THE EXPERIMENT WILL BE SAVED')

        logpath = LOG_FOLDER+params['run_name']

        env = CountryWideEnv(params['env_config'], mode=params['mode'])
        if hasattr(models, params['model']):
            model = getattr(models, params['model'])
        else:
            print(f'Error : {params["model"]} is not a valid model name,')
            return None

        agent = DQNAgent(env=env,
                      model=model,
                      criterion=params['criterion'],
                      lr=params['lr'],
                      epsilon=params['epsilon'],
                      gamma=params['gamma'],
                      buffer_size=params['buffer_size'],
                      batch_size=params['batch_size']
                      )

        if params['log']:
            writer = SummaryWriter(log_dir=logpath)
        else:
            writer = None

        CountryWideTrainer.train(env, agent, writer, params)
        
        
    @staticmethod
    def evaluate(params, weights_path, eval_iterations = 50, verbose=False):
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

        env = CountryWideEnv(params['env_config'], mode=params['mode'])
        if params['model']=='naive':
            pass
        elif hasattr(models, params['model']):
            model = getattr(models, params['model'])
        else:
            print(f'Error : {params["model"]} is not a valid model name,')
            return None

        if params['model']=='naive':
            agent = NaiveAgent(env=env,
                        threshold=params['threshold'],
                        confine_time=params['confine_time'],
                    )
        else:
            agent = DQNAgent(env=env,
                        model=model,
                        criterion=params['criterion'],
                        lr=params['lr'],
                        epsilon=params['epsilon'],
                        gamma=params['gamma'],
                        buffer_size=params['buffer_size'],
                        batch_size=params['batch_size']
                    )

            # Load the weights
            agent.model = torch.load(weights_path)
            agent.epsilon = 0 # we run in eval mode

        results = []
        for eval_iter in range(eval_iterations):
            
            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, loss_hist, Q_hist = CountryWideTrainer._log_init(
                obs, info)

            while not finished:
                if params['model']=='naive':
                    obs = info['parameters'][0]['infected']
                action, est_Q = agent.act(obs)
                obs, R, finished, info = env.step(action)
                CountryWideTrainer._log_hist(
                    R, info, obs, 0, est_Q, R_hist,
                    info_hist, obs_hist, loss_hist, Q_hist)
                
                if finished:
                    break
    
            if verbose:
                print("episode {}, avg reward = {}".format(
                eval_iter, np.mean(R_hist)))

            results.append({
                'R_hist': R_hist, 
                'obs_hist': obs_hist, 
                'info_hist': info_hist, 
                'loss_hist': loss_hist, 
                'Q_hist': Q_hist,
            })

        return results

