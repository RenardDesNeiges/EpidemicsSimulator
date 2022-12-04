"""Distributed implementation of DQN, implements paralel Q-learning of N independant deep-q-learning agents.

"""
from deep_q_learning.trainer import Trainer, SAVE_FOLDER, LOG_FOLDER
from epidemic_env.distributed import DistributedEnv
from epidemic_env.visualize import Visualize
from deep_q_learning.agent import Agent, DQNAgent, NaiveAgent
import deep_q_learning.model as models
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from copy import deepcopy

from typing import List, Dict, Any, Tuple


class DistributedTrainer(Trainer):
    """Distributed trainer class to train N Q-learning agents.
    """

    @staticmethod
    def render_log(writer: SummaryWriter, hist : List[Dict[str,Any]], episode:int):
        """Renders a plot to the tensorboard logs

        Args:
            writer (SummaryWriter): tensorboard summmary writer
            hist (List[Dict[str,Any]]): episode history
            episode (int): the episode number
        """
        print(f'Eval at episode {episode} Logging image')
        policy_img = Visualize.render_episode_country(hist).transpose(2, 0, 1)
        cities_img = Visualize.render_episode_city(hist).transpose(2, 0, 1)
        writer.add_image(f'Episode/PolicyView', policy_img, episode)
        writer.add_image(f'Episode/CityView', cities_img, episode)

    @staticmethod
    def log_init(info: Dict[str,Any], obs: Dict[str,torch.Tensor]):
        """Initializes logging

        Args:
            info (Dict[str,Any]): measured system variables
            obs (Dict[str,torch.Tensor]): last observation
        """

        info_hist = [info]
        obs_hist = [obs]
        rew_hist = []
        loss_hist = []
        Q_hist = []
        glob_R_hist = []
        return rew_hist, info_hist, obs_hist, loss_hist, Q_hist, glob_R_hist

    def log_hist(rew:torch.Tensor, info:Dict[str,Any], obs:Dict[str,torch.Tensor], loss:Dict[str,torch.Tensor], _actions:Dict[str,torch.Tensor], _glob_R, rew_hist, info_hist, obs_hist, loss_hist, Q_hist, glob_R_hist):
        rew_hist.append({c: r.detach().numpy()[0, 0] for c, r in rew.items()})
        obs_hist.append(obs)
        info_hist.append(info)
        loss_hist.append(loss)
        glob_R_hist.append(_glob_R)
        Q_hist.append({c: v[1] for c, v in _actions.items()})

    @staticmethod
    def tb_log(writer, episode, params, R_hist, info_hist, obs_hist, info, obs_next, loss_hist, agents, Q_hist, glob_R_hist):

        info_hist.append(info)
        if episode % params['reward_sample_rate'] == params['reward_sample_rate']-1:
            writer.add_scalar('Alg/Avg_Reward',
                              np.mean(glob_R_hist), episode)
            writer.add_scalar('Alg/Avg_IndividualRewards',
                              np.mean([np.mean([v for _, v in r.items()]) for r in R_hist]), episode)
            writer.add_scalar('Alg/Avg_Loss',
                              np.mean([np.mean([v for _, v in l.items()]) for l in loss_hist]), episode)
            writer.add_scalar('Alg/Avg_Q_values',
                              np.mean([np.mean([v for _, v in q.items()]) for q in Q_hist]), episode)
            writer.add_scalar('Alg/epsilon',
                              agents[list(agents.keys())[0]].epsilon, episode)
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

    @staticmethod
    def tb_eval_log(writer, episode, params, glob_R_hist, info_hist):
        """ Reward metric computation """
        avg_R = np.mean([np.mean(e) for e in glob_R_hist])
        var_R = np.var([np.mean(e) for e in glob_R_hist])

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

        avg_Confined = np.mean([np.sum([np.sum([int(c) for c in list(
            e['action']['confinement'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])
        var_Confined = np.var([np.sum([np.sum([int(c) for c in list(
            e['action']['confinement'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])

        avg_Isolated = np.mean([np.sum([np.sum([int(c) for c in list(
            e['action']['isolation'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])
        var_Isolated = np.var([np.sum([np.sum([int(c) for c in list(
            e['action']['isolation'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])

        avg_Hospital = np.mean([np.sum([np.sum([int(c) for c in list(
            e['action']['hospital'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])
        var_Hospital = np.var([np.sum([np.sum([int(c) for c in list(
            e['action']['hospital'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])

        avg_fVac = np.mean([np.sum([np.sum([int(c) for c in list(
            e['action']['vaccinate'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])
        var_fVac = np.var([np.sum([np.sum([int(c) for c in list(
            e['action']['vaccinate'].values())]) for e in i[:-1]])*(7/9) for i in info_hist])

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
    def train(env, agents:Dict[str,Agent], writer:SummaryWriter, params:Dict[str,Any]):
        """Trains the agent group.

        Args:
            env (_type_): simulation environment
            agents (_type_): dict
            writer (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """

        max_reward = -1000

        for episode in range(params['num_episodes']):

            for _, agent in agents.items():
                agent.epsilon = max(
                    params['epsilon'] - params['epsilon'] * episode/params['epsilon_decrease'], params['epsilon_floor'])

            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, loss_hist, Q_hist, glob_R_hist = DistributedTrainer.log_init(
                obs, info)

            while not finished:
                _actions = {c: agent.act(obs[c])
                            for (c, agent) in agents.items()}
                obs_next, _glob_R, R, finished, info = env.step(_actions)
                for c in env.dyn.cities:
                    agents[c].memory.push(
                        obs[c], _actions[c][0], obs_next[c], R[c])

                _losses = {c: agent.optimize_model()
                           for (c, agent) in agents.items()}
                DistributedTrainer.log_hist(
                    R, info, obs_next, _losses, _actions, _glob_R, R_hist, info_hist, obs_hist, loss_hist, Q_hist, glob_R_hist)

                obs = obs_next
                if finished:
                    break

            print("episode {}, avg individual reward = {}, epsilon = {}".format(
                episode, np.mean([np.mean([v for _, v in r.items()])
                                 for r in R_hist]),
                agents[list(agents.keys())[0]].epsilon
            ))
            if params['log']:
                DistributedTrainer.tb_log(
                    writer, episode, params, R_hist, info_hist, obs_hist,
                    info, obs_next, loss_hist, agents, Q_hist, glob_R_hist)

                if episode % params['eval_rate'] == 0 or (episode == params['num_episodes']-1):

                    last_reward = DistributedTrainer._eval_model(
                        agents, env, writer, episode, params, params['eval_samples'])
                    if last_reward > max_reward:
                        max_reward = last_reward
                        print(
                            f"    New maximum reward (E[R]={last_reward}, saving weights!")
                        [torch.save(agent.model, SAVE_FOLDER +
                                   params['run_name'] +'_'+c+ '.pkl')
                                for (c, agent) in agents.items()]

        return None

    @staticmethod
    def _eval_model(agents, env, writer, episode, params, iterations):
        for _, agent in agents.items():
            agent.targetModel.load_state_dict(agent.model.state_dict())

        R_container = []
        Info_container = []
        for _it in range(iterations):

            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, _, Q_hist, glob_R_hist = DistributedTrainer.log_init(
                obs, info)

            while not finished:
                _actions = {c: agent.act(obs[c])
                            for (c, agent) in agents.items()}
                obs_next, _glob_R, R, finished, info = env.step(_actions)
                for c in env.dyn.cities:
                    agents[c].memory.push(
                        obs[c], _actions[c][0], obs_next[c], R[c])

                DistributedTrainer.log_hist(
                    R, info, obs_next, [], _actions, _glob_R, R_hist, info_hist, obs_hist, [], Q_hist, glob_R_hist)

                obs = obs_next
                if finished:
                    break

            R_container.append(glob_R_hist)
            Info_container.append(info_hist)
            if _it == 0:
                DistributedTrainer.render_log(writer, info_hist, episode)

        return DistributedTrainer.tb_eval_log(
            writer, episode, params, R_container, Info_container)

    @staticmethod
    def run(params):
        if not params['log']:
            print(
                'WARNING LOGGING IS NOT ENABLED, NO TB LOGS OF THE EXPERIMENT WILL BE SAVED')

        logpath = LOG_FOLDER+params['run_name']

        env = DistributedEnv(params['env_config'], mode=params['mode'])
        if hasattr(models, params['model']):
            model = getattr(models, params['model'])
        else:
            print(f'Error : {params["model"]} is not a valid model name,')
            return None

        _agent = DQNAgent(env=env,
                       model=model,
                       criterion=params['criterion'],
                       lr=params['lr'],
                       epsilon=params['epsilon'],
                       gamma=params['gamma'],
                       buffer_size=params['buffer_size'],
                       batch_size=params['batch_size']
                       )
        agents = {c: deepcopy(_agent) for c in env.dyn.cities}

        if params['log']:
            writer = SummaryWriter(log_dir=logpath)
        else:
            writer = None

        DistributedTrainer.train(env, agents, writer, params)

    
