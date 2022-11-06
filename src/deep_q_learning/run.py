from epidemic_env.env import CountryWideEnv, DistributedEnv
from epidemic_env.visualize import Visualize
from deep_q_learning.agent import Agent
import deep_q_learning.model as models
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
import numpy as np
from copy import deepcopy

LOG_FOLDER = 'runs/'
SAVE_FOLDER = 'models/'


PARAMS = {
    'COUNTRY_WIDE_DEBUG': {
        'log': True,
        'mode': 'binary',
        'run_name': 'country_wide_debug_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 2,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'COUNTRY_WIDE_BINARY': {
        'log': True,
        'mode': 'binary',
        'run_name': 'country_wide_binary_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'COUNTRY_WIDE_BINARY_TOGGLE': {
        'log': True,
        'mode': 'toggle',
        'run_name': 'country_wide_binary_toggle_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512
    },
    'COUNTRY_WIDE_MULTI_TOGGLE': {
        'log': True,
        'mode': 'multi',
        'run_name': 'country_wide_multiaction_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 600,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.9,
        'buffer_size': 10000,
        'batch_size': 512
    },
    'DISTRIBUTED_DEBUG': {
        'log': True,
        'mode': 'binary',
        'run_name': 'decentralized_debug' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 2,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'DISTRIBUTED_BINARY': {
        'log': True,
        'mode': 'binary',
        'run_name': 'decentralized_binary_agents' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'DISTRIBUTED_BINARY_TOGGLE': {
        'log': True,
        'mode': 'toggle',
        'run_name': 'decentralized_binary_toggled_agents' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'DISTRIBUTED_MULTI_TOGGLE': {
        'log': True,
        'mode': 'multi',
        'run_name': 'decentralized_multiaction_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 600,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.9,
        'buffer_size': 10000,
        'batch_size': 512
    },
}


def getTrainer(name):
    if name == 'CountryWideTrainer':
        return CountryWideTrainer
    elif name == 'DistributedTrainer':
        return DistributedTrainer
    else:
        raise ValueError('Invalid trainer object!')


def getParams(name):
    if name not in PARAMS.keys():
        raise ValueError('Invalid Parameters')
    else:
        return PARAMS[name]


class DistributedTrainer():

    @staticmethod
    def render_log(writer, hist, episode):
        print(f'Eval at episode {episode} Logging image')
        policy_img = Visualize.render_episode_country(hist).transpose(2, 0, 1)
        cities_img = Visualize.render_episode_city(hist).transpose(2, 0, 1)
        writer.add_image(f'Episode/PolicyView', policy_img, episode)
        writer.add_image(f'Episode/CityView', cities_img, episode)

    @staticmethod
    def log_init(info, obs):
        info_hist = [info]
        obs_hist = [obs]
        rew_hist = []
        loss_hist = []
        Q_hist = []
        glob_R_hist = []
        return rew_hist, info_hist, obs_hist, loss_hist, Q_hist, glob_R_hist

    def log_hist(rew, info, obs, loss, _actions, _glob_R, rew_hist, info_hist, obs_hist, loss_hist, Q_hist, glob_R_hist):
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
    def train(env, agents, writer, params):

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
                        torch.save(agent.model, SAVE_FOLDER +
                                   params['run_name'] + '.pkl')

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

        _agent = Agent(env=env,
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


class CountryWideTrainer():

    @staticmethod
    def render_log(writer, hist, episode):
        print(f'Eval at episode {episode} Logging image')
        policy_img = Visualize.render_episode_country(hist).transpose(2, 0, 1)
        cities_img = Visualize.render_episode_city(hist).transpose(2, 0, 1)
        writer.add_image(f'Episode/PolicyView', policy_img, episode)
        writer.add_image(f'Episode/CityView', cities_img, episode)

    @staticmethod
    def log_init(info, obs):
        info_hist = [info]
        obs_hist = [obs]
        rew_hist = []
        loss_hist = []
        Q_hist = []
        return rew_hist, info_hist, obs_hist, loss_hist, Q_hist

    def log_hist(rew, info, obs, loss, distrib, rew_hist, info_hist, obs_hist, loss_hist, Q_hist):
        rew_hist.append(rew.detach().numpy()[0, 0])
        obs_hist.append(obs)
        info_hist.append(info)
        loss_hist.append(loss)
        Q_hist.append(distrib)

    @staticmethod
    def tb_log(writer, episode, params, R_hist, info_hist, obs_hist,
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

    def tb_eval_log(writer, episode, params, R_hist, info_hist):
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
            R_hist, obs_hist, info_hist, loss_hist, Q_hist = CountryWideTrainer.log_init(
                obs, info)

            while not finished:
                action, est_Q = agent.act(obs)
                obs_next, R, finished, info = env.step(action)
                agent.memory.push(obs, action, obs_next, R)

                loss = agent.optimize_model()
                CountryWideTrainer.log_hist(
                    R, info, obs_next, loss, est_Q, R_hist,
                    info_hist, obs_hist, loss_hist, Q_hist)

                obs = obs_next
                if finished:
                    break

            print("episode {}, avg reward = {}, epsilon = {}".format(
                episode, np.mean(R_hist), agent.epsilon))

            if params['log']:
                CountryWideTrainer.tb_log(  # We log at each time step
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
        agent.epsilon = 0

        R_container = []
        Info_container = []
        for _it in range(iterations):

            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, _, Q_hist = CountryWideTrainer.log_init(
                obs, info)
            while not finished:
                action, est_Q = agent.act(obs)
                obs_next, R, finished, info = env.step(action)
                agent.memory.push(obs, action, obs_next, R)
                CountryWideTrainer.log_hist(
                    R, info, obs_next, 0, est_Q,
                    R_hist, info_hist, obs_hist, [], Q_hist)
                obs = obs_next
                if finished:
                    break

            R_container.append(R_hist)
            Info_container.append(info_hist)

            if _it == 0:
                CountryWideTrainer.render_log(writer, info_hist, episode)

        return CountryWideTrainer.tb_eval_log(  # We log at each time step
            writer, episode, params, R_container, Info_container)

    @staticmethod
    def run(params):

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

        agent = Agent(env=env,
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
