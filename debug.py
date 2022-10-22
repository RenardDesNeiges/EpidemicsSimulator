import sys
sys.path.append('./src')

import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from src.epidemic_env.dynamics import ModelDynamics

def plot_per_city(city_history):
    plt.rcParams['figure.figsize'] = [12, 12]

    fig, ax = plt.subplots(len(city_history[0].keys()),1)
    i = 0
    for c in city_history[0].keys():
        c_suceptible = [h[c]['suceptible'] for h in city_history]
        c_infected = [h[c]['infected'] for h in city_history]
        c_recovered = [h[c]['recovered'] for h in city_history]
        c_dead = [h[c]['dead'] for h in city_history]
        ax[i].plot(c_infected)
        ax[i].plot(c_dead)
        ax[i].legend(['infected', 'dead'], loc='upper right')
        ax[i].set_title(c)
        i+= 1
    fig.tight_layout()
def plot(total_history, confinement_history=None):
    plt.rcParams['figure.figsize'] = [12, 12]
    
    fig, ax = plt.subplots(2,2)
    
    t_suceptible = np.array([t['suceptible'] for t in total_history])
    t_infected = np.array([t['infected'] for t in total_history])
    t_recovered = np.array([t['recovered'] for t in total_history])
    t_exposed = np.array([t['exposed'] for t in total_history])
    t_dead = np.array([t['dead'] for t in total_history])
    # plot 1 "fake SI"
    # ax[0,0].plot(np.array(t_suceptible) + np.array(t_recovered))
    ax[0,0].plot(t_infected)
    ax[0,0].plot(t_dead)
    if confinement_history is not None:
        ax[0,0].plot(np.array(confinement_history)*20000)
        ax[0,0].legend(['infected', 'dead', 'confinement'])
    else:
        ax[0,0].legend(['infected', 'dead'])
    ax[0,0].axhline(0,color='black')
    # ax[0,0].axhline(total_history[0]['initial population'],color='black')
    ax[0,0].set_title('Measurable')
    
    # plot 2 "SIR"
    # ax[0,1].plot(np.array(t_suceptible))
    ax[0,1].plot(t_infected)
    ax[0,1].plot(np.array(t_recovered))
    ax[0,1].plot(np.array(t_exposed))
    ax[0,1].plot(t_dead)
    ax[0,1].legend(['infected', 'recovered', 'exposed', 'dead'])
    ax[0,1].axhline(0,color='black')
    # ax[0,1].axhline(total_history[0]['initial population'],color='black')
    ax[0,1].set_title('SIR')
    
    # plot 3 "Phase plane 1"
    ax[1,0].plot(np.array(t_suceptible),t_infected)
    ax[1,0].set_xlabel('S')
    ax[1,0].set_ylabel('I')
    ax[1,0].set_title('SI plot')
    
    # plot 3 "Death rate 1"
    ax[1,1].plot(t_infected[1:],t_dead[1:]-t_dead[0:-1])
    ax[1,1].set_xlabel('I')
    ax[1,1].set_ylabel('D')
    ax[1,1].set_title('Death rate plot')
    
    fig.tight_layout()
    
    plt.show()
    
def naive_run(isolate=False,hospital=False, vaccinate=False, auto_plot=True,time=100,thresh=30000):

    total_history = []
    city_history = []
    confinement_history = []
    
    dyn = ModelDynamics('./config/switzerland.yaml')
    unconfined = {
        'confinement':False,
        'isolation':False,
        'hospital':False,
        'vaccinate':False,} 
    
    confined = {
        'confinement':True,
        'isolation':False,
        'hospital':False,
        'vaccinate':False,} 
    
    act = {c: unconfined for c in dyn.cities}
        
    dyn.start_epidemic(prop=0.1)
    confinement_time = 10
    timer = -1

    for i in range(time):
        total, cities = dyn.epidemic_parameters()
        if total['infected']>thresh or timer >= 0:
                if timer == -1:
                    timer = confinement_time
                timer -= 1
                confinement_history.append(1.0)   
                act = {c: confined for c in dyn.cities}
                for c in dyn.cities:
                    dyn.set_action(act[c],c)
                obs = dyn.step()
        else:
            act = {c: unconfined for c in dyn.cities}
            for c in dyn.cities:
                    dyn.set_action(act[c],c)
            obs = dyn.step()
            confinement_history.append(0.0)
            
        total, cities = dyn.epidemic_parameters()
        total_history.append(total)
        city_history.append(cities)
        
    if auto_plot:
        plot(total_history)
        plot_per_city(city_history)
        
    return total_history, confinement_history

total_history, confinement_history = naive_run(time=10,auto_plot=True)