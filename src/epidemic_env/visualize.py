import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

DPI = 80

def fig_2_numpy(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', DPI = DPI)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def plot_time_boolean(dataframe,ax, true_label, false_label):
    b_array = np.array(dataframe).transpose()*(1.0)
    ax.imshow(b_array,aspect='auto', interpolation=None, cmap='bwr', vmin=0, vmax=1)

    
    legend_elements = [ Patch(facecolor='blue',label=false_label),
                        Patch(facecolor='red',label=true_label), ]


    ax.legend(handles=legend_elements)
    
    y_label_list = dataframe.columns
    ax.set_yticks(list(range(b_array.shape[0])), y_label_list)

class Visualize():
    
    @staticmethod 
    def render_episode_country(info_hist):
        
        plt.close('all')
        
        infection_hist = pd.DataFrame([e['parameters'][0] for e in info_hist[0:-1]])
        confinement_hist = pd.DataFrame([e['action']['confinement'] for e in info_hist[0:-1]])
        isolation_hist = pd.DataFrame([e['action']['isolation'] for e in info_hist[0:-1]])
        hospital_hist = pd.DataFrame([e['action']['hospital'] for e in info_hist[0:-1]])
        vaccinate_hist = pd.DataFrame([e['action']['vaccinate'] for e in info_hist[0:-1]])
        
        fig, ax = plt.subplots(3,2, figsize=(9,9))
        infection_hist.plot(y='infected', use_index=True, ax=ax[0,0])
        infection_hist.plot(y='dead', x='day', ax=ax[0,0])
        plot_time_boolean(confinement_hist,ax[1,0], 'Confined', 'Not Confined')
        plot_time_boolean(isolation_hist,ax[0,1], 'Isolated', 'Not Isolated')
        plot_time_boolean(hospital_hist,ax[1,1], 'With additional hospital beds', 'Without additional hospital beds')
        plot_time_boolean(vaccinate_hist,ax[2,1], 'Vaccinate', 'Do not vaccinate')
        fig.tight_layout()
        return fig_2_numpy(fig)
    
    @staticmethod 
    def render_episode_fullstate(info_hist):
        
        plt.close('all')
        
        t_infected = np.array([e['parameters'][0]['infected'] for e in info_hist[0:-1]])
        t_suceptible = np.array([e['parameters'][0]['suceptible'] for e in info_hist[0:-1]])
        t_recovered = np.array([e['parameters'][0]['recovered'] for e in info_hist[0:-1]])
        t_exposed = np.array([e['parameters'][0]['exposed'] for e in info_hist[0:-1]])
        t_dead = np.array([e['parameters'][0]['dead'] for e in info_hist[0:-1]])
        if type(info_hist[0]['action']['confinement']) == bool:
            t_conf = [int(e['action']['confinement'])*10000 for e in info_hist[0:-1]]
        else:
            t_conf = [
                np.sum(
                    [int(e['action']['confinement'][c]) for c in  list(info_hist[0]['action']['confinement'].keys()) ]
                    )*10000/9
                    for e in info_hist[0:-1]
                ]

        fig, ax = plt.subplots(2,2, figsize=(9,9))

        ax[0,0].plot(t_infected)
        ax[0,0].plot(t_dead)
        ax[0,0].plot(t_conf)
        ax[0,0].legend(['infected', 'dead','confined'])

        ax[0,1].plot(t_infected)
        ax[0,1].plot(np.array(t_recovered))
        ax[0,1].plot(np.array(t_exposed))
        ax[0,1].plot(t_dead)
        ax[0,1].legend(['infected', 'recovered', 'exposed', 'dead'])
        ax[0,1].axhline(0,color='black')
        ax[0,1].set_title('SIR')

        ax[1,0].plot(np.array(t_suceptible),t_infected)
        ax[1,0].set_xlabel('S')
        ax[1,0].set_ylabel('I')
        ax[1,0].set_title('SI plot')

        ax[1,1].plot(t_infected[1:],t_dead[1:]-t_dead[0:-1])
        ax[1,1].set_xlabel('I')
        ax[1,1].set_ylabel('D')
        ax[1,1].set_title('Death rate plot')
        
        fig.tight_layout()
        return fig_2_numpy(fig)

    @staticmethod 
    def render_episode_city(info_hist):
        
        plt.close('all')

        cities = list(info_hist[0]['parameters'][1].keys())
        fig, ax = plt.subplots(len(cities), 1, figsize=(9,9))

        for _id,city in enumerate(cities):    
            c_infected = np.array([ e['parameters'][1][city]['infected'] for e in info_hist[0:-1]])
            c_dead =  np.array([ e['parameters'][1][city]['dead'] for e in info_hist[0:-1]])

            ax[_id].plot(c_infected)
            ax[_id].plot(c_dead)
            ax[_id].legend(['infected', 'dead'])

        fig.tight_layout()
        
        return fig_2_numpy(fig)
