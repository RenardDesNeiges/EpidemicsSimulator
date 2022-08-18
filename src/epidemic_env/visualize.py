import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

DPI = 300

def fig_2_numpy(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', DPI = DPI)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def plot_time_boolean(dataframe,ax, true_label, false_label):
    b_array = np.array(dataframe).transpose()
    ax.imshow(b_array,aspect='auto', interpolation=None, cmap='bwr')

    
    legend_elements = [ Patch(facecolor='blue',label=false_label),
                        Patch(facecolor='red',label=true_label), ]


    ax.legend(handles=legend_elements)
    
    y_label_list = dataframe.columns
    ax.set_yticks(list(range(b_array.shape[0])), y_label_list)

class Visualize():
    
    @staticmethod 
    def render_episode_country(info_hist):
        
        infection_hist = pd.DataFrame([e['parameters'][0] for e in info_hist[0:-1]])
        confinement_hist = pd.DataFrame([e['action']['confinement'] for e in info_hist[0:-1]])
        isolation_hist = pd.DataFrame([e['action']['isolation'] for e in info_hist[0:-1]])
        hospital_hist = pd.DataFrame([e['action']['hospital'] for e in info_hist[0:-1]])
        vaccinate_hist = pd.DataFrame([e['action']['vaccinate'] for e in info_hist[0:-1]])
        
        fig, ax = plt.subplots(2,2)
        infection_hist.plot(y='infected', use_index=True, ax=ax[0,0])
        infection_hist.plot(y='dead', x='day', ax=ax[0,0])
        plot_time_boolean(confinement_hist,ax[1,0], 'Confined', 'Not Confined')
        plot_time_boolean(isolation_hist,ax[0,1], 'Isolated', 'Not Isolated')
        plot_time_boolean(hospital_hist,ax[1,1], 'With additional hospital beds', 'Without additional hospital beds')
        
        return fig_2_numpy(fig)
