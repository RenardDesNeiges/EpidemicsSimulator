import sys
sys.path.append('./src')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from deep_q_learning import getTrainer, getParams

naive_runs = [
    ('CountryWideTrainer','COUNTRY_WIDE_NAIVE',None),

]

def _eval(trainer, params_path, ev_weights, iterations):
    _trainer = getTrainer(trainer)
    _params = getParams(params_path)
    return _trainer.evaluate(_params, ev_weights, eval_iterations = iterations)
    
iteratons = 2
run_logs = {
    params[1]:_eval(*params,iteratons) for params in naive_runs
}