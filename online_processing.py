from time import sleep
import os
from intanutil.read_header import read_header
import pandas as pd
import torch
import spikeinterface.toolkit as st
import spikeinterface.extractors as se
import sys
sys.path.append("../utils/")
sys.path.append("../model/")
from collections import Counter
import numpy as np
from waveform_loader import *
from tqdm import tqdm
from classifiersimple import *
from pathlib import Path
from scipy.signal import argrelextrema
from statistics import median, mean
import scipy
import time

freq_max = 3000
freq_min = 300
ch_max_simul_firing=10

day_id_str = ['0616', '0620']

set_shank_id = [1, 3, 5, 6, 7, 8, 14, 15, 16, 19, 20, 31, 33, 36, 37, 45]


set_channel_id = list(np.arange(32))
sensor_positions = np.array(
    [[120, 260], [440, 260], [120, 360], [440, 360], [200, 280], [360, 280], [200, 220], [360, 220],
     [200, 120], [360, 120], [240, 160], [320, 160], [240, 240], [320, 240], [240, 320], [320, 400], [0, 0],
     [560, 0], [0, 120], [560, 120], [0, 300], [560, 300], [80, 240], [480, 240], [80, 140], [480, 140],
     [80, 40], [480, 40], [120, 80], [440, 80], [120, 180], [440, 180]])




if __name__ == '__main__':
    path='C:/Users/yhe/Desktop/0925_intan_teset/'
    dir_list = os.listdir(path)


    ### load model

    tensor_size = 30 * len(set_channel_id)

    load_dir = "M:/online_spike_sorting/model_save/joystick_time_0616/"

    save_model_path_1 = load_dir + 'multitask_single_wave_noise_ae.pth'
    save_model_path_2 = load_dir + 'multitask_single_wave_clsfier_noise_clsfier.pth'
    save_model_path_3 = load_dir + 'multitask_single_wave_clsfier_label_clsfier.pth'
    model = AE(input_shape=tensor_size).to(device)
    clsfier_noise = clssimp(160, 2).to(device)
    clsfier_label = clssimp(160, len(set_shank_id)).to(device)
    min_valid_loss = np.inf
    load_model = True

    if load_model == True:
        model.load_state_dict(torch.load(save_model_path_1))
        clsfier_noise.load_state_dict(torch.load(save_model_path_2))
        clsfier_label.load_state_dict(torch.load(save_model_path_3))
    model.eval()
    clsfier_noise.eval()
    clsfier_label.eval()

    for i in dir_list:
        if 'good7' in i:
            print("Files and directories in '", path, "' :")
            print(i)
            data, header = read_data_online(path+i,model,clsfier_noise,clsfier_label)
