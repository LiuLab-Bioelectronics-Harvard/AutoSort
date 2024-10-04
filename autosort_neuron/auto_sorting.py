import sys, struct, math, os, time
import numpy as np
from tqdm import tqdm
import pickle
import scipy
from intanutil.read_header import read_header
from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
from intanutil.read_one_data_block import read_one_data_block
from intanutil.notch_filter import notch_filter
from intanutil.data_to_result import data_to_result
from scipy import signal
from pathlib import Path
from spikeinterface.sorters import WaveClusSorter, IronClustSorter, Kilosort3Sorter
import os 
import spikeinterface
spikeinterface.__version__
import spikeinterface
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import numpy as np
import shutil
import seaborn as sns
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from spikeinterface.core.npzsortingextractor import NpzSortingExtractor
import warnings
warnings.filterwarnings("ignore")
import pickle
from autosort_neuron.sorting import *

def find_trials(cont_trigger_all_all):
    timepoint = np.where(cont_trigger_all_all==1)[0]
    trial_end_t = np.where(np.diff(timepoint)>50)[0]
    trial_start_t = np.where(np.diff(timepoint)>50)[0]+1
    trial_start_t = np.insert(trial_start_t,0,0)

    trial_end_t = np.insert(trial_end_t,len(trial_end_t),len(timepoint)-1)

    trial_start = timepoint[trial_start_t]
    trial_end = timepoint[trial_end_t]
    return trial_start, trial_end



def read_AutoSort_data(date_id_all,
                       day_pth,
                       results_data_path,
                       file_name = 'AutoSort_sorting.npz',
                       save_pth=None,
                       Keep_id=None, 
                       unit_list_all=None,
                       trigger = True,
                        freq_max=3000,
                        freq_min=300):
    
    data_folder_all = day_pth+ f'/Ephys_concat_{date_id_all}/'

    if trigger:
        cont_trigger_all_all = np.load(data_folder_all+'cont_trigger_all_all.npy')  
        cont_trigger_all_all = cont_trigger_all_all.reshape(1,-1)
        cont_trigger_all_all = cont_trigger_all_all[0,:]

    trial_start, trial_end = find_trials(cont_trigger_all_all)
    print(f'{date_id_all} interval:',trial_end[10]-trial_start[10])
    
    
    recording_concat = spikeinterface.core.base.BaseExtractor.load_from_folder(data_folder_all)
    recording_f = spikeinterface.preprocessing.bandpass_filter(recording_concat, freq_min=freq_min, 
                                                       freq_max=freq_max)
    recording_cmr = spikeinterface.preprocessing.common_reference(recording_f, reference='global',operator='average')
    
    file_path = results_data_path+'sorting/'
    Path(file_path).mkdir(parents=True, exist_ok=True)
    
    if os.path.isfile(file_path+file_name):
        sorting = se.NpzSortingExtractor(file_path+file_name)
    else:
        # read AutoSort result
        # gt_all = pd.read_csv(results_data_path+'gt_all.csv', index_col=0)
        pred_all = pd.read_csv(results_data_path+'pred_all.csv', index_col=0)
        # gt_all = np.array(gt_all['0'])
        pred_all = np.array(pred_all['0'])

        # gt_class_all = pd.read_csv(results_data_path+'gt_class_all.csv', index_col=0)
        pred_class_all = pd.read_csv(results_data_path+'pred_class_all.csv', index_col=0)
        # gt_class_all = np.array(gt_class_all['0'])
        pred_class_all = np.array(pred_class_all['0'])

        # read gt spike time
        with (open(save_pth+'/generate_input_cmr/'+date_id_all+'/test_data/X_spiketrain_time.pkl', "rb")) as openfile:
            X_spiketrain_time_test = pickle.load(openfile)
        with (open(save_pth+'/generate_input_cmr/'+date_id_all+'/test_data/Y_spike_id_noise.pkl', "rb")) as openfile:
            Y_spiketrain_id_final_test = pickle.load(openfile)

        print(f'origin spike num:{pred_all.shape[0]}')

        spike_ind = pred_all.astype('bool')
        spike_time = X_spiketrain_time_test[spike_ind]
        neuron_id = pred_class_all[spike_ind]
        ch_num = Y_spiketrain_id_final_test[spike_ind]
        neuron_unitid = [Keep_id[i] for i in neuron_id]
        gt_ch_num = np.array([unit_list_all[i] for i in neuron_unitid])
        print(f'noise prediction after spike num:{spike_time.shape[0]}')

        pred_prob_all = pd.read_csv(results_data_path+'pred_prob_all.csv', index_col=0)
        pred_prob_all = np.array(pred_prob_all)
        pred_prob_all = pred_prob_all[spike_ind,:]

        neuron_ind = np.equal(gt_ch_num, ch_num)

        spike_time = spike_time[neuron_ind]
        neuron_id = neuron_id[neuron_ind]
        neuron_unitid = np.array(neuron_unitid)[neuron_ind]
        ch_num = ch_num[neuron_ind]
        gt_ch_num = gt_ch_num[neuron_ind]
        pred_prob_all=pred_prob_all[neuron_ind,:]
        print(f'channel correspondence after spike num:{spike_time.shape[0]}')

        prob_ind = np.max(pred_prob_all,axis=1)>=0.9
        spike_time = spike_time[prob_ind]
        neuron_id = neuron_id[prob_ind]
        neuron_unitid = neuron_unitid[prob_ind]
        print(f'label prob spike num:{spike_time.shape[0]}')

        # create sorting object
        sorting = se.NumpySorting.from_times_labels([spike_time], [neuron_unitid], 10000)
        se.NpzSortingExtractor.write_sorting(sorting, file_path+file_name)
    return sorting,  trial_start, trial_end,  cont_trigger_all_all,recording_cmr

