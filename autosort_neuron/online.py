from time import sleep
from intanutil.read_header import read_header
import os
import pandas as pd
import spikeinterface.toolkit as st
import spikeinterface.extractors as se
import sys
from collections import Counter
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.signal import argrelextrema
from statistics import median, mean
import scipy
import pickle5 as pickle
from collections import Counter
import numpy as np
import torch

ch_keep_list = np.arange(32)
ch_keep_list = np.delete(ch_keep_list, 23)
ch_keep_list = np.delete(ch_keep_list, 24)
save_pth = './AutoSort_data/'
ch_num=30
samplepoints=30
loc_dim=3
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
fs = 10000
freq_min=300
freq_max=3000



def location_cal(sensor_positions, batch_features):
    NumChannels = batch_features.shape[1]
    location_day = []

    b_max = batch_features.max(-1)
    b_min = batch_features.min(-1)
    amplitudes = b_max-b_min
    # amplitudes_multi = np.multiply(amplitudes,amplitudes)
    # amplitudes = np.multiply(amplitudes_multi,amplitudes)
    amplitudes =np.square(amplitudes)
    amplitudes = np.square(amplitudes)
    sum_square_amplitute=np.sum(amplitudes,axis=1)

    x=np.dot(sensor_positions[:, 0] , amplitudes.T)
    x=np.divide(x, sum_square_amplitute)

    y=np.dot(sensor_positions[:, 1] , amplitudes.T)
    y=np.divide(y, sum_square_amplitute)

    location_day=[x,y]

    return np.array(location_day)


def detect_spike_online(trace0_car):
    noise_std_detect = np.median(abs(trace0_car - np.mean(trace0_car, axis=0)[None, :]) / 0.6745, axis=0)
    # noise_std_detect = np.median(abs(trace0_car)/0.6745, axis=0)

    thr = 3 * noise_std_detect
    thrmax = 30 * noise_std_detect

    spikes = np.zeros(trace0_car.shape)
    for i in range(noise_std_detect.shape[0]):
        peaks, _ = scipy.signal.find_peaks(-trace0_car[:, i], thr[i], distance=30)
        spikes[peaks, i] = 1

    print(np.sum(spikes == 1))
    # larger value no more than thrmax
    points = trace0_car.shape[0]
    spike_coord = np.argwhere(spikes == 1)
    for i in range(spike_coord.shape[0]):
        near_start = spike_coord[i, 0] - 5
        near_end = spike_coord[i, 0] + 5
        if near_start < 0:
            near_start = 0
        if near_end >= points:
            near_end = points - 1
        if np.any(np.max(trace0_car[near_start:near_end, :], axis=0) >= thrmax):
            spikes[spike_coord[i, 0], spike_coord[i, 1]] = 0

    # no simultanous firing!!!!
    thres_cross = ch_max_simul_firing
    spikes[np.sum(spikes, axis=1) > thres_cross, :] = 0
    print(np.sum(spikes == 1))

    return spikes


def read_data_online(filename,model,clsfier_noise,clsfier_label):
    with open(filename+'/info.rhd', "rb") as fid:
        header = read_header(fid)
        # This file contains the data listed:
        # sampling rate, amplifier bandwidth,
        # channel names, and other useful information.
        num_channels = header['num_amplifier_channels']
        sample_rate = header['sample_rate']
        print(str(sample_rate) + ' is sample rate')
        print(str(num_channels) + ' is num channels')

    data_timestamp=[]  # = {'timestamp':[],
    data_ch_data=[]

    # spike_time={'time':[],'spike_num':[],'need_time':[]}
    # start_time = time.time()


    # now we open time.dat and amplifier.dat to read them
    with open(filename+'/time.dat', 'rb') as fid_time:
        with open(filename+'/amplifier.dat', 'rb') as fid_data:
            while True:
                try:
                    try:
                        read_time = fid_time.read(4)
                    except Exception:
                        print("waiting for more data")
                        sleep(0.1)
                    if not read_time:
                        print("waiting for more data 2")
                        # pd.DataFrame(spike_time).to_csv('with_buffer_1s_spike_num.csv')
                        sleep(0.1)
                    else:

                        t_stamp = int.from_bytes(read_time, "little") / sample_rate
                        amp_list = []
                        for ch in range(num_channels):
                            read_ch_data = fid_data.read(2)
                            ch_data = int.from_bytes(read_ch_data, "little",signed=True)
                            amp_list.append(ch_data * 0.195)  # Micro volts
                        data_timestamp.append(t_stamp)
                        data_ch_data.append(amp_list)
                        # print(amp_list)
                        # print(f' {data[-1]} \n')

                    if t_stamp%1==0:
                        if np.array(data_ch_data).shape[0] < 50:
                            continue
                        print('Second:',t_stamp)
                        recording_concat = se.NumpyRecording(traces_list=np.array(data_ch_data),
                                                      sampling_frequency=sample_rate)
                        recording_f = st.bandpass_filter(recording_concat, freq_min=freq_min,
                                                                       freq_max=freq_max)
                        recording_cmr = st.common_reference(recording_f, reference='global',
                                                                          operator='average')
                        trace0_car = recording_cmr.get_traces(segment_index=0)

                        spikes = detect_spike_online(trace0_car)


                        ### with 1s buffer
                        if spikes.shape[0]>10100:
                            spikes = spikes[-10100:-100,:]
                            data_ch_data = data_ch_data[-200:]

                        spike_coord = np.argwhere(spikes==1)
                        if spike_coord.shape[0]<1:
                            continue

                        ### record spike detection computation time
                        # spike_time['time'].append(t_stamp)
                        # spike_time['spike_num'].append(spike_coord.shape[0])
                        # spike_time['need_time'].append(time.time() - start_time)
                        # start_time = time.time()



                        ### spike classification
                        #prepare multimodal imput
                        spiketrain = {}
                        for i in range(spikes.shape[1]):
                            spike_loc = np.argwhere(spikes[:, i] == 1)
                            spiketrain[i] = spike_loc.flatten()


                        detected_spike = spiketrain[list(spiketrain.keys())[0]]
                        detected_spike_channel = np.zeros(spiketrain[list(spiketrain.keys())[0]].shape[0]) + \
                                                 list(spiketrain.keys())[0]
                        for i in list(spiketrain.keys())[1:]:
                            detected_spike = np.concatenate((detected_spike, spiketrain[i]))
                            detected_spike_channel = np.concatenate(
                                (detected_spike_channel, i + np.zeros(spiketrain[i].shape[0])))

                        X_spiketrain_time = detected_spike
                        Y_spiketrain_id = detected_spike_channel

                        for time_range in np.arange(-10, 20):
                            if time_range == -10:
                                waveform = trace0_car[X_spiketrain_time + time_range, :]
                            else:
                                waveform = np.dstack((waveform, trace0_car[X_spiketrain_time + time_range, :]))

                        waveform_single = waveform[np.arange(waveform.shape[0]), Y_spiketrain_id.astype('int'), :]
                        pred_location = location_cal(sensor_positions, waveform).T


                        #apply trained model
                        # classify_labels = classify_labels.to(device)
                        data = torch.Tensor(waveform).view(-1, 960).to(device)
                        # labels = labels.to(device)
                        pred_loc = torch.tensor(pred_location).to(device)


                        codes, target = model(data)
                        codes = torch.cat((codes, torch.Tensor(waveform_single).to(device)), axis=1)
                        codes = torch.cat((codes, pred_loc), axis=1)

                        cls_output = clsfier_noise(codes.float())
                        labels_pred = np.argmax(cls_output.cpu().detach().numpy(),axis=1)
                        test = labels_pred == 1
                        if sum(test) > 1:
                            cls_label_output = clsfier_label(codes.float()[test, :])
                            cls_label_pred = torch.argmax(cls_label_output, axis=1)
                            ch_loc = Y_spiketrain_id[test]
                            cls_label_pred_int = cls_label_pred.cpu().detach().numpy()

                            cls_label_pred_int = [set_shank_id[i] for i in cls_label_pred_int]
                            cls_label_pred_int = np.array(cls_label_pred_int)

                            unique, counts = np.unique(cls_label_pred_int, return_counts=True)
                            for i,j in zip(unique,counts):
                                # print('------neuron:',i,'ch:',ch_loc[cls_label_pred_int==i],'spikes:',j)
                                print('------neuron:', i, 'spikes:', j)

                            p=0
                            
                except KeyboardInterrupt:
                    break

    return data, header

