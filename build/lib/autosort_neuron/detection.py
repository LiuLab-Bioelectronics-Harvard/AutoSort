from spikeinterface.sorters import WaveClusSorter, IronClustSorter, Kilosort3Sorter
import pickle
import os
from pathlib import Path
import scipy
import spikeinterface

spikeinterface.__version__
import spikeinterface
import spikeinterface.extractors as se
# import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
from autosort_neuron.sorting import *


def detect_spike(
    trace0_car,
    thr_min=5,
    thr_max=30,
    distance=3,
    ch_max_simul_firing=3,
    wlen=5,
    prominence=10,
):
    noise_std_detect = np.median(abs(trace0_car) / 0.6745, axis=0)
    thr = thr_min * noise_std_detect
    thrmax = thr_max * noise_std_detect

    spikes = np.zeros(trace0_car.shape)
    if trace0_car.ndim > 1:
        for i in range(noise_std_detect.shape[0]):
            peaks, props = scipy.signal.find_peaks(
                -trace0_car[:, i],
                thr[i],
                distance=distance,
                wlen=wlen,
                prominence=prominence,
            )
            prominences = scipy.signal.peak_prominences(
                -trace0_car[:, i], peaks, wlen=7
            )[0]
            peaks = peaks[props["peak_heights"] > 10]
            prominences = prominences[props["peak_heights"] > 10]
            peaks = peaks[(prominences > 15)]

            spikes[peaks, i] = 1

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
    return spikes


def map_gt_annotation(detect_array, gt_array):
    gt_label_array1 = np.zeros((detect_array.shape[0],)) - 1

    for ind, i in enumerate(detect_array):
        f = 1
        indj = np.where(gt_array[:, 0] == i[0])[0]
        for j in indj:
            if gt_array[j, 1] == i[1]:
                f = 0
                break
        if f:
            indj = np.where(gt_array[:, 0] == i[0] - 1)[0]
            for j in indj:
                if gt_array[j, 1] == i[1]:
                    f = 0
                    break
        if f:
            indj = np.where(gt_array[:, 0] == i[0] + 1)[0]
            for j in indj:
                if gt_array[j, 1] == i[1]:
                    f = 0
                    break
        if f == 0:
            gt_label_array1[ind] = j

    return gt_label_array1


def save_obj(objt, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(objt, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def generate_autosort_input(
    date_id_all,
    raw_data_path,
    save_pth,
    day_pth,
    left_sample,
    right_sample,
    freq_min,
    freq_max,
    mesh_probe,
):
    all_save_folder_name = "_".join(date_id_all)

    for i in date_id_all:
        save_folder_name = i
        data_folder_all = f"./processed_data/Ephys_concat_{save_folder_name}/"
        if os.path.exists(data_folder_all) == False:
            _, _ = read_data_folder(
                data_folder_all,
                [i],
                raw_data_path,
                mesh_probe,
            )

        pth = f"./processed_data/Ephys_concat_{all_save_folder_name}/"
        extremum_channels_ids = pd.read_csv(
            pth + "mountainsort/extremum_channels_ids.csv", index_col=0
        )
        unit_list_all = {}
        for i, j in zip(
            extremum_channels_ids.index, extremum_channels_ids.values.flatten()
        ):
            unit_list_all[i] = j

        for set_time in np.arange(0, len(date_id_all)):

            date_id_all_i = date_id_all[set_time]
            if os.path.exists(save_pth + "/input/" + date_id_all_i + "/"):
                continue
            print("###", date_id_all_i)

            print("### 1. load raw data")
            recording_concat = spikeinterface.core.base.BaseExtractor.load_from_folder(
                day_pth + "Ephys_concat_" + date_id_all_i + "/"
            )
            recording_f = spikeinterface.preprocessing.bandpass_filter(
                recording_concat, freq_min=freq_min, freq_max=freq_max
            )
            recording_cmr = spikeinterface.preprocessing.common_reference(
                recording_f, reference="global", operator="average"
            )
            trace0_car = recording_cmr.get_traces(segment_index=0)

            print("### 2. detect spikes")
            spiketrain = {}
            all_spike_train = []
            spike_loc = []
            spikes = detect_spike(
                trace0_car,
                thr_min=3,
                thr_max=30,
                distance=3,
                ch_max_simul_firing=5,
                wlen=5,
                prominence=10,
            )
            for channel_num in range(trace0_car.shape[1]):
                spiketrain_loc = np.where(spikes[:, channel_num])[0]
                spiketrain[channel_num] = spiketrain_loc
                all_spike_train += list(spiketrain_loc)
                spike_loc += [channel_num] * len(spiketrain_loc)
            X_spiketrain_time = all_spike_train
            Y_spiketrain_id_final = spike_loc
            detect_array = np.array([X_spiketrain_time, Y_spiketrain_id_final]).T

            print("### 3. load ground truth")
            sorting = se.NpzSortingExtractor(
                pth + f"mountainsort/{date_id_all_i}/sorting/firings_merged.npz"
            )
            shank_id = sorting.unit_ids
            spike_train_all = []
            y_unit_id = []
            for i in tqdm(shank_id):
                add_unit = list(sorting.get_unit_spike_train(i))
                spike_train_all += add_unit
                y_unit_id += [i] * len(add_unit)
            gt_ch = [unit_list_all[i] for i in y_unit_id]
            gt_array = np.array([spike_train_all, gt_ch]).T

            print("### 4. map ground truth annotation")
            gt_label_array1 = map_gt_annotation(detect_array, gt_array)
            print(
                "---spike detection rate:",
                np.where(gt_label_array1 > -1)[0].shape[0] / gt_array.shape[0],
            )
            Y_spiketrain_id = np.zeros((detect_array.shape[0],)) - 1
            Y_spiketrain_id[np.where(gt_label_array1 > -1)[0]] = np.array(y_unit_id)[
                gt_label_array1[np.where(gt_label_array1 > -1)[0]].astype("int")
            ]

            print("### 4.5 add all gt")
            mapped_ind = gt_label_array1[np.where(gt_label_array1 > -1)[0]].astype(
                "int"
            )
            A = [i for i in np.arange(len(y_unit_id)) if i not in mapped_ind]

            X_spiketrain_time_train = list(X_spiketrain_time) + list(
                np.array(spike_train_all)[A]
            )
            Y_spiketrain_id_train = list(Y_spiketrain_id) + list(np.array(y_unit_id)[A])
            Y_spiketrain_id_final_train = list(Y_spiketrain_id_final) + list(
                np.array(gt_ch)[A]
            )

            print("### 5. find corresponding waveform")
            X_spiketrain_time = np.array(X_spiketrain_time)
            Y_spiketrain_id = np.array(Y_spiketrain_id)[
                X_spiketrain_time < trace0_car.shape[0] - (left_sample + right_sample)
            ]
            Y_spiketrain_id_final = np.array(Y_spiketrain_id_final)[
                X_spiketrain_time < trace0_car.shape[0] - (left_sample + right_sample)
            ]
            X_spiketrain_time = X_spiketrain_time[
                X_spiketrain_time < trace0_car.shape[0] - (left_sample + right_sample)
            ]
            for time_range in tqdm(np.arange(-left_sample, right_sample)):
                if time_range == -left_sample:
                    waveform = trace0_car[X_spiketrain_time + time_range, :]
                else:
                    waveform = np.dstack(
                        (waveform, trace0_car[X_spiketrain_time + time_range, :])
                    )

            print("### 6. save output")
            current_save_path = save_pth + "/input/" + date_id_all_i + "/test_data/"
            Path(current_save_path).mkdir(parents=True, exist_ok=True)
            save_obj(waveform, current_save_path + "/X_waveform")
            save_obj(Y_spiketrain_id, current_save_path + "/Y_spike_id")
            save_obj(Y_spiketrain_id_final, current_save_path + "/Y_spike_id_noise")
            save_obj(X_spiketrain_time, current_save_path + "/X_spiketrain_time")

            print("### 7. find corresponding waveform")
            X_spiketrain_time_train = np.array(X_spiketrain_time_train)
            Y_spiketrain_id_train = np.array(Y_spiketrain_id_train)[
                X_spiketrain_time_train
                < trace0_car.shape[0] - (left_sample + right_sample)
            ]
            Y_spiketrain_id_final_train = np.array(Y_spiketrain_id_final_train)[
                X_spiketrain_time_train
                < trace0_car.shape[0] - (left_sample + right_sample)
            ]
            X_spiketrain_time_train = X_spiketrain_time_train[
                X_spiketrain_time_train
                < trace0_car.shape[0] - (left_sample + right_sample)
            ]
            for time_range in tqdm(np.arange(-left_sample, right_sample)):
                if time_range == -left_sample:
                    waveform = trace0_car[X_spiketrain_time_train + time_range, :]
                else:
                    waveform = np.dstack(
                        (waveform, trace0_car[X_spiketrain_time_train + time_range, :])
                    )

            print("### 8. save output")
            current_save_path = save_pth + "/input/" + date_id_all_i + "/train_data/"
            Path(current_save_path).mkdir(parents=True, exist_ok=True)
            save_obj(waveform, current_save_path + "/X_waveform")
            save_obj(Y_spiketrain_id_train, current_save_path + "/Y_spike_id")
            save_obj(
                Y_spiketrain_id_final_train, current_save_path + "/Y_spike_id_noise"
            )
            save_obj(X_spiketrain_time_train, current_save_path + "/X_spiketrain_time")
