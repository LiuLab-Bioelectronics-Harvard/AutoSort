import numpy as np
import torch
from torch.utils import data
import pickle


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

    location_day=[]
    for ij in range(sensor_positions.shape[1]):
        x=np.dot(sensor_positions[:, ij] , amplitudes.T)
        x=np.divide(x, sum_square_amplitute)
        location_day.append(x)
    # y=np.dot(sensor_positions[:, 1] , amplitudes.T)
    # y=np.divide(y, sum_square_amplitute)
    #
    # # location_day = [x, y]
    # z=np.dot(sensor_positions[:, 2] , amplitudes.T)
    # z=np.divide(z, sum_square_amplitute)
    # location_day=[x,y,z]
    location_day=np.array(location_day).T
    return location_day


def location_cal_group(sensor_positions, batch_features,group_id):
    group_batch = sensor_positions[:,-1]
    location_day=np.zeros((batch_features.shape[0],3))
    for i in np.unique(group_batch):
        care_loc = np.where(group_batch==i)[0]
        look_spike_loc = np.nonzero(np.in1d(group_id, care_loc))[0]
        location_day_batch = location_cal(sensor_positions[care_loc,:], batch_features[look_spike_loc,:,:][:,care_loc,:])
        location_day[look_spike_loc,:] = location_day_batch
    return location_day



class waveformLoader(data.Dataset):
    def __init__(self, root, shank_channel ,
                 sensor_positions,Keep_id=None):
        with (open(root + "X_waveform.pkl", "rb")) as openfile:
            datafile = pickle.load(openfile)
        try:
            with (open(root + "Y_spike_id.pkl", "rb")) as openfile:
                GT = pickle.load(openfile)
        except FileNotFoundError:
            GT = np.zeros(datafile.shape[0])-1
        with (open(root + "Y_spike_id_noise.pkl", "rb")) as openfile:
            channel_id = np.array(pickle.load(openfile))

        if Keep_id is None:
            Keep_id = np.unique(GT)
            Keep_id = list(Keep_id[Keep_id != -1])
            self.keep_id = Keep_id

        mask = ~np.isin(GT, Keep_id)
        GT = np.array(GT)

        GT_binary = np.zeros((GT.shape[0], 2))
        GT_binary[list(mask), 0] = 1
        GT_binary[~mask, 1] = 1

        self.GT_unique = Keep_id + [-1]
        self.GT_binary = GT_binary
        
        self.Img_single = datafile[np.arange(datafile.shape[0]), np.array(channel_id).astype('int'), :]

        self.GT_LIST = GT

        GT_array = np.zeros((len(GT), len(Keep_id)))
        for idx, unique_id in enumerate(Keep_id):
            rmv_list = np.where(np.array(GT) == unique_id)[0]
            GT_array[rmv_list, idx] = 1
        self.GT = GT_array

        self.Img = datafile

        self.pos_weight_noise =torch.tensor( [-np.sum(self.GT_binary[:,0]-1)/np.sum(self.GT_binary[:,0]),
                                 -np.sum(self.GT_binary[:,1]-1)/np.sum(self.GT_binary[:,1])])
        self.pos_weight_label = torch.tensor([-(np.sum(self.GT[:,i]-1)+sum(np.sum(GT_array,axis=1)==0))/np.sum(self.GT[:,i]) for i in range(self.GT.shape[1])])

        pred_location = location_cal_group(sensor_positions, datafile, channel_id)

        self.pred_location = pred_location
        print('pred_location',pred_location.shape)
        self.n_classes = len(set(self.GT_unique))

    def __len__(self):
        return len(self.GT)

    def __getitem__(self, index):
        return self.Img[index, ...] ,  self.GT[index, ...], self.GT_binary[index, ...], self.Img_single[index, ...],self.pred_location[index,...]

