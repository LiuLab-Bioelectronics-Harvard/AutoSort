import os 
from pathlib import Path
import scipy
import spikeinterface
import spikeinterface.extractors as se
#import spikeinterface.toolkit as st
from spikeinterface.preprocessing import bandpass_filter, common_reference

import pickle
import pickle
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from pylab import *
ss.Kilosort3Sorter.set_kilosort3_path('/kilosort3')
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm
import os
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("../model/")

from tqdm import tqdm

from utils.sma_fun import *
from utils.spike_sorting import *
from utils.sma_fun import *
from utils.classifiersimple import *
from utils.waveform_loader import *

import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided

__all__ = ['view_as_blocks', 'view_as_windows']

from collections import Counter

import numpy as np

def list_substraction(a, b):

    ca = Counter(a)
    cb = Counter(b)

    result_b = sorted((cb - ca).elements())
    return result_b

def save_obj(objt, name):
    with(open(name+'.pkl','wb')) as f:
        pickle.dump(objt,f,pickle.HIGHEST_PROTOCOL)
        f.close()

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

def array2_in_array1(array2, array1):
    t=0
    miss_loc=[]
    for i in array2:
        f=1
        for j in array1[array1[:,0]==i[0],:]:
            if j[1]==i[1]:
                f=0
        if f:
            t=t+1
            miss_loc.append(i)
    miss_loc = np.array(miss_loc)
    return miss_loc, t

def compare_results(n,X_spiketrain_time,Y_spiketrain_id_final):
        
        int11 = n*10000-200
        int22 = (1+n)*10000

        #### compare if ground truth is almost included: -yes
        array1 = np.array((X_spiketrain_time+int11,Y_spiketrain_id_final)).T
        array2 = array1.copy()
        array2[:,0] = array2[:,0]+1
        array3 = array1.copy()
        array3[:,0] = array3[:,0]-1
        array1 = np.concatenate((array1,array2),axis=0)
        array1 = np.concatenate((array1,array3),axis=0)
        
        ind = np.where((X_spiketrain_time_all<int22)&(X_spiketrain_time_all>int11))
        array2 = np.array([X_spiketrain_time_all[ind],Y_spiketrain_id_final_all[ind]]).T

        miss_loc, t = array2_in_array1(array2, array1)
        

        print('miss:',t)      
        print('-'*10) 

def find_trials(cont_trigger_all_all):
    timepoint = np.where(cont_trigger_all_all==1)[0]
    trial_end_t = np.where(np.diff(timepoint)>50)[0]
    trial_start_t = np.where(np.diff(timepoint)>50)[0]+1
    trial_start_t = np.insert(trial_start_t,0,0)

    trial_end_t = np.insert(trial_end_t,len(trial_end_t),len(timepoint)-1)

    trial_start = timepoint[trial_start_t]
    trial_end = timepoint[trial_end_t]
    return trial_start, trial_end

def block_reduce(image, block_size=2, func=np.sum, cval=0, func_kwargs=None):
    """Downsample image by applying function `func` to local blocks.

    This function is useful for max and mean pooling, for example.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like or int
        Array containing down-sampling integer factor along each axis.
        Default block_size is 2.
    func : callable
        Function object which is used to calculate the return value for each
        local block. This function must implement an ``axis`` parameter.
        Primary functions are ``numpy.sum``, ``numpy.min``, ``numpy.max``,
        ``numpy.mean`` and ``numpy.median``.  See also `func_kwargs`.
    cval : float
        Constant padding value if image is not perfectly divisible by the
        block size.
    func_kwargs : dict
        Keyword arguments passed to `func`. Notably useful for passing dtype
        argument to ``np.mean``. Takes dictionary of inputs, e.g.:
        ``func_kwargs={'dtype': np.float16})``.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Examples
    --------
    >>> from skimage.measure import block_reduce
    >>> image = np.arange(3*3*4).reshape(3, 3, 4)
    >>> image # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]],
           [[24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35]]])
    >>> block_reduce(image, block_size=(3, 3, 1), func=np.mean)
    array([[[16., 17., 18., 19.]]])
    >>> image_max1 = block_reduce(image, block_size=(1, 3, 4), func=np.max)
    >>> image_max1 # doctest: +NORMALIZE_WHITESPACE
    array([[[11]],
           [[23]],
           [[35]]])
    >>> image_max2 = block_reduce(image, block_size=(3, 1, 4), func=np.max)
    >>> image_max2 # doctest: +NORMALIZE_WHITESPACE
    array([[[27],
            [31],
            [35]]])
    """

    if np.isscalar(block_size):
        block_size = (block_size,) * image.ndim
    elif len(block_size) != image.ndim:
        raise ValueError("`block_size` must be a scalar or have "
                         "the same length as `image.shape`")

    if func_kwargs is None:
        func_kwargs = {}

    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = np.pad(image, pad_width=pad_width, mode='constant',
                   constant_values=cval)

    blocked = view_as_blocks(image, block_size)

    return func(blocked, axis=tuple(range(image.ndim, blocked.ndim)),
                **func_kwargs)

def view_as_blocks(arr_in, block_shape):
    """Block view of the input n-dimensional array (using re-striding).

    Blocks are non-overlapping views of the input array.

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    block_shape : tuple
        The shape of the block. Each dimension must divide evenly into the
        corresponding dimensions of `arr_in`.

    Returns
    -------
    arr_out : ndarray
        Block view of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_blocks
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> B = view_as_blocks(A, block_shape=(2, 2))
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[2, 3],
           [6, 7]])
    >>> B[1, 0, 1, 1]
    13

    >>> A = np.arange(4*4*6).reshape(4,4,6)
    >>> A  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],
           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]],
           [[48, 49, 50, 51, 52, 53],
            [54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71]],
           [[72, 73, 74, 75, 76, 77],
            [78, 79, 80, 81, 82, 83],
            [84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95]]])
    >>> B = view_as_blocks(A, block_shape=(1, 2, 2))
    >>> B.shape
    (4, 2, 3, 1, 2, 2)
    >>> B[2:, 0, 2]  # doctest: +NORMALIZE_WHITESPACE
    array([[[[52, 53],
             [58, 59]]],
           [[[76, 77],
             [82, 83]]]])
    """
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.

    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.

    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_windows
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])

    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    >>> A = np.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out

def detect_spike(trace0_car,thr_min = 5, thr_max=30, threshold=20, 
                 distance=30, ch_max_simul_firing = 3,wlen=5, prominence=10):
    noise_std_detect = np.median(abs(trace0_car ) / 0.6745, axis=0)
    thr = thr_min * noise_std_detect
    thrmax = thr_max * noise_std_detect

    spikes = np.zeros(trace0_car.shape)
    if trace0_car.ndim>1:
        for i in range(noise_std_detect.shape[0]):
#             peaks, props = scipy.signal.find_peaks(-trace0_car[:, i], thr[i], 
#                                                 distance=distance,
#                                               wlen=wlen, prominence=prominence)
            peaks, props = scipy.signal.find_peaks(-trace0_car[:, i], thr[i],threshold = threshold,)        
            prominences = scipy.signal.peak_prominences(-trace0_car[:, i], peaks, wlen=7)[0]
            peaks = peaks[props['peak_heights']>10]
            prominences = prominences[props['peak_heights']>10]
            peaks = peaks[(prominences>20)]

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

def compare_spike_sorting_results(n, X_spiketrain_time,save_ind, pred_class, Y_spiketrain_id_final, y_id_wehave):
    int11 = n*10000-200
    int22 = (1+n)*10000

    #### compare if ground truth is almost included: -yes
    array1 = np.array((X_spiketrain_time+int11,Y_spiketrain_id_final)).T

    indd = np.where((X_spiketrain_time_all<int22-100)&(X_spiketrain_time_all>int11+100))
    array2 = np.array([X_spiketrain_time_all[indd],Y_spiketrain_id_final_all[indd]]).T
    
    
    gt_label_array1=np.zeros((array1.shape[0],))-1
    for ind,i in enumerate(array1):
        f=1
        indj = np.where(array2[:,0]==i[0])[0]
        for j in indj:
            if array2[j,1]==i[1]:
                f=0
                break
        if f:
            indj = np.where(array2[:,0]==i[0]-1)[0]
            for j in indj:
                if array2[j,1]==i[1]:
                    f=0   
                    break
        if f:
            indj = np.where(array2[:,0]==i[0]+1)[0]
            for j in indj:
                if array2[j,1]==i[1]:
                    f=0
                    break
        if f==0:
            gt_label_array1[ind] = j

    gt_label_noise = gt_label_array1.copy()
    gt_label_noise[gt_label_noise>-1]=1
    gt_label_noise[gt_label_noise==-1]=0
    
    labe_post_process_idx=np.zeros(save_ind.shape)+1
    for ind,(i,j) in enumerate(zip([Keep_id[i] for i in pred_class],array1[save_ind,1])):
        if unit_list_all[i]!=j:
            labe_post_process_idx[ind]=0
    save_ind=save_ind[labe_post_process_idx.astype('bool')]
    pred_class=pred_class[labe_post_process_idx.astype('bool')]        


    pred_label_noise = np.zeros(X_spiketrain_time.shape[0])
    pred_label_noise[save_ind]=1  

    gt_save_ind = [-1]
    miss_pct = [-1]
    
    if y_id_wehave: 
        y_true =Y_spiketrain_id_all[indd][gt_label_array1[save_ind[np.where(gt_label_array1[save_ind]>-1)[0]]].astype('int')]
        y_pred = np.array([Keep_id[i] for i in pred_class])[np.where(gt_label_array1[save_ind]>-1)[0]]
        acc1.append(accuracy_score(gt_label_noise, pred_label_noise))
        acc2.append(accuracy_score(y_true, y_pred))
        gt_save_ind = []
        for i in gt_label_array1[save_ind]:
            if i>-1:
                gt_save_ind.append(Y_spiketrain_id_all[indd][int(i)])
            else:
                gt_save_ind.append(-1)
        miss_pct = (array2.shape[0]-np.sum(gt_label_array1[save_ind]>-1))/array2.shape[0]

    
    return array1,array2,gt_save_ind,save_ind,pred_class, miss_pct

def apply_trained_model(waveform,waveform_single,pred_location):

        data = torch.Tensor(waveform).view(-1, samplepoints * ch_num).to(device)
        single_waveform  =  torch.Tensor(waveform_single).to(device)
        pred_loc = torch.tensor(pred_location).to(device)

        codes = torch.cat((data, single_waveform), axis=1)
        codes = torch.cat((codes, pred_loc), axis=1)

        cls_output = clsfier_noise(codes.float())
        pred = torch.argmax(cls_output, axis=1)
        probs_noise = torch.sigmoid(cls_output)
        probs_noise = probs_noise.cpu().detach().numpy()

        
        labels = pred.cpu().numpy()
        test = np.where(labels)[0]
        
        if sum(test)>1:
            cls_label_output = clsfier_label(codes.float()[test,:])
            pred_class = torch.argmax(cls_label_output,axis=1)
            pred_class = pred_class.cpu().detach().numpy()
            probs = torch.sigmoid(cls_label_output)
            probs = probs.cpu().detach().numpy()
        else:
            pred_class=torch.tensor([])
            pred_class = pred_class.cpu().detach().numpy()
            probs=np.zeros((cls_label_output.shape))

            
        #####second filtering
        second_prob = np.max(probs,axis=1)
        ind = np.where(second_prob>0.9)[0]
        test = test[ind]
        pred_class = pred_class[ind]
        probs = probs[ind,:]

        return test, pred_class, np.max(probs,axis=1),  probs_noise


sensor_positions_all = np.array([[150, 250,1],
                    [150,200,1],
                    [50, 0,0],
                    [50, 50,0],
                    [50, 100,0],
                    [0, 100,0],
                    [0, 50,0],
                    [0, 0,0],
                    [650, 0,4],
                    [650, 50,4],
                    [650, 100,4],
                    [600, 100,4],
                    [600, 50,4],
                    [600, 0,4],
                    [500, 200,3],
                    [500, 250,3],
                    [500, 300,3],
                    [450, 300,3],
                    [450, 250,3],
                    [450, 200,3],
                    [350, 400,2],
                    [350, 450,2],
                    [350, 500,2],
                    [300, 500,2],
                    [300, 450,2],
                    [300, 400,2],
                    [200, 200,1],
                    [200, 250,1],
                    [200, 300,1],
                    [150, 300,1] ])

''' 
day_id_str = ['0305_a',
              '0306', '0307', '0308', '0309',  # 5
              '0310', '0311', '0312', '0313', '0314', '0315',  # 11
              '0316', '0317',
               
              '0330',
              '0331', '0401', '0402', '0403', '0404',
              '0405', '0407', '0408', '0409', '0410', '0411',  # 25
              '0412', '0414',
              '0415', '0416', '0417', '0418', '0419', '0420', '0424', '0425'] #35
''' #this one works


day_id_str = ['0305_a',
              '0306', '0307', '0308', '0309',  # 5
              '0310', '0311', '0312', '0313', '0314', '0315',  # 11
              '0316', '0317', 
              
              #'0318', '0319', '0320', '0321', '0322', #these were not included!!!
               
              '0330', '0331', '0401', '0402', '0403', '0404', '0405', #25

              #'0406', #this was not included either
              
              '0407', '0408', '0409', '0410', '0411',  # 31
              '0412', '0414',
              '0415', '0416', '0417', '0418', '0419', '0420', '0424', '0425'] #41


extremum_channels_ids=pd.read_csv('/Volumes/Extreme SSD/yichun/AutoSort/m1/AutoSort_data/generate_input_cmr/data_all/extremum_channels_ids_change.csv',index_col=0)
unit_list_all={}

for i,j in zip(extremum_channels_ids.index,extremum_channels_ids.values.flatten()):
    unit_list_all[i]=j 

save_pth = '/Volumes/Extreme SSD/yichun/AutoSort/m1/AutoSort_data/'

num_bins = 32
group=np.arange(30)
Keep_id=list(np.arange(21))
goodchannel=list(np.arange(30)) 
num_neurons = len(Keep_id)

ch_num=30
samplepoints=30
loc_dim=3
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

clsfier_noise = clssimp((ch_num+1)*samplepoints+loc_dim , 2)
clsfier_label = clssimp((ch_num+1)*samplepoints+loc_dim , len(Keep_id))

clsfier_noise.load_state_dict(torch.load(save_pth+'/model_save_cmr/group_split_allshank_data_allmodel_AutoSort2_seed_0/'+'train_weight/multitask_single_wave_clsfier_noise_clsfier.pth', map_location='cpu'))
clsfier_label.load_state_dict(torch.load(save_pth+'/model_save_cmr/group_split_allshank_data_allmodel_AutoSort2_seed_0/'+'train_weight/multitask_single_wave_clsfier_label_clsfier.pth', map_location='cpu'))

clsfier_noise.eval()
clsfier_label.eval()


fs = 10000
freq_min=300
freq_max=3000

start_interval1 = 10000
start_interval2 = 10000
stimulus_times = [-1, 1]


num_trials_all=[0]
num_stimulus = start_interval1+start_interval2
day_trial_spike_train=[]
trial_num_all=[]
day_num_all=[]
firing_rate_day = np.zeros((len(day_id_str), num_neurons ,num_bins))
firing_rate_sd_day = np.zeros((len(day_id_str), num_neurons, num_bins))
print(f"num_day: {len(day_id_str)}, num_neurons: {num_neurons}, num_bins: {num_bins}")


num_day=-1
trial_test_all=[]

for set_time in range(0,len(day_id_str)):
    num_day = num_day+1   
    
    set_day_id_str = day_id_str[set_time]
    print(set_day_id_str)

    data_folder_all = f'/Volumes/Extreme SSD/yichun/AutoSort/m1/processed_data/Ephys_{set_day_id_str}/'
    root=save_pth+'generate_input_cmr/'+set_day_id_str+'/test_data/'
    with (open(root + "X_spiketrain_time.pkl", "rb")) as openfile:
        X_spiketrain_time_all = pickle.load(openfile)
    with (open(root + "Y_spike_id_noise.pkl", "rb")) as openfile:
        Y_spiketrain_id_final_all = np.array(pickle.load(openfile))

    y_id_wehave = False

    try: 
        with open(root + "Y_spike_id.pkl", "rb") as openfile:
            Y_spiketrain_id_all = pickle.load(openfile)
            y_id_wehave = True
            print('We have Y_spike_id')
    except FileNotFoundError:
        print('Y_spike_id.pkl not found.') 

    good_ch_ind = np.isin(Y_spiketrain_id_final_all,goodchannel)    
    X_spiketrain_time_all = X_spiketrain_time_all[good_ch_ind]
    Y_spiketrain_id_final_all = Y_spiketrain_id_final_all[good_ch_ind]

    good_ch_ind = np.isin(Y_spiketrain_id_final_all,goodchannel)    
    X_spiketrain_time_all = X_spiketrain_time_all[good_ch_ind]
    Y_spiketrain_id_final_all = Y_spiketrain_id_final_all[good_ch_ind]

    print('###  load raw data')

    recording_concat = spikeinterface.core.base.BaseExtractor.load_from_folder('/Volumes/Extreme SSD/yichun/AutoSort/m1/processed_data/Ephys_'+set_day_id_str+'/')
    trace_step0 = recording_concat.get_traces(segment_index=0)
    
    print_true = False
    data_ch_data=np.array([])

    spike_time_all=[]
    spike_channel_all=[]
    spike_label_all=[]
    gt_label_ind_all=[]

    acc1=[]
    acc2=[]
    acc3=[]
    acc4=[]

    ## this is for the trajectory
    cont_trigger_all_all = np.load(data_folder_all+'cont_trigger_all.npy')  
    cont_trigger_all_all = cont_trigger_all_all.reshape(1,-1)
    cont_trigger_all_all = cont_trigger_all_all[0,:]
    trial_start, trial_end = find_trials(cont_trigger_all_all)

    num_trials = len(trial_start)
    start_time_point=0
    end_time_point = cont_trigger_all_all.shape[0]

    onlinetraj_raster = np.zeros((end_time_point-start_time_point,num_neurons))
    print(onlinetraj_raster.shape)

    # now we start 'online'
    for n in tqdm(range(trace_step0.shape[0]//fs)):
        trace_step0_part = trace_step0[n*fs:(n+1)*fs,:]

        if n==0:
            data_ch_data =trace_step0_part
        else:
            data_ch_data = np.vstack((data_ch_data,trace_step0_part))


        recording_concat = se.NumpyRecording(traces_list=np.array(data_ch_data), sampling_frequency=fs)
        recording_f = bandpass_filter(recording_concat, freq_min=freq_min, freq_max=freq_max)
        
        trace1_car_part = recording_f.get_traces(segment_index=0)

        recording_cmr = common_reference(recording_f, reference='global', operator='average')
        trace0_car_part = recording_cmr.get_traces(segment_index=0)        
        
        spikes = detect_spike(trace0_car_part,thr_min=3,thr_max=30,threshold =2, 
                              distance=1,ch_max_simul_firing=5,
                              wlen=3, prominence=10)

        if n>0:
            ##### extract features    
            X_spiketrain_time = np.where(spikes)[0]
            Y_spiketrain_id = [-1]*X_spiketrain_time.shape[0]
            Y_spiketrain_id_final = np.where(spikes)[1]
            indexind = np.logical_and(X_spiketrain_time < spikes.shape[0]-100,X_spiketrain_time > 100)
            X_spiketrain_time = X_spiketrain_time[indexind]
            Y_spiketrain_id = np.array(Y_spiketrain_id)[indexind]
            Y_spiketrain_id_final = Y_spiketrain_id_final[indexind]

            for time_range in np.arange(-10,20):
                if time_range==-10:
                    waveform = trace0_car_part[X_spiketrain_time+time_range,:]
                else:
                    waveform = np.dstack((waveform, trace0_car_part[X_spiketrain_time+time_range,:] ))

            waveform_single = waveform[np.arange(waveform.shape[0]), Y_spiketrain_id_final.astype('int'), :]
            pred_location = location_cal_group(sensor_positions_all, waveform, Y_spiketrain_id_final)

            #apply trained model
            save_ind, pred_class, _, _ = apply_trained_model(waveform,waveform_single,pred_location)

            k = X_spiketrain_time[save_ind]+fs*n
            try:
                onlinetraj_raster[k, pred_class]=1
            except IndexError:
                print(IndexError)


        if n==0:
            data_ch_data = data_ch_data[-200:]
        else:
            data_ch_data = data_ch_data[-200:]

    idx1 = trial_start>start_interval1
    trial_start=trial_start[idx1]
    trial_end=trial_end[idx1]

    idx2 = trial_end<end_time_point-start_interval2
    trial_start=trial_start[idx2]
    trial_end=trial_end[idx2]

    # Generate some example data
    trial_test=np.zeros((num_trials, num_neurons, num_stimulus))
    time_bin = num_bins
    data=np.zeros((time_bin*(num_trials),onlinetraj_raster.shape[1]))
    data_datahigh=np.zeros((num_trials, num_neurons, int((start_interval1+start_interval2)/10) ))
    trial_num=[]

    for j in np.arange(len(trial_end)):
        trial_test[j,...] = onlinetraj_raster[trial_start[j]-start_interval1:trial_start[j]+start_interval2,:].T
        arr_reduced = block_reduce(trial_test[j], block_size=(1,int(trial_test[j].shape[1]/time_bin)),
                                   func=np.sum, cval=0)
        data[(j)*time_bin:(j+1)*time_bin,:] = arr_reduced.T
        arr_reduced_2 = block_reduce(trial_test[j], block_size=(1,10),
                                   func=np.max, cval=0)        
        data_datahigh[j,:,:] = arr_reduced_2
        trial_num+=[j]*num_bins
    
    trial_num_all.append(trial_num)
    day_num_all.append([set_day_id_str]*len(trial_num))
    day_trial_spike_train.append(data)
    num_trials_all.append(num_trials_all[-1]+num_trials)
    trial_test_all.append(trial_test)


    for neuron_idx in range(num_neurons):
        neuron_data = trial_test[:, neuron_idx, :]

        bin_edges = np.linspace(stimulus_times[0], stimulus_times[-1], num_bins + 1)
        bin_width = np.diff(bin_edges)[0]
        firing_rate_all = np.zeros(num_bins,)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        firing_rates = np.zeros((num_trials, num_bins))  # Store firing rates for each trial to calculate SD later

        for j in range(num_trials):
            peristimulus_raster_i = neuron_data[j, :]
            ind = np.where(peristimulus_raster_i > 0)[0]
            ind = ind / 10000 + stimulus_times[0]
            spikes_per_bin_i, _ = np.histogram(ind, bins=bin_edges)
            firing_rate_i = spikes_per_bin_i / (bin_width)
            firing_rate_all += firing_rate_i
            firing_rates[j, :] = firing_rate_i  # Store firing rate for this trial

        firing_rate = firing_rate_all / num_trials
        firing_rate_sd = np.std(firing_rates, axis=0)  # Calculate SD across trials for this neuron
        firing_rate_day[num_day, neuron_idx, :] = firing_rate
        firing_rate_sd_day[num_day, neuron_idx, :] = firing_rate_sd  # Store SD in the array

selected_neuron = Keep_id #this if all
#selected_neuron = [0,2,3,5,6,9,10,13,14,15,17,18,]
        
data = np.vstack(day_trial_spike_train)
data = data[:,selected_neuron]
data_sma = SMA(data,time_bin)
print(data_sma.shape, data.shape)

scipy.io.savemat(f'online_aml_ablation_{set_day_id_str}_{ch_num}ch_{len(selected_neuron)}_final.mat', {'data_sma': data_sma,'num_trials_all': num_trials_all})


color_palette=sns.color_palette('rainbow',len(day_id_str))
fig, axs = plt.subplots(nrows=int(ceil(num_neurons/3)), ncols=3, figsize=(10,10))
bin_edges = np.linspace(stimulus_times[0], stimulus_times[-1], num_bins+1)
bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

for ax, neuron_idx in zip(axs.flat,
             range(num_neurons)):
    
    #for day_idx in range(len(date_id_all_all)):
    for day_idx in range(len(day_id_str)):
        firing_rate = firing_rate_day[day_idx, neuron_idx, :]
        ax.plot(bin_centers, firing_rate,c=color_palette[day_idx],alpha=0.5)
        
    ax.set_title(f'neuron {neuron_idx}')

plt.subplots_adjust(hspace=0.5)
#plt.savefig('firing_rates_selectedneuron_online.png', format='png', dpi=300)  
plt.show()

list_neuron = selected_neuron
n_neurons = len(list_neuron)
n_rows = 4
n_cols = int(np.ceil(n_neurons / n_rows))
plt.figure(figsize=(n_cols * 3, 20)) # Adjust the figure size as needed

with open('onlinetraj_2024_raster.pickle', 'wb') as file:
    pickle.dump(onlinetraj_raster, file)

with open('onlinetraj_2024_data.pickle', 'wb') as file:
    pickle.dump(data, file)
