import math
import scipy
import numpy as np
from tqdm import tqdm

from tqdm import tqdm
from skimage.measure import block_reduce
import scipy
import matplotlib.pyplot as plt



class FSM2:
    def __init__(self, k, d, tau0=None, Minv0=None, W0=None, learning_rate=None):
        self.k = k
        self.d = d
        if W0 is None:
            self.W = np.eye(k, d) / d
        else:
            self.W = W0;
        self.M = np.eye(k)
        if Minv0 is None:
            self.Minv = np.eye(k)
        else:
            self.Minv = Minv0
        if tau0 is None:
            self.tau = 0.5
        else:
            self.tau = tau0
        if learning_rate is None:
            self.lr = lambda x: 1.0 / (2 * (x + 100) + 5)
        else:
            self.lr = lambda x: learning_rate

        self.t = 0
        self.outer_W = np.zeros(np.shape(self.W))
        self.outer_Minv = np.zeros(np.shape(self.Minv))
        self.y = []

    def fit_next(self, x):
        step = self.lr(self.t)
        #         print('step',step)

        z = np.dot(self.W, x.T)
        y = np.zeros(np.shape(z))
        #         print('z',z)
        #         print('y',y)

        # The loop below solves the linear system My = Wx
        y[0] = z[0] / self.M[0, 0]
        #         print('y[0]',y[0])
        for i in np.arange(1, self.k):
            y[i] = z[i]
            #             print('i',i,'y[i]',y[i])
            for j in np.arange(0, i):
                y[i] = y[i] - y[j] * self.M[i, j]
            #                 print('i',i,'j',j,'y[i]',y[i])

            y[i] = y[i] / self.M[i, i]
        #             print('i',i,'y[i]',y[i])

        y_up = np.tril(
            np.dot(y.reshape(y.shape[0], 1), y.reshape(y.shape[0], 1).T))  # extract lower triangular component of y
        #         print('y_up',y_up)
        self.W = self.W + step * (y.reshape(y.shape[0], 1) * x.reshape(x.shape[0], 1).T - self.W)  # update W matrix
        #         print('W',self.W)
        self.M = self.M + step * (y_up - self.M)  # update M matrix
        #         print('M',self.M)
        self.t = self.t + 1;
        #         print('t',self.t)
        return y


def offline_smoother(yIn, kernSD, stepSize):
    causal = False;

    if (kernSD == 0) or (yIn.shape[1]==1):
        yOut = yIn
        return yOut


    # Filter half length
    # Go 3 standard deviations out
    fltHL = math.ceil(3 * kernSD / stepSize)

    # Length of flt is 2*fltHL + 1
    flt = scipy.stats.norm.pdf(np.arange(-fltHL*stepSize ,fltHL*stepSize,stepSize), loc=0, scale = kernSD)


    [yDim, T] = np.shape(yIn)
    yOut      = np.zeros((yDim, T))

    nm = scipy.signal.convolve(flt.reshape(1,-1), np.ones((1,T)), mode='full', method='auto')

    for i in range(yDim):

        ys = np.divide(scipy.signal.convolve(flt.reshape(1,-1),yIn[i,:].reshape(1,-1)) , nm)
    #     # Cut off edges so that result of convolution is same length
    #     # as original data
        yOut[i,:] = ys[0,fltHL:ys.shape[1]-fltHL+1]
    return yOut

    
def SMA(data, time_bin,kernSD_coef=0.52, stepSize_coef=0.15):
    N = data.shape[0]
    D = data.shape[1]
    k=3;
    scal = 1;


    a = -scal
    b = scal
    W0 = a + (b-a)* np.eye(k,D)
    W0 = W0 / scal

    W_hist = np.zeros((k, D))
    M_hist = np.zeros((k, k))
    y_hist = np.zeros((k, N))

    ### apply FSM
    fsm = FSM2(k=k, d=D, W0 =W0, learning_rate = 0.01)
    for i in tqdm(range(N)):
        bb=data[i,:]
        y_hist[:,i] = fsm.fit_next(bb);

    kernSD = time_bin*kernSD_coef;
    stepSize = time_bin*stepSize_coef


    ### smooth
    data_sma = np.zeros((3,time_bin,int(y_hist.shape[1]/time_bin)))
    for j in np.arange(math.floor(y_hist.shape[1]/time_bin)):
        data_sma[:,:,int(j)] = offline_smoother(y_hist[:,int(j*time_bin):int((j+1)*time_bin)], kernSD, stepSize)
    return data_sma


def extract_trial_data(trial_start, 
                       trial_end, 
                       onlinetraj_raster, 
                       start_interval1, 
                       start_interval2,
                       num_neurons,
                       num_bins):
    num_trials = len(trial_start)
    num_stimulus=start_interval1+start_interval2
    trial_test=np.zeros((num_trials, num_neurons, num_stimulus))#len(trial_start)
    time_bin = num_bins
    data=np.zeros((time_bin*(num_trials),onlinetraj_raster.shape[1]))
    for j in np.arange(len(trial_end)):
        trial_test[j,...] = onlinetraj_raster[trial_start[j]-start_interval1:trial_start[j]+start_interval2,:].T
        arr_reduced = block_reduce(trial_test[j], block_size=(1,int(trial_test[j].shape[1]/time_bin)),
                                    func=np.sum, cval=0)
        data[(j)*time_bin:(j+1)*time_bin,:] = arr_reduced.T
    return data,trial_test
            
    # num_trials = len(trial_start)
    # trial_test=np.zeros((num_trials, num_neurons, num_stimulus))#len(trial_start)
    # time_bin = num_bins
    # data=np.zeros((time_bin*(num_trials),onlinetraj_raster.shape[1]))
    # data_datahigh=np.zeros((num_trials, num_neurons, int((start_interval1+start_interval2)/10) ))
    # trial_num=[]
    # for j in np.arange(len(trial_end)):
    #     trial_test[j,...] = onlinetraj_raster[trial_start[j]-start_interval1:trial_start[j]+start_interval2,:].T
    #     arr_reduced = block_reduce(trial_test[j], block_size=(1,int(trial_test[j].shape[1]/time_bin)),
    #                                 func=np.sum, cval=0)
    #     data[(j)*time_bin:(j+1)*time_bin,:] = arr_reduced.T
    #     arr_reduced_2 = block_reduce(trial_test[j], block_size=(1,10),
    #                                 func=np.max, cval=0)        
    #     data_datahigh[j,:,:] = arr_reduced_2
    #     trial_num+=[j]*num_bins


def plot_neuron_spike_train(spike_train, 
                            num_trials, 
                            num_neurons, 
                            num_bins, 
                            start_interval1, 
                            start_interval2):
    stimulus_times=[-start_interval1/10000, start_interval2/10000]
    fig, axs = plt.subplots(nrows=4, ncols=int(np.ceil(num_neurons/4)), figsize=(10, 10), sharey='row')
    for ax, neuron_idx in zip(axs.flat,
                     range(num_neurons)):
        neuron_data = spike_train[:, neuron_idx, :]
        bin_edges = np.linspace(stimulus_times[0], stimulus_times[-1], num_bins+1)

        for i in range(num_trials):
            ind = np.where(neuron_data[i,:])[0]
            ax.eventplot(ind,lineoffsets=i,linewidths=0.5, colors='black')
            ax.set_ylabel('Trial')
            ax.set_ylim(-0.5, num_trials + 0.5)
            ax.set_title(f'Neuron {neuron_idx+1}')
            ax.set_xticks([0,start_interval1,start_interval1+15000,start_interval1+start_interval2],
                          [stimulus_times[0], 0,(start_interval1+15000)/10000-1, stimulus_times[-1]])
    plt.subplots_adjust(hspace=0.5)    