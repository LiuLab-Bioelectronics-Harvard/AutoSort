from autosort_neuron import *

# positions=np.array([
#                 [150, 250], ### electrode 1 x,y
#                 [150,200], ### electrode 2 x,y
#                 [50, 0], ### electrode 3 x,y
#                 [50, 50],
#                 [50, 100], 
#                 [0, 100],
#                 [0, 50], 
#                 [0, 0],
#                 [650, 0], 
#                 [650, 50],
#                 [650, 100], 
#                 [600, 100],
#                 [600, 50], 
#                 [600, 0],
#                 [500, 200],
#                 [500, 250],
#                 [500, 300],
#                 [450, 300],
#                 [450, 250], 
#                 [450, 200],
#                 [350, 400], 
#                 [350, 450],
#                 [350, 500], 
#                 [300, 500],
#                 [300, 450], 
#                 [300, 400], 
#                 [200, 200],
#                 [200, 250],
#                 [200, 300],
#                 [150, 300]
#                     ])
# electrode_group=[1, 1, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
# electrode_position=np.hstack([positions,np.array(electrode_group).reshape(-1,1)])

# args=config()
# args.day_id_str=['0305','0415'] ### all days
# args.cluster_path='./AutoSort_data/' ### path of input data
# args.set_time=0  ### set 0411 data as training data
# args.test_time=[1] ### set 0420 data as testing data
# args.group=np.arange(30)  ### all electrodes
# args.samplepoints=10+20#+right_sample ### 30 points for each waveform
# args.sensor_positions_all=electrode_position
# args.mode='test'

# run(args)

day_pth = './processed_data/'
save_pth = './AutoSort_data/'
raw_data_path = './raw_data/'

results_data_path =  f'./AutoSort_data/model_save/train_day0305_0/offline_result/0415/'
extremum_channels_ids_pth='./processed_data/Ephys_concat_0305_0310/mountainsort/extremum_channels_ids.csv'
(sorting,  
 trial_start, 
 trial_end,  
 cont_trigger_all_all,
 recording_cmr) = read_AutoSort_data('0415',
                                     day_pth,
                                     results_data_path, 
                                     save_pth=save_pth,
                                     extremum_channels_ids_pth=extremum_channels_ids_pth
                                     )  