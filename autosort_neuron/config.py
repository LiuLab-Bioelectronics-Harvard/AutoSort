import argparse
import numpy as np

# parser = argparse.ArgumentParser(description='AutoSort')

# parser.add_argument('--day_id_str', type=list, default=['0411','0416'], help='day id every day.')
# parser.add_argument('--shank_id', type=list, default= [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20]], help='neuron id')
# parser.add_argument('--group', type=np.array, default=np.arange(30), help='keep channels')

# parser.add_argument('--samplepoints', type=int, default=30, help='samplepoints')
# parser.add_argument('--cluster_path', type=str, default='./AutoSort_data/',
#                     help='cluster path.')
# parser.add_argument('--seed_all', type=int, default=0, help='seed')
# parser.add_argument('--set_time', type=int, default=0, help='set_time')
# parser.add_argument('--test_time', type=int, default=1, help='test_time')
# parser.add_argument('--load_model', type=str, default='1', help='load_model')

# args = parser.parse_args()

class config:
    def __init__(self):
        self.day_id_str=['0411','0416']
        self.shank_id=[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20]]
        self.group=np.arange(30)
        self.samplepoints=30
        self.cluster_path='./AutoSort_data/'
        self.seed_all=0
        self.set_time=0
        self.test_time=1
        self.load_model='1'
        self.mode='train'
    
    

