import argparse
import numpy as np


parser = argparse.ArgumentParser(description='AutoSort')

parser.add_argument('--day_id_str', type=list, default=['0411','0416'], help='day id every day.')
parser.add_argument('--group', type=np.array, default=np.arange(30), help='keep channels')
parser.add_argument('--samplepoints', type=int, default=30, help='samplepoints')
parser.add_argument('--cluster_path', type=str, default='./AutoSort_data/',
                    help='cluster path.')
parser.add_argument('--seed_all', type=int, default=0, help='seed')
parser.add_argument('--set_time', type=int, default=0, help='set_time')
parser.add_argument('--test_time', type=list, default=[1], help='test_time')

args = parser.parse_args()


