import argparse
from ast import Global
from dis import dis
from glob import glob
from itertools import count
from math import dist

import gym
from matplotlib.pyplot import axis
import scipy.optimize

import pdb
import torch
from torch.autograd import Variable

from jax_rl.agents import AWACLearner, SACLearner
from jax_rl.datasets import ReplayBuffer
from jax_rl.evaluation import evaluate
from jax_rl.utils import make_env

import numpy as np

import pickle
import random
import copy

# import ant
# import swimmer
# import reacher
# import walker
# import halfcheetah
# import inverted_double_pendulum
import sys
sys.path.append('../all_envs')
import swimmer
import walker
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save_path', type=str, default= 'temp', metavar='N',
                    help='path to save demonstrations on')
parser.add_argument('--xml', type=str, default= None, metavar='N',
                    help='For diffent dynamics')
parser.add_argument('--demo_files', nargs='+')
parser.add_argument('--test_demo_files', nargs='+')
parser.add_argument('--ratio', type=float, nargs='+')
parser.add_argument('--eval-interval', type=int, default=1000)
parser.add_argument('--restore_model', default=None)
parser.add_argument('--mode')
parser.add_argument('--discount', type=float, default=0.9)
parser.add_argument('--discount_train', action='store_true')
parser.add_argument('--fixed_train', action='store_true')
parser.add_argument('--algo', default='sac', help='the algorithm of RL')
parser.add_argument('--max_steps', type=int, default=int(1e6), help='the maximum number of steps')
parser.add_argument('--start_training', type=int, default=int(1e4), help='Number of training steps to start training.')
args = parser.parse_args()

def load_demos(demo_files, ratio):
    all_demos = []
    all_init_obs = []
    for demo_file in demo_files:
        raw_demos = pickle.load(open(demo_file, 'rb'))
        # use fix ratio for every domain
        use_num = int(len(raw_demos['obs'])*ratio[0])
        all_demos = all_demos + raw_demos['obs'][:use_num]
        if 'init_obs' in raw_demos:
            all_init_obs = all_init_obs + raw_demos['init_obs'][:use_num]
    return all_demos, all_init_obs

def load_pairs(demo_files, ratio):
    all_pairs = []
    for demo_file in demo_files:
        raw_demos = pickle.load(open(demo_file, 'rb'))
        for i in range(int(len(raw_demos['obs'])*ratio)):
            obs = np.array(raw_demos['obs'][i])
            next_obs = np.array(raw_demos['next_obs'][i])
            all_pairs.append(np.reshape(np.concatenate([obs, next_obs], axis=1), (obs.shape[0], 2, -1)))
    return np.concatenate(all_pairs, axis=0)

global iters

def Max_Clique(t, size):
    global bestx #最大团节点编号
    global bestn #最大团节点数
    global iters
    if iters > 9000000: # can be seen as 收敛
        return
    iters += 1
    if t >= size:
        sum = 0
        for value in x:
            sum = sum + value
        if sum > bestn:
            bestx = copy.deepcopy(x)
            bestn = sum
            print("iters: ", iters)
            iters = 0
            # pdb.set_trace()
            # if bestn > int(0.5 * size):
            #     return
    else:
        #判断与当前节点是否形成团
        flag = 1
        for i in range(0,t):
            if x[i] == 1 and graph_matrix[t][i] == 0:
                flag = 0
        if flag == 0:# 不形成团，则判断下一个节点
            x[t] = 0
            Max_Clique(t+1, size)
        else:        # 形成团，则依次判断该节点加入/不加入  
            for j in range(0,2):
                x[t] = j
                Max_Clique(t+1, size)
def update_graph(G, nodes_to_delete):
    # node_id = 0
    # for exist in nodes_to_delete:
    #     if exist == 1:
    #         G[:, node_id] = 0 # clear all edges of node node_id
    #         G[node_id, :] = 0 # clear all edges of node node_id
    #     node_id += 1
    for node_id in nodes_to_delete:
        G[:, node_id] = 0 # clear all edges of node node_id
        G[node_id, :] = 0 # clear all edges of node node_id
    # G[:, nodes_to_delete] = 0 # clear all edges of node node_id
    # G[nodes_to_delete, :] = 0 # clear all edges of node node_id

def is_all_clear(Graph):
    return np.count_nonzero(Graph) == 0

def greedy_independ(Graph):
    global x #最大团节点编号
    global bestn #最大团节点数
    global bestx # list
    degree_mat = np.sum(Graph, axis=-1)
    degree_mat = np.where(degree_mat==0, 10000, degree_mat)
    x = degree_mat.argmin()
    bestn += 1
    bestx[x] = 1
# if __name__ == '__main__':
#     #输入一个图，用二维数组存储
#     #输入节点数量
#     print('图中节点个数为：')
#     n = 4
#     G_list = [[1,1,1,0], [1,1,1,0], [1,1,1,1], [0,0,1,1]]
#     print('图的邻接矩阵为：')
#     # for i in range(n):
#     #     G_list.append(input().split(','))
#     x = [0 for i in range(n)]
#     G_list = np.array(G_list)
#     while(not is_all_clear(G_list)):
#         print(G_list)
#         global bestn
#         bestn = 0
#         Max_Clique(0)
#         print(bestx,bestn)
#         pdb.set_trace()
#         update_graph(G=G_list, nodes_to_delete=bestx)

# def re_split_demos(demos_all):
#     size = len(demos_all)
#     traj_len = len(demos_all[0])
#     pdb.set_trace()
#     dist_matrix = np.zeros((size, size)) # 200 * 200
#     look_1 = np.expand_dims(np.array(demos_all), axis=0) # 1 * 200 * 1000 * 18
#     look_2 = np.expand_dims(np.array(demos_all), axis=1) # 200 * 1 * 1000 * 18
#     dist_matrix = np.sum(abs(look_1 - look_2), axis=-1)  # 200 * 200 * 1000
#     # dist_matrix = np.linalg.norm(look_1 - look_2, axis=-1)
#     dist_matrix = np.mean(dist_matrix, dim=-1)
#     # for i in range(size):
#     #     for j in range(size):
#     #         dist_matrix[i][j] = calculate_traj_dist(demos_all[i], demos_all[j])
#     global graph_matrix
#     # # clique
#     # graph_matrix = dist_matrix < (dist_matrix.mean() * 1.1)
#     # independent
#     graph_matrix = dist_matrix > (dist_matrix.mean() * 0.9)
#     print("sample graph:", graph_matrix[0])
#     graph_matrix = graph_matrix.astype(int)
#     split_done = False
#     split_clique=[]
#     while(not split_done):
#         global x
#         # print(G_list)
#         global bestn
#         global iters
#         x = [0 for i in range(size)]
#         bestn = 0
#         iters = 0
#         # pdb.set_trace()
#         Max_Clique(0, size=size)
#         print(bestx, bestn)
#         update_graph(G=graph_matrix, nodes_to_delete=bestx)
#         # pdb.set_trace()
#         clique = [i for i, x in enumerate(bestx) if x == 1]
#         if len(clique) > int(0.1 * size):
#             split_clique.append(clique)
#         split_done = is_all_clear(graph_matrix)
#     print('re_cluster id:', split_clique)
#     pdb.set_trace()
#     # save new demo clique
#     raw_demos = {}
#     for i in range(len(split_clique)):
#         save_demo_path = '../demo/walker2d/re_split_{}_batch_00.pkl'.format(i)
#         raw_demos['obs'] = [demos_all[idx] for idx in split_clique[i]]
#         pickle.dump(raw_demos, open(save_demo_path, 'wb'))

def re_split_demos(demos_all):
    size = len(demos_all)
    traj_len = len(demos_all[0])
    # pdb.set_trace()
    dist_matrix = np.zeros((size, size)) # 200 * 200
    look_1 = np.expand_dims(np.array(demos_all), axis=0) # 1 * 200 * 1000 * 18
    look_2 = np.expand_dims(np.array(demos_all), axis=1) # 200 * 1 * 1000 * 18
    # dist_matrix = np.sum(abs(look_1 - look_2), axis=-1)  # 200 * 200 * 1000
    dist_matrix = np.linalg.norm(look_1 - look_2, axis=-1)
    dist_matrix = np.mean(dist_matrix, axis=-1)
    save_dist_path = '../demo/walker2d/dist_matrix.pkl'
    pickle.dump(dist_matrix, open(save_dist_path, 'wb'))
   # for i in range(size):
    #     for j in range(size):
    #         dist_matrix[i][j] = calculate_traj_dist(demos_all[i], demos_all[j])
    global graph_matrix
    # # clique
    # graph_matrix = dist_matrix < (dist_matrix.mean() * 1.1)
    # independent
    graph_matrix = dist_matrix > (dist_matrix.mean() * 1.)
    print("sample graph:", graph_matrix[0])
    graph_matrix = graph_matrix.astype(int)
    split_clique=[]
    global bestx
    bestx = [0 for i in range(size)]
    all_clear = False
    # remember the true graph
    graph_memory = copy.deepcopy(graph_matrix)
    decay_step = 0
    while(not all_clear):
        decay_step += 1
        bestx = [0 for i in range(size)]
        graph_matrix = copy.deepcopy(graph_memory)
        split_done = False
        # look for independent set for one time
        while(not split_done):
            global x
            # print(G_list)
            global bestn
            global iters
            bestn = 0
            # pdb.set_trace()
            # Max_Clique(0, size=size)
            greedy_independ(graph_matrix) # set bestx = bestx U x
            update_graph(G=graph_matrix, nodes_to_delete=find_neighbor_x(node_id=x, Graph=graph_matrix))
            split_done = is_all_clear(graph_matrix)
        # print(bestx, bestn)
        # find one independent set(approximately)
        clique = [i for i, x in enumerate(bestx) if x == 1]
        if len(clique) > int(0.05 * size):
            split_clique.append(clique)
        print('re_cluster id:', split_clique)
        # to contain more nodes
        graph_memory = (dist_matrix > (dist_matrix.mean() * (1. + 0.1 * decay_step))).astype(int)
        update_graph(G=graph_memory, nodes_to_delete=[x for clique in split_clique for x in clique])
        # check if all the demos have been selected
        all_clear = is_all_clear(graph_memory)
        # pdb.set_trace()

    # pdb.set_trace()
    # save new demo clique
    raw_demos = {}
    for i in range(len(split_clique)):
        save_demo_path = '../demo/walker2d/re_split_{}_batch_00.pkl'.format(i)
        raw_demos['obs'] = [demos_all[idx] for idx in split_clique[i]]
        pickle.dump(raw_demos, open(save_demo_path, 'wb'))

# return a list of neigjbor of x (including self)
def find_neighbor_x(node_id, Graph):
    neighbor_list = [node_id]
    for i in range(len(Graph)):
        if Graph[node_id][i] == 1:
            neighbor_list.append(i)
    return neighbor_list

def calculate_traj_dist(traj1, traj2):
    steps = len(traj1)
    assert steps == len(traj2)
    diff = np.zeros(steps)
    for i in range(steps):
        diff[i] = abs(np.linalg.norm(traj1[i] - traj2[i], ord=2))
    return np.min(diff)
# main
if args.mode == 'pair':
    demos = [load_pairs(args.demo_files[i:i+1], args.ratio[i]) for i in range(len(args.test_demo_files))]
elif args.mode == 'traj':
    # load all demos
    demos_all, init_obs_all = load_demos(args.demo_files, args.ratio)
test_demos = []
test_init_obs = []

# clean dataset
not_expert = []
for i in range(len(demos_all)):
    if len(demos_all[i]) < 1000:
        not_expert.append(i) # not expert traj?
    if i % 5 == 0:
        print("len demos {}:{}".format(i, len(demos_all[i])))
# pdb.set_trace()
for i in reversed(not_expert):
    del demos_all[i]
re_split_demos(demos_all=demos_all)

for i in range(len(args.test_demo_files)):
    demos_single, init_obs_single = load_demos(args.test_demo_files[i:i+1], args.ratio)
    test_demos.append(demos_single) # 4 * 50 * 1000 * 18
    test_init_obs.append(init_obs_single) # 4 * 0?
pdb.set_trace()
