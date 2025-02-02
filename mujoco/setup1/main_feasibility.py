import argparse
from itertools import count
from logger import *
import json
import gym
from matplotlib import pyplot as plt
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
from tensorboardX import SummaryWriter

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
parser.add_argument('--mode', default=None)
parser.add_argument('--discount', type=float, default=0.9)
parser.add_argument('--discount_train', action='store_true')
parser.add_argument('--fixed_train', action='store_true')
parser.add_argument('--algo', default='sac', help='the algorithm of RL')
parser.add_argument('--max_steps', type=int, default=int(1e6), help='the maximum number of steps')
parser.add_argument('--start_training', type=int, default=int(1e4), help='Number of training steps to start training.')
parser.add_argument('--n_domains', type=int, default=5, help='Number of resplit-domains')
args = parser.parse_args()
# re-define datasets
if args.mode == 'resplit':
    print("=> Use re-split datasets x{}".format(args.n_domains))
    args.demo_files = []
    for i in range(args.n_domains):
        args.demo_files.append('../demo/'+ os.path.splitext(args.xml)[0] + '/re_split_{}_batch_00.pkl'.format(i)+args.mode)
    args.test_demo_files = copy.deepcopy(args.demo_files)
elif args.mode == 'random-split':
    print("=> Use random re-split datasets x{}".format(args.n_domains))
    args.demo_files = []
    for i in range(args.n_domains):
        args.demo_files.append('../demo/'+ os.path.splitext(args.xml)[0] + '/re_split_random_{}_batch_00.pkl'.format(i)+args.mode)
    args.test_demo_files = copy.deepcopy(args.demo_files)
elif args.mode == 'resplit-far':
    print("=> Use far re-split datasets x{}".format(args.n_domains))
    args.demo_files = []
    for i in range(args.n_domains):
        args.demo_files.append('../demo/'+ os.path.splitext(args.xml)[0] + '/re_split_far_{}_batch_00.pkl'.format(i))
    args.test_demo_files = copy.deepcopy(args.demo_files)
logger = CompleteLogger('log/'+ args.env_name + '/'+ os.path.splitext(args.xml)[0] + '_' + str(len(args.demo_files))+args.mode)
writer=SummaryWriter(log_dir=logger.root + '/runs')

json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)
def evaluate_pair_between(trajs1, trajs2):
    look_1 = np.expand_dims(np.array(trajs1), axis=0) # 1 * a * 1000 * 18
    look_2 = np.expand_dims(np.array(trajs2), axis=1) # b * 1 * 1000 * 18
    dist_matrix = np.linalg.norm(look_1 - look_2, axis=-1)  # 
    dist_matrix = np.mean(dist_matrix, axis=-1) # b * a
    # dist_matrix[np.arange(dist_matrix.shape[0]), np.arange(dist_matrix.shape[0])] = 10000 
    print('min_between_traj: ', np.min(dist_matrix))
    print('max_between_traj: ', np.max(dist_matrix))

def load_far_pairs(trajs1, trajs2, idx=0, topk=5):
    # look_1 = np.expand_dims(np.expand_dims(trajs1[idx], axis=0)) # 1 * 1 * 1000 * 18
    # look_2 = np.expand_dims(np.array(trajs2), axis=1) # b * 1 * 1000 * 18
    pdb.set_trace()
    dist_matrix = np.linalg.norm(trajs1[idx] - np.array(trajs2), axis=-1)  # 
    dist_matrix = np.mean(dist_matrix, axis=-1) # b * a
    # get max
    max_idx = np.argmax(dist_matrix)
    v, max_idx =  torch.topk(torch.from_numpy(dist_matrix), k=topk)
    print('max_between_traj to {}: {}, id:{} '.format(idx, v, max_idx))
    # get min
    dist_matrix[idx] = 10000.
    min_idx = np.argmin(dist_matrix)
    v, min_idx =  torch.topk(torch.from_numpy(-dist_matrix), k=topk)
    print('min_between_traj to {}: {}, id:{}'.format(idx, v,min_idx))
    # return trajs2[max_idx], trajs2[min_idx] 
    return [trajs2[i] for i in max_idx], [trajs2[i] for i in min_idx]



def load_demos(demo_files, ratio):
    all_demos = []
    all_init_obs = []
    for demo_file in demo_files:
        raw_demos = pickle.load(open(demo_file, 'rb'))
        use_num = int(len(raw_demos['obs'])*ratio)
        all_demos = all_demos + raw_demos['obs'][:use_num]
        if 'init_obs' in raw_demos:
            all_init_obs = all_init_obs + raw_demos['init_obs'][:use_num]
            print("=>set init obs.")
    print("=>load demos size:", len(all_demos))
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

# make demos and test_demos:
if args.mode == 'pair':
    demos = [load_pairs(args.demo_files[i:i+1], args.ratio[0]) for i in range(len(args.test_demo_files))]
elif args.mode == 'test':
    demos = [[], []] # max, min
    # for idx in range(5):
    idx = 0
    demos_single, init_obs_single = load_demos(args.demo_files[0:1], args.ratio[0])
    # norm
    demos_all_normed = (np.array(demos_single) / np.expand_dims(np.linalg.norm(np.array(demos_single),axis=1), axis=1))
    max_demo, min_demo = load_far_pairs(demos_all_normed, demos_all_normed, idx=idx)
    demos[0].append(demos_single[idx])
    demos[1].append(demos_single[idx])
    for i in range(len(max_demo)):
        demos[0].append(max_demo[i])
        demos[1].append(min_demo[i])
    test_demos = copy.deepcopy(demos)
    test_init_obs = [[]] * len(demos)
    args.demo_files *= 2
else:
    demos = []
    init_obs = []
    # pdb.set_trace()
    for i in range(len(args.test_demo_files)):
        demos_single, init_obs_single = load_demos(args.demo_files[i:i+1], args.ratio[0])
        demos.append(demos_single)
        init_obs.append(init_obs_single)
if args.mode != 'test':
    test_demos = []
    test_init_obs = []
    for i in range(len(args.test_demo_files)): # 4
        # pdb.set_trace()
        demos_single, init_obs_single = load_demos(args.test_demo_files[i:i+1], args.ratio[0])
        test_demos.append(demos_single) # 4 * 50 * 1000 * 18
        test_init_obs.append(init_obs_single) # 4 * 0?
# pdb.set_trace()
# for test
    
# make envs
if 'Ant' in args.env_name or 'Swimmer' in args.env_name or 'Walker' in args.env_name or 'HalfCheetah' in args.env_name:
    env_list = [gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos[i]) for i in range(len(args.demo_files))]
    test_env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos[0][0:3]) # demos[0:3]?
elif 'InvertedDoublePendulum' in args.env_name:
    env_list = [gym.make(args.env_name, xml_file=args.xml, demos=demos[i], init_obs=init_obs[i]) for i in range(len(args.demo_files))]
    test_env = gym.make(args.env_name, xml_file=args.xml, demos=demos[0][0:3], init_obs=init_obs[0][0:3])
else:
    env_list = [gym.make(args.env_name, xml_file=args.xml, demos=demos[i]) for i in range(len(args.demo_files))]
    test_env = gym.make(args.env_name, xml_file=args.xml, demos=demos[0][0:3])

num_inputs = env_list[0].observation_space.shape[0] # 18
num_actions = env_list[0].action_space.shape[0] # 6

for i in range(len(env_list)):
    env_list[i].seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)


# Create agents for each demo
agents = []
if args.restore_model is not None:
    model_dict = torch.load(args.restore_model)
    for i in range(len(model_dict)):
        agents[i].append(model_dict['policy_'+str(i)])
else:
    if args.algo == 'sac':
        from configs.sac_default import get_config
        kwargs = dict(get_config())
        algo = kwargs.pop('algo')
        replay_buffer_size = kwargs.pop('replay_buffer_size')
        for i in range(len(demos)):
            agents.append(SACLearner(args.seed,
                          np.concatenate([env_list[0].observation_space.sample()[np.newaxis], env_list[0].observation_space.sample()[np.newaxis]], axis=1),
                          env_list[0].action_space.sample()[np.newaxis], **kwargs))
    elif args.algo == 'awac':
        from configs.awac_default import get_config
        kwargs = dict(get_config())
        algo = kwargs.pop('algo')
        replay_buffer_size = kwargs.pop('replay_buffer_size')
        for i in range(len(demos)):
            agents.append(AWACLearner(args.seed,
                        np.concatenate([env_list[0].observation_space.sample()[np.newaxis], env_list[0].observation_space.sample()[np.newaxis]], axis=1),
                        env_list[0].action_space.sample()[np.newaxis], **kwargs))
    else:
        raise NotImplementedError()

replay_buffer_list = []
best_reward_list = []
save_model_dict = {}
for i in range(len(demos)):
    action_dim = env_list[i].action_space.shape[0]
    # What??
    replay_buffer_list.append(ReplayBuffer(gym.spaces.Box(np.concatenate([env_list[i].observation_space.low, env_list[i].observation_space.low], axis=0),
                                                          np.concatenate([env_list[i].observation_space.high, env_list[i].observation_space.high], axis=0),
                                                          [num_inputs*2]), 
                                           action_dim, replay_buffer_size or args.max_steps)) # capacity
    best_reward_list.append(-10000000)
    save_model_dict['policy_'+str(i)] = None

def evaluate(test_demos, agents, test_env, best_reward_list, num_episode, test_init_obs):
    reward_list = []
    # test_demo_id = 0, 1, 2, 3
    for test_demo_id in range(len(test_demos)):
        all_reward = []
        for ii in range(num_episode):
            test_env.reset()
            if len(test_init_obs[test_demo_id]) > 0:
                state = test_env.set_initial_state(test_demos[test_demo_id][ii], test_init_obs[test_demo_id][ii])
            else:
                state = test_env.set_initial_state(test_demos[test_demo_id][ii])
            state0 = state
            done = False
            test_reward = 0
            step_id = 0
            while not done:
                # sample from where? state==state0
                action = agents[test_demo_id].sample_actions(np.concatenate([state,state0], axis=0), temperature=0.0)
                next_state, reward, done, _ = test_env.step(action)
                if args.mode == 'pair':
                    test_reward += reward
                    state = test_demos[test_demo_id][ii][test_env.step_]
                    test_env.set_observation(state)
                else:
                    test_reward += reward * (args.discount**step_id)
                    state = next_state
                step_id += 1
            # test_reward /= len(test_demos[test_demo_id][ii])
            all_reward.append(test_reward)     
        reward_list.append(np.mean(all_reward))
        print('reward', test_demo_id, ' ', np.mean(all_reward))
        if best_reward_list[test_demo_id] < np.mean(all_reward):
            best_reward_list[test_demo_id] = np.mean(all_reward)
            save_model_dict['policy_'+str(test_demo_id)] = copy.deepcopy(agents[test_demo_id])
        print('best reward', test_demo_id, ' ', best_reward_list[test_demo_id])
    torch.save(save_model_dict, logger.get_checkpoint_path('seed_{}model'.format(args.seed)))
    # torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
    torch.save({'policy_'+str(i):agents[i] for i in range(len(test_demos))}, logger.get_checkpoint_path('seed_{}_instant_model'.format(args.seed)))
    return reward_list
observation_list = []
done_list = []
step_id_list = []

state0_list = []
for j in range(len(demos)):
    observation, done = env_list[j].reset(), False
    observation_list.append(observation)
    done_list.append(done)
    step_id_list.append(0)
    state0_list.append(observation)
reward_eval_all = [[]] * len(args.demo_files)
reward_sum = [0.] * len(args.demo_files)
for i in range(1, args.max_steps + 1):
    # reward_sum = 0
    for j in range(len(demos)):
        # What's start_training for?
        if i < args.start_training:
            action = env_list[j].action_space.sample() # exploration
        else:
            action = agents[j].sample_actions(np.concatenate([observation_list[j], state0_list[j]], axis=0))
        next_observation, reward, done, info = env_list[j].step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0
        if args.discount_train:
            reward_sum[j] += reward * (args.discount**i)
        else:
            reward_sum[j] += reward

        if args.discount_train:
            replay_buffer_list[j].insert(np.concatenate([observation_list[j], state0_list[j]], axis=0), action, reward * (args.discount**step_id_list[j]), mask, np.concatenate([next_observation, state0_list[j]], axis=0))
        else:
            replay_buffer_list[j].insert(np.concatenate([observation_list[j], state0_list[j]], axis=0), action, reward, mask, np.concatenate([next_observation, state0_list[j]], axis=0))
        step_id_list[j] += 1

        observation_list[j] = next_observation
        done_list[j] = done

        if done:
            observation_list[j], done_list[j] = env_list[j].reset(), False # 
            # reward_eval_all[j].append(reward_sum[j])
            reward_sum[j] = 0
            step_id_list[j] = 0
            state0_list[j] = observation_list[j]

        if i >= args.start_training:
            batch = replay_buffer_list[j].sample(args.batch_size)
            update_info = agents[j].update(batch)
    num_episode = 10 if args.mode != 'test' else 1
    if i % args.eval_interval == 0 and i >= args.start_training:
        print("step [{}]:".format(i))
        r = evaluate(test_demos, agents, test_env, best_reward_list, num_episode, test_init_obs)
        for j in range(len(demos)):
            # reward_eval_all[j].append(r[j])
            print('reward_all[{}]: {}'.format(j, r[j]))
            if args.mode == 'test':
                toy = 'far' if j == 0 else 'near'
                writer.add_scalar('reward/toy_{}'.format(toy), r[j], i)
            else:
                writer.add_scalar('reward/clusterID_{}'.format(j), r[j], i)
            # writer.add_scalar('reward/test_fit{}'.format(j), r[j], i)
        # plt.figure(figsize=(6,6), dpi=80)
        # plt.figure(1)
        # ax1 = plt.subplot(231)
        # plt.plot(reward_eval_all[0], color="r",linestyle = "--")
        # ax2 = plt.subplot(232)
        # plt.plot(reward_eval_all[1],color="y",linestyle = "--")
        # ax3 = plt.subplot(233)
        # plt.plot(reward_eval_all[2],color="g",linestyle = "--")
        # ax4 = plt.subplot(234)
        # plt.plot(reward_eval_all[3],color="b",linestyle = "--")
        # ax5 = plt.subplot(235)
        # plt.plot(reward_eval_all[4],color="r",linestyle = "--")
        # plt.savefig(logger.get_image_path("reward_curve.png"))
logger.close()