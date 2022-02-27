from logger import *
import json
from matplotlib import pyplot as plt
import argparse
from itertools import count

import gym
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
logger = CompleteLogger('log/'+ args.env_name + '/'+ os.path.splitext(args.xml)[0] + '_pretrain')
json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)
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



def evaluate(test_demos, agents, test_env, best_reward_list, num_episode, test_init_obs):
    # test_demo_id = 0, 1, 2, 3
    # if not isinstance(agents, list):
    # agents = [agents] * len(test_demos)
    # pdb.set_trace()
    domain_reward = np.zeros(len(test_demos))
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
                elif args.mode == 'traj':
                    test_reward += reward * (args.discount**step_id)
                    state = next_state
                step_id += 1
            # test_reward /= len(test_demos[test_demo_id][ii])
            all_reward.append(test_reward)       
        print('reward', test_demo_id, ' ', np.mean(all_reward))
        domain_reward[test_demo_id] = np.mean(all_reward)
    # use average reward on all domians
    if sum(best_reward_list) / len(test_demos) < np.mean(domain_reward):
        for i in range(len(test_demos)):
            best_reward_list[i] = domain_reward[i]
            save_model_dict['policy_'+str(i)] = copy.deepcopy(agents[i])
        print('best reward', i, ' ', best_reward_list[i])
        torch.save(save_model_dict, logger.get_checkpoint_path('seed_{}_pretrain_model'.format(args.seed)))
    torch.save({'policy_'+str(i):agents[i] for i in range(len(test_demos))}, logger.get_checkpoint_path('seed_{}_pretrain_instant_model'.format(args.seed)))
    return domain_reward

def re_split_demos(demos_all):
    pass

# main
if args.mode == 'pair':
    demos = [load_pairs(args.demo_files[i:i+1], args.ratio[i]) for i in range(len(args.test_demo_files))]
elif args.mode == 'traj':
    # load all demos
    demos_all, init_obs_all = load_demos(args.demo_files, args.ratio)
test_demos = []
test_init_obs = []
for i in range(len(args.test_demo_files)):
    demos_single, init_obs_single = load_demos(args.test_demo_files[i:i+1], args.ratio)
    test_demos.append(demos_single) # 4 * 50 * 1000 * 18
    test_init_obs.append(init_obs_single) # 4 * 0?
# pdb.set_trace()
if 'Ant' in args.env_name or 'Swimmer' in args.env_name or 'Walker' in args.env_name or 'HalfCheetah' in args.env_name:
    env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos_all)
    # env_list = [gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos[i]) for i in range(len(args.demo_files))]
    test_env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos_all[0:3]) # demos[0:3]?
# elif 'InvertedDoublePendulum' in args.env_name:
#     env_list = [gym.make(args.env_name, xml_file=args.xml, demos=demos[i], init_obs=init_obs[i]) for i in range(len(args.demo_files))]
#     test_env = gym.make(args.env_name, xml_file=args.xml, demos=demos[0][0:3], init_obs=init_obs[0][0:3])
else:
    env = gym.make(args.env_name, xml_file=args.xml, demos=demos_all)
    test_env = gym.make(args.env_name, xml_file=args.xml, demos=demos[0:3])

num_inputs = env.observation_space.shape[0] # 18
num_actions = env.action_space.shape[0] # 6

# for i in range(len(env_list)):
#     env.seed(args.seed)
env.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)


# Create single agent for all_demo
if args.restore_model is not None:
    model_dict = torch.load(args.restore_model)
    agent = (model_dict['policy_'+str(i)])
else:
    if args.algo == 'sac':
        from configs.sac_default import get_config
        kwargs = dict(get_config())
        algo = kwargs.pop('algo')
        replay_buffer_size = kwargs.pop('replay_buffer_size')
        agent = SACLearner(args.seed,
                    np.concatenate([env.observation_space.sample()[np.newaxis], env.observation_space.sample()[np.newaxis]], axis=1),
                    env.action_space.sample()[np.newaxis], **kwargs)
    elif args.algo == 'awac':
        from configs.awac_default import get_config
        kwargs = dict(get_config())
        algo = kwargs.pop('algo')
        replay_buffer_size = kwargs.pop('replay_buffer_size')
        agent = AWACLearner(args.seed,
                    np.concatenate([env.observation_space.sample()[np.newaxis], env.observation_space.sample()[np.newaxis]], axis=1),
                    env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

replay_buffer_list = []
best_reward_list = []
for i in range(len(test_demos)):
    best_reward_list.append(-10000000)
save_model_dict = {}
action_dim = env.action_space.shape[0]
print("action dim:", action_dim)
# What??
replay_buffer = ReplayBuffer(gym.spaces.Box(np.concatenate([env.observation_space.low, env.observation_space.low], axis=0),
                                                    np.concatenate([env.observation_space.high, env.observation_space.high], axis=0),
                                                    [num_inputs*2]), 
                                    action_dim, replay_buffer_size or args.max_steps) # capacity
# best_reward = -10000000.
save_model_dict['policy_'+str(i)] = None


observation_list = []
done_list = []
step_id_list = []

state0_list = []
observation, done = env.reset(), False
step_id = 0
state0 = observation
reward_eval_all = [[]] * len(test_demos)
for i in range(1, args.max_steps + 1):
    # What's start_training for?
    if i < args.start_training:
        action = env.action_space.sample() # exploration
    else:
        action = agent.sample_actions(np.concatenate([observation, state0], axis=0))
    next_observation, reward, done, info = env.step(action)

    if not done or 'TimeLimit.truncated' in info:
        mask = 1.0
    else:
        mask = 0.0

    if args.discount_train:
        replay_buffer.insert(np.concatenate([observation, state0], axis=0), action, reward * (args.discount**step_id), mask, np.concatenate([next_observation, state0], axis=0))
    else:
        replay_buffer.insert(np.concatenate([observation, state0], axis=0), action, reward, mask, np.concatenate([next_observation, state0], axis=0))
    step_id += 1

    observation = next_observation
    done = done

    if done:
        observation, done = env.reset(), False # 
        step_id = 0
        state0 = observation

    if i >= args.start_training:
        batch = replay_buffer.sample(args.batch_size)
        update_info = agent.update(batch)

    # if i % args.eval_interval == 0 and i >= args.start_training:
    #     reward_eval = evaluate(test_demos, [agent] * len(test_demos), test_env, best_reward_list, 10, test_init_obs)
    #     reward_eval_all.append(reward_eval)
    if i % (args.eval_interval ) == 0 and i >= args.start_training:
        # plt.plot(reward_eval_all)
        # plt.savefig(logger.get_image_path("pretrain_reward.png"))
        print("step [{}]:".format(i))
        r = evaluate(test_demos, [agent] * len(test_demos), test_env, best_reward_list, 10, test_init_obs)
        for i in range(len(test_demos)):
            reward_eval_all[i].append(r[i])
        print('reward_all[0]: ',reward_eval_all[0])
        plt.figure(figsize=(6,6), dpi=80)
        plt.figure(1)
        ax1 = plt.subplot(221)
        plt.plot(reward_eval_all[0], color="r",linestyle = "--")
        ax2 = plt.subplot(222)
        plt.plot(reward_eval_all[1],color="y",linestyle = "-")
        ax3 = plt.subplot(223)
        plt.plot(reward_eval_all[2],color="g",linestyle = "-.")
        ax4 = plt.subplot(224)
        plt.plot(reward_eval_all[3],color="b",linestyle = ":")
        plt.savefig(logger.get_image_path("reward_curve.png"))
logger.close()