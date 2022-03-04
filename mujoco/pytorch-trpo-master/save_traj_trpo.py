# import free_mjc
import pdb
import sys
from logger import *
import json
sys.path.append('../all_envs')
import swimmer
import walker
from running_state import ZFilter

# import envs.swimmer
# import envs.ant
# import envs.params.swimmer
# import envs.params.hopper
# import envs.params.half_cheetah
# import envs.params.walker2d
# from utils import *
from itertools import count
import argparse
import gym
import os
import sys
import pickle
import imageio
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import numpy as np
from models import Policy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-episode', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
parser.add_argument('--dump', default=False, action='store_true')
parser.add_argument('--xml', type=str, default='', metavar='N',
                    help='xml of env')
args = parser.parse_args()
logger = CompleteLogger('log/'+ args.env_name + '/'+ os.path.splitext(args.xml)[0] + '_save_traj')
json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)

# creat envs
dtype = torch.float32
torch.set_default_dtype(dtype)
env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False)
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print('state dim:', state_dim)
print('action dim:', num_actions)

# load models
save_demo_dir = '../demo/' + os.path.splitext(args.xml)[0]
save_demo_path = '../demo/' + os.path.splitext(args.xml)[0]  + '/batch_00_test.pkl'
if not os.path.exists(save_demo_dir):
    os.mkdir(save_demo_dir)
save_path = '../checkpoints/' + os.path.splitext(args.xml)[0]  + '_model.pth'
policy_net = Policy(num_inputs, num_actions)
# pdb.set_trace()
# policy_net.to(dtype)
state_dict =  torch.load(save_path, map_location='cpu')
policy_net.load_state_dict(state_dict)

# using running state
print('use running state')
# policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
# running_state = ZFilter((num_inputs,), clip=5)
running_state = torch.load("../checkpoints/running_state")
raw_demos = {}
def main_loop():

    num_steps = 0
    
    raw_demos['obs'] = []
    raw_demos['next_obs'] = []
    for i_episode in count():
        expert_traj = []
        # expert_traj_init = []
        state = env.reset()
        # state = torch.tensor(running_state(state, update=False)).unsqueeze(0).to(dtype)
        reward_episode = 0
        episode_steps = 0

        rewards = []
        frames = []
        reward_sum = 0.
        for t in trange(10000):
            # state_var = torch.tensor(running_state(state, update=False)).unsqueeze(0).to(dtype)
            # # choose mean action
            # action = policy_net(state_var)[0][0].detach().numpy()
            # # choose stochastic action
            # # action = policy_net.select_action(state_var)[0].cpu().numpy()
            # action = int(action) if is_disc_action else action.astype(np.float64)

            # action_mean, _, action_std = policy_net(next_state)
            # action = torch.normal(action_mean, action_std)

            action = policy_net(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
            action = (torch.normal(action[0], action[2])).detach().numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state, update=False)
            reward_episode += reward
            rewards.append(reward)
            # num_steps += 1
            episode_steps += 1

            expert_traj.append(running_state.reverse_norm_state(state))
            # expert_traj.append(np.hstack([state[2:], action])) # special for custom swimmer
            # pdb.set_trace()
            if args.render:
                frames.append(env.render(mode='rgb_array', height=256, width=256))
            if done:
                # print("rewards: ", reward_episode)
                if episode_steps >= 1000: # 1000 steps -> done
                    # expert_traj = np.stack(expert_traj)
                    print("total steps: ",episode_steps)
                    print("save traj[{}] with rewards: {}".format(len(raw_demos['obs']), reward_episode))
                    raw_demos['obs'].append(expert_traj)
                    raw_demos['next_obs'].append(expert_traj[1:] + next_state)
                    num_steps += episode_steps
                    if len(raw_demos['obs']) == 500:
                        return
                break

            state = next_state
        # raw_demos['obs'].append(expert_traj)

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))
        if args.render:
            imageio.mimsave(f'demo_{i_episode}.mp4', frames, fps=120)
            plt.clf()
            plt.plot(rewards)
            plt.savefig(f'demo_{i_episode}.png')
        # raw_demos['obs'].append(expert_traj)

        if i_episode >= args.max_expert_episode:
            break


main_loop()
if args.dump:
    pickle.dump(raw_demos,
                open(save_demo_path, 'wb'))
