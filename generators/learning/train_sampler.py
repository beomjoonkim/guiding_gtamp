import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
import random

from AdMon import AdversarialMonteCarlo
from Qmse import Qmse


def load_data(traj_dir):
    traj_files = os.listdir(traj_dir)
    cache_file_name = 'cache.pkl'
    if os.path.isfile(traj_dir + cache_file_name):
        return pickle.load(open(traj_dir + cache_file_name, 'r'))

    all_states = []
    all_actions = []
    all_sum_rewards = []
    for traj_file in traj_files:
        if 'pidx' not in traj_file: continue
        traj = pickle.load(open(traj_dir + traj_file, 'r'))
        if len(traj.states) == 0:
            continue
        states = np.array([s.state_vec for s in traj.states])
        actions = [a[1] for a in traj.actions]
        rewards = traj.rewards
        sum_rewards = np.array([np.sum(traj.rewards[t:]) for t in range(len(rewards))])

        all_states.append(states)
        all_actions.append(actions)
        all_sum_rewards.append(sum_rewards)

    all_states = np.vstack(all_states).squeeze(axis=1)
    all_actions = np.vstack(all_actions)
    all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
    pickle.dump((all_states, all_actions, all_sum_rewards), open(traj_dir + cache_file_name, 'wb'))
    return all_states, all_actions, all_sum_rewards[:, None]


def train_admon(config):
    # Loads the processed data
    states, actions, sum_rewards = load_data('./planning_experience/processed/domain_two_arm_mover/'
                                             'n_objs_pack_1/irsc/sampler_trajectory_data/')
    savedir = './generators/learning/learned_weights/'
    n_key_configs = 618
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = actions.shape[1]
    admon = AdversarialMonteCarlo(dim_action=dim_action, dim_state=dim_state, save_folder=savedir, tau=config.tau,
                                  explr_const=0.0)
    admon.train(states, actions, sum_rewards)


def train_mse(config):
    # Loads the processed data
    states, actions, sum_rewards = load_data('./planning_experience/processed/domain_two_arm_mover/'
                                             'n_objs_pack_1/irsc/sampler_trajectory_data/')
    savedir = './generators/learning/learned_weights/'
    n_key_configs = 618
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = actions.shape[1]
    model = Qmse(dim_action=dim_action, dim_state=dim_state, save_folder=savedir, tau=config.tau,
                 explr_const=0.0)
    model.train(states, actions, sum_rewards)


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-a', default='ddpg')
    parser.add_argument('-g', action='store_true')
    parser.add_argument('-n_trial', type=int, default=-1)
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-tau', type=float, default=0.999)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-algo', type=str, default='admon')
    parser.add_argument('-n_score', type=int, default=5)
    parser.add_argument('-otherpi', default='uniform')
    parser.add_argument('-explr_p', type=float, default=0.3)
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    configs = parse_args()
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    tf.set_random_seed(configs.seed)

    if configs.algo == 'admon':
        train_admon(configs)
    elif configs.algo == 'mse':
        train_mse(configs)


if __name__ == '__main__':
    main()
