import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
import random

from Qmse import Qmse, QmseWithPose
from generators.learning.train_sampler import load_data


def train_mse(config):
    # Loads the processed data
    states, poses, actions, sum_rewards = load_data('./planning_experience/processed/domain_two_arm_mover/'
                                                    'n_objs_pack_1/irsc/sampler_trajectory_data/')
    savedir = './generators/learning/learned_weights/'
    n_key_configs = 618
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = actions.shape[1]
    model = QmseWithPose(dim_action=dim_action, dim_collision=dim_state, save_folder=savedir, tau=config.tau)

    n_train = 5000
    test_states = states[n_train:, :]
    test_poses = poses[n_train:, :]
    test_actions = actions[n_train:, :]
    test_sum_rewards = sum_rewards[n_train:, :]

    train_states = states[:n_train, :]
    train_poses = poses[:n_train, :]
    train_actions = actions[:n_train, :]
    train_sum_rewards = sum_rewards[:n_train, :]
    model.train(train_states, train_poses, train_actions, train_sum_rewards)

    import pdb;pdb.set_trace()


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
    train_mse(configs)

    # todo evaluation


if __name__ == '__main__':
    main()
