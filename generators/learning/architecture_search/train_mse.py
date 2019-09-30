import argparse
import numpy as np
import tensorflow as tf
import random

from Qmse import QmseWithPose
from generators.learning.train_sampler import load_data
from sklearn.preprocessing import StandardScaler


def create_model(config, dim_action):
    savedir = './generators/learning/learned_weights/'
    n_key_configs = 618
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    model = QmseWithPose(dim_action=dim_action, dim_collision=dim_state, save_folder=savedir, tau=config.tau,
                         config=config)
    return model


def train_mse(config):
    # Loads the processed data
    states, poses, actions, sum_rewards = load_data('./planning_experience/processed/domain_two_arm_mover/'
                                                    'n_objs_pack_1/irsc/sampler_trajectory_data/')
    actions = StandardScaler().fit_transform(actions)
    model = create_model(config, actions.shape[1])

    n_train = 5000

    train_states = states[:n_train, :]
    train_poses = poses[:n_train, :]
    train_actions = actions[:n_train, :]
    train_sum_rewards = sum_rewards[:n_train, :]
    model.train(train_states, train_poses, train_actions, train_sum_rewards)


def test_mse(config):
    states, poses, actions, sum_rewards = load_data('./planning_experience/processed/domain_two_arm_mover/'
                                                    'n_objs_pack_1/irsc/sampler_trajectory_data/')
    actions = StandardScaler().fit_transform(actions)
    model = create_model(config, actions.shape[1])
    n_train = config.n_data
    test_states = states[n_train:, :]
    test_poses = poses[n_train:, :]
    test_actions = actions[n_train:, :]
    test_sum_rewards = sum_rewards[n_train:, :]
    diff = test_sum_rewards - model.disc.predict([test_actions, test_states, test_poses])
    mse = np.mean(diff * diff)
    print 'mse ', mse


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-tau', type=float, default=1.0)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-algo', type=str, default='admon')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-test', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    configs = parse_args()
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    tf.set_random_seed(configs.seed)

    if configs.test:
        test_mse(configs)
    else:
        train_mse(configs)
        test_mse(configs)


if __name__ == '__main__':
    main()
