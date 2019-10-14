import argparse
import os
import pickle
import numpy as np
import random
import socket


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
    parser.add_argument('-algo', type=str, default='rel_konf_place_admon')
    parser.add_argument('-n_score', type=int, default=5)
    parser.add_argument('-otherpi', default='uniform')
    parser.add_argument('-explr_p', type=float, default=0.3)
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    return args


configs = parse_args()
np.random.seed(configs.seed)
random.seed(configs.seed)
os.environ['PYTHONHASHSEED'] = str(configs.seed)

import tensorflow as tf

tf.set_random_seed(configs.seed)

from AdMon import AdversarialMonteCarlo
from PlaceAdMonWithPose import PlaceAdmonWithPose
from CMAESAdMonWithPose import CMAESAdversarialMonteCarloWithPose
from RelKonfMSEWithPose import RelKonfMSEPose
from RelKonfIMLE import RelKonfIMLEPose
from utils.data_processing_utils import get_processed_poses_from_state, get_processed_poses_from_action, \
    state_data_mode, action_data_mode, make_konfs_relative_to_pose

from gtamp_utils import utils


def load_data(traj_dir):
    traj_files = os.listdir(traj_dir)
    cache_file_name = 'cache_state_data_mode_%s_action_data_mode_%s.pkl' % (state_data_mode, action_data_mode)
    if os.path.isfile(traj_dir + cache_file_name):
        print "Loading the cache file", traj_dir + cache_file_name
        return pickle.load(open(traj_dir + cache_file_name, 'r'))

    print 'caching file...'
    all_states = []
    all_actions = []
    all_sum_rewards = []
    all_poses = []
    all_rel_konfs = []

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)

    for traj_file in traj_files:
        if 'pidx' not in traj_file:
            continue
        traj = pickle.load(open(traj_dir + traj_file, 'r'))
        if len(traj.states) == 0:
            continue

        # states = np.array([s.state_vec for s in traj.states])  # collision vectors
        states = []
        for s in traj.states:
            # state_vec = np.delete(s.state_vec, [415, 586, 615, 618, 619], axis=1)
            state_vec = s.collision_vector
            n_key_configs = state_vec.shape[1]

            is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([s.obj in s.goal_entities]))
            is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
            is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([s.region in s.goal_entities]))
            is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
            state_vec = np.concatenate([state_vec, is_goal_obj, is_goal_region], axis=2)
            states.append(state_vec)

        states = np.array(states)
        poses = np.array([get_processed_poses_from_state(s) for s in traj.states])
        actions = np.array([get_processed_poses_from_action(s, a)
                            for s, a in zip(traj.states, traj.actions)])
        for s in traj.states:
            rel_konfs = make_konfs_relative_to_pose(s.abs_obj_pose, key_configs)
            all_rel_konfs.append(np.array(rel_konfs).reshape((1, 615, 4, 1)))

        rewards = traj.rewards
        sum_rewards = np.array([np.sum(traj.rewards[t:]) for t in range(len(rewards))])

        all_poses.append(poses)
        all_states.append(states)
        all_actions.append(actions)
        all_sum_rewards.append(sum_rewards)

    all_rel_konfs = np.vstack(all_rel_konfs)
    all_states = np.vstack(all_states).squeeze(axis=1)
    all_actions = np.vstack(all_actions)
    all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
    all_poses = np.vstack(all_poses).squeeze()
    pickle.dump((all_states, all_poses, all_rel_konfs, all_actions, all_sum_rewards),
                open(traj_dir + cache_file_name, 'wb'))
    return all_states, all_poses, all_rel_konfs, all_actions, all_sum_rewards[:, None]


def get_data():
    if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp/planning_experience/processed/'
    states, poses, rel_konfs, actions, sum_rewards = load_data(
        root_dir + '/planning_experience/processed/domain_two_arm_mover/'
                   'n_objs_pack_1/irsc/sampler_trajectory_data/')
    is_goal_flag = states[:, :, 2:, :]
    states = states[:, :, :2, :]  # collision vector

    n_data = 5000
    states = states[:5000, :]
    poses = poses[:n_data, :]
    actions = actions[:5000, :]
    sum_rewards = sum_rewards[:5000]
    is_goal_flags = is_goal_flag[:5000, :]

    print "Number of data", len(states)
    return states, poses, rel_konfs, is_goal_flags, actions, sum_rewards


def train_admon(config):
    # Loads the processed data
    states, _, actions, sum_rewards = get_data()
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    n_key_configs = 618  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = actions.shape[1]
    savedir = './generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/admon/' % (
        state_data_mode, action_data_mode)
    admon = AdversarialMonteCarlo(dim_action=dim_action, dim_state=dim_state, save_folder=savedir, tau=config.tau)
    admon.train(states, actions, sum_rewards, epochs=100)


def train_admon_with_pose(config):
    states, poses, actions, sum_rewards = get_data()
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    n_key_configs = 618  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = actions.shape[1]
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/admon_with_pose/' % (
        state_data_mode, action_data_mode)
    admon = PlaceAdmonWithPose(dim_action=dim_action, dim_collision=dim_state,
                               save_folder=savedir, tau=config.tau, config=config)
    admon.train(states, poses, actions, sum_rewards, epochs=500)


def train_place_admon_with_pose(config):
    states, poses, key_configs, actions, sum_rewards = get_data()
    actions = actions[:, 4:]
    n_key_configs = states.shape[1]  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs, 6, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/place_admon/' % (
        state_data_mode, action_data_mode)
    admon = PlaceAdmonWithPose(dim_action=dim_action, dim_collision=dim_state,
                               save_folder=savedir, tau=config.tau, config=config)

    is_mse_pretrained = os.path.isfile(admon.save_folder + admon.pretraining_file_name)
    if not is_mse_pretrained:
        admon.pretrain_discriminator_with_mse(states, poses, actions, sum_rewards)

    # But I have not loaded the weight?
    admon.train(states, poses, actions, sum_rewards, epochs=500)


def train_cmaes_place_admon_with_pose(config):
    states, poses, rel_konfs, actions, sum_rewards = get_data()
    n_key_configs = 615
    dim_state = (n_key_configs, 6, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/cmaes_place_admon/' % (
        state_data_mode, action_data_mode)
    admon = CMAESAdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                               save_folder=savedir, tau=config.tau, config=config)

    actions = actions[:, 4:]
    is_mse_pretrained = os.path.isfile(admon.save_folder + admon.pretraining_file_name)
    if not is_mse_pretrained:
        admon.pretrain_discriminator_with_mse(states, poses, actions, sum_rewards)
    admon.disc_mse_model.load_weights(admon.save_folder + admon.pretraining_file_name)

    # But I have not loaded the weight?
    admon.train(states, poses, actions, sum_rewards, epochs=500)


def train_rel_konf_place_mse(config):
    n_key_configs = 615
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/rel_konf_place_mse/' % (
        state_data_mode, action_data_mode)
    admon = RelKonfMSEPose(dim_action=dim_action, dim_collision=dim_state,
                           save_folder=savedir, tau=config.tau, config=config)
    admon.policy_model.summary()

    states, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data()
    actions = actions[:, 4:]
    poses = poses[:, :8]  # now include relative goal pose

    admon.train_policy(states, poses, rel_konfs, goal_flags, actions, sum_rewards)
    pred = admon.w_model.predict([goal_flags, rel_konfs, states, poses])
    import pdb;pdb.set_trace()


def train_rel_konf_place_admon(config):
    n_key_configs = 615
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/rel_konf_place_admon/' % (
        state_data_mode, action_data_mode)
    admon = RelKonfIMLEPose(dim_action=dim_action, dim_collision=dim_state,
                            save_folder=savedir, tau=config.tau, config=config)
    print "Created IMLE-admon"

    states, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data()
    actions = actions[:, 4:]
    poses = poses[:, :8]
    #pred = admon.w_model.predict([goal_flags, rel_konfs, states, poses])
    #import pdb;pdb.set_trace()
    admon.train(states, poses, rel_konfs, goal_flags, actions, sum_rewards)


def main():
    if configs.algo == 'admon':
        train_admon(configs)
    elif configs.algo == 'admonpose':
        train_admon_with_pose(configs)
    elif configs.algo == 'placeadmonpose':
        print "Training place only"
        train_place_admon_with_pose(configs)
    elif configs.algo == 'cmaes_placeadmonpose':
        train_cmaes_place_admon_with_pose(configs)
    elif configs.algo == 'rel_konf_place_mse':
        train_rel_konf_place_mse(configs)
    elif configs.algo == 'rel_konf_place_admon':
        train_rel_konf_place_admon(configs)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
