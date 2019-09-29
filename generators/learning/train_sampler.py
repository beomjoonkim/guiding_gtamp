import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
import random

from gtamp_utils import utils
from AdMon import AdversarialMonteCarlo
from AdMonWithPose import AdversarialMonteCarloWithPose


def get_processed_poses_from_state(state, data_mode):
    if data_mode == 'absolute':
        obj_pose = utils.encode_pose_with_sin_and_cos_angle(state.obj_pose)
        robot_pose = utils.encode_pose_with_sin_and_cos_angle(state.robot_pose)
    elif data_mode == 'robot_rel_to_obj':
        obj_pose = utils.encode_pose_with_sin_and_cos_angle(state.obj_pose)
        robot_pose = utils.get_relative_pose1_wrt_pose2(state.robot_pose, state.obj_pose)
        robot_pose = utils.encode_pose_with_sin_and_cos_angle(robot_pose)
    else:
        raise not NotImplementedError

    pose = np.hstack([obj_pose, robot_pose])
    return pose


def get_processed_poses_from_action(state, action, data_mode):
    if data_mode == 'absolute':
        pick_pose = utils.encode_pose_with_sin_and_cos_angle(action['pick_abs_base_pose'])
        place_pose = utils.encode_pose_with_sin_and_cos_angle(action['place_abs_base_pose'])
    elif data_mode == 'pick_relative':
        pick_pose = action['pick_abs_base_pose']
        pick_pose = utils.get_relative_pose1_wrt_pose2(pick_pose, state.obj_pose)
        pick_pose = utils.encode_pose_with_sin_and_cos_angle(pick_pose)
        place_pose = utils.encode_pose_with_sin_and_cos_angle(action['place_abs_base_pose'])
    elif data_mode == 'pick_relative_place_relative_to_region':
        pick_pose = action['pick_abs_base_pose']
        pick_pose = utils.get_relative_pose1_wrt_pose2(pick_pose, state.obj_pose)
        pick_pose = utils.encode_pose_with_sin_and_cos_angle(pick_pose)
        place_pose = action['place_abs_base_pose']
        if action['region_name'] == 'home_region':
            place_pose[0:2] -= [-1.75, 5.25]
        elif action['region_name'] == 'loading_region':
            place_pose[0:2] -= [-0.7, 4.3]
        else:
            raise NotImplementedError
        place_pose = utils.encode_pose_with_sin_and_cos_angle(place_pose)
    elif data_mode == '':
        raise NotImplementedError

    action = np.hstack([pick_pose, place_pose])
    return action


def load_data(traj_dir, state_data_mode='robot_rel_to_obj', action_data_mode='pick_relative_place_relative_to_region'):
    traj_files = os.listdir(traj_dir)
    cache_file_name = 'cache_state_data_mode_%s_action_data_mode_%s.pkl' % (state_data_mode, action_data_mode)
    if os.path.isfile(traj_dir + cache_file_name):
        return pickle.load(open(traj_dir + cache_file_name, 'r'))
    print 'caching file...'
    all_states = []
    all_actions = []
    all_sum_rewards = []
    all_poses = []

    for traj_file in traj_files:
        if 'pidx' not in traj_file:
            continue
        traj = pickle.load(open(traj_dir + traj_file, 'r'))
        if len(traj.states) == 0:
            continue

        states = np.array([s.state_vec for s in traj.states])  # collision vectors
        poses = np.array([get_processed_poses_from_state(s, state_data_mode) for s in traj.states])
        actions = np.array([get_processed_poses_from_action(s, a, action_data_mode) for s,a in zip(traj.states, traj.actions)])

        rewards = traj.rewards
        sum_rewards = np.array([np.sum(traj.rewards[t:]) for t in range(len(rewards))])

        all_poses.append(poses)
        all_states.append(states)
        all_actions.append(actions)
        all_sum_rewards.append(sum_rewards)

    all_states = np.vstack(all_states).squeeze(axis=1)
    all_actions = np.vstack(all_actions)
    all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
    all_poses = np.vstack(all_poses).squeeze()
    pickle.dump((all_states, all_poses, all_actions, all_sum_rewards), open(traj_dir + cache_file_name, 'wb'))
    return all_states, all_poses, all_actions, all_sum_rewards[:, None]


def get_data():
    states, poses, actions, sum_rewards = load_data('./planning_experience/processed/domain_two_arm_mover/'
                                                    'n_objs_pack_1/irsc/sampler_trajectory_data/')
    n_data = 5000
    states = states[:5000, :]
    poses = poses[:n_data, :]
    actions = actions[:5000, :]
    sum_rewards = sum_rewards[:5000]
    print "Number of data", len(states)
    return states, poses, actions, sum_rewards


def train_admon(config):
    # Loads the processed data
    states, _, actions, sum_rewards = get_data()
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    n_key_configs = 618  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = actions.shape[1]
    savedir = './generators/learning/learned_weights/'
    admon = AdversarialMonteCarlo(dim_action=dim_action, dim_state=dim_state, save_folder=savedir, tau=config.tau)
    admon.train(states, actions, sum_rewards, epochs=20)


def train_admon_with_pose(config):
    states, poses, actions, sum_rewards = get_data()
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    n_key_configs = 618  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = actions.shape[1]
    savedir = './generators/learning/learned_weights/'
    admon = AdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                          save_folder=savedir, tau=config.tau)
    admon.train(states, poses, actions, sum_rewards, epochs=20)


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
    elif configs.algo == 'admonpose':
        train_admon_with_pose(configs)
    elif configs.algo == 'mse':
        train_mse(configs)


if __name__ == '__main__':
    main()
