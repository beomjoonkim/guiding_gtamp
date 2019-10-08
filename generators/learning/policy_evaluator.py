from test_scripts.run_greedy import get_problem_env
from generators.learned_generator import LearnedGenerator
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from trajectory_representation.operator import Operator
from generators.learning.train_sampler import get_processed_poses_from_state, state_data_mode, action_data_mode
from generators.learning.RelKonfAdMonWithPose import RelKonfIMLEPose

import numpy as np
import collections
import pickle
import os
import openravepy
import sys

smpler_processed_path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/' \
                        'sampler_trajectory_data/'
abs_plan_path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/trajectory_data/mc/'
cached_env_path = './generators/learning/evaluation_pidxs/'


def get_pidx(processed_file_name):
    pidx = processed_file_name.split('_')[-1].split('.pkl')[0]
    return int(pidx)


def load_pose_file(pidx):
    poses = pickle.load(open(cached_env_path + 'pidx_%d.pkl' % pidx, 'r'))['body_poses']
    return poses


def get_smpler_state(pidx):
    state = pickle.load(open(cached_env_path + 'pidx_%d.pkl' % pidx, 'r'))['state']
    return state


def generate(obj, state_vec, smpler_state, policy):
    n_key_configs = 615
    utils.set_color(obj, [1, 0, 0])
    is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([obj in smpler_state.goal_entities]))
    is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
    is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([smpler_state.region in smpler_state.goal_entities]))
    is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
    state_vec = np.concatenate([state_vec, is_goal_obj, is_goal_region], axis=2)

    poses = np.hstack(
        [utils.encode_pose_with_sin_and_cos_angle(utils.get_body_xytheta(obj).squeeze()), 0, 0, 0, 0]).reshape((1, 8))

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    rel_konfs = []
    for k in key_configs:
        konf = utils.clean_pose_data(k)
        obj_pose = utils.clean_pose_data(smpler_state.obj_pose)
        rel_konf = utils.subtract_pose2_from_pose1(konf, obj_pose)
        rel_konfs.append(rel_konf)
    rel_konfs = np.array(rel_konfs).reshape((1, 615, 3, 1))

    places = []
    for _ in range(20):
        goal_flags = state_vec[:, :, 2:, :]
        poses = poses[:, :4]
        collisions = state_vec[:, :, :2, :]
        placement = policy.generate(goal_flags, rel_konfs, collisions, poses)
        if 'place_relative_to_obj' in action_data_mode:
            placement = utils.get_absolute_pose_from_relative_pose(placement, utils.get_body_xytheta(obj).squeeze())
        if 'place_relative_to_region' in action_data_mode:
            if smpler_state.region == 'home_region':
                placement[0:2] += [-1.75, 5.25]
            elif smpler_state.region == 'loading_region':
                placement[0:2] += [-0.7, 4.3]

        places.append(placement)
    return places


def get_pidxs_to_evaluate_policy(n_evals):
    smpler_processed_files = [f for f in os.listdir(smpler_processed_path) if 'pap_traj' in f]

    pidxs = []
    for smpler_processed_file in smpler_processed_files:
        pidx = get_pidx(smpler_processed_file)
        abs_plan_fname = 'pap_traj_seed_0_pidx_%d.pkl' % pidx
        if not os.path.isfile(abs_plan_path + abs_plan_fname):
            continue
        else:
            pidxs.append(pidx)
    return pidxs[:n_evals]


def visualize_samples(policy):
    n_evals = 10
    pidxs = get_pidxs_to_evaluate_policy(n_evals)
    pidx = pidxs[0]
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')
    problem_env = get_problem_env(config)
    pidx_poses = load_pose_file(pidx)

    problem_env.set_body_poses(pidx_poses)
    smpler_state = get_smpler_state(pidx)
    state_vec = np.delete(smpler_state.state_vec, [415, 586, 615, 618, 619], axis=1)

    obj = 'square_packing_box2'

    print 'generating..'
    places = generate(obj, state_vec, smpler_state, policy)

    utils.viewer()
    utils.visualize_path(places)
    import pdb;pdb.set_trace()


def main():
    n_key_configs = 615  # indicating whether it is a goal obj and goal region
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/rel_konf_place_admon/' % (
        state_data_mode, action_data_mode)

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'tau seed')

    config = mconfig_type(
        tau=1.0,
        seed=int(sys.argv[1])
    )

    dim_action = 3
    fname = 'imle_pose_seed_%d.h5' % config.seed
    dim_state = (n_key_configs, 2, 1)
    policy = RelKonfIMLEPose(dim_action, dim_state, savedir, 1.0, config)
    policy.policy_model.load_weights(policy.save_folder + fname)

    visualize_samples(policy)


if __name__ == '__main__':
    main()
