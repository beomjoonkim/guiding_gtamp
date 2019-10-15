from generators.learning.utils.model_creation_utils import create_imle_model, load_weights
from generators.learning.utils.sampler_utils import generate_smpls, generate_w_values, generate_transformed_key_configs, \
    generate_smpls_using_noise
from trajectory_representation.concrete_node_state import ConcreteNodeState
from test_scripts.run_greedy import get_problem_env
from gtamp_utils import utils

import collections
import pickle
import sys
import os
import numpy as np

smpler_processed_path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/' \
                        'sampler_trajectory_data/'
abs_plan_path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/trajectory_data/mc/'
cached_env_path = './generators/learning/evaluation_pidxs/'


def load_pose_file(pidx):
    poses = pickle.load(open(cached_env_path + 'pidx_%d.pkl' % pidx, 'r'))['body_poses']
    return poses


def compute_state(obj, region, problem_env):
    goal_entities = ['square_packing_box1', 'home_region']
    return ConcreteNodeState(problem_env, obj, region, goal_entities)


def get_smpler_state(pidx, obj, problem_env):
    fname = cached_env_path + 'policy_eval_pidx_%d.pkl' % pidx
    if os.path.isfile(fname):
        state = pickle.load(open(fname, 'r'))
    else:
        state = compute_state(obj, 'loading_region', problem_env)
        pickle.dump(state, open(fname, 'wb'))

    state.obj = obj
    state.goal_flags = state.get_goal_flags()
    state.abs_obj_pose = utils.get_body_xytheta(obj)
    return state


def load_problem(pidx):
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')
    problem_env = get_problem_env(config)
    return problem_env


def visualize_key_configs_with_top_k_w_vals(w_values, key_configs, k):
    utils.visualize_path(key_configs[np.argsort(w_values.squeeze())[-k:], :])


def visualize_smpl(smpler_state, policy, noise):
    policy_smpl = generate_smpls_using_noise(smpler_state, policy, noise)
    utils.visualize_path(policy_smpl)


def visualize_samples(policy, pidx):
    problem_env = load_problem(pidx)
    utils.viewer()

    obj = problem_env.object_names[1]
    utils.set_color(obj, [1, 0, 0])
    smpler_state = get_smpler_state(pidx, obj, problem_env)
    smpler_state.abs_obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    """
    w_values = generate_w_values(smpler_state, policy)
    transformed_konfs = generate_transformed_key_configs(smpler_state, policy)
    print "Visualizing top-k transformed konfs..."
    visualize_key_configs_with_top_k_w_vals(w_values, transformed_konfs, k=5)
    print "Visualizing top-k konfs..."
    visualize_key_configs_with_top_k_w_vals(w_values, smpler_state.key_configs, k=10)
    """

    place_smpls = []
    noises_used = []
    for i in range(10):
        noise = i/10.0 * np.random.normal(size=(1, 4)).astype('float32')
        smpl = generate_smpls_using_noise(smpler_state, policy, noise)[0]
        place_smpls.append(smpl)
        noises_used.append(noise)
    import pdb;pdb.set_trace()
    utils.visualize_path(place_smpls[0:10])


def visualize_samples_at_noises(problem_env, policy, pidx, noises):
    utils.viewer()

    obj = problem_env.object_names[1]
    utils.set_color(obj, [1, 0, 0])
    smpler_state = get_smpler_state(pidx, obj, problem_env)
    smpler_state.abs_obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    place_smpls = []
    for noise in noises:
        smpl = generate_smpls_using_noise(smpler_state, policy, noise)[0]
        place_smpls.append(smpl)
    utils.visualize_path(place_smpls)


def main():
    seed = int(sys.argv[1])
    pidx = int(sys.argv[2])
    # python generators/learning/policy_evaluator.py 0 0 - can you generate different samples?
    policy = create_imle_model(seed)
    problem_env = load_problem(pidx)

    #noises = [np.random.normal(size=(1, 4)).astype('float32') for _ in range(50)]
    noises = []
    for i in range(30):
        if i <= 10:
            noise = 0.01*np.random.normal(size=(1, 4)).astype('float32')
        elif 10 < i <= 20:
            noise = 0.1 * np.random.normal(size=(1, 4)).astype('float32')
        elif 20 <= i <= 30:
            noise = 1 * np.random.normal(size=(1, 4)).astype('float32')
        noises.append(noise)
    visualize_samples_at_noises(problem_env, policy, pidx, noises)

    unregularized_weight_fname = 'without_regularizer/imle_pose_seed_0.h5'
    load_weights(policy, None, unregularized_weight_fname)
    visualize_samples_at_noises(problem_env, policy, pidx, noises)

    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
