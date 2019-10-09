from generators.learning.utils.model_creation_utils import create_imle_model
from generators.learning.utils.sampler_utils import generate_smpls
from trajectory_representation.concrete_node_state import ConcreteNodeState
from test_scripts.run_greedy import get_problem_env
from gtamp_utils import utils

import numpy as np
import collections
import pickle
import os
import sys

smpler_processed_path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/' \
                        'sampler_trajectory_data/'
abs_plan_path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/trajectory_data/mc/'
cached_env_path = './generators/learning/evaluation_pidxs/'

"""

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


def get_pidx(processed_file_name):
    pidx = processed_file_name.split('_')[-1].split('.pkl')[0]
    return int(pidx)
"""


def load_pose_file(pidx):
    poses = pickle.load(open(cached_env_path + 'pidx_%d.pkl' % pidx, 'r'))['body_poses']
    return poses


def compute_state(obj, region, problem_env, key_configs):
    goal_entities = ['square_packing_box1', 'home_region']
    return ConcreteNodeState(problem_env, obj, region, goal_entities, key_configs)


def get_smpler_state(pidx, obj):
    fname = cached_env_path + 'pidx_%d.pkl' % pidx
    state = pickle.load(open(fname, 'r'))['state']
    state.obj = obj
    return state


def load_problem(pidx):
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')
    problem_env = get_problem_env(config)
    return problem_env


def visualize_samples(policy):
    pidxs = range(20000, 20010)
    pidx = pidxs[3]
    problem_env = load_problem(pidx)
    utils.viewer()

    obj = problem_env.object_names[1]
    smpler_state = get_smpler_state(pidx, obj)

    print 'generating..'
    places = generate_smpls(smpler_state, policy, n_data=20)

    utils.visualize_path(places)


def main():
    seed = int(sys.argv[1])
    policy = create_imle_model(seed)
    visualize_samples(policy)


if __name__ == '__main__':
    main()
