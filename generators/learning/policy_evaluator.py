from generators.learning.utils.model_creation_utils import create_imle_model
from generators.learning.utils.sampler_utils import generate_smpls
from trajectory_representation.concrete_node_state import ConcreteNodeState
from test_scripts.run_greedy import get_problem_env
from gtamp_utils import utils

import collections
import pickle
import sys
import os

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


def visualize_samples(policy, pidx):
    problem_env = load_problem(pidx)
    utils.viewer()

    obj = problem_env.object_names[5]
    smpler_state = get_smpler_state(pidx, obj, problem_env)

    places = generate_smpls(smpler_state, policy, n_data=20)
    utils.visualize_path(places)


def main():
    seed = int(sys.argv[1])
    pidx = int(sys.argv[2])
    # python generators/learning/policy_evaluator.py 0 0 - can you generate different samples?
    policy = create_imle_model(seed)
    visualize_samples(policy, pidx)


if __name__ == '__main__':
    main()
