from test_scripts.run_greedy import get_problem_env
from generators.learned_generator import LearnedGenerator
from gtamp_utils import utils
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.concrete_node_state import ConcreteNodeState
from trajectory_representation.operator import Operator
from AdMonWithPose import AdversarialMonteCarloWithPose
from generators.learning.train_sampler import get_processed_poses_from_state

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


def get_smpler_and_abstract_action_trajectories(pidx):
    abs_plan_fname = smpler_plan_fname = 'pap_traj_seed_0_pidx_%d.pkl' % pidx
    print "Loading the plan ", abs_plan_fname
    smpler_traj = pickle.load(open(smpler_processed_path + smpler_plan_fname, 'r'))
    abs_traj = pickle.load(open(abs_plan_path + abs_plan_fname, 'r'))
    return smpler_traj, abs_traj


def get_smpler_state(pidx):
    state = pickle.load(open(cached_env_path + 'pidx_%d.pkl' % pidx, 'r'))['state']
    return state


def load_pose_file(pidx):
    poses = pickle.load(open(cached_env_path + 'pidx_%d.pkl' % pidx, 'r'))['body_poses']
    return poses


def evaluate_in_problem_instance(policy, pidx, problem_env):
    pidx_poses = load_pose_file(pidx)
    problem_env.set_body_poses(pidx_poses)
    #smpler_traj, abs_traj = get_smpler_and_abstract_action_trajectories(pidx)
    smpler_state = get_smpler_state(pidx)

    #abs_plan = abs_traj.actions
    #abs_states = abs_traj.states
    #smpler_states = smpler_traj.states
    #smpler_state_idx = 0

    #abs_state = abs_states[0]
    #abs_action = abs_plan[0]

    # Evaluate it in the first state
    utils.set_color(smpler_state.obj, [1, 0, 0])
    abs_action = Operator('two_arm_pick_two_arm_place',
                          discrete_parameters={'object': smpler_state.obj, 'region': smpler_state.region})
    generator = LearnedGenerator(abs_action, problem_env, policy, smpler_state)
    base_poses = np.array(
        [(generator.generate_base_poses(abs_action)[0], generator.generate_base_poses(abs_action)[1]) for _ in range(10)])

    place_poses = []
    poses = get_processed_poses_from_state(smpler_state, 'robot_rel_to_obj').reshape((1, 8))
    pose_scaler = pickle.load(open('scalers.pkl', 'r'))['pose']
    action_scaler = pickle.load(open('scalers.pkl', 'r'))['action']
    poses = pose_scaler.transform(poses)
    for _ in range(20):
        pap_base_poses = generator.sampler.generate(smpler_state.state_vec, poses)  # I need grasp parameters;
        pap_base_poses = action_scaler.inverse_transform(pap_base_poses)
        place_poses.append(pap_base_poses[0, 4:])
    print np.mean(place_poses, axis=0)
    import pdb;pdb.set_trace()
    utils.viewer()
    # utils.visualize_path([abs_action.continuous_parameters['pick']['q_goal']])
    # utils.visualize_path([abs_action.continuous_parameters['place']['q_goal']])
    picks = base_poses[:, 0]
    places = base_poses[:, 1]
    import pdb;pdb.set_trace()
    utils.visualize_path(places)
    utils.visualize_path(picks)
    import pdb;
    pdb.set_trace()
    """
    abs_action.discrete_parameters['region'] = abs_action.discrete_parameters['two_arm_place_region']
    smpler_state = smpler_states[smpler_state_idx]
    smpler = LearnedGenerator(abs_action, problem_env, policy, smpler_state)
    smpled_param = smpler.sample_next_point(abs_action, n_iter=50, n_parameters_to_try_motion_planning=3,
                                            cached_collisions=abs_state.collides,
                                            cached_holding_collisions=None,
                                            dont_check_motion_existence=True)
    """
    utils.set_color(abs_action.discrete_parameters['object'], [0, 0, 0])
    smpled_param = {'is_feasible': True}
    print smpled_param['is_feasible']
    if smpled_param['is_feasible']:
        return True
    else:
        return False


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


def cache_poses_of_robot_and_objs(pidxs):
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    for pidx in pidxs:
        config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')
        problem_env = get_problem_env(config)

        obj = 'square_packing_box1'
        region = 'loading_region'
        smpler_state = ConcreteNodeState(problem_env, obj, region, problem_env.goal, key_configs)
        body_poses = {}
        for o in problem_env.objects:
            body_poses[o.GetName()] = utils.get_body_xytheta(o)
        body_poses['pr2'] = utils.get_body_xytheta(problem_env.robot)
        smpler_state.problem_env = None
        pickle.dump({'state': smpler_state,
                     'body_poses': body_poses},
                    open(cached_env_path + 'pidx_%d.pkl' % pidx, 'wb'))
        problem_env.env.Destroy()
        openravepy.RaveDestroy()


def evaluate_policy(policy):
    n_evals = 10
    pidxs = get_pidxs_to_evaluate_policy(n_evals)
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    config = config_type(pidx=437, n_objs_pack=1, domain='two_arm_mover')
    problem_env = get_problem_env(config)
    n_successes = [evaluate_in_problem_instance(policy, pidx, problem_env) for pidx in pidxs[1:]]
    return n_successes


def main():
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    n_key_configs = 618  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = 8
    savedir = './generators/learning/learned_weights/'

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'tau seed')

    config = mconfig_type(
        tau=1.0,
        seed=int(sys.argv[1])
    )

    policy = AdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                           save_folder=savedir, tau=config.tau, config=config)
    epoch_number = int(sys.argv[2])
    print "Trying epoch number ", epoch_number
    policy.load_weights(additional_name='_epoch_%d' % epoch_number)
    n_successes = evaluate_policy(policy)
    print n_successes


if __name__ == '__main__':
    main()
