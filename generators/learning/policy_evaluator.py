from test_scripts.run_greedy import get_problem_env
from planners.sahs.helper import compute_q_bonus, compute_hcount
from learn.data_traj import extract_individual_example
from generators.learned_generator import LearnedGenerator

from AdMonWithPose import AdversarialMonteCarloWithPose

import collections
import pickle
import os
import numpy as np


def qlearned_hcount_old_number_in_goal(state, action, problem_env, pap_model, config):
    is_two_arm_domain = 'two_arm_place_object' in action.discrete_parameters
    if is_two_arm_domain:
        target_o = action.discrete_parameters['two_arm_place_object']
        target_r = action.discrete_parameters['two_arm_place_region']
    else:
        target_o = action.discrete_parameters['object'].GetName()
        target_r = action.discrete_parameters['region'].name

    region_is_goal = state.nodes[target_r][8]

    if 'two_arm' in problem_env.name:
        goal_objs = [tmp_o for tmp_o in state.goal_entities if 'box' in tmp_o]
        goal_region = 'home_region'
    else:
        goal_objs = [tmp_o for tmp_o in state.goal_entities if 'region' not in tmp_o]
        goal_region = 'rectangular_packing_box1_region'

    number_in_goal = 0
    for i in state.nodes:
        if i == target_o:
            continue
        for tmpr in problem_env.regions:
            if tmpr in state.nodes:
                is_r_goal_region = state.nodes[tmpr][8]
                if is_r_goal_region:
                    is_i_in_r = state.binary_edges[(i, tmpr)][0]
                    if is_r_goal_region:
                        number_in_goal += is_i_in_r
    number_in_goal += int(region_is_goal)  # encourage moving goal obj to goal region

    nodes, edges, actions, _ = extract_individual_example(state, action)
    nodes = nodes[..., 6:]

    q_bonus = compute_q_bonus(state, nodes, edges, actions, pap_model, problem_env)
    hcount = compute_hcount(state, problem_env)
    obj_already_in_goal = state.binary_edges[(target_o, goal_region)][0]

    if config.n_objs_pack == 1:
        hval = -number_in_goal + obj_already_in_goal + hcount - q_bonus
    else:
        hval = -number_in_goal + obj_already_in_goal + hcount - 100.0 * q_bonus

    return hval


def get_pidx(processed_file_name):
    pidx = processed_file_name.split('_')[-1].split('.pkl')[0]
    return int(pidx)


def evaluate_in_problem_instance(policy, smpler_processed_path, smpler_processed_file, config_type):
    pidx = get_pidx(smpler_processed_file)
    config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')

    mover = get_problem_env(config)
    smpler_traj = pickle.load(open(smpler_processed_path + smpler_processed_file, 'r'))

    #raw_path = './planning_experience/raw/two_arm_mover/n_objs_pack_1//'
    #raw_file = 'seed_0_pidx_%d.pkl' % pidx
    #raw_plan = pickle.load(open(raw_path + raw_file, 'r'))['plan']

    path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/trajectory_data/mc/'
    fname = 'pap_traj_seed_0_pidx_%d.pkl' % pidx
    traj = pickle.load(open(path + fname, 'r'))
    plan = traj.actions
    states = traj.states

    smpler_states = smpler_traj.states
    smpler_state_idx = 0
    for state, action in zip(states, plan):
        smpler_state = smpler_states[smpler_state_idx]
        smpler = LearnedGenerator(action, mover, policy, smpler_state)
        smpled_param = smpler.sample_next_point(action, n_iter=200, n_parameters_to_try_motion_planning=3,
                                                cached_collisions=state.collides,
                                                cached_holding_collisions=None)
        import pdb;pdb.set_trace()


def evaluate_policy(policy):
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    smpler_processed_path = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/' \
                            'sampler_trajectory_data/'
    smpler_processed_files = np.random.permutation(os.listdir(smpler_processed_path))

    for smpler_processed_file in smpler_processed_files:
        hcount = evaluate_in_problem_instance(policy, smpler_processed_path, smpler_processed_file, config_type)


def main():
    n_goal_flags = 2  # indicating whether it is a goal obj and goal region
    n_key_configs = 618  # indicating whether it is a goal obj and goal region
    dim_state = (n_key_configs + n_goal_flags, 2, 1)
    dim_action = 8
    savedir = './generators/learning/learned_weights/'
    policy = AdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                           save_folder=savedir, tau=1.0, explr_const=0.0)

    epoch_number = 10
    policy.load_weights(agen_file='a_gen_epoch_%d.h5' % epoch_number)
    evaluate_policy(policy)


if __name__ == '__main__':
    main()
