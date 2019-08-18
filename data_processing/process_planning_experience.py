from trajectory_representation.trajectory import Trajectory

import pickle
import os
import argparse
import socket


hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/tamp_q_results/'



def get_save_dir(parameters):
    if parameters.scenario is None:
        save_dir = ROOTDIR+'/planning_experience/irsc/two_arm_mover/n_objs_pack_1/trajectory_data/'
    else:
        save_dir = ROOTDIR+'/planning_experience/irsc/two_arm_mover/n_objs_pack_1/trajectory_data/special_cases/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_raw_dir(parameters):
    if parameters.scenario is None:
        raw_dir = ROOTDIR+'/planning_experience/irsc/two_arm_mover/n_objs_pack_1/'
    else:
        raw_dir = ROOTDIR+'/planning_experience/irsc/two_arm_mover/n_objs_pack_1/'
    return raw_dir


def get_p_idx(fname):
    return int(fname.split('.pkl')[0])


def save_traj(traj, save_fname):
    for state in traj.states:
        state.make_pklable()
    pickle.dump(traj, open(save_fname, 'wb'))


def process_plan_file(filename, pidx, goal_entities, parameters):
    if parameters.scenario == 0:
        scenario = 'reachable_goal_entities'
    elif parameters.scenario == 1:
        scenario = 'reachable_goal_object_unreachable_goal_region'
    elif parameters.scenario == 2:
        scenario = 'unreachable_goal_object_unreachable_goal_region'
    else:
        scenario = ''

    print "Plan file name", filename
    plan_data = pickle.load(open(filename, 'r'))
    if plan_data['plan'] is None:
        raise IOError

    plan = plan_data['plan']
    traj = Trajectory(pidx, scenario)
    traj.add_trajectory(plan, goal_entities)
    return traj


def parse_parameters():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-scenario', type=int, default=None)
    parameters = parser.parse_args()

    return parameters


def get_processed_fname(parameters, save_dir, raw_fname):
    print "Processing", save_dir + raw_fname
    traj_fname = 'pap_traj_' + raw_fname

    return traj_fname


def get_goal_entities(parameters):
    if parameters.scenario is None:
        goal_entities = ['square_packing_box1', 'home_region']
    else:
        if parameters.scenario == 0:
            goal_obj = 'rectangular_packing_box1'
        elif parameters.scenario == 1:
            goal_obj = 'rectangular_packing_box2'
        elif parameters.scenario == 2:
            goal_obj = 'square_packing_box3'
        else:
            raise ValueError
        goal_entities = [goal_obj, 'home_region']

    return goal_entities


def get_raw_fname(parameters):
    if parameters.scenario is None:
        return 'seed_0_pidx_' + str(parameters.pidx) + '.pkl'
    else:
        if parameters.scenario == 0:
            scenario = 'reachable_goal_entities'
        elif parameters.scenario == 1:
            scenario = 'reachable_goal_object_unreachable_goal_region'
        elif parameters.scenario == 2:
            scenario = 'unreachable_goal_object_unreachable_goal_region'
        else:
            raise ValueError
        return scenario + '.pkl'


def quit_if_already_done(fpath):
    if os.path.isfile(fpath):
        print "Already done"


def main():
    parameters = parse_parameters()

    raw_dir = get_raw_dir(parameters)
    raw_fname = get_raw_fname(parameters)
    save_dir = get_save_dir(parameters)
    processed_fname = get_processed_fname(parameters, save_dir, raw_fname)
    print "Raw fname", raw_dir+raw_fname
    print "Processed fname ", save_dir+processed_fname
    quit_if_already_done(save_dir + processed_fname)

    goal_entities = get_goal_entities(parameters)
    traj = process_plan_file(raw_dir + raw_fname, parameters.pidx, goal_entities, parameters)
    save_traj(traj, save_dir + processed_fname)


if __name__ == '__main__':
    main()
