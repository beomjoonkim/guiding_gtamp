from trajectory_representation.trajectory import Trajectory
from trajectory_representation.hcount_trajectory import HCountExpTrajectory

import pickle
import os
import argparse
import socket

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def get_save_dir(parameters):
    save_dir = ROOTDIR + '/planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/%s/trajectory_data/%s/' \
                %(parameters.planner, parameters.statetype)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_raw_dir(parameters):
    if parameters.planner == 'hcount':
        #raw_dir = ROOTDIR + '/planning_experience/hcount/domain_two_arm_mover/n_objs_pack_1/hcount/'
        raw_dir = ROOTDIR + '/planning_experience/domain_two_arm_mover/n_objs_pack_1/hcount/'
    else:
        raw_dir = ROOTDIR + '/planning_experience/irsc/two_arm_mover/n_objs_pack_1//'
    return raw_dir


def get_p_idx(fname):
    return int(fname.split('.pkl')[0])


def save_traj(traj, save_fname):
    for state in traj.states:
        state.make_pklable()
    pickle.dump(traj, open(save_fname, 'wb'))


def process_plan_file(filename, pidx, goal_entities, parameters):
    scenario = ''

    print "Plan file name", filename
    plan_data = pickle.load(open(filename, 'r'))
    #if plan_data['plan'] is None:
    #    raise IOError
    if parameters.planner == 'hcount':
        if isinstance(plan_data, dict):
            plan = plan_data['plan']
        else:
            plan = plan_data.actions
    else:
        plan = plan_data['plan']

    if parameters.planner == 'hcount':
        traj = HCountExpTrajectory(pidx, scenario, parameters.statetype)
    else:
        traj = Trajectory(pidx, scenario, parameters.statetype)
    traj.add_trajectory(plan, goal_entities)
    return traj


def parse_parameters():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])  # used for threaded runs
    parser.add_argument('-planner', type=str, default="hcount")
    parser.add_argument('-statetype', type=str, default="mc")
    parameters = parser.parse_args()

    return parameters


def get_processed_fname(parameters, save_dir, raw_fname):
    traj_fname = 'pap_traj_' + raw_fname

    return traj_fname


def get_goal_entities(parameters):
    goal_entities = ['square_packing_box1', 'home_region']

    return goal_entities


def get_raw_fname(parameters):
    if parameters.planner == 'hcount':
        #return 'pidx_%d_planner_seed_0.pkl' % parameters.pidx
        return 'pidx_%d_planner_seed_0_train_seed_0_domain_two_arm_mover.pkl' % parameters.pidx
    else:
        return 'seed_0_pidx_' + str(parameters.pidx) + '.pkl'


def quit_if_already_done(fpath):
    if os.path.isfile(fpath):
        print "Already done"


def main():
    parameters = parse_parameters()

    raw_dir = get_raw_dir(parameters)
    raw_fname = get_raw_fname(parameters)
    save_dir = get_save_dir(parameters)
    processed_fname = get_processed_fname(parameters, save_dir, raw_fname)
    print "Raw fname", raw_dir + raw_fname
    print "Processed fname ", save_dir + processed_fname
    quit_if_already_done(save_dir + processed_fname)

    goal_entities = get_goal_entities(parameters)
    traj = process_plan_file(raw_dir + raw_fname, parameters.pidx, goal_entities, parameters)

    save_traj(traj, save_dir + processed_fname)


if __name__ == '__main__':
    main()
