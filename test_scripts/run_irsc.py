from gtamp_problem_environments.mover_env import Mover
from planners.rsc.resolve_spatial_constraints import ResolveSpatialConstraints
from planners.rsc.one_arm_resolve_spatial_constraints import OneArmResolveSpatialConstraints
from planners.rsc.planner_without_reachability import PlannerWithoutReachability
from planners.rsc.one_arm_planner_without_reachability import OneArmPlannerWithoutReachability
from gtamp_problem_environments.one_arm_mover_env import OneArmMover
from trajectory_representation.swept_volume import PickAndPlaceSweptVolume

from gtamp_utils import utils

import os
import sys
import argparse
import pickle
import numpy as np
import random
import time
import socket

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def make_and_get_save_dir(parameters):
    save_dir = ROOTDIR + '/planning_experience/irsc/'
    save_dir += parameters.domain + '/n_objs_pack_'
    save_dir += str(parameters.n_objs_pack)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def quit_if_already_tested(file_path, force_test):
    if os.path.isfile(file_path):
        print "Already done"
        if not force_test:
            sys.exit(-1)


def parse_parameters():
    parser = argparse.ArgumentParser(description='HPN parameters')

    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-timelimit', type=int, default=1000)
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-n_feasibility_checks', type=int, default=500)
    parser.add_argument('-n_parameters_to_test_each_sample_time', type=int, default=10)
    parser.add_argument('-n_motion_plan_trials', type=int, default=10)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-domain', type=str, default='two_arm_mover')

    # dummy variables
    parser.add_argument('-loss', type=str, default='asdf')
    parser.add_argument('-train_seed', type=int, default=1000)
    parser.add_argument('-num_train', type=int, default=1000)
    parameters = parser.parse_args()
    return parameters


def attach_q_goal_as_low_level_motion(target_op_inst):
    target_op_inst.low_level_motion = {}
    target_op_inst.low_level_motion['pick'] = target_op_inst.continuous_parameters['pick']['q_goal']
    target_op_inst.low_level_motion['place'] = target_op_inst.continuous_parameters['place']['q_goal']
    return target_op_inst


def find_plan_for_obj(obj_name, target_op_inst, environment, stime, timelimit):
    is_one_arm_environment = environment.name.find('one_arm') != -1
    if is_one_arm_environment:
        target_op_inst = attach_q_goal_as_low_level_motion(target_op_inst)
        swept_volumes = PickAndPlaceSweptVolume(environment, None)
        swept_volumes.add_pap_swept_volume(target_op_inst)
        obstacles_to_remove = swept_volumes.get_objects_in_collision()
        print len(obstacles_to_remove)
        if len(obstacles_to_remove) == 0:
            return [target_op_inst], 1, "HasSolution"
        rsc = OneArmResolveSpatialConstraints(problem_env=environment,
                                              goal_object_name=obj_name,
                                              goal_region_name='rectangular_packing_box1_region')
        obstacle_to_remove_idx = 0

    else:
        rsc = ResolveSpatialConstraints(problem_env=environment,
                                        goal_object_name=obj_name,
                                        goal_region_name='home_region',
                                        misc_region_name='loading_region')
        plan = None

    plan_found = False
    status = 'NoSolution'
    while not plan_found and rsc.get_num_nodes() < 100 and time.time() - stime < timelimit:
        if is_one_arm_environment:
            obj_to_move = obstacles_to_remove[obstacle_to_remove_idx]
            tmp_obstacles_to_remove = set(obstacles_to_remove).difference(set([obj_to_move]))
            tmp_obstacles_to_remove = list(tmp_obstacles_to_remove)
            top_level_plan = [target_op_inst]
            plan, status = rsc.search(object_to_move=obj_to_move,
                                      parent_swept_volumes=swept_volumes,
                                      obstacles_to_remove=tmp_obstacles_to_remove,
                                      objects_moved_before=[target_op_inst.discrete_parameters['object']],
                                      plan=top_level_plan,
                                      stime=stime,
                                      timelimit=timelimit)
        else:
            plan, status = rsc.search(obj_name,
                                      parent_swept_volumes=None,
                                      obstacles_to_remove=[],
                                      objects_moved_before=[],
                                      plan=[],
                                      stime=stime,
                                      timelimit=timelimit)
        plan_found = status == 'HasSolution'
        if plan_found:
            print "Solution found"
        else:
            if is_one_arm_environment:
                obstacle_to_remove_idx += 1
                if obstacle_to_remove_idx == len(obstacles_to_remove):
                    break
            print "Restarting..."
    if plan_found:
        return plan, rsc.get_num_nodes(), status
    else:
        return [], rsc.get_num_nodes(), status


def execute_plan(plan):
    for p in plan:
        p.execute()


def save_plan(total_plan, total_n_nodes, n_remaining_objs, found_solution, file_path, goal_entities, time_taken):
    [p.make_pklable() for p in total_plan]
    pickle.dump({"plan": total_plan,
                 'n_nodes': total_n_nodes,
                 'n_remaining_objs': n_remaining_objs,
                 'goal_entities': goal_entities,
                 'time_taken': time_taken,
                 'found_solution': found_solution},
                open(file_path, 'wb'))


def find_plan_without_reachability(problem_env, goal_object_names):
    if problem_env.name.find('one_arm_mover') != -1:
        planner = OneArmPlannerWithoutReachability(problem_env, goal_object_names,
                                                   goal_region='rectangular_packing_box1_region')
    else:
        planner = PlannerWithoutReachability(problem_env, goal_object_names, goal_region='home_region')
    goal_obj_order_plan, plan = planner.search()

    goal_obj_order_plan = [o.GetName() for o in goal_obj_order_plan]
    return goal_obj_order_plan, plan


def main():
    parameters = parse_parameters()

    save_dir = make_and_get_save_dir(parameters)
    file_path = save_dir + '/seed_' + str(parameters.planner_seed) + '_pidx_' + str(parameters.pidx) + '.pkl'
    quit_if_already_tested(file_path, parameters.f)

    # for creating problem
    np.random.seed(parameters.pidx)
    random.seed(parameters.pidx)
    is_one_arm_env = parameters.domain.find('two_arm') != -1
    if is_one_arm_env:
        environment = Mover(parameters.pidx)
        goal_region = ['home_region']
    else:
        environment = OneArmMover(parameters.pidx)
        goal_region = ['rectangular_packing_box1_region']

    goal_object_names = [obj.GetName() for obj in environment.objects[:parameters.n_objs_pack]]
    goal_entities = goal_object_names + goal_region

    # for randomized algorithms
    np.random.seed(parameters.planner_seed)
    random.seed(parameters.planner_seed)

    if parameters.v:
        environment.env.SetViewer('qtcoin')

    # from manipulation.bodies.bodies import set_color
    # set_color(environment.env.GetKinBody(goal_object_names[0]), [1, 0, 0])
    stime = time.time()

    goal_object_names, high_level_plan = find_plan_without_reachability(environment, goal_object_names)  # finds the plan

    total_n_nodes = 0
    total_plan = []
    idx = 0
    total_time_taken = 0
    found_solution = False
    timelimit = parameters.timelimit
    timelimit = np.inf
    while total_n_nodes < 1000 and total_time_taken < timelimit:
        goal_obj_name = goal_object_names[idx]
        plan, n_nodes, status = find_plan_for_obj(goal_obj_name, high_level_plan[idx], environment, stime, timelimit)
        total_n_nodes += n_nodes
        total_time_taken = time.time() - stime
        print goal_obj_name, goal_object_names, total_n_nodes
        print "Time taken: %.2f" % total_time_taken
        if status == 'HasSolution':
            execute_plan(plan)
            environment.initial_robot_base_pose = utils.get_body_xytheta(environment.robot)
            total_plan += plan
            save_plan(total_plan, total_n_nodes, len(goal_object_names) - idx, found_solution, file_path, goal_entities,
                      total_time_taken)
            idx += 1
        else:
            # Note that HPN does not have any recourse if this happens. We re-plan at the higher level.
            goal_object_names, plan = find_plan_without_reachability(environment, goal_object_names)  # finds the plan
            total_plan = []
            idx = 0

        if idx == len(goal_object_names):
            found_solution = True
            break
        else:
            idx %= len(goal_object_names)

    save_plan(total_plan, total_n_nodes, len(goal_object_names) - idx, found_solution, file_path, goal_entities,
              total_time_taken)
    print 'plan saved'


if __name__ == '__main__':
    main()
