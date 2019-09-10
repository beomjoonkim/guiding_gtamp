import os
import time
import pickle
import random
import argparse
import Queue

import cProfile
import pstats
from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import OneArmMover
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from generators.uniform import UniformPaPGenerator

from trajectory_representation.operator import Operator
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.one_arm_pap_state import OneArmPaPState
from trajectory_representation.trajectory import Trajectory

from mover_library.utils import set_robot_config, set_obj_xytheta, visualize_path, two_arm_pick_object, \
    two_arm_place_object, get_body_xytheta, grab_obj, release_obj, fold_arms, one_arm_pick_object, one_arm_place_object

import numpy as np
import tensorflow as tf
import openravepy



from mover_library import utils
from manipulation.primitives.display import set_viewer_options, draw_line, draw_point, draw_line
from manipulation.primitives.savers import DynamicEnvironmentStateSaver

from learn.data_traj import extract_individual_example
from learn.pap_gnn import PaPGNN

from planners.heuristics import compute_hcount_with_action, compute_hcount
import collections

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
# prm_edges = [set(l) - {i} for i,l in enumerate(prm_edges)]
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0


def get_actions(mover, goal, config):
    return mover.get_applicable_ops()


def compute_heuristic(state, action, pap_model, problem_env, config):
    is_two_arm_domain = 'two_arm_place_object' in action.discrete_parameters
    if is_two_arm_domain:
        o = action.discrete_parameters['two_arm_place_object']
        r = action.discrete_parameters['two_arm_place_region']
    else:
        o = action.discrete_parameters['object'].GetName()
        r = action.discrete_parameters['region'].name

    nodes, edges, actions = extract_individual_example(state, action)
    nodes = nodes[..., 6:]

    region_is_goal = state.nodes[r][8]
    number_in_goal = 0

    for i in state.nodes:
        if i == o:
            continue
        for tmpr in problem_env.regions:
            if tmpr in state.nodes:
                is_r_goal_region = state.nodes[tmpr][8]
                if is_r_goal_region:
                    is_i_in_r = state.binary_edges[(i, tmpr)][0]
                    if is_r_goal_region:
                        number_in_goal += is_i_in_r
    number_in_goal += int(region_is_goal)

    if config.hcount:
        hcount = compute_hcount_with_action(state, action, problem_env)
        print "%s %s %.4f" % (o, r, hcount)
        return hcount
    elif config.state_hcount:
        hcount = compute_hcount(state, problem_env)
        print "state_hcount %s %s %.4f" % (o, r, hcount)
        return hcount
    elif config.hadd:
        goal_regions = [goal_r for goal_r in state.goal_entities if 'region' in goal_r]
        assert len(goal_regions) == 1
        goal_r = goal_regions[0]
        goal_objects = [goal_o for goal_o in state.goal_entities if 'region' not in goal_o]
        hadd = 0
        for goal_o in goal_objects:
            for entity, features in state.nodes.items():
                features[8] = False
            state.nodes[goal_o][8] = True
            state.nodes[goal_r][8] = True

            nodes, edges, actions, _ = extract_individual_example(state, action)
            nodes = nodes[..., 6:]
            gnn_pred = -pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...], actions[None, ...])
            hadd += gnn_pred
        return hadd
    else:
        #hcount = compute_hcount_with_action(state, action, problem_env)

        gnn_pred = -pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...], actions[None, ...])
        hval = gnn_pred - number_in_goal

        """
        Vpre_free = state.nodes[action.discrete_parameters['object']][9]
        Vmanip_free = state.binary_edges[(action.discrete_parameters['object'], action.discrete_parameters['region'])][2]
        Vpre_occ = state.pick_entities_occluded_by(action.discrete_parameters['object'])
        Vmanip_occ = state.place_entities_occluded_by(action.discrete_parameters['object'])
        print "Vpre_free", Vpre_free
        print "Vmanip_free", Vmanip_free
        print "Vpre_occ ", Vpre_occ
        print "Vmanip_occ ", Vmanip_occ
        print "Total occ " + str(len(Vpre_occ + Vmanip_occ))
        print "Occ place to goal " + str(len([objregion for objregion in Vmanip_occ if objregion[1] == 'home_region']))

        print "%s %s hval: %.9f hcount: %d" % (o, r, hval, hcount)
        print "====================="
        """
        return hval


def search(mover, config):
    tt = time.time()

    obj_names = [obj.GetName() for obj in mover.objects]
    n_objs_pack = config.n_objs_pack

    if config.domain == 'two_arm_mover':
        statecls = ShortestPathPaPState
        # goal = ['home_region'] + [obj.GetName() for obj in mover.objects[:n_objs_pack]]
    elif config.domain == 'one_arm_mover':
        def create_one_arm_pap_state(*args, **kwargs):
            while True:
                state = OneArmPaPState(*args, **kwargs)
                if len(state.nocollision_place_op) > 0:
                    return state
                else:
                    print('failed to find any paps, trying again')

        statecls = create_one_arm_pap_state
    else:
        raise NotImplementedError

    goal = mover.goal
    if config.visualize_plan:
        mover.env.SetViewer('qtcoin')
        set_viewer_options(mover.env)

    # state.make_pklable()

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'operator n_msg_passing n_layers num_fc_layers n_hidden no_goal_nodes top_k optimizer lr use_mse batch_size seed num_train val_portion num_test mse_weight diff_weight_msg_passing same_vertex_model weight_initializer loss')

    assert config.num_train <= 5000
    pap_mconfig = mconfig_type(
        operator='two_arm_pick_two_arm_place',
        n_msg_passing=1,
        n_layers=2,
        num_fc_layers=2,
        n_hidden=32,
        no_goal_nodes=False,

        top_k=1,
        optimizer='adam',
        lr=1e-4,
        use_mse=True,

        batch_size='32',
        seed=config.train_seed,
        num_train=config.num_train,
        val_portion=.1,
        num_test=1882,
        mse_weight=1.0,
        diff_weight_msg_passing=False,
        same_vertex_model=False,
        weight_initializer='glorot_uniform',
        loss=config.loss,
    )
    if config.domain == 'two_arm_mover':
        num_entities = 11
        n_regions = 2
    elif config.domain == 'one_arm_mover':
        num_entities = 12
        n_regions = 2
    else:
        raise NotImplementedError
    num_node_features = 10
    num_edge_features = 44
    entity_names = mover.entity_names

    with tf.variable_scope('pap'):
        pap_model = PaPGNN(num_entities, num_node_features, num_edge_features, pap_mconfig, entity_names, n_regions)
    pap_model.load_weights()

    mover.reset_to_init_state_stripstream()
    depth_limit = 60

    class Node(object):
        def __init__(self, parent, action, state, reward=0):
            self.parent = parent  # parent.state is initial state
            self.action = action
            self.state = state  # resulting state
            self.reward = reward  # resulting reward

            if parent is None:
                self.depth = 1
            else:
                self.depth = parent.depth + 1

        def backtrack(self):
            node = self
            while node is not None:
                yield node
                node = node.parent

    state = statecls(mover, goal)

    # lowest valued items are retrieved first in PriorityQueue
    action_queue = Queue.PriorityQueue()  # (heuristic, nan, operator skeleton, state. trajectory);
    initnode = Node(None, None, state)
    initial_state = state
    actions = get_actions(mover, goal, config)
    for a in actions:
        hval = compute_heuristic(state, a, pap_model, mover, config)
        action_queue.put((hval, float('nan'), a, initnode))  # initial q

    iter = 0
    # beginning of the planner
    while True:
        if time.time() - tt > config.timelimit:
            return None, iter

        iter += 1
        # if 'one_arm' in mover.name:
        #   time.sleep(3.5) # gauged using max_ik_attempts = 20

        if iter > 3000:
            print('failed to find plan: iteration limit')
            return None, iter

        if action_queue.empty():
            actions = get_actions(mover, goal, config)
            for a in actions:
                action_queue.put((compute_heuristic(initial_state, a, pap_model, mover, config), float('nan'), a,
                                  initnode))  # initial q

        curr_hval, _, action, node = action_queue.get()
        state = node.state
        print "Curr hval", curr_hval

        print('\n'.join([str(parent.action.discrete_parameters.values()) for parent in list(node.backtrack())[-2::-1]]))
        print("{}".format(action.discrete_parameters.values()))

        if node.depth >= 2 and action.type == 'two_arm_pick' and node.parent.action.discrete_parameters['object'] == \
                action.discrete_parameters['object']:  # and plan[-1][1] == r:
            print('skipping because repeat', action.discrete_parameters['object'])
            continue

        if node.depth > depth_limit:
            print('skipping because depth limit', node.action.discrete_parameters.values())

        # reset to state
        state.restore(mover)
        # utils.set_color(action.discrete_parameters['object'], [1, 0, 0])  # visualization purpose

        if action.type == 'two_arm_pick_two_arm_place':
            smpler = UniformPaPGenerator(None, action, mover, None,
                                         n_candidate_params_to_smpl=3,
                                         total_number_of_feasibility_checks=200,
                                         dont_check_motion_existence=False)

            smpled_param = smpler.sample_next_point(cached_collisions=state.collisions_at_all_obj_pose_pairs,
                                                    cached_holding_collisions=None)
            if smpled_param['is_feasible']:
                action.continuous_parameters = smpled_param
                action.execute()
                print "Action executed"
            else:
                print "Failed to sample an action"
                # utils.set_color(action.discrete_parameters['object'], [0, 1, 0])  # visualization purpose
                continue

            is_goal_achieved = \
                np.all([mover.regions['home_region'].contains(mover.env.GetKinBody(o).ComputeAABB()) for o in
                        obj_names[:n_objs_pack]])
            if is_goal_achieved:
                print("found successful plan: {}".format(n_objs_pack))
                trajectory = Trajectory(mover.seed, mover.seed)
                plan = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                #trajectory.states = [nd.state for nd in plan]
                trajectory.actions = [nd.action for nd in plan[1:]] + [action]
                #trajectory.rewards = [nd.reward for nd in plan[1:]] + [0]
                #trajectory.state_prime = [nd.state for nd in plan[1:]]
                trajectory.seed = mover.seed
                print(trajectory)
                return trajectory, iter
            else:
                newstate = statecls(mover, goal, node.state, action)
                print "New state computed"
                newnode = Node(node, action, newstate)
                newactions = get_actions(mover, goal, config)
                for newaction in newactions:
                    # What's this?
                    # hval = compute_heuristic(newstate, newaction, pap_model, mover, config) - 1. * newnode.depth
                    hval = compute_heuristic(newstate, newaction, pap_model, mover, config)
                    # print "New state h value %.4f for %s %s" % (
                    # hval, newaction.discrete_parameters['object'], newaction.discrete_parameters['region'])
                    action_queue.put(
                        (hval, float('nan'), newaction, newnode))
            # utils.set_color(action.discrete_parameters['object'], [0, 1, 0])  # visualization purpose

        elif action.type == 'one_arm_pick_one_arm_place':
            success = False

            obj = action.discrete_parameters['object']
            region = action.discrete_parameters['region']
            o = obj.GetName()
            r = region.name

            if (o, r) in state.nocollision_place_op:
                pick_op, place_op = node.state.nocollision_place_op[(o, r)]
                pap_params = pick_op.continuous_parameters, place_op.continuous_parameters
            else:
                mover.enable_objects()
                current_region = mover.get_region_containing(obj).name
                papg = OneArmPaPUniformGenerator(action, mover, cached_picks=(
                node.state.iksolutions[current_region], node.state.iksolutions[r]))
                pick_params, place_params, status = papg.sample_next_point(500)
                if status == 'HasSolution':
                    pap_params = pick_params, place_params
                else:
                    pap_params = None

            if pap_params is not None:
                pick_params, place_params = pap_params
                action = Operator(
                    operator_type='one_arm_pick_one_arm_place',
                    discrete_parameters={
                        'object': obj,
                        'region': mover.regions[r],
                    },
                    continuous_parameters={
                        'pick': pick_params,
                        'place': place_params,
                    }
                )
                action.execute()

                success = True

                is_goal_achieved = \
                    np.all([mover.regions['rectangular_packing_box1_region'].contains(
                        mover.env.GetKinBody(o).ComputeAABB()) for o in obj_names[:n_objs_pack]])

                if is_goal_achieved:
                    print("found successful plan: {}".format(n_objs_pack))
                    trajectory = Trajectory(mover.seed, mover.seed)
                    plan = list(node.backtrack())[::-1]
                    trajectory.states = [nd.state for nd in plan]
                    for s in trajectory.states:
                        s.pap_params = None
                        s.pick_params = None
                        s.place_params = None
                        s.nocollision_pick_op = None
                        s.collision_pick_op = None
                        s.nocollision_place_op = None
                        s.collision_place_op = None
                    trajectory.actions = [nd.action for nd in plan[1:]] + [action]
                    for op in trajectory.actions:
                        op.discrete_parameters = {
                            key: value.name if 'region' in key else value.GetName()
                            for key, value in op.discrete_parameters.items()
                        }
                    trajectory.rewards = [nd.reward for nd in plan[1:]]
                    trajectory.state_prime = [nd.state for nd in plan[1:]]
                    trajectory.seed = mover.seed
                    print(trajectory)
                    return trajectory, iter
                else:
                    newstate = statecls(mover, goal, node.state, action)
                    print "New state computed"
                    newnode = Node(node, action, newstate)
                    newactions = get_actions(mover, goal, config)
                    print "Old h value", curr_hval
                    for newaction in newactions:
                        hval = compute_heuristic(newstate, newaction, pap_model, mover, config) - 1. * newnode.depth
                        action_queue.put((hval, float('nan'), newaction, newnode))

            if not success:
                print('failed to execute action')
            else:
                print('action successful')

        else:
            raise NotImplementedError

