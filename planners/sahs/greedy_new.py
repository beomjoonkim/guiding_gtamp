import time
import pickle
import Queue
import numpy as np
from node import Node
from gtamp_utils import utils

from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from generators.uniform import UniformPaPGenerator, PaPUniformGenerator

from trajectory_representation.operator import Operator
from trajectory_representation.trajectory import Trajectory

from helper import get_actions, compute_heuristic, get_state_class

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0

from learn.data_traj import extract_individual_example


def compute_heuristic(state, action, pap_model, problem_env, config):
    is_two_arm_domain = 'two_arm_place_object' in action.discrete_parameters
    if is_two_arm_domain:
        o = action.discrete_parameters['two_arm_place_object']
        r = action.discrete_parameters['two_arm_place_region']
    else:
        o = action.discrete_parameters['object'].GetName()
        r = action.discrete_parameters['region'].name

    nodes, edges, actions, _ = extract_individual_example(state, action)
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
        hcount = compute_hcount(state, action, pap_model, problem_env)
        print "%s %s %.4f" % (o, r, hcount)
        return hcount
    elif config.dont_use_gnn:
        return -number_in_goal
    elif config.dont_use_h:
        gnn_pred = -pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...],
                                                            actions[None, ...])
        return gnn_pred
    else:
        gnn_pred = -pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...], actions[None, ...])
        hval = -number_in_goal + gnn_pred

        o_reachable = state.is_entity_reachable(o)
        o_r_manip_free = state.binary_edges[(o, r)][-1]
        print '%s %s prefree %d manipfree %d numb_in_goal %d qval %.4f hval %.4f' % (
            o, r, o_reachable, o_r_manip_free, number_in_goal, gnn_pred, hval)
        if not is_two_arm_domain:
            obj_name = action.discrete_parameters['object'].GetName()
            region_name = action.discrete_parameters['region'].name
            is_reachable = state.nodes[obj_name][-2] #state.is_entity_reachable(obj_name)
            is_placeable = state.binary_edges[(obj_name, region_name)][2]
            is_goal = state.nodes[obj_name][-3]
            isgoal_region = state.nodes[region_name][-3]
            is_in_region = state.binary_edges[(obj_name, region_name)][0]
            in_way_of_goal_pap = obj_name in state.get_entities_in_way_to_goal_entities()
            print "%15s %35s reachable %d placeable_in_region %d isgoal %d isgoal_region %d is_in_region %d  num_in_goal %d in_way_of_goal_pap %d gnn %.4f hval %.4f" \
                  % (obj_name, region_name, is_reachable, is_placeable, is_goal, isgoal_region, is_in_region, number_in_goal, in_way_of_goal_pap, -gnn_pred, hval)

        return hval

"""
def get_actions(mover, goal, config):
    actions = []
    for o in mover.entity_names:
        if 'region' in o:
            continue
        for r in mover.entity_names:
            if 'region' not in r:
                continue
            if o not in goal and r in goal:
                # you cannot place non-goal object in the goal region
                continue
            if 'entire' in r: #and config.domain == 'two_arm_mover':
                continue

            if config.domain == 'two_arm_mover':
                action = Operator('two_arm_pick_two_arm_place', {'two_arm_place_object': o, 'two_arm_place_region': r})
                # following two lines are for legacy reasons, will fix later
                action.discrete_parameters['object'] = action.discrete_parameters['two_arm_place_object']
                action.discrete_parameters['region'] = action.discrete_parameters['two_arm_place_region']
            elif config.domain == 'one_arm_mover':
                action = Operator('one_arm_pick_one_arm_place',
                                  {'object': mover.env.GetKinBody(o), 'region': mover.regions[r]})

            else:
                raise NotImplementedError
            actions.append(action)

    return actions
"""

def search(mover, config, pap_model):
    tt = time.time()

    obj_names = [obj.GetName() for obj in mover.objects]
    n_objs_pack = config.n_objs_pack
    statecls = get_state_class(config.domain)
    goal = mover.goal
    mover.reset_to_init_state_stripstream()
    depth_limit = 60

    state = statecls(mover, goal)

    # lowest valued items are retrieved first in PriorityQueue
    action_queue = Queue.PriorityQueue()  # (heuristic, nan, operator skeleton, state. trajectory);
    initnode = Node(None, None, state)
    initial_state = state
    actions = get_actions(mover, goal, config)
    for a in actions:
        hval = compute_heuristic(state, a, pap_model, mover, config)
        discrete_params = (a.discrete_parameters['object'], a.discrete_parameters['region'])
        initnode.set_heuristic(discrete_params, hval)
        action_queue.put((hval, float('nan'), a, initnode))  # initial q

    """
    other = pickle.load(open('/home/beomjoon/Documents/github/qqq/tmp.pkl','r'))
    other_binary = other[1]
    keys = state.binary_edges.keys()
    for k in keys:
        #print k, state.binary_edges[k], other_binary[k], np.array(state.binary_edges[k]) == np.array(other_binary[k])
        if not np.all( np.array(state.binary_edges[k]) == np.array(other_binary[k]) ):
            print "Wrong!", k, state.binary_edges[k], np.array(other_binary[k])
    keys = state.ternary_edges.keys()
    other_ternary = other[2]
    for k in keys:
        if not np.all( np.array(state.ternary_edges[k]) == np.array(other_ternary[k]) ):
            print "Wrong!", k, state.ternary_edges[k], np.array(other_ternary[k])
    import pdb;pdb.set_trace()
    """

    iter = 0
    # beginning of the planner
    while True:
        iter += 1
        curr_time = time.time() - tt
        print "Time %.2f / %.2f " % (curr_time, config.timelimit)
        print "Iter %d / %d" % (iter, config.num_node_limit)
        if curr_time > config.timelimit or iter > config.num_node_limit:
            return None, iter

        if action_queue.empty():
            actions = get_actions(mover, goal, config)
            for a in actions:
                discrete_params = (a.discrete_parameters['object'], a.discrete_parameters['region'])
                hval = initnode.heuristic_vals[discrete_params]
                action_queue.put((hval, float('nan'), a, initnode))  # initial q

        curr_hval, _, action, node = action_queue.get()
        state = node.state
        print "Curr hval", curr_hval

        if node.depth > depth_limit:
            print('skipping because depth limit', node.action.discrete_parameters.values())

        # reset to state
        state.restore(mover)

        if action.type == 'two_arm_pick_two_arm_place':
            print("Sampling for {}".format(action.discrete_parameters.values()))
            a_obj = action.discrete_parameters['two_arm_place_object']
            a_region = action.discrete_parameters['two_arm_place_region']
            smpler = PaPUniformGenerator(action, mover, None)
            smpled_param = smpler.sample_next_point(action, n_iter=200, n_parameters_to_try_motion_planning=3,
                                                    cached_collisions=state.collides, cached_holding_collisions=None)
            """
            smpler = UniformPaPGenerator(None, action, mover, None,
                                         n_candidate_params_to_smpl=3,
                                         total_number_of_feasibility_checks=200,
                                         dont_check_motion_existence=False)
            smpled_param = smpler.sample_next_point(cached_collisions=state.collides,
                                                    cached_holding_collisions=None)
            """

            if smpled_param['is_feasible']:
                action.continuous_parameters = smpled_param
                action.execute()
                print "Action executed"
            else:
                print "Failed to sample an action"
                continue

            is_goal_achieved = \
                np.all([mover.regions['home_region'].contains(mover.env.GetKinBody(o).ComputeAABB()) for o in
                        obj_names[:n_objs_pack]])
            if is_goal_achieved:
                print("found successful plan: {}".format(n_objs_pack))
                plan = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                plan = [nd.action for nd in plan[1:]] + [action]
                return plan, iter
            else:
                newstate = statecls(mover, goal, node.state, action)
                print "New state computed"
                newnode = Node(node, action, newstate)
                newactions = get_actions(mover, goal, config)
                for newaction in newactions:
                    hval = compute_heuristic(newstate, newaction, pap_model, mover, config) #- 1. * newnode.depth
                    action_queue.put((hval, float('nan'), newaction, newnode))
                #import pdb;pdb.set_trace()

        elif action.type == 'one_arm_pick_one_arm_place':
            print("Sampling for {}".format(action.discrete_parameters.values()))
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
                    plan = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                    plan = [nd.action for nd in plan[1:]] + [action]
                    return plan, iter
                else:
                    newstate = statecls(mover, goal, node.state, action)
                    print "New state computed"
                    newnode = Node(node, action, newstate)
                    newactions = get_actions(mover, goal, config)
                    print "Old h value", curr_hval
                    for newaction in newactions:
                        hval = compute_heuristic(newstate, newaction, pap_model, mover, config)
                        action_queue.put((hval, float('nan'), newaction, newnode))
            if not success:
                print('failed to execute action')
            else:
                print('action successful')

        else:
            raise NotImplementedError
