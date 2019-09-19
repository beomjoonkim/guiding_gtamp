import os
import time
import pickle
import random
import argparse
import Queue

import cProfile
import pstats
from gtamp_problem_environments.mover_env import Mover
from gtamp_problem_environments.one_arm_mover_env import OneArmMover
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator

from trajectory_representation.operator import Operator
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.one_arm_pap_state import OneArmPaPState
from trajectory_representation.trajectory import Trajectory

from gtamp_utils.utils import set_robot_config, set_obj_xytheta, visualize_path, two_arm_pick_object, \
    two_arm_place_object, get_body_xytheta, grab_obj, release_obj, fold_arms, one_arm_pick_object, one_arm_place_object

import numpy as np
import tensorflow as tf
import openravepy

from generators.uniform import UniformGenerator, PaPUniformGenerator

from manipulation.primitives.display import set_viewer_options, draw_line, draw_point, draw_line
from manipulation.primitives.savers import DynamicEnvironmentStateSaver

from learn.data_traj import extract_individual_example
from learn.pap_gnn import PaPGNN
import collections

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
# prm_edges = [set(l) - {i} for i,l in enumerate(prm_edges)]
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0


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


def compute_heuristic(state, action, pap_model, problem_env):
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


def compute_hcount(state, action, pap_model, problem_env):
    objects_to_move = set()
    queue = Queue.Queue()
    goal_r = [entity for entity in state.goal_entities if 'region' in entity][0]
    for entity in state.goal_entities:
        if 'region' not in entity and not problem_env.regions[goal_r].contains(problem_env.env.GetKinBody(entity).ComputeAABB()):
            queue.put(entity)
    while not queue.empty():
        o = queue.get()
        if o not in objects_to_move:
            objects_to_move.add(o)
            for o2 in problem_env.entity_names:
                if state.binary_edges[(o2,o)][1] or any(state.ternary_edges[(o,o2,r)][0] for r in state.goal_entities
                                                        if 'region' in r and (r not in state.goal_entities or o2 in state.goal_entities)):
                    queue.put(o2)

    if 'two_arm' in problem_env.name:
        a_obj = action.discrete_parameters['two_arm_place_object']
        a_region = action.discrete_parameters['two_arm_place_region']
    else:
        a_obj = action.discrete_parameters['object'].GetName()
        a_region = action.discrete_parameters['region'].name

    if state.nodes[a_obj][9] and (a_obj not in state.goal_entities or a_region in state.goal_entities):
        objects_to_move -= {a_obj}

    return -len(objects_to_move)


def get_problem(mover):
    tt = time.time()

    obj_names = [obj.GetName() for obj in mover.objects]
    n_objs_pack = config.n_objs_pack

    if config.domain == 'two_arm_mover':
        statecls = ShortestPathPaPState
        goal = ['home_region'] + [obj.GetName() for obj in mover.objects[:n_objs_pack]]
    elif config.domain == 'one_arm_mover':
        def create_one_arm_pap_state(*args, **kwargs):
            while True:
                state = OneArmPaPState(*args, **kwargs)
                if len(state.nocollision_place_op) > 0:
                    return state
                else:
                    print('failed to find any paps, trying again')
        statecls = create_one_arm_pap_state
        goal = ['rectangular_packing_box1_region'] + [obj.GetName() for obj in mover.objects[:n_objs_pack]]
    else:
        raise NotImplementedError

    if config.visualize_plan:
        mover.env.SetViewer('qtcoin')
        set_viewer_options(mover.env)

    pr = cProfile.Profile()
    pr.enable()
    state = statecls(mover, goal)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(30)
    pstats.Stats(pr).sort_stats('cumtime').print_stats(30)

    #state.make_pklable()

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'operator n_msg_passing n_layers num_fc_layers n_hidden no_goal_nodes top_k optimizer lr use_mse batch_size seed num_train val_portion num_test mse_weight diff_weight_msg_passing same_vertex_model weight_initializer loss use_region_agnostic')

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
        use_region_agnostic=False
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

    action_queue = Queue.PriorityQueue()  # (heuristic, nan, operator skeleton, state. trajectory)
    initnode = Node(None, None, state)
    initial_state = state
    actions = get_actions(mover, goal, config)
    for a in actions:
        hval = compute_heuristic(state, a, pap_model, mover)
        action_queue.put((hval, float('nan'), a, initnode))  # initial q

    iter = 0
    # beginning of the planner
    while True:
        if time.time() - tt > config.timelimit:
            return None, iter

        iter += 1
        #if 'one_arm' in mover.name:
        #   time.sleep(3.5) # gauged using max_ik_attempts = 20

        if iter > 3000:
            print('failed to find plan: iteration limit')
            return None, iter

        if action_queue.empty():
            actions = get_actions(mover, goal, config)
            for a in actions:
                action_queue.put((compute_heuristic(initial_state, a, pap_model, mover), float('nan'), a, initnode))  # initial q
        #import pdb;pdb.set_trace()
        curr_hval, _, action, node = action_queue.get()
        state = node.state

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
        #utils.set_color(action.discrete_parameters['object'], [1, 0, 0])  # visualization purpose

        if action.type == 'two_arm_pick_two_arm_place':
            smpler = PaPUniformGenerator(action, mover, None)
            smpled_param = smpler.sample_next_point(action, n_iter=200, n_parameters_to_try_motion_planning=3,
                                                    cached_collisions=state.collides, cached_holding_collisions=None)
            if smpled_param['is_feasible']:
                action.continuous_parameters = smpled_param
                action.execute()
                print "Action executed"
            else:
                print "Failed to sample an action"
                #utils.set_color(action.discrete_parameters['object'], [0, 1, 0])  # visualization purpose
                continue

            is_goal_achieved = \
                np.all([mover.regions['home_region'].contains(mover.env.GetKinBody(o).ComputeAABB()) for o in
                        obj_names[:n_objs_pack]])
            if is_goal_achieved:
                print("found successful plan: {}".format(n_objs_pack))
                print iter, time.time()-tt
                import pdb;pdb.set_trace()
                """
                trajectory = Trajectory(mover.seed, mover.seed)
                plan = list(node.backtrack())[::-1] # plan of length 0 is possible I think
                trajectory.states = [nd.state for nd in plan]
                trajectory.actions = [nd.action for nd in plan[1:]] + [action]
                trajectory.rewards = [nd.reward for nd in plan[1:]] + [0]
                trajectory.state_prime = [nd.state for nd in plan[1:]]
                trajectory.seed = mover.seed
                print(trajectory)
                return trajectory, iter
                """
            else:
                newstate = statecls(mover, goal, node.state, action)
                print "New state computed"
                newstate.make_pklable()
                newnode = Node(node, action, newstate)
                newactions = get_actions(mover, goal, config)
                print "Old h value", curr_hval
                for newaction in newactions:
                    hval = compute_heuristic(newstate, newaction, pap_model, mover) - 1. * newnode.depth
                    print "New state h value %.4f for %s %s" % (
                    hval, newaction.discrete_parameters['object'], newaction.discrete_parameters['region'])
                    action_queue.put(
                        (hval, float('nan'), newaction, newnode))
                import pdb;pdb.set_trace()
            #utils.set_color(action.discrete_parameters['object'], [0, 1, 0])  # visualization purpose

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
                papg = OneArmPaPUniformGenerator(action, mover, cached_picks=(node.state.iksolutions[current_region], node.state.iksolutions[r]))
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
                    pr = cProfile.Profile()
                    pr.enable()
                    newstate = statecls(mover, goal, node.state, action)
                    pr.disable()
                    pstats.Stats(pr).sort_stats('tottime').print_stats(30)
                    pstats.Stats(pr).sort_stats('cumtime').print_stats(30)
                    print "New state computed"
                    newnode = Node(node, action, newstate)
                    newactions = get_actions(mover, goal, config)
                    print "Old h value", curr_hval
                    for newaction in newactions:
                        hval = compute_heuristic(newstate, newaction, pap_model, mover) - 1. * newnode.depth
                        action_queue.put((hval, float('nan'), newaction, newnode))

            if not success:
                print('failed to execute action')
            else:
                print('action successful')

        else:
            raise NotImplementedError


##################################################


##################################################

from planners.subplanners.motion_planner import BaseMotionPlanner
import socket


def generate_training_data_single():
    np.random.seed(config.pidx)
    random.seed(config.pidx)
    if config.domain == 'two_arm_mover':
        mover = Mover(config.pidx, config.problem_type)
    elif config.domain == 'one_arm_mover':
        mover = OneArmMover(config.pidx)
    else:
        raise NotImplementedError
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
    mover.set_motion_planner(BaseMotionPlanner(mover, 'prm'))
    mover.seed = config.pidx

    # todo change the root node
    mover.init_saver = DynamicEnvironmentStateSaver(mover.env)

    hostname = socket.gethostname()
    if hostname in {'dell-XPS-15-9560', 'phaedra', 'shakey', 'lab', 'glaucus', 'luke-laptop-1'}:
        root_dir = './'
        #root_dir = '/home/beomjoon/Dropbox (MIT)/cloud_results/'
    else:
        root_dir = '/data/public/rw/pass.port/tamp_q_results/'

    solution_file_dir = root_dir + '/test_results/greedy_results_on_mover_domain/domain_%s/n_objs_pack_%d'\
                        % (config.domain, config.n_objs_pack)

    if config.dont_use_gnn:
        solution_file_dir += '/no_gnn/'
    elif config.dont_use_h:
        solution_file_dir += '/gnn_no_h/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) + '/'
    elif config.hcount:
        solution_file_dir += '/hcount/'
    elif config.hadd:
        solution_file_dir += '/gnn_hadd/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) + '/'
    else:
        solution_file_dir += '/gnn/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) + '/'

    solution_file_name = 'pidx_' + str(config.pidx) + \
                         '_planner_seed_' + str(config.planner_seed) + \
                         '_train_seed_' + str(config.train_seed) + \
                         '_domain_' + str(config.domain) + '.pkl'
    if not os.path.isdir(solution_file_dir):
        os.makedirs(solution_file_dir)

    solution_file_name = solution_file_dir + solution_file_name

    is_problem_solved_before = os.path.isfile(solution_file_name)
    if is_problem_solved_before and not config.plan:
        with open(solution_file_name, 'rb') as f:
            trajectory = pickle.load(f)
            success = trajectory.metrics['success']
            tottime = trajectory.metrics['tottime']
    else:
        t = time.time()
        trajectory, num_nodes = get_problem(mover)
        tottime = time.time() - t
        success = trajectory is not None
        plan_length = len(trajectory.actions) if success else 0
        if not success:
            trajectory = Trajectory(mover.seed, mover.seed)
        trajectory.states = [s.get_predicate_evaluations() for s in trajectory.states]
        trajectory.state_prime = None
        trajectory.metrics = {
            'n_objs_pack': config.n_objs_pack,
            'tottime': tottime,
            'success': success,
            'plan_length': plan_length,
            'num_nodes': num_nodes,
        }

        #with open(solution_file_name, 'wb') as f:
        #    pickle.dump(trajectory, f)
    print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d' % (tottime, success, trajectory.metrics['plan_length'],
                                                                    trajectory.metrics['num_nodes'])
    import pdb;pdb.set_trace()

    """
    print("time: {}".format(','.join(str(trajectory.metrics[m]) for m in [
        'n_objs_pack',
        'tottime',
        'success',
        'plan_length',
        'num_nodes',
    ])))
    """

    print('\n'.join(str(a.discrete_parameters.values()) for a in trajectory.actions))

    def draw_robot_line(env, q1, q2):
        return draw_line(env, list(q1)[:2] + [.5], list(q2)[:2] + [.5], color=(0,0,0,1), width=3.)

    mover.reset_to_init_state_stripstream()
    if not config.dontsimulate:
        if config.visualize_sim:
            mover.env.SetViewer('qtcoin')
            set_viewer_options(mover.env)

            raw_input('Start?')
        handles = []
        for step_idx, action in enumerate(trajectory.actions):
            if action.type == 'two_arm_pick_two_arm_place':
                def check_collisions(q=None):
                    if q is not None:
                        set_robot_config(q, mover.robot)
                    collision = False
                    if mover.env.CheckCollision(mover.robot):
                        collision = True
                    for obj in mover.objects:
                        if mover.env.CheckCollision(obj):
                            collision = True
                    if collision:
                        print('collision')
                        if config.visualize_sim:
                            raw_input('Continue after collision?')

                check_collisions()
                o = action.discrete_parameters['two_arm_place_object']
                pick_params = action.continuous_parameters['pick']
                place_params = action.continuous_parameters['place']

                full_path = [get_body_xytheta(mover.robot)[0]] + pick_params['motion'] + [pick_params['q_goal']] + \
                            place_params['motion'] + [place_params['q_goal']]
                for i, (q1, q2) in enumerate(zip(full_path[:-1], full_path[1:])):
                    if np.linalg.norm(np.squeeze(q1)[:2] - np.squeeze(q2)[:2]) > 1:
                        print(i, q1, q2)
                        import pdb;
                        pdb.set_trace()

                pickq = pick_params['q_goal']
                pickt = pick_params['motion']
                check_collisions(pickq)
                for q in pickt:
                    check_collisions(q)
                obj = mover.env.GetKinBody(o)
                if len(pickt) > 0:
                    for q1, q2 in zip(pickt[:-1], pickt[1:]):
                        handles.append(
                            draw_robot_line(mover.env, q1.squeeze(), q2.squeeze()))
                # set_robot_config(pickq, mover.robot)
                # if config.visualize_sim:
                #   raw_input('Continue?')
                # set_obj_xytheta([1000,1000,0], obj)
                two_arm_pick_object(obj, pick_params)
                if config.visualize_sim:
                    raw_input('Place?')

                o = action.discrete_parameters['two_arm_place_object']
                r = action.discrete_parameters['two_arm_place_region']
                placeq = place_params['q_goal']
                placep = place_params['object_pose']
                placet = place_params['motion']
                check_collisions(placeq)
                for q in placet:
                    check_collisions(q)
                obj = mover.env.GetKinBody(o)
                if len(placet) > 0:
                    for q1, q2 in zip(placet[:-1], placet[1:]):
                        handles.append(
                            draw_robot_line(mover.env, q1.squeeze(), q2.squeeze()))
                # set_robot_config(placeq, mover.robot)
                # if config.visualize_sim:
                #   raw_input('Continue?')
                # set_obj_xytheta(placep, obj)
                two_arm_place_object(place_params)
                check_collisions()

                if config.visualize_sim:
                    raw_input('Continue?')

            elif action.type == 'one_arm_pick_one_arm_place':
                def check_collisions(q=None):
                    if q is not None:
                        set_robot_config(q, mover.robot)
                    collision = False
                    if mover.env.CheckCollision(mover.robot):
                        collision = True
                    for obj in mover.objects:
                        if mover.env.CheckCollision(obj):
                            collision = True
                    if collision:
                        print('collision')
                        if config.visualize_sim:
                            raw_input('Continue after collision?')

                check_collisions()
                pick_params = action.continuous_parameters['pick']
                place_params = action.continuous_parameters['place']

                one_arm_pick_object(mover.env.GetKinBody(action.discrete_parameters['object']), pick_params)

                check_collisions()

                if config.visualize_sim:
                    raw_input('Place?')

                one_arm_place_object(place_params)

                check_collisions()

                if config.visualize_sim:
                    raw_input('Continue?')
            else:
                raise NotImplementedError

        n_objs_pack = config.n_objs_pack
        if config.domain == 'two_arm_mover':
            goal_region = 'home_region'
        elif config.domain == 'one_arm_mover':
            goal_region = 'rectangular_packing_box1_region'
        else:
            raise NotImplementedError

        if all(mover.regions[goal_region].contains(o.ComputeAABB()) for o in
               mover.objects[:n_objs_pack]):
            print("successful plan length: {}".format(len(trajectory.actions)))
        else:
            print('failed to find plan')
        if config.visualize_sim:
            raw_input('Finish?')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-train_seed', type=int, default=0)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-timelimit', type=float, default=600)
    parser.add_argument('-visualize_plan', action='store_true', default=False)
    parser.add_argument('-visualize_sim', action='store_true', default=False)
    parser.add_argument('-dontsimulate', action='store_true', default=False)
    parser.add_argument('-plan', action='store_true', default=False)
    parser.add_argument('-dont_use_gnn', action='store_true', default=False)
    parser.add_argument('-dont_use_h', action='store_true', default=False)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-problem_type', type=str, default='normal') # supports normal, nonmonotonic
    parser.add_argument('-hcount', action='store_true', default=False)
    parser.add_argument('-hadd', action='store_true', default=False)

    config = parser.parse_args()
    generate_training_data_single()
