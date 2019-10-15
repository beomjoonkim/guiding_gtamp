import time
import pickle
import Queue
import numpy as np
from node import Node
from gtamp_utils import utils

from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from generators.uniform import PaPUniformGenerator
from generators.learned_generator import LearnedGenerator

from trajectory_representation.operator import Operator
from trajectory_representation.concrete_node_state import ConcreteNodeState

from helper import get_actions, compute_heuristic, get_state_class

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0


def create_state_vec(key_config_obstacles, action, goal_entities):
    obj = action.discrete_parameters['object']
    region = action.discrete_parameters['region']
    one_hot = utils.convert_binary_vec_to_one_hot(key_config_obstacles)
    is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([obj in goal_entities]))
    is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([region in goal_entities]))

    state_vec = np.vstack([one_hot, is_goal_obj, is_goal_region])
    state_vec = state_vec.reshape((1, len(state_vec), 2, 1))

    return state_vec


def get_pick_base_poses(action, smples):
    pick_base_poses = []
    for smpl in smples:
        smpl = smpl[0:4]
        sin_cos_encoding = smpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        smpl = np.hstack([smpl[0:2], decoded_angle])
        abs_base_pose = utils.get_absolute_pick_base_pose_from_ir_parameters(smpl, action.discrete_parameters['object'])
        pick_base_poses.append(abs_base_pose)
    return pick_base_poses


def get_place_base_poses(action, smples, mover):
    place_base_poses = smples[:, 4:]
    to_return = []
    for bsmpl in place_base_poses:
        sin_cos_encoding = bsmpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        bsmpl = np.hstack([bsmpl[0:2], decoded_angle])
        to_return.append(bsmpl)
    to_return = np.array(to_return)
    to_return[:, 0:2] += mover.regions[action.discrete_parameters['region']].box[0]
    return to_return


def search(mover, config, pap_model, learned_smpler=None):
    tt = time.time()

    obj_names = [obj.GetName() for obj in mover.objects]
    n_objs_pack = config.n_objs_pack
    statecls = get_state_class(config.domain)
    goal = mover.goal
    mover.reset_to_init_state_stripstream()
    depth_limit = 60

    state = statecls(mover, goal)

    # utils.viewer()
    """
    actions = get_actions(mover, goal, config)
    action = actions[0]
    utils.set_color(action.discrete_parameters['object'], [1, 0, 0])
    state_vec = create_state_vec(state.key_config_obstacles, action, goal)
    smpler = LearnedGenerator(action, mover, learned_smpler, state_vec)
    smples = np.vstack([smpler.sampler.generate(state_vec) for _ in range(10)])
    pick_base_poses = get_pick_base_poses(action, smples)
    place_base_poses = get_place_base_poses(action, smples, mover)
    utils.visualize_path(place_base_poses)
    #utils.visualize_path(pick_base_poses)
    import pdb;
    pdb.set_trace()

    smpled_param = smpler.sample_next_point(action, n_iter=200, n_parameters_to_try_motion_planning=3,
                                            cached_collisions=state.collides,
                                            cached_holding_collisions=None)
    """

    # lowest valued items are retrieved first in PriorityQueue
    action_queue = Queue.PriorityQueue()  # (heuristic, nan, operator skeleton, state. trajectory);
    initnode = Node(None, None, state)
    initial_state = state
    actions = get_actions(mover, goal, config)
    nodes = [initnode]
    for a in actions:
        hval = compute_heuristic(state, a, pap_model, mover, config)
        discrete_params = (a.discrete_parameters['object'], a.discrete_parameters['region'])
        initnode.set_heuristic(discrete_params, hval)
        action_queue.put((hval, float('nan'), a, initnode))  # initial q

    iter = 0
    # beginning of the planner
    while True:
        iter += 1
        curr_time = time.time() - tt
        print "Time %.2f / %.2f " % (curr_time, config.timelimit)
        print "Iter %d / %d" % (iter, config.num_node_limit)
        if curr_time > config.timelimit or iter > config.num_node_limit:
            return None, iter, nodes

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
            if learned_smpler is None:
                smpler = PaPUniformGenerator(action, mover, max_n_iter=200)
                stime = time.time()
                smpled_param = smpler.sample_next_point(action, n_parameters_to_try_motion_planning=3,
                                                        cached_collisions=state.collides,
                                                        cached_holding_collisions=None)
                print 'smpling time', time.time() - stime
            else:
                smpler = LearnedGenerator(action, mover, learned_smpler, state, max_n_iter=200)
                # How can I change the state.collides to the one_hot? How long would it take?
                stime = time.time()
                smpled_param = smpler.sample_next_point(action, n_parameters_to_try_motion_planning=3,
                                                        cached_collisions=state.collides,
                                                        cached_holding_collisions=None)
                print 'smpling time', time.time() - stime

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
                return plan, iter, nodes
            else:
                newstate = statecls(mover, goal, node.state, action)
                print "New state computed"
                newnode = Node(node, action, newstate)
                newactions = get_actions(mover, goal, config)
                for newaction in newactions:
                    hval = compute_heuristic(newstate, newaction, pap_model, mover, config) - 1. * newnode.depth
                    action_queue.put((hval, float('nan'), newaction, newnode))
                nodes.append(newnode)

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
                    return plan, iter, nodes
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
