import time
import pickle
import Queue
import numpy as np
from node import Node

from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from generators.uniform import UniformPaPGenerator

from trajectory_representation.operator import Operator
from trajectory_representation.trajectory import Trajectory

from helper import get_actions, compute_heuristic, get_state_class

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
# prm_edges = [set(l) - {i} for i,l in enumerate(prm_edges)]
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0


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
        action_queue.put((hval, float('nan'), a, initnode))  # initial q

    iter = 0
    # beginning of the planner
    while True:
        print "Time limit is ", config.timelimit
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
            print("Sampling for {}".format(action.discrete_parameters.values()))
            a_obj = action.discrete_parameters['two_arm_place_object']
            a_region = action.discrete_parameters['two_arm_place_region']

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
                plan = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                #states = [nd.state for nd in plan]
                plan = [nd.action for nd in plan[1:]] + [action]
                return plan, iter
            else:
                newstate = statecls(mover, goal, node.state, action)
                print "New state computed"
                newnode = Node(node, action, newstate)
                newactions = get_actions(mover, goal, config)
                for newaction in newactions:
                    hval = compute_heuristic(newstate, newaction, pap_model, mover, config)
                    action_queue.put(
                        (hval, float('nan'), newaction, newnode))
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
