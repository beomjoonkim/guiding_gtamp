from learn.data_traj import extract_individual_example
from planners.heuristics import compute_hcount_with_action, compute_hcount
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.one_arm_pap_state import OneArmPaPState
from learn.data_traj import get_actions as convert_action_to_predictable_form
import numpy as np
import time


def get_actions(mover, goal, config):
    return mover.get_applicable_ops()


def get_state_class(domain):
    if domain == 'two_arm_mover':
        statecls = ShortestPathPaPState
    elif domain == 'one_arm_mover':
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
    return statecls


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
    goal_objs = [tmp_o for tmp_o in state.goal_entities if 'box' in tmp_o]
    if 'two_arm' in problem_env.name:
        goal_region = 'home_region'
        for obj_name in goal_objs:
            is_obj_in_goal_region = state.binary_edges[(obj_name, goal_region)][0]
            if is_obj_in_goal_region:
                number_in_goal += 1
    else:
        raise NotImplementedError
    if config.hcount:
        hcount = compute_hcount_with_action(state, action, problem_env)
        print "%s %s %.4f" % (o, r, hcount)
        return hcount
    elif config.state_hcount:
        hcount = compute_hcount(state, problem_env)
        print "state_hcount %s %s %.4f" % (o, r, hcount)
        return hcount
    elif config.qlearned_hcount:
        all_actions = get_actions(problem_env, None, None)
        entity_names = list(state.nodes.keys())[::-1]
        # todo clean this up later;
        q_vals = []
        for a in all_actions:
            a_raw_form = convert_action_to_predictable_form(a, entity_names)
            q_vals.append(np.exp(pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...], a_raw_form[None, ...])))
        q_val_on_curr_a = pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...], actions[None, ...])
        q_bonus = np.exp(q_val_on_curr_a) / np.sum(q_vals)

        # hval = -number_in_goal + gnn_pred
        hcount = compute_hcount(state, problem_env)

        hval = hcount - config.mixrate * q_bonus
        o_reachable = state.is_entity_reachable(o)
        o_r_manip_free = state.binary_edges[(o, r)][-1]

        print 'n_in_goal %d %s %s prefree %d manipfree %d hcount %d qbonus %.4f hval %.4f' % (
            number_in_goal, o, r, o_reachable, o_r_manip_free, hcount, -q_bonus, hval)
        return hval
    else:
        qval = pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...], actions[None, ...])
        hval = -number_in_goal - qval
        o_reachable = state.is_entity_reachable(o)
        o_r_manip_free = state.binary_edges[(o, r)][-1]

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
        print 'n_in_goal %d %s %s prefree %d manipfree %d qval %.4f hval %.4f' % (
            number_in_goal, o, r, o_reachable, o_r_manip_free, qval, hval)
        return hval
