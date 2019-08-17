import numpy as np
import pdb

from . import data_traj
from .pap_gnn import PaPGNN
from .train import parse_args

is_goal_idx = 2
is_holding_goal_entity_idx = -1
inway_idx = 0


def get_used_entities(nodes, edges, actions):
    entity_nodes = []

    for node, edge, action in zip(nodes, edges, actions):
        selected_entity_node = node[action].squeeze()
        entity_nodes.append(selected_entity_node)

    return np.array(entity_nodes)


def get_n_holding_goal_entity_and_moved_to_goal_region(nodes, edges, actions):
    global is_goal_idx
    global is_holding_goal_entity_idx

    numb = 0
    for node, edge, action in zip(nodes, edges, actions):
        selected_entity_node = node[action].squeeze()
        is_holding_goal_entity = selected_entity_node[is_holding_goal_entity_idx]
        is_goal = selected_entity_node[is_goal_idx]
        if is_holding_goal_entity and is_goal:
            numb += 1

    return float(numb) / len(nodes)


def print_predicate_values(predicate):
    print "is_obj %.5f is_region %.5f is_goal %.5f is_reachable %.5f is_holding_goal_entity %.5f" \
          % (predicate[0], predicate[1], predicate[2], predicate[3], predicate[4])


def get_n_moving_non_goal_inway_to_goal(nodes, edges, actions):
    global is_goal_idx
    global is_holding_goal_entity_idx
    global inway_idx

    numb = 0
    features_of_entities_in_way_to = []
    for node, edge, action in zip(nodes, edges, actions):
        selected_entity_node = node[action].squeeze()
        is_goal = selected_entity_node[is_goal_idx]

        if is_goal:
            continue

        is_in_way_to_any_entity = np.any(edge[action].squeeze()[:, inway_idx])
        if is_in_way_to_any_entity:
            entity_indices_in_way_to = np.nonzero(edge[action].squeeze()[:, inway_idx])[0]
            for idx in entity_indices_in_way_to:
                in_way_to = node[idx].squeeze()
                features_of_entities_in_way_to.append(in_way_to)
                if in_way_to[is_goal_idx]:
                    numb += 1
    features_of_entities_in_way_to = np.array(features_of_entities_in_way_to)
    print "Features of entities occluded by the moved non-goal entities:"
    print_predicate_values(features_of_entities_in_way_to.sum(axis=0) / len(nodes))
    import pdb;
    pdb.set_trace()
    return float(numb) / len(nodes)


def print_operator_entity_statistics(operator):
    print "======Operator: ", operator
    nodes, edges, actions, costs = data_traj.load_data(
        './test_results/mcts_results_on_mover_domain/widening_5/uct_1.0/trajectory_data', operator, )
    nodes = nodes[:, :, 6:]
    print "n data %d" % (len(nodes))

    entity_nodes = get_used_entities(nodes, edges, actions)

    if operator == 'two_arm_place':
        n_holding_goal_and_moved_to_goal = get_n_holding_goal_entity_and_moved_to_goal_region(nodes, edges, actions)
        print 'n_holding_goal_and_moved_to_goal:', n_holding_goal_and_moved_to_goal
    else:
        stat = get_n_moving_non_goal_inway_to_goal(nodes, edges, actions)
        print 'n moving non-goal entity not in way to a goal entity', stat

    """
    0 - entity not in self.problem_env.regions,  # IsObj
    1 - entity in self.problem_env.regions,  # IsRoom
    2 - entity in goal,  # IsGoal
    3 - is_entity_reachable,
    4 - blocks_key_configs(entity, goal),
    5 - self.is_holding()
    """
    predicate = entity_nodes.mean(axis=0)
    print_predicate_values(predicate)


def print_node_predicates(node):
    print "is_obj %d is_region %d is_goal %d is_reachable %d is_holding_goal_entity %d" \
          % (node[1], node[3], node[5], node[7], node[9])
    return node[7], node[5]


def print_src_edge_predicates(edge, selected_region, goal_entity_idx):
    # following indices indicate predicate evaluating to True
    pick_inway1_idx = 1  # uses the selected object as being in the way, PickInWay(selected, other_obj)
    in_region1_idx = 3  # InRegion(selected, other_entity)
    pick_inway2_idx = 5  # uses the target object as the object to pickup, PickInWay(other_obj, selected)
    in_region2_idx = 7  # InRegion(other_entity, selected)
    place_inway1_idx = 9  # use the selected object as holding, PlaceInWay(selected_obj, other_obj, selected_region)
    place_inway2_idx = 11  # use the selected object as, PlaceInWay(other_obj, selected_obj, selected_region)

    assert np.all(edge[:, 0, :8] == edge[:, 1, :8])
    clear_path_to_pick_object = np.all(edge[:, :, pick_inway2_idx] == 0)
    clear_path_to_region_with_obj_in_hand_to_home = np.all(edge[:, 0, place_inway1_idx] == 0)
    clear_path_to_region_with_obj_in_hand_to_loading = np.all(edge[:, 1, place_inway1_idx] == 0)

    print "clear_path_to_pick_obj: %d \n" \
          "clear_path_to_place_obj_to_home: %d \n" \
          "clear_path_to_place_obj_to_loading: %d" \
          % (clear_path_to_pick_object,
             clear_path_to_region_with_obj_in_hand_to_home,
             clear_path_to_region_with_obj_in_hand_to_loading)

    is_object_in_way_to_pick_path_of_any_object = np.any(edge[goal_entity_idx, :, pick_inway1_idx] == 1)
    is_object_in_way_to_place_path_of_any_object_to_home = np.any(edge[goal_entity_idx, 0, place_inway2_idx] == 1)
    is_object_in_way_to_place_path_of_any_object_to_loading = np.any(edge[goal_entity_idx, 1, place_inway2_idx] == 1)
    print "is_obj_in_way_to_pick_path_of_any_object %d \n" \
          "is_obj_in_way_to_place_path_of_any_object_to_home %d \n" \
          "is_obj_in_way_to_place_path_of_any_object_to_loading %d" \
          % (is_object_in_way_to_pick_path_of_any_object,
             is_object_in_way_to_place_path_of_any_object_to_home,
             is_object_in_way_to_place_path_of_any_object_to_loading)

    selected_region = 'home' if selected_region == 0 else 'loading'
    print "Selected region: %s \n" % selected_region

    return clear_path_to_pick_object


def pap_print_entity_statistics():
    nodes, edges, actions, costs = data_traj.load_data(
        './test_results/hpn_results_on_mover_domain/1/trajectory_data/special_cases', 'two_arm_pick_two_arm_place')
    nodes = nodes[:, :, 6:]
    print "n data %d" % (len(nodes))

    num_entities = 11

    n_disagree = 0
    n_reachable = 0
    n_clear_way = 0
    n_goal = 0
    for data_idx in range(len(nodes)):
        action = actions[data_idx]
        node = nodes[data_idx]
        edge = edges[data_idx]
        goal_obj = node[:, 1] * node[:, 5]
        goal_entity_idx = np.where(goal_obj)[0][0]

        chosen_obj = np.where(action)[0]
        chosen_region = np.where(action)[1]
        chosen_obj_node = node[chosen_obj].squeeze()
        chosen_obj_edge_src = edge[chosen_obj, :, :, :].squeeze()

        is_reachable, is_goal = print_node_predicates(chosen_obj_node)
        nothing_in_way_to_pick = print_src_edge_predicates(chosen_obj_edge_src, chosen_region, goal_entity_idx)

        if is_reachable != nothing_in_way_to_pick:
            n_disagree += 1

        n_reachable += is_reachable
        n_clear_way += nothing_in_way_to_pick
        n_goal += is_goal

    print 'reachable %.2f clear_way %.2f disagree_reachable_and_clear_way ' \
          '%.2f, n_goal %.2f' % (float(n_reachable) / len(nodes),
                                 float(n_clear_way) / len(nodes),
                                 float(n_disagree) / len(nodes),
                                 float(n_goal) / len(nodes))



def main():
    # print_operator_entity_statistics('two_arm_pick')
    # print_operator_entity_statistics('two_arm_place')
    pap_print_entity_statistics()


if __name__ == '__main__':
    main()
