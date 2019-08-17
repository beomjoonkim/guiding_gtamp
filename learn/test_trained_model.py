#!/bin/env python2

import argparse
import random
import tensorflow as tf
import numpy as np

from . import data_traj
from .gnn import GNN
import pickle


def get_predicates_of_object_entities_with_best_q_value(model, nodes, edges, actions):
    entity_name_to_idx = pickle.load(open('entity_name_to_idx.pkl', 'r'))
    idx_to_entity_name = {}
    for k, v in zip(entity_name_to_idx.keys(), entity_name_to_idx.values()):
        idx_to_entity_name[v] = k

    object_entity_idxs = range(8)
    region_idxs = [8, 9]

    best_entity_node_values = []
    for node, edge, action in zip(nodes, edges, actions):
        # print "selected entity ", idx_to_entity_name[action[0]]
        # print "selected entity q_val", model.predict_with_raw_input_format(node, edge, action)
        # print 'is selected entity reachable ', node[action][0, -3]

        best_value = -np.inf
        for idx in object_entity_idxs + region_idxs:
            entity_val = model.predict_with_raw_input_format(node, edge, idx)
            # print idx_to_entity_name[idx], entity_val
            if entity_val > best_value:
                best_value = entity_val
                best_entity_idx = idx

        # print "best entity ", idx_to_entity_name[best_entity_idx]
        # print "best entity q_val", best_value
        # print 'is best entity reachable ', node[best_entity_idx][-3]

        best_entity_node_values.append(node[best_entity_idx])

    return best_entity_node_values


def get_msg_values(q_fcn, nodes, edges, actions, data_idx):
    is_goal_idx = 2
    is_holding_goal_entity_idx = -1
    inway_idx = 0
    is_reachable_idx = -2
    is_obj = -5

    vals = q_fcn.value_model.predict([nodes, edges, actions])[data_idx]
    node_edge_concat = q_fcn.concat_model.predict([nodes, edges, actions])[data_idx]
    edge = edges[data_idx]
    tedge = np.transpose(edge, (1, 0, 2))

    msgs = q_fcn.msg_model.predict([nodes, edges, actions])[data_idx]
    msg_aggregation = q_fcn.aggregation_model.predict([nodes, edges, actions])[data_idx]

    reachable_vals = []
    reachable_and_in_way = []
    reachable_and_in_way_to_goal = []
    reachable_and_not_in_way = []
    unreachable_vals = []
    node_idx = 0
    reachable_non_goal_obj_nodes = []
    for n, val in zip(nodes[data_idx], vals):
        if n[is_goal_idx] or not n[is_obj]:
            node_idx += 1
            continue
        print node_idx, n[is_reachable_idx], nodes[data_idx][node_idx][is_reachable_idx]
        if n[is_reachable_idx]:
            reachable_non_goal_obj_nodes.append(n)
            reachable_vals.append((node_idx, val[0]))
            if np.any(edge[node_idx][:, inway_idx]):
                reachable_and_in_way.append((node_idx, val[0]))
                occluded_entity = np.nonzero(edge[node_idx][:, inway_idx])[0]
                for i in occluded_entity:
                    if nodes[data_idx][i][is_goal_idx]:
                        reachable_and_in_way_to_goal.append(val[0])
            else:
                reachable_and_not_in_way.append((node_idx, val[0]))
        else:
            unreachable_vals.append(val[0])
        node_idx += 1

    print "Reachable and in way:", reachable_and_in_way
    print "Reachable and in way to goal:", reachable_and_in_way_to_goal
    print "Reachable and not in way:", reachable_and_not_in_way
    # print "Non-goal object nodes:"
    # print np.array(reachable_non_goal_obj_nodes)
    entity_in_way = 0
    entity_not_in_way = 4
    if len(reachable_and_in_way_to_goal) > 0:
        # print edges[data_idx]
        edge_vals = q_fcn.edge_model.predict(edges)[data_idx]
        """
        print "inway edge network val:"
        print edge_vals[0][2] # src: 0th entity
        print 'not in way edge network val:'
        print edge_vals[0][1]
        print 'concat inway edge match:'
        print np.all(node_edge_concat[0][2][-32:] == edge_vals[0][2])
        print 'concat not inway edge match:'
        print np.all(node_edge_concat[0][1][-32:] == edge_vals[0][1])
        print 'inway msg value'
        print msgs[0][2]
        print 'not inway msg value'
        print msgs[0][1]
        """
        print np.sum(msgs[:, entity_in_way])
        print np.sum(msgs[:, entity_not_in_way])

        print 'inway aggregated msg value'
        print msg_aggregation[0]
        print 'not inway aggregated msg value'
        print msg_aggregation[3]

        import pdb;
        pdb.set_trace()


def test_trained_model(config):
    seed = config.seed
    np.random.seed(seed)
    random.seed(seed)
    num_test = config.num_test

    nodes, edges, actions, costs = data_traj.load_data(
        './test_results/mcts_results_on_mover_domain/widening_5/uct_1.0/trajectory_data',
        operator=config.operator,
    )
    nodes = nodes[:, :, 6:]

    num_entities = nodes.shape[1]
    num_operators = 1
    q_fcn = GNN(num_entities, nodes.shape[-1], edges.shape[-1], config)
    q_weight_file = q_fcn.weight_file_name
    print q_weight_file

    tnodes = nodes[-num_test:]
    tedges = edges[-num_test:]
    tactions = actions[-num_test:]

    #q_fcn.loss_model.load_weights(q_weight_file)
    # best_entity_node_values = get_predicates_of_object_entities_with_best_q_value(q_fcn, tnodes, tedges, tactions)
    # print np.mean(best_entity_node_values)

    data_idx = 0
    for data_idx in range(len(nodes)):
        msg_values = get_msg_values(q_fcn, tnodes, tedges, tactions, data_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_hidden', type=int, default=32)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_test', type=int, default=600)
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-val_portion', type=float, default=0.1)
    parser.add_argument('-top_k', type=int, default=3)
    parser.add_argument('-use_mse', action='store_true', default=False)
    parser.add_argument('-donttrain', action='store_true', default=False)
    parser.add_argument('-same_vertex_model', action='store_true', default=False)
    parser.add_argument('-diff_weight_msg_passing', action='store_true', default=False)
    parser.add_argument('-operator', type=str, default='two_arm_pick')
    parser.add_argument('-num_fc_layers', type=int, default=2)
    parser.add_argument('-no_goal_nodes', action='store_true', default=False)
    parser.add_argument('-n_msg_passing', type=int, default=1)
    parser.add_argument('-weight_initializer', type=str, default='glorot_uniform')
    parser.add_argument('-mse_weight', type=float, default=0.2)
    parameters = parser.parse_args()

    configs = parser.parse_args()
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    tf.set_random_seed(configs.seed)

    donttrain = configs.donttrain
    configs.donttrain = False

    # test_trained_model(configs)
    test_trained_model(configs)
