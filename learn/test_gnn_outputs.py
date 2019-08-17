#!/bin/env python2

import argparse
import random
import tensorflow as tf
import numpy as np
import pdb

from . import data_traj
from .gnn import GNN
from .pap_gnn import PaPGNN
import pickle
from .train import top_k_accuracy


def get_values_of_object_entities(model, nodes, edges, actions):
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


def test_trained_model(config):
    seed = config.seed
    np.random.seed(seed)
    random.seed(seed)
    num_test = config.num_test

    nodes, edges, actions, _, costs = data_traj.load_data(
        './test_results/mcts_results_on_mover_domain/widening_5/uct_1.0/trajectory_data',
        operator=config.operator,
    )
    nodes = nodes[:, :, 6:]

    num_entities = nodes.shape[1]
    num_operators = 1
    q_fcn = GNN(num_entities, nodes.shape[-1], edges.shape[-1], num_operators, config)
    q_weight_file = q_fcn.weight_file_name
    print q_weight_file

    tnodes = nodes[-num_test:]
    tedges = edges[-num_test:]
    tactions = actions[-num_test:]

    q_fcn.loss_model.load_weights(q_weight_file)
    best_entity_node_values = get_values_of_object_entities(q_fcn, tnodes, tedges, tactions)
    import pdb;
    pdb.set_trace()
    print np.mean(best_entity_node_values)


def make_src_dest_edge(sender, dest, edge):
    n_entities = 11
    sender = np.expand_dims(sender, -2)
    sender_repetitons = [1, 1, n_entities, 1]  # same across columns
    sender = np.tile(sender, sender_repetitons)

    dest = np.expand_dims(dest, 1)
    dest_repetitions = [1, n_entities, 1, 1]  # same across rows
    dest = np.tile(dest, dest_repetitions)

    np_concat = np.concatenate([sender, dest, edge], axis=-1)
    return np_concat


def test_loss_function(q_fcn, nodes, edges, actions, costs):
    final_q_vals = q_fcn.value_model.predict([nodes, edges, actions]).squeeze()
    # final_q_vals = final_q_vals.sum(axis=1)

    values = q_fcn.value_model.predict([nodes, edges, actions])
    q_vals = q_fcn.predict_with_raw_input_format(nodes, edges, actions)
    losses = q_fcn.loss_model.predict([nodes, edges, actions, costs])
    alt_msgs = q_fcn.alt_msg_layer.predict([nodes, edges, actions, costs])

    for data_idx in range(len(values)):
        value = values[data_idx]
        q_val = q_vals[data_idx]
        loss = losses[data_idx]
        alt_msg = alt_msgs[data_idx]
        # print value.squeeze()
        # print alt_msg
        # print np.isclose(value.squeeze(), alt_msg)
        # value = np.delete(value, actions[data_idx])
        min_of_top_k = np.sort(value.squeeze())[-q_fcn.top_k]
        q_delta = q_val - min_of_top_k
        action_ranking_cost = np.maximum(1 - q_delta, 0)

        if not np.isclose(loss, action_ranking_cost):
            try:
                raise AssertionError  # this worked before adding tf.reduce_mean
            except:
                print "Not close enough?"
                print 'tf loss:', loss
                print 'np loss:', action_ranking_cost  # the loss of tf is computed based on batches
                continue


def test_concat_layer(config):
    nodes, edges, actions, costs = data_traj.load_data(
        './test_results/hpn_results_on_mover_domain/1/trajectory_data/special_cases/',
        desired_operator_type=config.operator)

    num_entities = nodes.shape[1]
    q_fcn = PaPGNN(num_entities, nodes.shape[-1], edges.shape[-1], config)
    q_fcn.create_concat_model_for_verification()
    pred = q_fcn.concat_model_verifier.predict([nodes, edges, actions])

    ## true value computation
    src_nodes = np.tile(np.expand_dims(nodes, -2), [1, 1, 11, 1])
    dest_nodes = np.tile(np.expand_dims(nodes, 1), [1, 11, 1, 1])

    src_dest_concat = np.concatenate([src_nodes, dest_nodes], axis=-1)

    repetitions = [1, 1, 1, 2, 1]
    repeated_src_dest_concatenated = np.tile(np.expand_dims(src_dest_concat, -2), repetitions)
    all_concat = np.concatenate([repeated_src_dest_concatenated, edges], axis=-1)
    assert np.all(np.isclose(all_concat))

    # predictions = q_fcn.predict_with_raw_input_format(nodes, edges, actions)
    # test_loss_function(q_fcn, nodes, edges, actions, costs)


def change_action_format(actions, n_entities, n_regions):
    n_data = len(actions)
    new_format = np.zeros((n_data, n_entities, n_regions))

    for data_idx, action in enumerate(actions):
        obj_idx = action[0]
        region_idx = action[1]
        new_format[data_idx, obj_idx, region_idx] = 1
    return new_format


def test_prediction(config):
    nodes, edges, actions, costs = data_traj.load_data(
        './test_results/hpn_results_on_mover_domain/1/trajectory_data/special_cases/',
        desired_operator_type=config.operator)

    num_entities = nodes.shape[1]
    q_fcn = PaPGNN(num_entities, nodes.shape[-1], edges.shape[-1], config)
    # q_fcn.predict_with_raw_input_format(nodes, edges, actions)
    vals = q_fcn.value_model.predict([nodes, edges, actions])

    print vals, vals.shape
    q_vals = q_fcn.predict_with_raw_input_format(nodes, edges, actions)
    import pdb;
    pdb.set_trace()


def test_loss_model(config):
    nodes, edges, actions, costs = data_traj.load_data(
        './test_results/hpn_results_on_mover_domain/1/trajectory_data/special_cases/',
        desired_operator_type=config.operator)
    num_entities = nodes.shape[1]
    q_fcn = PaPGNN(num_entities, nodes.shape[-1], edges.shape[-1], config)
    losses = q_fcn.loss_model.predict([nodes, edges, actions, costs])
    vals = q_fcn.value_model.predict([nodes, edges, actions, costs])
    vals = vals.squeeze()
    qvals = q_fcn.q_model.predict([nodes, edges, actions, costs])
    alt_msgs = q_fcn.alt_msg_layer.predict([nodes, edges, actions, costs])

    for i in range(len(actions)):
        action = actions[i]
        tf_val = vals[i]
        tf_loss = losses[i]
        tf_qval = qvals[i]
        alt_msg = alt_msgs[i]

        tf_val = tf_val.reshape((11*2))
        min_of_top_k = np.sort(tf_val)[-config.top_k:][0]
        #print 'manual_loss',1-(tf_qval - min_of_top_k)
        #print 'loss', tf_loss
        print np.isclose(1-(tf_qval - min_of_top_k), tf_loss)

def test_x(configs):
    nodes, edges, actions, costs = data_traj.load_data(
        './test_results/hpn_results_on_mover_domain/1/trajectory_data//', 'two_arm_pick_two_arm_place')
    nodes = nodes[:, :, 6:]
    print "n data %d" % (len(nodes))

    num_entities = 11
    q_fcn = PaPGNN(num_entities, nodes.shape[-1], edges.shape[-1], configs)
    concated_data = q_fcn.concat_model_verifier.predict([nodes, edges, actions])
    values = q_fcn.msg_model.predict([nodes, edges, actions])
    for action, concated_point in zip(actions, concated_data):
        object_idx = np.where(action)[0]
        region_idx = np.where(action)[1]

        as_src = concated_point[object_idx, :, region_idx, :].squeeze()
        as_dest = concated_point[:, object_idx, region_idx, :].squeeze()
        import pdb;
        pdb.set_trace()


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
    parser.add_argument('-operator', type=str, default='two_arm_pick_two_arm_place')
    parser.add_argument('-num_fc_layers', type=int, default=2)
    parser.add_argument('-no_goal_nodes', action='store_true', default=False)
    parser.add_argument('-n_msg_passing', type=int, default=1)
    parser.add_argument('-weight_initializer', type=str, default='glorot_uniform')
    parser.add_argument('-loss', type=str, default='not_dql')
    parser.add_argument('-mse_weight', type=float, default=0)

    configs = parser.parse_args()
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    tf.set_random_seed(configs.seed)

    donttrain = configs.donttrain
    configs.donttrain = False

    # test_trained_model(configs)
    # test_concat_layer(configs)
    # test_prediction(configs)
    test_loss_model(configs)
