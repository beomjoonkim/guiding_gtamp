#!/bin/env python2

import argparse
import random
import tensorflow as tf
import numpy as np
import sys
import os

from . import data_traj
from .pap_gnn import PaPGNN
import csv
import pickle


def top_k_accuracy(q_model, nodes, edges, actions, k):
    print "Testing on %d number of data" % len(nodes)
    q_target_action = q_model.predict_with_raw_input_format(nodes, edges, actions)
    n_data = len(nodes)
    q_all_actions = q_model.alt_msg_layer.predict([nodes, edges, actions])
    accuracy = []
    top_zero_accuracy = []
    top_one_accuracy = []
    top_two_accuracy = []
    for i in range(n_data):
        n_actions_bigger_than_target = np.sum(q_target_action[i] < q_all_actions[i])
        accuracy.append(n_actions_bigger_than_target <= k)
        top_zero_accuracy.append(n_actions_bigger_than_target == 0)
        top_one_accuracy.append(n_actions_bigger_than_target <= 1)
        top_two_accuracy.append(n_actions_bigger_than_target <= 2)

    return np.mean(accuracy), np.mean(top_zero_accuracy), np.mean(top_one_accuracy), np.mean(top_two_accuracy)


def create_callbacks(q_weight_file):
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=100, ),
        tf.keras.callbacks.ModelCheckpoint(filepath=q_weight_file, verbose=False, save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.TensorBoard()
    ]
    return callbacks


# def create_gnn_model(nodes, edges, config):
def create_gnn_model(config, nodes, edges):
    num_entities = nodes.shape[1]
    m = PaPGNN(num_entities, nodes.shape[-1], edges.shape[-1], config)
    if os.path.isfile(m.weight_file_name) and not config.donttrain and not config.f:
        print "Quitting because we've already trained with the given configuration"
        sys.exit(-1)
    return m


def create_train_data(nodes, edges, actions, costs, num_training):
    training_inputs = [nodes[:num_training], edges[:num_training], actions[:num_training], costs[:num_training]]
    training_targets = np.zeros(num_training, dtype=np.float32)
    return training_inputs, training_targets


def train(config):
    seed = config.seed

    nodes, edges, actions, rewards = data_traj.load_data(
        #'./planning_experience/irsc/two_arm_mover/n_objs_pack_1/trajectory_data/',
        #'./planning_experience/hcount/domain_two_arm_mover/n_objs_pack_1/trajectory_data/',
        #'./planning_experience/irsc/mc/domain_two_arm_mover/n_objs_pack_1/trajectory_data/',
        #'./planning_experience/hcount/mc/domain_two_arm_mover/n_objs_pack_1/trajectory_data/',
        #'./planning_experience/domain_two_arm_mover/n_objs_pack_1/hcount/trajectory_data/shortest/',
        #'./planning_experience/domain_two_arm_mover/n_objs_pack_1/irsc/trajectory_data/shortest/',
        './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/trajectory_data/mc/',
        desired_operator_type=config.operator)
    """
    print "Loading data..."
    nodes, edges, actions, rewards = pickle.load(open('tmp.pkl', 'r'))
    """

    print "Total number of data", len(nodes)

    # num_test = min(config.num_test, len(nodes))
    num_training = config.num_train
    num_test = len(nodes) - num_training
    assert num_training > 0
    config.num_train = num_training
    config.num_test = num_test

    nodes = nodes[:, :, 6:]
    model = create_gnn_model(config, nodes, edges)
    callbacks = create_callbacks(model.weight_file_name)
    training_inputs, training_targets = create_train_data(nodes, edges, actions, rewards, num_training)
    tnodes = nodes[-num_test:]
    tedges = edges[-num_test:]
    tactions = actions[-num_test:]

    if not donttrain:
        model.loss_model.fit(
            training_inputs, training_targets, config.batch_size, epochs=500, verbose=2,
            callbacks=callbacks,
            validation_split=config.val_portion)

    model.load_weights()
    _, post_top_zero_acc, post_top_one_acc, post_top_two_acc = top_k_accuracy(model, tnodes, tedges, tactions,
                                                                              config.top_k)

    write_test_results_in_csv(post_top_zero_acc, post_top_one_acc, post_top_two_acc, seed, num_training, config.loss)
    print "Post-training top-0 accuracy %.2f" % post_top_zero_acc
    print "Post-training top-1 accuracy %.2f" % post_top_one_acc
    print "Post-training top-2 accuracy %.2f" % post_top_two_acc


def write_test_results_in_csv(top0, top1, top2, seed, num_training, loss_fcn):
    with open('./learn/top_k_results/' + loss_fcn + '_seed_' + str(seed) + '.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([num_training, top0, top1, top2])


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_hidden', type=int, default=32)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_test', type=int, default=1000)
    parser.add_argument('-num_train', type=int, default=7000)
    parser.add_argument('-val_portion', type=float, default=0.1)
    parser.add_argument('-top_k', type=int, default=1)
    parser.add_argument('-donttrain', action='store_true', default=False)
    parser.add_argument('-same_vertex_model', action='store_true', default=False)
    parser.add_argument('-diff_weight_msg_passing', action='store_true', default=False)
    parser.add_argument('-operator', type=str, default='two_arm_pick_two_arm_place')
    parser.add_argument('-num_fc_layers', type=int, default=2)
    parser.add_argument('-no_goal_nodes', action='store_true', default=False)
    parser.add_argument('-use_region_agnostic', action='store_true', default=False)
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-n_msg_passing', type=int, default=1)
    parser.add_argument('-weight_initializer', type=str, default='glorot_uniform')
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-mse_weight', type=float, default=0.0)

    configs = parser.parse_args()
    return configs


if __name__ == '__main__':
    configs = parse_args()
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    tf.set_random_seed(configs.seed)

    donttrain = configs.donttrain

    train(configs)
