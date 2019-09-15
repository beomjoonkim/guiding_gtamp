import pickle
import os
import sys
import numpy as np


def get_time_taken(test_dir, stat):
    if test_dir.find('irsc') != -1:
        return stat['time_taken']
    elif test_dir.find('sahs') != -1:
        if isinstance(stat, dict):
            return stat['tottime']
        else:
            return stat.metrics['tottime']
    elif test_dir.find('mcts') != -1:
        return stat['search_time_to_reward'][-1][0]


def get_success(test_dir, stat):
    if test_dir.find('irsc') != -1:
        return stat['found_solution']
    elif test_dir.find('sahs') != -1:
        if isinstance(stat, dict):
            return stat['success']
        else:
            return stat.metrics['success']
    elif test_dir.find('mcts') != -1:
        return stat['search_time_to_reward'][-1][-1]


"""
def get_num_nodes(test_dir, stat):
    if test_dir.find('irsc') != -1:
        return stat['n_nodes']
    elif test_dir.find('sahs') != -1:
        return stat.metrics['num_nodes']
"""


def get_pidx(test_dir, filename):
    if test_dir.find('hpn') != -1:
        return int(filename.split('pidx_')[1].split('.pkl')[0])
    else:
        return int(filename.split('pidx_')[1].split('_planne')[0])


def get_num_node_from_file(test_dir, stat):
    if test_dir.find('irsc') != -1:
        return stat['time_taken']
    elif test_dir.find('sahs') != -1:
        if isinstance(stat, dict):
            return stat['num_nodes']
        else:
            return stat.metrics['num_nodes']
    elif test_dir.find('mcts') != -1:
        return stat['search_time_to_reward'][-1][0]


def get_num_nodes(test_dir, test_files):
    successes = []
    all_num_nodes = []
    print "Getting test stats from %d files in %s" % (len(test_files), test_dir)
    for filename in test_files:
        pidx = get_pidx(test_dir, filename)

        stat = pickle.load(open(test_dir + filename, 'r'))
        num_nodes = get_num_node_from_file(test_dir, stat)
        all_num_nodes.append(num_nodes)
    CI95 = 1.96 * np.std(all_num_nodes) / np.sqrt(len(all_num_nodes))
    print "Number of data", len(all_num_nodes)
    print "Num nodes %.3f +- %.3f" % (np.mean(all_num_nodes), CI95)


def get_plan_times(test_dir, test_files, t_limit):
    successes = []
    time_taken = []
    print "Getting test stats from %d files in %s" % (len(test_files), test_dir)
    for filename in test_files:
        pidx = get_pidx(test_dir, filename)
        # if pidx < 20000 or pidx > 20100:
        #    continue

        # print filename

        # if 'train_seed_3' not in filename:
        #    continue
        stat = pickle.load(open(test_dir + filename, 'r'))
        ftime_taken = get_time_taken(test_dir, stat)
        fsuccess = get_success(test_dir, stat)

        if ftime_taken < t_limit:
            time_taken.append(ftime_taken)
            successes.append(fsuccess)
        else:
            time_taken.append(t_limit)
            successes.append(False)
    CI95 = 1.96 * np.std(time_taken) / np.sqrt(len(time_taken))
    print "Number of data", len(time_taken)
    print "Time taken %.3f +- %.3f" % (np.mean(time_taken), CI95)
    print "Success rate %.3f" % np.mean(successes)


def main():
    n_objs = int(sys.argv[1])
    t_limit = 300 * n_objs
    num_nodes = 500
    domain = 'two_arm_mover'

    # Customize the below
    # test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/state_hcount/' % (domain, n_objs)
    # test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/new_hcount/' % (domain, n_objs)

    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/state_hcount/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/gnn_and_hcount/loss_largemargin/num_train_5000/' % (
    domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/hcount_after_submission/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qbonus_and_hcount/loss_largemargin/num_train_5000/' % (
    domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/gnn_after_submission/loss_largemargin/num_train_5000/' % (
    domain, n_objs)

    test_dir = './test_results/mcts_results_with_q_bonus/domain_%s/n_objs_pack_%d/' \
               'sampling_strategy_uniform/n_mp_trials_3/widening_30.0/uct_0.1/switch_frequency_50/' \
               'reward_shaping_False/learned_q_False/' % (domain, n_objs)
    test_dir = './test_results/mcts_results_with_q_bonus/domain_%s/n_objs_pack_%d/' \
               'sampling_strategy_voo/n_mp_trials_3/widening_20.0/uct_0.1/switch_frequency_50/' \
               'reward_shaping_False/learned_q_False/explr_p_0.3/' % (domain, n_objs)
    test_dir = './test_results/mcts_results_with_q_bonus/domain_%s/n_objs_pack_%d/' \
               'sampling_strategy_uniform/n_mp_trials_3/widening_20.0/uct_0.1/switch_frequency_50/' \
               'reward_shaping_False/learned_q_False/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/hcount/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount/loss_largemargin/' \
               'num_train_7000/mse_weight_1.0/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount/loss_largemargin/num_train_7000/1.0/' % (
    domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount/loss_largemargin/' \
               'num_train_7000/mse_weight_1.0/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount/loss_largemargin/' \
               'num_train_7000/mse_weight_1.0/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount_obj_already_in_goal/loss_largemargin/' \
               'num_train_7000/mse_weight_1.0/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount_obj_already_in_goal/loss_largemargin/' \
               'num_train_7000/mse_weight_1.0/use_region_agnostic_True/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount_obj_already_in_goal/loss_largemargin/' \
               'num_train_1805/mse_weight_0.0/use_region_agnostic_False/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/gnn/loss_largemargin/num_train_1805/mse_weight_1.0/use_region_agnostic_False/' % (
    domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount_obj_already_in_goal/loss_mse/' \
               'num_train_10000/mse_weight_0.0/use_region_agnostic_False/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/hcount/' % (domain, n_objs)
    test_files = os.listdir(test_dir)
    get_plan_times(test_dir, test_files, t_limit)
    get_num_nodes(test_dir, test_files)


if __name__ == '__main__':
    main()
