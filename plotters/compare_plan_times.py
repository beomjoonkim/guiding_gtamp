import pickle
import os
import sys
import numpy as np


def get_time_taken(test_dir, stat):
    if test_dir.find('sahs') != -1:
        if isinstance(stat, dict):
            return stat['tottime']
        else:
            return stat.metrics['tottime']
    elif test_dir.find('irsc') != -1:
        return stat['time_taken']

    elif test_dir.find('mcts') != -1:
        return stat['search_time_to_reward'][-1][0]


def get_success(test_dir, stat):
    if test_dir.find('sahs') != -1:
        if isinstance(stat, dict):
            return stat['success']
        else:
            return stat.metrics['success']
    elif test_dir.find('irsc') != -1:
        return stat['found_solution']
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
    if test_dir.find('sahs') != -1:
        if isinstance(stat, dict):
            return stat['num_nodes']
        else:
            return stat.metrics['num_nodes']
    elif test_dir.find('mcts') != -1:
        return stat['search_time_to_reward'][-1][0]
    elif test_dir.find('irsc') != -1:
        return stat['time_taken']


def get_num_nodes(test_dir, test_files):
    successes = []
    all_num_nodes = []
    print "Getting test stats from %d files in %s" % (len(test_files), test_dir)
    for filename in test_files:
        pidx = get_pidx(test_dir, filename)

        stat = pickle.load(open(test_dir + filename, 'r'))
        #if not stat['success']:
        #    continue

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
        if pidx < 20000 or pidx > 20009:
            continue

        #if 'train_seed_1' in filename: #or 'train_seed_0' not in filename:
        #    continue

        if 'train_seed_0' not in filename:
            continue

        print filename

        stat = pickle.load(open(test_dir + filename, 'r'))
        ftime_taken = get_time_taken(test_dir, stat)
        fsuccess = get_success(test_dir, stat)

        if ftime_taken < t_limit:
            time_taken.append(ftime_taken)
            successes.append(fsuccess)
        else:
            #if not stat['success']:
            #    continue
            time_taken.append(t_limit)
            successes.append(False)
            #print 'Failed',filename
    CI95 = 1.96 * np.std(time_taken) / np.sqrt(len(time_taken))
    print "Number of data", len(time_taken)
    print "Time taken %.3f +- %.3f" % (np.mean(time_taken), CI95)
    print "Success rate %.3f" % np.mean(successes)


def main():
    n_objs = int(sys.argv[1])
    if n_objs == 4:
        t_limit = 100 * n_objs
    elif n_objs == 1:
        t_limit = 300 * n_objs
    else:
        #t_limit = A00 * n_objs
        t_limit = 2400


    domain = 'two_arm_mover'
    if domain == 'one_arm_mover':
        t_limit = 1000

    # Customize the below

    if n_objs == 4 or n_objs==1:
        test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount_obj_already_in_goal/shortest_irsc/' \
                   'loss_largemargin/num_train_5000/mse_weight_1.0/use_region_agnostic_False/mix_rate_1.0/' % (domain, n_objs)
    else:
        test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/qlearned_hcount_obj_already_in_goal/shortest_irsc/' \
                   'loss_largemargin/num_train_5000/mse_weight_1.0/use_region_agnostic_False/mix_rate_100.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/domain_%s/n_objs_pack_%d/hcount/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/irsc_mc/domain_%s/n_objs_pack_%d/qlearned_hcount_obj_already_in_goal/shortest_irsc/' \
               'loss_largemargin/num_train_5000/mse_weight_1.0/use_region_agnostic_False/mix_rate_1.0/' % (domain, n_objs)
    test_dir = './test_results/sahs_results/using_weights_for_submission/n_objs_pack_%d/gnn/shortest_irsc/' \
               'loss_largemargin/num_train_5000/mse_weight_1.0/use_region_agnostic_False/' % (n_objs)

    test_files = os.listdir(test_dir)
    get_plan_times(test_dir, test_files, t_limit)
    get_num_nodes(test_dir, test_files)


if __name__ == '__main__':
    main()
