import pickle
import os
import numpy as np


def get_time_taken(test_dir, stat):
    if test_dir.find('irsc') != -1:
        return stat['time_taken']
    elif test_dir.find('sahs') != -1:
        return stat.metrics['tottime']
    elif test_dir.find('mcts') != -1:
        return stat['search_time_to_reward'][-1][0]


def get_success(test_dir, stat):
    if test_dir.find('irsc') != -1:
        return stat['found_solution']
    elif test_dir.find('sahs') != -1:
        return stat.metrics['success']
    elif test_dir.find('mcts') != -1:
        return stat['search_time_to_reward'][-1][-1]


def get_num_nodes(test_dir, stat):
    if test_dir.find('irsc') != -1:
        return stat['n_nodes']
    elif test_dir.find('sahs') != -1:
        return stat.metrics['num_nodes']


def get_pidx(test_dir, filename):
    if test_dir.find('hpn') != -1:
        return int(filename.split('pidx_')[1].split('.pkl')[0])
    else:
        return int(filename.split('pidx_')[1].split('_planne')[0])


def get_plan_times(test_dir, test_files, t_limit):
    successes = []
    time_taken = []
    print "Getting test stats from %d files in %s" % (len(test_files), test_dir)
    for filename in test_files:
        pidx = get_pidx(test_dir, filename)
        if pidx < 20000:
            continue

        stat = pickle.load(open(test_dir + filename, 'r'))
        ftime_taken = get_time_taken(test_dir, stat)
        fsuccess = get_success(test_dir, stat)

        if ftime_taken < t_limit:
            time_taken.append(ftime_taken)
            successes.append(fsuccess)
        else:
            time_taken.append(t_limit)
            successes.append(False)

    import pdb;pdb.set_trace()

    CI95 = 1.96 * np.std(time_taken) / np.sqrt(len(time_taken))
    print "Time taken %.3f +- %.3f" % (np.mean(time_taken), CI95)
    print "Success rate %.3f" % np.mean(successes)


def save_summary(stat_summary, test_dir, n_data, n_objs):
    if test_dir.find('hpn') != -1:
        pickle.dump(stat_summary, open('./plotters/stats/hpn_n_objs_%d.pkl' % n_objs, 'wb'))
    elif test_dir.find('greedy') != -1:
        pickle.dump(stat_summary, open('./plotters/stats/greedy_n_objs_%d_n_data_%d.pkl' % (n_objs, n_data), 'wb'))


def get_metrics(test_dir, test_files, n_objs, n_data=None):
    successes = []
    time_taken = []
    num_nodes = []
    for fidx, filename in enumerate(test_files):
        print "%d / %d" % (fidx, len(test_files))
        pidx = get_pidx(test_dir, filename)
        if pidx < 20000:
            continue

        stat = pickle.load(open(test_dir + filename, 'r'))
        ftime_taken = get_time_taken(test_dir, stat)
        fsuccess = get_success(test_dir, stat)
        fnodes = get_num_nodes(test_dir, stat)

        time_taken.append(ftime_taken)
        successes.append(fsuccess)
        num_nodes.append(fnodes)

    stat_summary = {'times': time_taken, 'successes': successes, 'num_nodes': num_nodes}
    save_summary(stat_summary, test_dir, n_data, n_objs)


def main():
    n_objs = 1
    t_limit = 300*n_objs
    domain = 'two_arm_mover'

    # Customize the below
    algo_name = 'sahs'
    #test_dir += 'sampling_strategy_uniform/n_mp_trials_3/widening_3/uct_0.1/reward_shaping_True/learned_q_False/'
    test_dir = './test_results/%s_results/domain_%s/n_objs_pack_%d/' % (algo_name, domain, n_objs)
    test_dir += 'hcount/'
    #test_dir += 'dql/n_train_5000/'

    test_files = os.listdir(test_dir)
    get_plan_times(test_dir, test_files, t_limit)




if __name__ == '__main__':
    main()
