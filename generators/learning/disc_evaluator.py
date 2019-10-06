from CMAESAdMonWithPose import CMAESAdversarialMonteCarloWithPose
from RelKonfCMAESAdMonWithPose import RelKonfCMAESAdversarialMonteCarloWithPose
from policy_evaluator import get_pidxs_to_evaluate_policy, load_pose_file, get_smpler_state
from data_processing.utils import state_data_mode, action_data_mode, convert_pose_rel_to_region_to_abs_pose, \
    unnormalize_pose_wrt_region

from gtamp_utils import utils
from test_scripts.run_greedy import get_problem_env

import sys
import numpy as np
import collections
import time


def get_augmented_state_vec_and_poses(obj, state_vec, smpler_state):
    n_key_configs = 615
    utils.set_color(obj, [1, 0, 0])
    is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([obj in smpler_state.goal_entities]))
    is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
    is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([smpler_state.region in smpler_state.goal_entities]))
    is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
    state_vec = np.concatenate([state_vec, is_goal_obj, is_goal_region], axis=2)

    poses = np.hstack(
        [utils.encode_pose_with_sin_and_cos_angle(utils.get_body_xytheta(obj).squeeze()), 0, 0, 0, 0]).reshape((1, 8))
    return state_vec, poses


import copy
from gtamp_utils.utils import *


def make_body(height, i, x, y):
    env = openravepy.RaveGetEnvironments()[0]
    new_body = box_body(env, 0.1, 0.1, height,
                        name='value_obj%s' % i,
                        color=(0, 1, 0))
    env.AddKinBody(new_body)
    trans = np.eye(4)
    trans[2, -1] = 1.0
    trans[0, -1] = x
    trans[1, -1] = y
    new_body.SetTransform(trans)


def get_placements(state, poses, admon, smpler_state):
    stime = time.time()
    # placement, value = admon.get_max_x(state, poses)
    max_x = None
    max_val = -np.inf
    placement = np.array([0., 0., 1., 1.])
    exp_val = {}
    for x in np.linspace(0., 1, 10):
        for y in np.linspace(0, 1, 10):
            placement[0] = x
            placement[1] = y
            val = admon.disc_mse_model.predict([placement[None, :], state, poses])
            if val > max_val:
                max_x = copy.deepcopy(placement)
                max_val = val
            exp_val[(x, y)] = np.exp(val)
            print x, y, val

    total = np.sum(exp_val.values())
    i = 0
    utils.viewer()
    for x in np.linspace(0, 1, 10):
        for y in np.linspace(0, 1, 10):
            height = exp_val[(x, y)] / total + 1
            print x, y, height
            placement[0] = x
            placement[1] = y
            absx, absy = unnormalize_pose_wrt_region(placement, 'loading_region')[0:2]
            make_body(height, i, absx, absy)
            i += 1
    placement = max_x
    print placement, max_val
    print 'maximizing x time', time.time() - stime

    placement = utils.decode_pose_with_sin_and_cos_angle(placement)
    if 'place_relative_to_obj' in action_data_mode:
        obj = smpler_state.obj
        placement = utils.get_absolute_pose_from_relative_pose(placement, utils.get_body_xytheta(obj).squeeze())
    elif 'place_relative_to_region' in action_data_mode:
        region = smpler_state.region
        placement = convert_pose_rel_to_region_to_abs_pose(placement, region)
    elif 'place_normalized_relative_to_region' in action_data_mode:
        region = smpler_state.region
        placement = unnormalize_pose_wrt_region(placement, region)

    return [placement]


def visualize_samples(q_fcn):
    n_evals = 10
    pidxs = get_pidxs_to_evaluate_policy(n_evals)
    pidx = pidxs[5]
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')
    problem_env = get_problem_env(config)
    pidx_poses = load_pose_file(pidx)

    problem_env.set_body_poses(pidx_poses)
    smpler_state = get_smpler_state(pidx)
    state_vec = np.delete(smpler_state.state_vec, [415, 586, 615, 618, 619], axis=1)

    obj = 'rectangular_packing_box2'
    state_vec, poses = get_augmented_state_vec_and_poses(obj, state_vec, smpler_state)

    places = get_placements(state_vec, poses, q_fcn, smpler_state)
    utils.viewer()
    utils.visualize_path(places)
    import pdb;
    pdb.set_trace()


def main():
    n_key_configs = 615  # indicating whether it is a goal obj and goal region
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/cmaes_place_admon/' % (
        state_data_mode, action_data_mode)

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'tau seed')

    config = mconfig_type(
        tau=1.0,
        seed=int(sys.argv[1])
    )
    epoch_number = int(sys.argv[2])

    use_rel_konf = True
    dim_action = 4
    if use_rel_konf:
        dim_state = (n_key_configs, 2, 1)
        policy = RelKonfCMAESAdversarialMonteCarloWithPose(dim_action, dim_state, savedir, 1.0, config)
    else:
        dim_state = (n_key_configs, 6, 1)
        policy = CMAESAdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                                    save_folder=savedir, tau=1.0, config=config)
    print "Trying epoch number ", epoch_number
    fname = 'admonpose_seed_3_epoch_57_drop_in_mse_-0.29760.h5'
    fname = 'admonpose_seed_0_epoch_57_drop_in_mse_-0.44833.h5'
    policy.disc.load_weights(policy.save_folder + fname)
    visualize_samples(policy)


if __name__ == '__main__':
    main()
