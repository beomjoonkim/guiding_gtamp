from generators.learning.utils import data_processing_utils
from gtamp_utils import utils

import time
import numpy as np


def generate_smpls(smpler_state, policy, n_data, noise_smpls_tried=None):
    stime = time.time()
    obj = smpler_state.obj

    utils.set_color(obj, [1, 0, 0])
    poses = np.hstack(
        [utils.encode_pose_with_sin_and_cos_angle(utils.get_body_xytheta(obj).squeeze()), 0, 0, 0, 0]).reshape((1, 8))

    # todo compute this only once, and store it in smpler state
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)
    if smpler_state.rel_konfs is None:
        key_configs = smpler_state.key_configs
        rel_konfs = data_processing_utils.make_konfs_relative_to_pose(obj_pose, key_configs)
        rel_konfs = np.array(rel_konfs).reshape((1, 615, 4, 1))
        smpler_state.rel_konfs = rel_konfs
    else:
        rel_konfs = smpler_state.rel_konfs

    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.collision_vector

    places = []
    noises_used = []
    for _ in range(n_data):
        poses = poses[:, :4]
        smpls, noises_used = policy.generate(goal_flags, rel_konfs, collisions, poses, noises_used)
        placement = data_processing_utils.get_unprocessed_placement(smpls.squeeze(), obj_pose)
        places.append(placement)
    # print "Time taken", time.time()-stime
    if noise_smpls_tried is not None:
        return places, noises_used
    else:
        return places


def generate_policy_smpl_batch(smpler_state, policy, noise_batch):
    obj = smpler_state.obj

    # utils.set_color(obj, [1, 0, 0])
    poses = np.hstack(
        [utils.encode_pose_with_sin_and_cos_angle(utils.get_body_xytheta(obj).squeeze()), 0, 0, 0, 0]).reshape((1, 8))

    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)
    smpler_state.abs_obj_pose = obj_pose
    if smpler_state.rel_konfs is None:
        key_configs = smpler_state.key_configs
        rel_konfs = data_processing_utils.make_konfs_relative_to_pose(obj_pose, key_configs)
        rel_konfs = np.array(rel_konfs).reshape((1, 615, 4, 1))
        smpler_state.rel_konfs = rel_konfs
    else:
        rel_konfs = smpler_state.rel_konfs
    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.collision_vector

    n_smpls = len(noise_batch)
    goal_flags = np.tile(goal_flags, (n_smpls, 1, 1, 1))
    rel_konfs = np.tile(rel_konfs, (n_smpls, 1, 1, 1))
    collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
    poses = poses[:, :4]
    poses = np.tile(poses, (n_smpls, 1))
    noise_batch = np.array(noise_batch).squeeze()
    pred_batch = policy.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_batch])
    return pred_batch
