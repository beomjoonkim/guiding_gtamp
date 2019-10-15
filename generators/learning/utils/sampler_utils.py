from generators.learning.utils import data_processing_utils
from gtamp_utils import utils

import time
import numpy as np


def prepare_input(smpler_state):
    # poses = np.hstack(
    #    [utils.encode_pose_with_sin_and_cos_angle(utils.get_body_xytheta(obj).squeeze()), 0, 0, 0, 0]).reshape((1, 8))
    poses = data_processing_utils.get_processed_poses_from_state(smpler_state)[None, :]
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    # todo compute this only once, and store it in smpler state

    if smpler_state.rel_konfs is None:
        key_configs = smpler_state.key_configs
        rel_konfs = data_processing_utils.make_konfs_relative_to_pose(obj_pose, key_configs)
        rel_konfs = np.array(rel_konfs).reshape((1, 615, 4, 1))
        smpler_state.rel_konfs = rel_konfs
    else:
        rel_konfs = smpler_state.rel_konfs

    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.collision_vector
    poses = poses[:, :8]

    return goal_flags, rel_konfs, collisions, poses


def generate_smpls(smpler_state, policy, n_data, noise_smpls_tried=None):
    goal_flags, rel_konfs, collisions, poses = prepare_input(smpler_state)
    obj = smpler_state.obj
    utils.set_color(obj, [1, 0, 0])
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    places = []
    noises_used = []
    for _ in range(n_data):
        smpls, noises_used = policy.generate(goal_flags, rel_konfs, collisions, poses, noises_used)
        placement = data_processing_utils.get_unprocessed_placement(smpls.squeeze(), obj_pose)
        places.append(placement)
    if noise_smpls_tried is not None:
        return places, noises_used
    else:
        return places


def generate_smpls_using_noise(smpler_state, policy, noises):
    goal_flags, rel_konfs, collisions, poses = prepare_input(smpler_state)
    obj = smpler_state.obj
    utils.set_color(obj, [1, 0, 0])
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)

    places = []
    smpls = policy.generate_given_noise(goal_flags, rel_konfs, collisions, poses, noises)
    placement = data_processing_utils.get_unprocessed_placement(smpls.squeeze(), obj_pose)
    places.append(placement)
    return places


def generate_w_values(smpler_state, policy):
    goal_flags, rel_konfs, collisions, poses = prepare_input(smpler_state)
    w_vals = policy.w_model.predict([goal_flags, rel_konfs, collisions, poses])
    return w_vals


def generate_transformed_key_configs(smpler_state, policy):
    obj_pose = utils.clean_pose_data(smpler_state.abs_obj_pose)
    goal_flags, rel_konfs, collisions, poses = prepare_input(smpler_state)
    n_data = len(goal_flags)
    a_dim = 4
    noise_smpls = np.random.normal(size=(n_data, a_dim)).astype('float32')
    smpls = policy.value_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls]).squeeze()
    transformed = [data_processing_utils.get_unprocessed_placement(s, obj_pose) for s in smpls]
    return np.array(transformed)


def generate_policy_smpl_batch(smpler_state, policy, noise_batch):
    goal_flags, rel_konfs, collisions, poses = prepare_input(smpler_state)
    obj = smpler_state.obj
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
    poses = poses[:, :8]
    poses = np.tile(poses, (n_smpls, 1))
    noise_batch = np.array(noise_batch).squeeze()
    pred_batch = policy.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_batch])
    return pred_batch
