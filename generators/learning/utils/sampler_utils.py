import numpy as np
from generators.learning.utils import data_processing_utils
from gtamp_utils import utils


def generate_smpls(smpler_state, policy, n_data):
    obj = smpler_state.obj

    utils.set_color(obj, [1, 0, 0])
    poses = np.hstack(
        [utils.encode_pose_with_sin_and_cos_angle(utils.get_body_xytheta(obj).squeeze()), 0, 0, 0, 0]).reshape((1, 8))

    obj_pose = utils.clean_pose_data(smpler_state.obj_pose)
    key_configs = smpler_state.key_configs
    rel_konfs = data_processing_utils.make_konfs_relative_to_pose(obj_pose, key_configs)
    rel_konfs = np.array(rel_konfs).reshape((1, 615, 3, 1))
    goal_flags = smpler_state.goal_flags
    collisions = smpler_state.collision_vec

    places = []
    for _ in range(n_data):
        poses = poses[:, :4]
        placement = data_processing_utils.get_unprocessed_placement(
            policy.generate(goal_flags, rel_konfs, collisions, poses).squeeze(), obj_pose)
        places.append(placement)
    return places
