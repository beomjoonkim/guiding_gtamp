from generators.learning.utils import data_processing_utils
from gtamp_utils import utils

import time
import numpy as np


def generate_smpls(smpler_state, policy, n_data, noise_smpls_tried=None):
    stime = time.time()
    obj = smpler_state.obj

    # utils.set_color(obj, [1, 0, 0])
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
    for _ in range(n_data):
        poses = poses[:, :4]
        import pdb;pdb.set_trace()
        smpls, noises_used = policy.generate(goal_flags, rel_konfs, collisions, poses, noise_smpls_tried)
        placement = data_processing_utils.get_unprocessed_placement(smpls.squeeze(), obj_pose)
        places.append(placement)
    #print "Time taken", time.time()-stime
    if noise_smpls_tried is not None:
        return places, noises_used
    else:
        return places
