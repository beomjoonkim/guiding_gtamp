from gtamp_utils import utils
import numpy as np

state_data_mode = 'robot_rel_to_obj'
action_data_mode = 'pick_parameters_place_normalized_relative_to_region'
action_data_mode = 'pick_parameters_place_relative_to_object'


# action_data_mode = 'absolute'
def make_konfs_relative_to_pose(obj_pose, key_configs):
    rel_konfs = []
    utils.clean_pose_data(obj_pose)
    for k in key_configs:
        konf = utils.clean_pose_data(k)
        rel_konf = utils.get_relative_robot_pose_wrt_body_pose(konf, obj_pose)
        rel_konf = utils.encode_pose_with_sin_and_cos_angle(rel_konf)
        rel_konfs.append(rel_konf)
    return np.array(rel_konfs)


def get_processed_poses_from_state(state):
    if state_data_mode == 'absolute':
        obj_pose = utils.encode_pose_with_sin_and_cos_angle(state.abs_obj_pose)
        robot_pose = utils.encode_pose_with_sin_and_cos_angle(state.robot_pose)
        goal_obj_pose = utils.encode_pose_with_sin_and_cos_angle(state.abs_goal_obj_pose)
    elif state_data_mode == 'robot_rel_to_obj':
        obj_pose = utils.encode_pose_with_sin_and_cos_angle(state.abs_obj_pose)

        robot_pose = utils.get_relative_robot_pose_wrt_body_pose(state.abs_robot_pose, state.abs_obj_pose)
        robot_pose = utils.encode_pose_with_sin_and_cos_angle(robot_pose)

        goal_obj_pose = utils.get_relative_robot_pose_wrt_body_pose(state.abs_goal_obj_pose, state.abs_obj_pose)
        recovered = utils.clean_pose_data(utils.get_absolute_pose_from_relative_pose(goal_obj_pose, state.abs_obj_pose.squeeze()))
        assert np.all(np.isclose(recovered, state.abs_goal_obj_pose.squeeze()))
        goal_obj_pose = utils.encode_pose_with_sin_and_cos_angle(goal_obj_pose)
    else:
        raise not NotImplementedError

    pose = np.hstack([obj_pose, goal_obj_pose, robot_pose])
    return pose


def put_pose_wrt_region(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError

    pose[0] = pose[0] - region_box[0][0]
    pose[1] = pose[1] - region_box[0][1]
    return pose


def convert_pose_rel_to_region_to_abs_pose(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError

    pose[0] = pose[0] + region_box[0][0]
    pose[1] = pose[1] + region_box[0][1]
    return pose


def unnormalize_pose_wrt_region(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError
    size_x, size_y = get_box_size(region_box)
    pose[0] = pose[0] * size_x
    pose[1] = pose[1] * size_y
    pose = convert_pose_rel_to_region_to_abs_pose(pose, region)
    return pose


def get_place_pose_wrt_region(pose, region):
    place_pose = put_pose_wrt_region(pose, region)
    return place_pose


def get_box_size(box):
    box_size_x = box[1][0] - box[0][0]
    box_size_y = box[1][1] - box[0][1]
    return box_size_x, box_size_y


def normalize_place_pose_wrt_region(pose, region):
    if region == 'home_region':
        region_box = [[-1.75, -3.16], [5.25, 3.16]]
    elif region == 'loading_region':
        region_box = [[-0.7, -8.55], [4.3, -4.85]]
    else:
        raise NotImplementedError
    place_pose = put_pose_wrt_region(pose, region)
    size_x, size_y = get_box_size(region_box)
    place_pose[0] = place_pose[0] / size_x
    place_pose[1] = place_pose[1] / size_y
    return place_pose


def get_unprocessed_placement(placement, obj_abs_pose):
    placement = utils.decode_pose_with_sin_and_cos_angle(placement)
    if action_data_mode == 'pick_parameters_place_relative_to_object':
        #abs_place = placement.squeeze() + obj_abs_pose.squeeze()
        abs_place = utils.get_absolute_pose_from_relative_pose(placement, obj_abs_pose)

    else:
        raise NotImplementedError

    return abs_place


def get_processed_poses_from_action(state, action):
    if action_data_mode == 'absolute':
        pick_pose = utils.encode_pose_with_sin_and_cos_angle(action['pick_abs_base_pose'])
        place_pose = utils.encode_pose_with_sin_and_cos_angle(action['place_abs_base_pose'])
    elif action_data_mode == 'pick_relative':
        pick_pose = action['pick_abs_base_pose']
        pick_pose = utils.get_relative_robot_pose_wrt_body_pose(pick_pose, state.abs_obj_pose)
        pick_pose = utils.encode_pose_with_sin_and_cos_angle(pick_pose)
        place_pose = utils.encode_pose_with_sin_and_cos_angle(action['place_abs_base_pose'])
    elif action_data_mode == 'pick_relative_place_relative_to_region':
        pick_pose = action['pick_abs_base_pose']
        pick_pose = utils.get_relative_robot_pose_wrt_body_pose(pick_pose, state.abs_obj_pose)
        pick_pose = utils.encode_pose_with_sin_and_cos_angle(pick_pose)
        place_pose = get_place_pose_wrt_region(action['place_abs_base_pose'], action['region_name'])
    elif action_data_mode == 'pick_parameters_place_relative_to_region':
        pick_pose = action['pick_abs_base_pose']
        portion, base_angle, facing_angle_offset \
            = utils.get_ir_parameters_from_robot_obj_poses(pick_pose, state.abs_obj_pose)
        base_angle = utils.encode_angle_in_sin_and_cos(base_angle)
        pick_pose = np.hstack([portion, base_angle, facing_angle_offset])
        place_pose = get_place_pose_wrt_region(action['place_abs_base_pose'], action['region_name'])
        place_pose = utils.encode_pose_with_sin_and_cos_angle(place_pose)
    elif action_data_mode == 'pick_parameters_place_normalized_relative_to_region':
        pick_pose = action['pick_abs_base_pose']
        portion, base_angle, facing_angle_offset \
            = utils.get_ir_parameters_from_robot_obj_poses(pick_pose, state.abs_obj_pose)
        base_angle = utils.encode_angle_in_sin_and_cos(base_angle)
        pick_pose = np.hstack([portion, base_angle, facing_angle_offset])
        place_pose = normalize_place_pose_wrt_region(action['place_abs_base_pose'], action['region_name'])
        place_pose = utils.encode_pose_with_sin_and_cos_angle(place_pose)
    elif action_data_mode == 'pick_parameters_place_relative_to_pick':
        pick_pose = action['pick_abs_base_pose']
        portion, base_angle, facing_angle_offset \
            = utils.get_ir_parameters_from_robot_obj_poses(pick_pose, state.abs_obj_pose)
        base_angle = utils.encode_angle_in_sin_and_cos(base_angle)
        pick_params = np.hstack([portion, base_angle, facing_angle_offset])
        place_pose = action['place_abs_base_pose']
        place_pose = utils.get_relative_robot_pose_wrt_body_pose(place_pose, pick_pose)
        pick_pose = pick_params
        place_pose = utils.encode_pose_with_sin_and_cos_angle(place_pose)
    elif action_data_mode == 'pick_parameters_place_relative_to_object':
        pick_pose = action['pick_abs_base_pose']
        portion, base_angle, facing_angle_offset \
            = utils.get_ir_parameters_from_robot_obj_poses(pick_pose, state.abs_obj_pose)
        base_angle = utils.encode_angle_in_sin_and_cos(base_angle)
        pick_params = np.hstack([portion, base_angle, facing_angle_offset])
        pick_pose = pick_params

        place_pose = action['place_abs_base_pose']
        obj_pose = state.abs_obj_pose
        rel_place_pose = utils.get_relative_robot_pose_wrt_body_pose(place_pose, obj_pose)
        place_pose = utils.encode_pose_with_sin_and_cos_angle(rel_place_pose)

    unprocessed_place = utils.clean_pose_data(get_unprocessed_placement(place_pose, obj_pose))
    target = utils.clean_pose_data(action['place_abs_base_pose'])

    is_recovered = np.all(np.isclose(unprocessed_place, target))
    try:
        assert is_recovered
    except:
        import pdb;pdb.set_trace()
    action = np.hstack([pick_pose, place_pose])

    return action
