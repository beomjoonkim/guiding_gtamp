from manipulation.primitives.transforms import *
from manipulation.bodies.bodies import *
import manipulation
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.primitives.transforms import get_point, set_point, pose_from_quat_point, unit_quat

from manipulation.primitives.utils import mirror_arm_config
import openravepy
import numpy as np
import math
import time

FOLDED_LEFT_ARM = [0.0, 1.29023451, 0.0, -2.121308, 0.0, -0.69800004, 0.0]
PR2_ARM_LENGTH = 0.9844


def convert_collision_vec_to_one_hot(c_data):
    n_konf = c_data.shape[1]
    onehot_cdata = []
    for cvec in c_data:
        one_hot_cvec = np.zeros((n_konf, 2))
        for boolean_collision, onehot_collision in zip(cvec, one_hot_cvec):
            onehot_collision[boolean_collision] = 1
        assert (np.all(np.sum(one_hot_cvec, axis=1) == 1))
        onehot_cdata.append(one_hot_cvec)

    onehot_cdata = np.array(onehot_cdata)
    return onehot_cdata


def compute_angle_to_be_set(target_xy, src_xy):
    target_dirn = target_xy - src_xy
    target_dirn = target_dirn / np.linalg.norm(target_dirn)
    if target_dirn[1] < 0:
        # rotation from x-axis, because that is the default rotation
        angle_to_be_set = -math.acos(np.dot(target_dirn, np.array(([1, 0]))))
    else:
        angle_to_be_set = math.acos(np.dot(target_dirn, np.array(([1, 0]))))
    return angle_to_be_set


def convert_rel_to_abs_base_pose(rel_xytheta, src_xy):
    if len(rel_xytheta.shape) == 1: rel_xytheta = rel_xytheta[None, :]
    assert (len(src_xy.shape) == 1)
    ndata = rel_xytheta.shape[0]
    abs_base_pose = np.zeros((ndata, 3))
    abs_base_pose[:, 0:2] = rel_xytheta[:, 0:2] + src_xy[0:2]
    for i in range(ndata):
        th_to_be_set = compute_angle_to_be_set(src_xy[0:2], abs_base_pose[i, 0:2])
        abs_base_pose[i, -1] = th_to_be_set + rel_xytheta[i, -1]
    return abs_base_pose


def set_body_transparency(body, transparency):
    for link in body.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(transparency)


def set_obj_xytheta(xytheta, obj):
    if isinstance(xytheta, list) or isinstance(xytheta, tuple):
        xytheta = np.array(xytheta)
    xytheta = xytheta.squeeze()
    set_quat(obj, quat_from_angle_vector(xytheta[-1], np.array([0, 0, 1])))
    set_xy(obj, xytheta[0], xytheta[1])


def set_active_config(conf, robot=None):
    if robot is None:
        env = openravepy.RaveGetEnvironments()[0]
        robot = env.GetRobot('pr2')
    robot.SetActiveDOFValues(conf.squeeze())


def draw_robot_at_conf(conf, transparency, name, robot, env, color=None):
    held_obj = robot.GetGrabbed()
    before = get_body_xytheta(robot)
    newrobot = RaveCreateRobot(env, '')
    newrobot.Clone(robot, 0)
    newrobot.SetName(name)
    env.Add(newrobot, True)
    set_active_config(conf, newrobot)
    newrobot.Enable(False)
    if color is not None:
        set_color(newrobot, color)
    print get_body_xytheta(robot)

    if len(held_obj) > 0:
        print get_body_xytheta(held_obj[0])
        set_robot_config(conf)
        held_obj = robot.GetGrabbed()[0]
        held_obj_trans = held_obj.GetTransform()
        release_obj()
        new_obj = RaveCreateKinBody(env, '')
        new_obj.Clone(held_obj, 0)
        new_obj.SetName(name + '_obj')
        env.Add(new_obj, True)
        new_obj.SetTransform(held_obj_trans)
        for link in new_obj.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)
        print get_body_xytheta(robot)
        grab_obj(held_obj)
        set_robot_config(before)

    for link in newrobot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(transparency)


def visualize_pick_and_place(pick, place):
    env = openravepy.RaveGetEnvironments()[0]
    obj = pick.discrete_parameters['object']
    if type(obj) == unicode or type(object) == str:
        obj = env.GetKinBody(obj)

    saver = CustomStateSaver(env)
    visualize_path(pick.low_level_motion)
    two_arm_pick_object(obj, pick.continuous_parameters)
    visualize_path(place.low_level_motion)
    saver.Restore()


def visualize_path(path):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    dim_path = len(path[0])
    if dim_path == 3:
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    elif dim_path == 11:
        manip = robot.GetManipulator('rightarm_torso')
        robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    else:
        assert path[0].shape[0] == robot.GetActiveDOF(), 'robot and path should have same dof'

    env = robot.GetEnv()
    if len(path) > 1000:
        path_reduced = path[0:len(path) - 1:int(len(path) * 0.1)]
    else:
        path_reduced = path
    for idx, conf in enumerate(path_reduced):
        is_goal_config = idx == len(path_reduced) - 1
        if is_goal_config:
            draw_robot_at_conf(conf, 0.5, 'path' + str(idx), robot, env)
        else:
            draw_robot_at_conf(conf, 0.9, 'path' + str(idx), robot, env)
    raw_input("Continue?")
    remove_drawn_configs('path', env)


def open_gripper(robot=None):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')
    robot.SetDOFValues(np.array([0.54800022]), robot.GetActiveManipulator().GetGripperIndices())


def close_gripper(robot):
    robot.SetDOFValues(np.array([0]), robot.GetActiveManipulator().GetGripperIndices())
    # taskprob = interfaces.TaskManipulation(robot)
    # robot.GetEnv().StopSimulation()
    # taskprob.CloseFingers()
    # robot.GetEnv().StartSimulation(0.01)


def check_collision_except(exception_body, env):
    # todo make this more efficient
    assert exception_body != env.GetRobots()[0], 'Collision exception cannot be the robot'

    # exception_body.Enable(False)  # todo what happens to the attached body when I enable and disable the held object?
    # col = env.CheckCollision(env.GetRobots()[0])
    # exception_body.Enable(True)
    # todo optimize this later
    return np.any([env.CheckCollision(env.GetRobots()[0], body) for body in env.GetBodies() if body != exception_body])
    # return col


def set_robot_config(base_pose, robot=None):
    if robot is None:
        env = openravepy.RaveGetEnvironments()[0]
        robot = env.GetRobot('pr2')

    base_pose = np.array(base_pose)
    base_pose = clean_pose_data(base_pose.astype('float'))

    robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    base_pose = np.array(base_pose).squeeze()
    """
    while base_pose[-1] < 0:
      try:
        factor = -int(base_pose[-1] /(2*np.pi))
      except:
        import pdb;pdb.set_trace()
      if factor == 0: factor = 1
      base_pose[-1] += factor*2*np.pi
    while base_pose[-1] > 2*np.pi:
      factor = int(base_pose[-1] /(2*np.pi))
      base_pose[-1] -= factor*2*np.pi
    
    if base_pose[-1] <
    if base_pose[-1] > 1.01:
      base_pose[-1] = 1.01
    elif base_pose[-1] < 0.99:
      base_pose[-1] = 0.99
    """
    # print base_pose
    robot.SetActiveDOFValues(base_pose)


def trans_from_xytheta(obj, xytheta):
    rot = rot_from_quat(quat_from_z_rot(xytheta[-1]))
    z = get_point(obj)[-1]
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, -1] = [xytheta[0], xytheta[1], z]
    return trans


def remove_drawn_configs(name, env):
    for body in env.GetBodies():
        if body.GetName().find(name) != -1:
            env.Remove(body)


def draw_robot_base_configs(configs, robot, env, name='bconf', transparency=0.7):
    for i in range(len(configs)):
        config = configs[i]
        draw_robot_at_conf(config, transparency, name + str(i), robot, env)


def draw_configs(configs, env, name='point', colors=None, transparency=0.1):
    # assert configs[0].shape==(6,), 'Config shape must be (6,)'
    if colors is None:
        for i in range(len(configs)):
            config = configs[i]
            new_body = box_body(env, 0.1, 0.05, 0.05, \
                                name=name + '%d' % i, \
                                color=(1, 0, 0), \
                                transparency=transparency)
            env.Add(new_body);
            set_point(new_body, np.append(config[0:2], 0.075))
            new_body.Enable(False)
            th = config[2]
            set_quat(new_body, quat_from_z_rot(th))
    else:
        for i in range(len(configs)):
            config = configs[i]
            if isinstance(colors, tuple):
                color = colors
            else:
                color = colors[i]
            new_body = box_body(env, 0.1, 0.05, 0.05, \
                                name=name + '%d' % i, \
                                color=color, \
                                transparency=transparency)
            """
            new_body = load_body(env,'mug.xml')
            set_name(new_body, name+'%d'%i)
            set_transparency(new_body, transparency)
            """
            env.Add(new_body);
            set_point(new_body, np.append(config[0:2], 0.075))
            new_body.Enable(False)
            th = config[2]
            set_quat(new_body, quat_from_z_rot(th))


def get_trajectory_length(trajectory):
    dists = 0
    for i in range(len(trajectory) - 1):
        dists += se2_distance(trajectory[i + 1], trajectory[i], 1, 1)
    return dists


def clean_pose_data(pose_data):
    if len(pose_data.shape) > 0 and pose_data.size > 0:
        # fixes angle to be between 0 to 2pi
        if len(pose_data.shape) == 1:
            pose_data = pose_data[None, :]

        data_idx_neg_angles = pose_data[:, -1] < 0
        data_idx_big_angles = pose_data[:, -1] > 2 * np.pi
        pose_data[data_idx_neg_angles, -1] += 2 * np.pi
        pose_data[data_idx_big_angles, -1] -= 2 * np.pi

        # assert( np.all(pose_data[:,-1]>=0) and np.all(pose_data[:,-1] <2*np.pi))
        return pose_data
    else:
        return np.array([])[None, :]


def compute_occ_vec(key_configs, robot, env):
    occ_vec = []
    with robot:
        for config in key_configs:
            set_robot_config(config, robot)
            collision = env.CheckCollision(robot) * 1
            occ_vec.append(collision)
    return np.array(occ_vec)


def get_robot_xytheta(robot):
    with robot:
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_xytheta = robot.GetActiveDOFValues()
    robot_xytheta = robot_xytheta[None, :]
    clean_pose_data(robot_xytheta)
    return robot_xytheta


def get_body_xytheta(body):
    Tbefore = body.GetTransform()
    body_quat = get_quat(body)
    th1 = np.arccos(body_quat[0]) * 2
    th2 = np.arccos(-body_quat[0]) * 2
    th3 = -np.arccos(body_quat[0]) * 2
    quat_th1 = quat_from_angle_vector(th1, np.array([0, 0, 1]))
    quat_th2 = quat_from_angle_vector(th2, np.array([0, 0, 1]))
    quat_th3 = quat_from_angle_vector(th3, np.array([0, 0, 1]))
    if np.all(np.isclose(body_quat, quat_th1)):
        th = th1
    elif np.all(np.isclose(body_quat, quat_th2)):
        th = th2
    elif np.all(np.isclose(body_quat, quat_th3)):
        th = th3
    else:
        print "This should not happen. Check if object is not standing still"
        import pdb;
        pdb.set_trace()
    if th < 0: th += 2 * np.pi
    assert (th >= 0 and th < 2 * np.pi)

    # set the quaternion using the one found
    set_quat(body, quat_from_angle_vector(th, np.array([0, 0, 1])))
    Tafter = body.GetTransform()
    assert (np.all(np.isclose(Tbefore, Tafter)))
    body_xytheta = np.hstack([get_point(body)[0:2], th])
    body_xytheta = body_xytheta[None, :]
    clean_pose_data(body_xytheta)
    return body_xytheta


def grab_obj(obj):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    robot.Grab(obj)


def release_obj():
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')
    obj = robot.GetGrabbed()[0]
    robot.Release(obj)


def convert_to_kin_body(obj):
    env = openravepy.RaveGetEnvironments()[0]
    if isinstance(obj, openravepy.KinBody):
        obj = obj
    else:
        obj = env.GetKinBody(obj)
    return obj


def one_arm_pick_object(obj, pick_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    open_gripper(robot)
    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    pick_config = pick_action['q_goal']
    set_active_config(pick_config)
    grab_obj(obj)


def one_arm_place_object(place_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    place_config = place_action['q_goal']
    set_active_config(place_config)
    release_obj()
    open_gripper(robot)


def two_arm_place_object(place_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    try:
        place_base_pose = place_action['q_goal']
    except KeyError:
        place_base_pose = place_action['base_pose']
    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')

    set_robot_config(place_base_pose, robot)
    release_obj()
    set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), rightarm_manip.GetArmIndices())


def fold_arms():
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    leftarm_manip = robot.GetManipulator('leftarm')
    rightarm_manip = robot.GetManipulator('rightarm')
    FOLDED_LEFT_ARM = [0.0, 1.29023451, 0.0, -2.12, 0.0, -0.69800004, 0.0]

    set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), rightarm_manip.GetArmIndices())


def two_arm_pick_object(obj, pick_action):
    assert len(openravepy.RaveGetEnvironments()) == 1
    env = openravepy.RaveGetEnvironments()[0]
    robot = env.GetRobot('pr2')

    if type(obj) == str or type(obj) == unicode:
        obj = robot.GetEnv().GetKinBody(obj)
    try:
        base_pose = pick_action['q_goal']
    except KeyError:
        base_pose = pick_action['base_pose']
    set_robot_config(base_pose, robot)

    if 'g_config' in pick_action.keys():
        g_config = pick_action['g_config']
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_torso_manip = robot.GetManipulator('rightarm_torso')
        set_config(robot, g_config[0], leftarm_manip.GetArmIndices())
        set_config(robot, g_config[1], rightarm_torso_manip.GetArmIndices())
    grab_obj(obj)


def viewer():
    env = openravepy.RaveGetEnvironments()[0]
    env.SetViewer('qtcoin')


def set_color(obj, color):
    env = openravepy.RaveGetEnvironments()[0]
    if type(obj) == str or type(obj) == unicode:
        obj = env.GetKinBody(obj)

    manipulation.bodies.bodies.set_color(obj, color)


def simulate_path(robot, path, timestep=0.001):
    for p in path:
        set_robot_config(p, robot)
        time.sleep(timestep)


def pick_distance(a1, a2, curr_obj):
    grasp_a1 = np.array(a1['grasp_params']).squeeze()
    base_a1 = clean_pose_data(np.array(a1['q_goal'])).squeeze()

    grasp_a2 = np.array(a2['grasp_params']).squeeze()
    base_a2 = clean_pose_data(np.array(a2['q_goal'])).squeeze()

    # normalize grasp distance
    grasp_max_diff = [1 / 2.356, 1., 1.]
    grasp_distance = np.sum(np.dot(abs(grasp_a1 - grasp_a2), grasp_max_diff))

    # bas_distance_max_diff = np.array([1./(2*2.51), 1./(2*2.51), 1/np.pi])
    base_distance_max_diff = np.array([1, 1, 1 / np.pi])
    base_distance = np.sum(np.dot(base_conf_diff(base_a1, base_a2), base_distance_max_diff))

    # base distance more important the grasp
    return grasp_distance + 2 * base_distance


def base_conf_diff(x, y):
    base_diff = abs(x - y)
    base_diff[-1] = base_diff[-1] if base_diff[-1] <= np.pi else 2 * np.pi - base_diff[-1]
    return base_diff


def place_distance(a1, a2):
    base_a1 = np.array(a1['q_goal'])
    base_a1 = clean_pose_data(base_a1).squeeze()

    base_a2 = np.array(a2['q_goal'])
    base_a2 = clean_pose_data(np.array(base_a2)).squeeze()

    base_distance_max_diff = np.array([1. / 2.51, 1. / 2.51, 1 / np.pi])
    base_distance = np.sum(np.dot(base_conf_diff(base_a1, base_a2), base_distance_max_diff))

    return base_distance


def compute_robot_xy_given_ir_parameters(portion, angle, obj, radius=PR2_ARM_LENGTH):
    dist_to_obj = radius * portion  # how close are you to obj?
    x = dist_to_obj * np.cos(angle)
    y = dist_to_obj * np.sin(angle)
    robot_wrt_o = np.array([x, y, 0, 1])
    return np.dot(obj.GetTransform(), robot_wrt_o)[:-1]


def get_pick_base_pose_and_grasp_from_pick_parameters(obj, pick_parameters):
    grasp_params = pick_parameters[0:3]
    portion = pick_parameters[3]
    base_angle = pick_parameters[4]
    facing_angle = pick_parameters[5]

    pick_base_pose = compute_robot_xy_given_ir_parameters(portion, base_angle, obj)
    obj_xy = get_body_xytheta(obj).squeeze()[:-1]
    robot_xy = pick_base_pose[0:2]
    angle_to_be_set = compute_angle_to_be_set(obj_xy, robot_xy)
    pick_base_pose[-1] = angle_to_be_set + facing_angle
    return grasp_params, pick_base_pose


def pick_parameter_distance(obj, param1, param2):
    grasp_params1, pick_base_pose1 = get_pick_base_pose_and_grasp_from_pick_parameters(obj, param1)
    grasp_params2, pick_base_pose2 = get_pick_base_pose_and_grasp_from_pick_parameters(obj, param2)

    base_pose_distance = se2_distance(pick_base_pose1, pick_base_pose2, 1, 1)
    grasp_distance = np.linalg.norm(grasp_params2 - grasp_params1)

    c1 = 2
    c2 = 1
    distance = c1 * base_pose_distance + c2 * grasp_distance
    return distance


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def se2_distance(base_a1, base_a2, c1, c2):
    base_a1 = base_a1.squeeze()
    base_a2 = base_a2.squeeze()

    x1, y1 = pol2cart(1, base_a1[-1])
    x2, y2 = pol2cart(1, base_a2[-1])

    angle_distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    base_distance = np.linalg.norm(base_a1[0:2] - base_a2[0:2])

    distance = c1 * base_distance + c2 * angle_distance
    return distance


def are_base_confs_close_enough(q1, q2, xy_threshold, th_threshold):
    diff = base_conf_diff(q1, q2)
    th_threshold = th_threshold * np.pi / 180.0
    if np.linalg.norm(diff[0:2]) < xy_threshold and diff[-1] < th_threshold:
        return True
    else:
        return False


def convert_base_pose_to_se2(base_pose):
    base_pose = base_pose.squeeze()
    a, b = pol2cart(1, base_pose[-1])
    x, y = base_pose[0], base_pose[1]
    return np.array([x, y, a, b])


def convert_se2_to_base_pose(basepose_se2):
    basepose_se2 = basepose_se2.squeeze()

    phi = cart2pol(basepose_se2[2], basepose_se2[3])
    return np.array([basepose_se2[0], basepose_se2[1], phi])


def place_parameter_distance(param1, param2, c1=1):
    return se2_distance(param1, param2, c1, 0)


def get_place_domain(region):
    box = np.array(region.box)
    x_range = np.array([[box[0, 0]], [box[0, 1]]])
    y_range = np.array([[box[1, 0]], [box[1, 1]]])
    th_range = np.array([[0], [2 * np.pi]])
    domain = np.hstack([x_range, y_range, th_range])
    return domain


def get_pick_domain():
    portion_domain = [[0.4], [0.9]]
    base_angle_domain = [[0], [2 * np.pi]]
    facing_angle_domain = [[-30 * np.pi / 180.0], [30 * np.pi / 180]]
    base_pose_domain = np.hstack([portion_domain, base_angle_domain, facing_angle_domain])

    grasp_param_domain = np.array([[45 * np.pi / 180, 0.5, 0.1], [180 * np.pi / 180, 1, 0.9]])
    domain = np.hstack([grasp_param_domain, base_pose_domain])
    return domain


def set_obj_xyztheta(xyztheta, obj):
    set_point(obj, xyztheta[0:-1])
    set_obj_xytheta(np.hstack([xyztheta[0:2], xyztheta[-1]]), obj)


def get_body_with_name(obj_name):
    env = openravepy.RaveGetEnvironments()[0]
    return env.GetKinBody(obj_name)


def randomly_place_region(body, region):
    env = openravepy.RaveGetEnvironments()[0]
    if env.GetKinBody(get_name(body)) is None:
        env.Add(body)
    while True:
        set_quat(body, quat_from_z_rot(uniform(0, 2 * PI)))
        aabb = aabb_from_body(body)
        cspace = region.cspace(aabb)
        if cspace is None: continue
        set_point(body, np.array([uniform(*range) for range in cspace] + [
            region.z + aabb.extents()[2] + BODY_PLACEMENT_Z_OFFSET]) - aabb.pos() + get_point(body))
        if not body_collision(env, body):
            return


class CustomStateSaver:
    def __init__(self, env):
        objects_in_env = env.GetBodies()
        self.env_id = 1

        self.robot_name = 'pr2'
        robot = env.GetRobot(self.robot_name)
        self.object_poses = {o.GetName(): get_body_xytheta(o) for o in objects_in_env}
        self.object_poses = {}
        for o in objects_in_env:
            xyz = get_point(o)
            xytheta = get_body_xytheta(o)
            xyztheta = np.hstack([xyz, xytheta.squeeze()[-1]])
            self.object_poses[o.GetName()] = xyztheta
            # todo set xyztheta

        self.robot_base_pose = get_body_xytheta(robot)
        self.robot_dof_values = robot.GetDOFValues()
        self.is_holding = len(robot.GetGrabbed()) > 0
        if self.is_holding:
            self.held_object = robot.GetGrabbed()[0].GetName()
        else:
            self.held_object = None

    def Restore(self):
        assert len(openravepy.RaveGetEnvironments()) == 1
        env = openravepy.RaveGetEnvironment(self.env_id)
        robot = env.GetRobot(self.robot_name)

        currently_holding = len(robot.GetGrabbed()) > 0
        if currently_holding:
            held_obj = robot.GetGrabbed()[0]
            release_obj()

        for obj_name, obj_pose in zip(self.object_poses.keys(), self.object_poses.values()):
            set_obj_xyztheta(obj_pose, env.GetKinBody(obj_name))
        set_robot_config(self.robot_base_pose, robot)
        robot.SetDOFValues(self.robot_dof_values)

        if self.is_holding:
            held_obj = env.GetKinBody(self.held_object)
            grab_obj(held_obj)

        # print "After-restoration"
