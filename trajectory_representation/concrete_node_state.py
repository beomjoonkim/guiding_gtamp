from gtamp_utils import utils
import numpy as np


class ConcreteNodeState:
    def __init__(self, problem_env, obj, region, goal_entities, key_configs=None, collision_vector=None):
        self.obj = obj
        self.region = region
        self.goal_entities = goal_entities
        self.problem_env = problem_env

        if collision_vector is None:
            self.key_configs = key_configs
            self.collision_vector = utils.compute_occ_vec(self.key_configs)
        else:
            self.collision_vector = collision_vector
            self.key_configs = None

        self.one_hot = utils.convert_binary_vec_to_one_hot(self.collision_vector)
        is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([obj in goal_entities]))
        is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([region in goal_entities]))

        self.state_vec = np.vstack([self.one_hot, is_goal_obj, is_goal_region])
        self.state_vec = self.state_vec.reshape((1, len(self.state_vec), 2, 1))

        if type(obj) == str or type(obj) == unicode:
            obj = problem_env.env.GetKinBody(obj)
        #robot_xy_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), problem_env.robot.GetTransform())[:-2, 3]
        #self.robot_wrt_obj = robot_xy_wrt_o.reshape((1, 2))
        #self.obj_pose = utils.get_body_xytheta(obj)[0, 0:2].reshape((1, 2))

        self.robot_pose = utils.get_body_xytheta(self.problem_env.robot)
        self.obj_pose = utils.get_body_xytheta(obj)

        """
        t_robot = utils.get_transform_from_pose(self.robot_pose, 'robot')
        t_obj = utils.get_transform_from_pose(self.obj_pose, 'kinbody')
        assert np.all(np.isclose(t_robot, self.problem_env.robot.GetTransform()))
        assert np.all(np.isclose(t_obj, obj.GetTransform()))
        rel_t = utils.get_relative_transform_T1_wrt_T2(t_robot, t_obj)
        true_rel_t = np.dot(np.linalg.inv(t_obj), t_robot)
        assert np.all(np.isclose(rel_t, true_rel_t))
        """
        rel_pick_pose = utils.get_relative_robot_pose_wrt_body_pose(self.robot_pose, self.obj_pose)
        recovered = utils.clean_pose_data(utils.get_global_pose_from_relative_pose_to_body(obj, rel_pick_pose))
        self.robot_pose = utils.clean_pose_data(self.robot_pose)
        # why does z not match?
        try:
            assert np.all(np.isclose(recovered, self.robot_pose.squeeze()))
        except:
            import pdb;pdb.set_trace()




