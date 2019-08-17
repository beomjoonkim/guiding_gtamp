from openravepy import DOFAffine
from gtamp_utils.motion_planner import collision_fn, base_extend_fn, base_sample_fn, base_distance_fn, \
    rrt_connect, prm_connect, rrt_region, arm_base_sample_fn, arm_base_distance_fn, \
    arm_base_extend_fn


class MotionPlanner:
    def __init__(self, problem_env):
        self.problem_env = problem_env

    def get_motion_plan(self, goal_configuration):
        raise NotImplementedError


class BaseMotionPlanner(MotionPlanner):
    def __init__(self, problem_env, algorithm):
        MotionPlanner.__init__(self, problem_env)
        self.algorithm = algorithm

    def get_motion_plan(self, goal, region_name='entire_region', n_iterations=None, cached_collisions=None):
        self.problem_env.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

        if region_name == 'bridge_region':
            region_name = 'entire_region'

        region_x = self.problem_env.problem_config[region_name + '_xy'][0]
        region_y = self.problem_env.problem_config[region_name + '_xy'][1]
        region_x_extents = self.problem_env.problem_config[region_name + '_extents'][0]
        region_y_extents = self.problem_env.problem_config[region_name + '_extents'][1]
        d_fn = base_distance_fn(self.problem_env.robot, x_extents=region_x_extents, y_extents=region_y_extents)
        s_fn = base_sample_fn(self.problem_env.robot, x_extents=region_x_extents, y_extents=region_y_extents,
                              x=region_x, y=region_y)

        e_fn = base_extend_fn(self.problem_env.robot)

        if cached_collisions is not None:
            c_fn = set()
            for tmp in cached_collisions.values():
                c_fn = c_fn.union(tmp)
        else:
            c_fn = collision_fn(self.problem_env.env, self.problem_env.robot)

        q_init = self.problem_env.robot.GetActiveDOFValues()

        if n_iterations is None:
            n_iterations = [20, 50, 100, 500, 1000]

        path = None
        status = 'NoSolution'
        if self.algorithm == 'rrt':
            planning_algorithm = rrt_connect
            assert cached_collisions is None
            if not isinstance(goal, list):
                goal = [goal]
            for n_iter in n_iterations:
                print n_iter
                for g in goal:
                    path = planning_algorithm(q_init, g, d_fn, s_fn, e_fn, c_fn, iterations=n_iter)
                    if path is not None:
                        return path, 'HasSolution'
        else:
            planning_algorithm = prm_connect
            path = planning_algorithm(q_init, goal, c_fn)
            if path is not None:
                status = "HasSolution"

        return path, status


class ArmBaseMotionPlanner(MotionPlanner):
    def __init__(self, problem_env, algorithm):
        MotionPlanner.__init__(self, problem_env)
        self.algorithm = algorithm

    def get_motion_plan(self, goal, region_name='entire_region', manip_name='rightarm_torso', cached_collisions=None):
        region_x = self.problem_env.problem_config[region_name + '_xy'][0]
        region_y = self.problem_env.problem_config[region_name + '_xy'][1]
        region_x_extents = self.problem_env.problem_config[region_name + '_extents'][0]
        region_y_extents = self.problem_env.problem_config[region_name + '_extents'][1]
        d_fn = arm_base_distance_fn(self.problem_env.robot, region_x_extents, region_y_extents)
        s_fn = arm_base_sample_fn(self.problem_env.robot, region_x_extents, region_y_extents, region_x, region_y)
        e_fn = arm_base_extend_fn(self.problem_env.robot)
        if cached_collisions is not None:
            c_fn = cached_collisions
        else:
            c_fn = collision_fn(self.problem_env.env, self.problem_env.robot)

        manip = self.problem_env.robot.GetManipulator('rightarm_torso')
        self.problem_env.robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis,
                                             [0, 0, 1])

        q_init = self.problem_env.robot.GetActiveDOFValues()
        n_iterations = [20, 50, 100, 500, 1000]
        print 'Arm-base motion planning...'
        if self.algorithm == 'rrt':
            planning_algorithm = rrt_connect
        elif self.algorithm == 'rrt_region':
            planning_algorithm = rrt_region
        else:
            planning_algorithm = prm_connect

        path = None
        status = 'NoSolution'
        for n_iter in n_iterations:
            if self.algorithm == 'rrt' and isinstance(goal, list):
                for g in goal:  # todo make rrt to be able to take multiple goals like prm
                    path = planning_algorithm(q_init, g, d_fn, s_fn, e_fn, c_fn, iterations=n_iter)
            else:
                path = planning_algorithm(q_init, goal, d_fn, s_fn, e_fn, c_fn, iterations=n_iter)
            if path is not None:
                status = "HasSolution"

        return path, status

    def set_operator_instance(self, operator_instance):
        self.operator_instance = operator_instance


class OperatorBaseMotionPlanner(BaseMotionPlanner):
    def __init__(self, problem_env, planning_algorithm):
        BaseMotionPlanner.__init__(self, problem_env, planning_algorithm)
        self.operator_instance = None

    def set_operator_instance(self, operator_instance):
        self.operator_instance = operator_instance
