from generators.uniform import UniformGenerator


class MinimumConstraintGoalSampler:
    def __init__(self, problem_env, operator_skeleton):
        self.problem_env = problem_env
        self.operator_skeleton = operator_skeleton
        self.is_place_op = self.operator_skeleton.type.find('pick') == -1

    def sample_pick_config(self):
        target_object = self.operator_skeleton.discrete_parameters['object']
        target_object.Enable(True)
        generator = UniformGenerator(self.operator_skeleton, self.problem_env, None)
        param = generator.sample_next_point(self.operator_skeleton,
                                            n_iter=1000,
                                            n_parameters_to_try_motion_planning=1,
                                            dont_check_motion_existence=True,
                                            cached_collisions=None)
        return param

    def sample_place_config(self):
        generator = UniformGenerator(self.operator_skeleton, self.problem_env, None)
        param = generator.sample_next_point(self.operator_skeleton,
                                            n_iter=1000,
                                            n_parameters_to_try_motion_planning=1,
                                            dont_check_motion_existence=True,
                                            cached_collisions=None)
        return param

    def get_motion_plan(self):
        self.problem_env.enable_objects_in_region('entire_region')
        idx = 0
        for _ in range(100):
            if self.is_place_op:
                param = self.sample_place_config()
            else:
                param = self.sample_pick_config()
            if param['is_feasible']:
                break
            self.problem_env.objects[idx].Enable(False)
        self.problem_env.enable_objects_in_region('entire_region')
        if param['is_feasible']:
            return param['motion'], "HasSolution"
        else:
            return None, "NoSolution"



