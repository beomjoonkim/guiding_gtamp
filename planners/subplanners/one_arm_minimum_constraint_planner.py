from generators.uniform import UniformPaPGenerator
from gtamp_utils import utils


class OneArmMinimumConstraintPlanner:
    def __init__(self, problem_env, operator_skeleton):
        self.problem_env = problem_env
        self.operator_skeleton = operator_skeleton

    def sample_pick_config(self):
        target_object = self.operator_skeleton.discrete_parameters['object']
        target_object.Enable(True)

        generator = UniformPaPGenerator(self.operator_skeleton,
                                        self.problem_env,
                                        None,
                                        n_candidate_params_to_smpl=1,
                                        total_number_of_feasibility_checks=50,
                                        dont_check_motion_existence=True)

        param = generator.sample_next_point(self.operator_skeleton)
        return param

    def approximate_minimal_collision_path(self, naive_path, naive_path_collisions):
        enabled_objects = {obj.GetName() for obj in self.problem_env.objects}
        enabled_objects -= {obj.GetName() for obj in naive_path_collisions}

        [o.Enable(False) for o in naive_path_collisions]
        minimal_objects_in_way = []
        minimal_collision_path = naive_path
        for obj in naive_path_collisions:
            obj.Enable(True)
            [o.Enable(False) for o in minimal_objects_in_way]
            enabled_objects.add(obj.GetName())
            enabled_objects -= {obj.GetName() for obj in minimal_objects_in_way}
            param = self.sample_pick_config()
            is_param_feasible = param['motion'] is not None
            if not is_param_feasible:
                minimal_objects_in_way.append(obj)
            else:
                minimal_collision_path = param['motion']
        self.problem_env.enable_objects_in_region('entire_region')
        return minimal_collision_path

    def get_motion_plan(self):
        self.problem_env.disable_objects_in_region('entire_region')
        param = self.sample_pick_config()
        self.problem_env.enable_objects_in_region('entire_region')
        naive_path = [param['motion']]
        is_param_feasible = param['motion'] is not None
        if is_param_feasible:
            naive_path_collisions = self.problem_env.get_objs_in_collision(naive_path, 'entire_region')
            no_obstacle = len(naive_path_collisions) == 0
            if no_obstacle:
                return naive_path, 'HasSolution'
            assert len(naive_path_collisions) > 0
            minimal_collision_path = self.approximate_minimal_collision_path(naive_path, naive_path_collisions)
            self.problem_env.enable_objects_in_region('entire_region')
            return minimal_collision_path, "HasSolution"
        else:
            return None, "NoSolution"
