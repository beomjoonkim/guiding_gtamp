from problem_environments.reward_functions.reward_function import RewardFunction
from gtamp_utils.utils import two_arm_pick_object, two_arm_place_object


class ObstacleClearingProblemRewardFunction(RewardFunction):
    def __init__(self, problem_env):
        RewardFunction.__init__(self, problem_env)
        self.infeasible_reward = -2
        self.swept_volume = None
        self.plan_skeleton = None
        self.picked = False
        self.placed = False

    def set_swept_volume(self, swept_volume):
        self.swept_volume = swept_volume

    def apply_operator_instance_and_get_reward(self, operator_instance, is_op_feasible):
        if not is_op_feasible:
            print "Infeasible parameters"
            return self.infeasible_reward
        else:
            # apply the action
            objects_in_collision = self.problem_env.get_objs_in_collision(operator_instance.low_level_motion,
                                                                          'entire_region')
            if operator_instance.type == 'two_arm_pick':
                two_arm_pick_object(operator_instance.discrete_parameters['object'],
                                    self.robot, operator_instance.continuous_parameters)
            elif operator_instance.type == 'two_arm_place':
                object_held = self.problem_env.robot.GetGrabbed()[0]
                two_arm_place_object(object_held, self.robot, operator_instance.continuous_parameters)

            return -len(objects_in_collision) / 8.0

    def apply_operator_skeleton_and_get_reward(self, operator_skeleton):
        return 0

    def set_plan_skeleton(self, target_pick_skeleton, target_place_skeleton):
        self.plan_skeleton = [target_pick_skeleton, target_place_skeleton]

    def is_goal_reached(self):
        target_object = self.plan_skeleton[0].discrete_parameters['object']
        target_region = self.plan_skeleton[1].discrete_parameters['region']

        if type(target_region) == str:
            target_region = self.problem_env.regions[target_region]

        if type(target_object) == str:
            target_object = self.problem_env.env.GetKinBody(target_object)

        try:
            is_plan_skeleton_satisfied = self.problem_env.get_region_containing(
                target_object).name == target_region.name
        except:
            pass

        print "Is plan skeleton satisfied?", is_plan_skeleton_satisfied
        if is_plan_skeleton_satisfied and self.swept_volume.is_swept_volume_cleared(target_object):
            return True
        else:
            return False

    def is_optimal_plan_found(self, best_traj_reward):
        return best_traj_reward >= 0.0 / 8.0
