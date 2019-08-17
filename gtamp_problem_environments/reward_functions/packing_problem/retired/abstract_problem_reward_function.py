from problem_environments.reward_functions.reward_function import RewardFunction
from gtamp_utils.utils import two_arm_pick_object, two_arm_place_object, one_arm_pick_object, one_arm_place_object


class AbstractProblemRewardFunction(RewardFunction):
    def __init__(self, problem_env):
        RewardFunction.__init__(self, problem_env)

    def apply_operator_instance_and_get_reward(self, state, operator_instance, is_op_feasible):
        if not is_op_feasible:
            reward = self.infeasible_reward
        else:
            if operator_instance.type == 'two_arm_pick':
                two_arm_pick_object(operator_instance.discrete_parameters['object'],
                                    self.robot, operator_instance.continuous_parameters)
                reward = 0
            elif operator_instance.type == 'two_arm_place':
                object_held = self.robot.GetGrabbed()[0]
                previous_region = self.problem_env.get_region_containing(object_held)
                two_arm_place_object(object_held, self.robot, operator_instance.continuous_parameters)
                current_region = self.problem_env.get_region_containing(object_held)

                if current_region.name == 'home_region' and previous_region != current_region:
                    task_reward = 1
                elif current_region.name == 'loading_region' and previous_region.name == 'home_region':
                    task_reward = -1.5
                else:
                    task_reward = 0  # placing it in the same region

                reward = task_reward
            elif operator_instance.type == 'one_arm_pick':
                one_arm_pick_object(operator_instance.discrete_parameters['object'],
                                    operator_instance.continuous_parameters)
                reward = 1
            elif operator_instance.type == 'one_arm_place':
                one_arm_place_object(operator_instance.discrete_parameters['object'],
                                     operator_instance.continuous_parameters)
            else:
                raise NotImplementedError
        return reward

    def is_goal_reached(self):
        return False

