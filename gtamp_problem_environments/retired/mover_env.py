import numpy as np

from openravepy import DOFAffine
from problem_environments.problem_environment import ProblemEnvironment
from problem_environments.mover_problem import MoverProblem
from trajectory_representation.operator import Operator

from gtamp_utils.utils import two_arm_pick_object, two_arm_place_object, set_robot_config, get_body_xytheta, \
    visualize_path, CustomStateSaver
from gtamp_utils.operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class Mover(ProblemEnvironment):
    def __init__(self):
        ProblemEnvironment.__init__(self)
        problem = MoverProblem(self.env)
        self.problem_config = problem.get_problem_config()
        self.robot = self.env.GetRobots()[0]
        self.objects = self.problem_config['packing_boxes']
        self.object_init_poses = {o.GetName(): get_body_xytheta(o).squeeze() for o in self.objects}
        self.init_base_conf = np.array([0, 1.05, 0])
        self.regions = {'entire_region': self.problem_config['entire_region'],
                        'home_region': self.problem_config['home_region'],
                        'loading_region': self.problem_config['loading_region']}
        self.placement_regions = {'home_region': self.problem_config['home_region'],
                                  'loading_region': self.problem_config['loading_region']}

        self.entity_names = [obj.GetName() for obj in self.objects] + list(self.regions)
        self.entity_idx = {name: idx for idx, name in enumerate(self.entity_names)}

        self.is_init_pick_node = True
        self.name = 'mover'
        self.init_saver = CustomStateSaver(self.env)
        self.problem_config['env'] = self.env
        self.operator_names = ['two_arm_pick', 'two_arm_place']
        self.reward_function = None
        self.applicable_op_constraint = None
        self.two_arm_pick_continuous_constraint = None
        self.two_arm_place_continuous_constraint = None
        self.objects_to_check_collision = None

    def set_exception_objs_when_disabling_objects_in_region(self, p):
        self.objects_to_check_collision = p

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def set_action_constraint(self, pick_constraint, place_constraint):
        target_object = pick_constraint.discrete_parameters['object']
        target_region = place_constraint.discrete_parameters['region']
        self.set_applicable_op_constraint({'object': target_object, 'region': target_region})

        if pick_constraint.continuous_parameters is not None:
            self.set_op_continuous_constraint(pick_constraint.type, pick_constraint.continuous_parameters)
        else:
            self.two_arm_pick_continuous_constraint = None

        if place_constraint.continuous_parameters is not None:
            self.set_op_continuous_constraint(place_constraint.type, place_constraint.continuous_parameters)
        else:
            self.two_arm_place_continuous_constraint = None

    def set_op_continuous_constraint(self, op_type, parameters):
        if op_type == 'two_arm_pick':
            self.two_arm_pick_continuous_constraint = parameters
        elif op_type == 'two_arm_place':
            self.two_arm_place_continuous_constraint = parameters
        else:
            raise NotImplementedError

    def set_applicable_op_constraint(self, constraint):
        # to handle both pkled abstract plan and raw
        if type(constraint['object']) == str:
            constraint['object'] = self.env.GetKinBody(constraint['object'])
        if type(constraint['region']) == str:
            constraint['region'] = self.regions[constraint['region']]

        self.applicable_op_constraint = constraint

    def get_objs_in_region(self, region_name):
        movable_objs = [o for o in self.objects if not (self.env.GetKinBody(o.GetName()) is None)]
        objs_in_region = []
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                objs_in_region.append(obj)
        return objs_in_region

    def get_region_containing(self, obj):
        if type(obj) == str:
            obj = self.env.GetKinBody(obj)

        if self.regions['home_region'].contains(obj.ComputeAABB()):
            return self.regions['home_region']
        elif self.regions['loading_region'].contains(obj.ComputeAABB()):
            return self.regions['loading_region']
        else:
            assert False, "This should not happen"

    def reset_to_init_state(self, node):
        saver = node.state_saver
        # print "Restoring from mover_env"
        saver.Restore()  # this call re-enables objects that are disabled
        # todo I need to re-grab the object if the object was grabbed, because Restore destroys the grabs

        if node.parent_action is not None:
            # todo
            #  When we are switching to a place, none-operator skeleton node, then we need to pick the object
            #  We need to determine the node's operator type. If it is a pick
            #  Corner case: node is a non-operator skeleton node, and it is time to place

            if node.parent_action.type is 'two_arm_pick' and node.is_operator_skeleton_node:  # place op-skeleton node
                # pick parent's object
                grabbed_object = node.parent_action.discrete_parameters['object']
                two_arm_pick_object(grabbed_object, self.robot, node.parent_action.continuous_parameters)
            elif node.parent_action.type is 'two_arm_place' and not node.is_operator_skeleton_node:  # place op-instance node
                # pick grand-parent's object
                grabbed_object = node.parent.parent_action.discrete_parameters['object']
                two_arm_pick_object(grabbed_object, self.robot, node.parent.parent_action.continuous_parameters)

        self.curr_state = self.get_state()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def set_init_namo_object_names(self, object_names):
        self.namo_planner.init_namo_object_names = object_names

    def disable_objects_in_region(self, region_name):
        if len(self.robot.GetGrabbed()) > 0:
            held_obj = self.robot.GetGrabbed()[0]
        else:
            held_obj = None

        movable_objs = [o for o in self.objects if not (self.env.GetKinBody(o.GetName()) is None)]
        for obj in movable_objs:
            if obj == held_obj:
                continue
            obj.Enable(False)
        if self.objects_to_check_collision is not None:
            [o.Enable(True) for o in self.objects_to_check_collision]

    def enable_objects_in_region(self, region_name):
        movable_objs = [o for o in self.objects if not (self.env.GetKinBody(o.GetName()) is None)]
        for object in movable_objs:
            object.Enable(True)

    def disable_objects(self):
        for object in self.objects:
            if object is None:
                continue
            object.Enable(False)

    def enable_objects(self):
        for object in self.objects:
            if object is None:
                continue
            object.Enable(True)

    def check_base_pose_feasible(self, base_pose, obj, region):
        if base_pose is None:
            return False
        if not self.is_collision_at_base_pose(base_pose, obj) \
                and self.is_in_region_at_base_pose(base_pose, obj, robot_region=region,
                                                   obj_region=region):
            return True
        return False

    def apply_two_arm_pick_action_stripstream(self, action, obj=None, do_check_reachability=False):
        if obj is None:
            obj_to_pick = self.curr_obj
        else:
            obj_to_pick = obj

        pick_base_pose, grasp_params = action
        set_robot_config(pick_base_pose, self.robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj_to_pick,
                                       robot=self.robot)
        g_config = solveTwoArmIKs(self.env, self.robot, obj_to_pick, grasps)
        try:
            assert g_config is not None
        except:
            pass
            # import pdb; pdb.set_trace()

        action = {'base_pose': pick_base_pose, 'g_config': g_config}
        two_arm_pick_object(obj_to_pick, self.robot, action)

        curr_state = self.get_state()
        reward = 0
        pick_path = None
        return curr_state, reward, g_config, pick_path

    def check_reachability_precondition(self, operator_instance, ignore_collision=False):
        if operator_instance.type.find('one') != -1:
            goal_config = operator_instance.continuous_parameters['g_config'].squeeze().tolist() \
                          + operator_instance.continuous_parameters['q_goal'].squeeze().tolist()
        else:
            goal_config = operator_instance.continuous_parameters['q_goal']

        if operator_instance.low_level_motion is not None:
            motion = operator_instance.low_level_motion
            status = 'HasSolution'
            return motion, status

        if ignore_collision:
            self.disable_objects_in_region('entire_region')
            if operator_instance.type.find('pick') != -1:
                obj = operator_instance.discrete_parameters['object']
                obj.Enable(True)

        self.motion_planner.set_operator_instance(operator_instance)
        motion, status = self.motion_planner.get_motion_plan(goal_config)

        if ignore_collision:
            self.enable_objects_in_region('entire_region')

        return motion, status

    @staticmethod
    def check_parameter_feasibility_precondition(operator_instance):
        if operator_instance.continuous_parameters['motion'] is None:
            print "Parameter values infeasible"
            return False
        else:
            return True

    def apply_operator_skeleton(self, operator_skeleton):
        reward = self.reward_function.apply_operator_skeleton_and_get_reward(operator_skeleton)
        return reward

    def apply_operator_instance(self, operator_instance, check_reachability=True):
        if self.check_parameter_feasibility_precondition(operator_instance):
            """
            if operator_instance.low_level_motion is None:
                motion_plan, status = self.check_reachability_precondition(operator_instance)
                if status != 'HasSolution':
                    print "Motion planning failed"
            else:
                motion_plan = operator_instance.low_level_motion
                status = "HasSolution"
            operator_instance.update_low_level_motion(motion_plan)
            """

            reward = self.reward_function.apply_operator_instance_and_get_reward(operator_instance, True)
        else:
            reward = self.reward_function.apply_operator_instance_and_get_reward(operator_instance, False)
        return reward

    def is_region_contain_object(self, region, obj):
        return region.contains(obj.ComputeAABB())

    def is_goal_reached(self):
        return self.reward_function.is_goal_reached()

    def check_holding_object_precondition(self):
        if len(self.robot.GetGrabbed()) == 0:
            return False
        else:
            return True

    def get_applicable_ops(self, parent_op=None):
        applicable_ops = []
        for op_name in self.operator_names:
            if op_name.find('place') != -1:
                if self.check_holding_object_precondition():
                    object_held = self.robot.GetGrabbed()[0]
                    if self.applicable_op_constraint is None:
                        for region in self.placement_regions.values():
                            if op_name == 'one_arm_place':
                                assert parent_op is not None
                                op = Operator(operator_type=op_name,
                                              discrete_parameters={'region': region,
                                                                   'object': object_held},
                                              continuous_parameters={
                                                  'grasp_params': parent_op.continuous_parameters['grasp_params']})
                            else:
                                op = Operator(operator_type=op_name,
                                              discrete_parameters={'region': region,
                                                                   'object': object_held})
                            applicable_ops.append(op)
                    else:
                        op = Operator(operator_type=op_name,
                                      discrete_parameters={'region': self.applicable_op_constraint['region'],
                                                           'object': object_held})
                        applicable_ops.append(op)
            else:
                if not self.check_holding_object_precondition():
                    if self.applicable_op_constraint is None:
                        for obj in self.objects:
                            op = Operator(operator_type=op_name,
                                          discrete_parameters={'object': obj})
                            applicable_ops.append(op)
                    else:
                        op = Operator(operator_type=op_name,
                                      discrete_parameters={'object': self.applicable_op_constraint['object']})
                        applicable_ops.append(op)
        return applicable_ops
