import numpy as np

from openravepy import DOFAffine
from gtamp_problem_environments.problem_environment import ProblemEnvironment
from gtamp_problem_environments.mover_environment_definition import MoverEnvironmentDefinition
from trajectory_representation.operator import Operator

from gtamp_utils.utils import two_arm_pick_object, set_robot_config, get_body_xytheta, \
    CustomStateSaver, set_obj_xytheta

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class Mover(ProblemEnvironment):
    def __init__(self, problem_idx, problem_type='normal'):
        ProblemEnvironment.__init__(self, problem_idx)
        problem_defn = MoverEnvironmentDefinition(self.env)
        self.problem_config = problem_defn.get_problem_config()
        self.robot = self.env.GetRobots()[0]
        self.objects = self.problem_config['packing_boxes']

        self.set_problem_type(problem_type)

        self.object_init_poses = {o.GetName(): get_body_xytheta(o).squeeze() for o in self.objects}
        self.initial_robot_base_pose = get_body_xytheta(self.robot)
        self.regions = {'entire_region': self.problem_config['entire_region'],
                        'home_region': self.problem_config['home_region'],
                        'loading_region': self.problem_config['loading_region']}
        self.region_names = ['entire_region', 'home_region', 'loading_region']
        self.object_names = [obj.GetName() for obj in self.objects]
        self.placement_regions = {'home_region': self.problem_config['home_region'],
                                  'loading_region': self.problem_config['loading_region']}

        #self.entity_names = self.object_names + ['home_region', 'loading_region','entire_region']
        self.entity_names = self.object_names + ['home_region', 'loading_region', 'entire_region']
        self.entity_idx = {name: idx for idx, name in enumerate(self.entity_names)}

        self.is_init_pick_node = True
        self.name = 'two_arm_mover'
        self.init_saver = CustomStateSaver(self.env)
        self.problem_config['env'] = self.env
        self.operator_names = ['two_arm_pick', 'two_arm_place']
        self.reward_function = None
        self.applicable_op_constraint = None
        self.two_arm_pick_continuous_constraint = None
        self.two_arm_place_continuous_constraint = None
        self.objects_to_check_collision = None
        self.goal = None

    def set_problem_type(self, problem_type):
        if problem_type == 'normal':
            pass
        elif problem_type == 'nonmonotonic':
            #from manipulation.primitives.display import set_viewer_options
            #self.env.SetViewer('qtcoin')
            #set_viewer_options(self.env)

            set_color(self.objects[0], [1,1,1])
            set_color(self.objects[4], [0,0,0])

            poses = [
                [2.3, -6.4, 0],
                [3.9, -6.2, 0],
                [1.5, -6.3, 0],
                [3.9, -5.5, 0],
                [0.8, -5.5, 0],
                [3.2, -6.2, 0],
                [1.5, -5.5, 0],
                [3.2, -5.5, 0],
            ]

            q = [2.3, -5.5, 0]

            set_robot_config(q)

            for obj, pose in zip(self.objects, poses):
                set_obj_xytheta(pose, obj)

            #import pdb; pdb.set_trace()

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
        else:
            constraint['object'] = constraint['object']

        if type(constraint['region']) == str:
            constraint['region'] = self.regions[constraint['region']]
        else:
            constraint['region'] = constraint['region']

        self.applicable_op_constraint = constraint

    def get_objs_in_region(self, region_name):
        movable_objs = [o for o in self.objects if not (self.env.GetKinBody(o.GetName()) is None)]
        objs_in_region = []
        for obj in movable_objs:
            if self.regions[region_name].contains(obj.ComputeAABB()):
                objs_in_region.append(obj)
        return objs_in_region

    def get_region_containing(self, obj):
        if type(obj) == str or type(obj) == unicode:
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
            for o in self.objects_to_check_collision:
                if isinstance(o, str) or isinstance(o, unicode):
                    self.env.GetKinBody(o).Enable(True)
                else:
                    o.Enable(True)

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
        if operator_instance.continuous_parameters['is_feasible']:
            return True
        else:
            print "Parameter values infeasible"
            return False

    def apply_operator_skeleton(self, state, operator_skeleton):
        return True

    def apply_operator_instance(self, state, operator_instance, check_reachability=True):
        is_op_instance_feasible = self.check_parameter_feasibility_precondition(operator_instance)
        if is_op_instance_feasible:
            operator_instance.execute()
        return is_op_instance_feasible

    @staticmethod
    def is_region_contain_object(region, obj):
        return region.contains(obj.ComputeAABB())

    def is_goal_reached(self):
        return self.reward_function.is_goal_reached()

    def check_holding_object_precondition(self):
        if len(self.robot.GetGrabbed()) == 0:
            return False
        else:
            return True

    def get_applicable_ops(self, parent_op=None):
        operator_name = 'two_arm_pick_two_arm_place'
        applicable_ops = [Operator(operator_name, {'object': o, 'region': r})
                          for o in self.object_names for r in self.region_names]

        return applicable_ops


class PaPMoverEnv(Mover):
    def __init__(self, problem_idx):
        Mover.__init__(self, problem_idx)

    def get_applicable_ops(self, parent_op=None):
        actions = []
        for o in self.entity_names:
            if 'region' in o:
                continue
            for r in self.entity_names:
                if 'region' not in r or 'entire' in r:
                    continue

                if o not in self.goal and r in self.goal:
                    # you cannot place non-goal object in the goal region
                    continue

                action = Operator('two_arm_pick_two_arm_place',
                                  {'two_arm_place_object': o, 'two_arm_place_region': r})
                # following two lines are for legacy reasons, will fix later
                action.discrete_parameters['object'] = action.discrete_parameters['two_arm_place_object']
                action.discrete_parameters['region'] = action.discrete_parameters['two_arm_place_region']

                actions.append(action)

        return actions

    def set_goal(self, goal):
        self.goal = goal

    def reset_to_init_state(self, node):
        saver = node.state_saver
        saver.Restore()  # this call re-enables objects that are disabled

        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
