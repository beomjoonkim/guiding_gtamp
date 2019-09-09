from gtamp_problem_environments.mover_env import Mover
from gtamp_utils.utils import set_robot_config, set_obj_xytheta
from trajectory_representation.operator import Operator
from manipulation.regions import AARegion
from gtamp_utils import utils


class OneArmMover(Mover):
    def __init__(self, problem_idx):
        Mover.__init__(self, problem_idx)
        self.operator_names = ['one_arm_pick', 'one_arm_place']
        set_robot_config([4.19855789, 2.3236321, 5.2933337], self.robot)
        set_obj_xytheta([3.35744004, 2.19644156, 3.52741118], self.objects[1])
        self.boxes = self.objects
        self.objects = self.problem_config['shelf_objects']
        self.objects = [k for v in self.objects.values() for k in v]
        self.objects[0], self.objects[1] = self.objects[1], self.objects[0]

        self.target_box = self.env.GetKinBody('rectangular_packing_box1')
        utils.randomly_place_region(self.target_box, self.regions['home_region'])
        self.regions['rectangular_packing_box1_region'] = self.compute_box_region(self.target_box)
        self.shelf_regions = self.problem_config['shelf_regions']
        self.target_box_region = self.regions['rectangular_packing_box1_region']
        self.regions.update(self.shelf_regions)
        self.entity_names = [obj.GetName() for obj in self.objects] + ['rectangular_packing_box1_region',
                                                                       'center_shelf_region']
        self.name = 'one_arm_mover'
        self.init_saver = utils.CustomStateSaver(self.env)

        self.object_names = self.entity_names

        # fix incorrectly named regions
        self.regions = {
            region.name: region
            for region in self.regions.values()
        }

    def compute_box_region(self, box):
        box_region = AARegion.create_on_body(box)
        box_region.color = (1., 1., 0., 0.25)
        box_region.draw(self.env)
        box_region.name += '_region'
        return box_region

    def reset_to_init_state(self, node):
        saver = node.state_saver
        saver.Restore()
        # todo finish implementing this function

    def get_region_containing(self, obj):
        if type(obj) == str or type(obj) == unicode:
            obj = self.env.GetKinBody(obj)

        for shelf_region in sorted(self.shelf_regions.values(), key=lambda r: 'top' not in r.name):
            if shelf_region.contains(obj.ComputeAABB()):
                return shelf_region

        if self.target_box_region.contains(obj.ComputeAABB()):
            return self.target_box_region

        assert False, "An object must belong to one of shelf or object regions"

    def get_applicable_ops(self, parent_op=None):
        applicable_ops = []
        for op_name in self.operator_names:
            if op_name.find('place') != -1:
                if self.check_holding_object_precondition():
                    object_held = self.robot.GetGrabbed()[0]
                    if parent_op.type.find('pick') != -1:
                        grasp_params = parent_op.continuous_parameters['grasp_params']
                    else:
                        grasp_params = None
                    if self.applicable_op_constraint is None:
                        for region in self.regions.values():
                            if region.name.find('box') == -1:
                                continue
                            assert parent_op is not None

                            op = Operator(operator_type=op_name,
                                          discrete_parameters={'region': region,
                                                               'object': object_held},
                                          continuous_parameters={
                                              'grasp_params': grasp_params})
                            applicable_ops.append(op)
                    else:
                        op = Operator(operator_type=op_name,
                                      discrete_parameters={'region': self.applicable_op_constraint['region'],
                                                           'object': object_held},
                                      continuous_parameters={
                                          'grasp_params': grasp_params})

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


class PaPOneArmMoverEnv(OneArmMover):
    def __init__(self, problem_idx):
        OneArmMover.__init__(self, problem_idx)

    def set_goal(self, goal):
        self.goal = goal

    def get_applicable_ops(self, parent_op=None):
        actions = []
        for o in self.entity_names:
            if 'region' in o:
                continue
            for r in self.entity_names:
                if 'region' not in r:
                    continue
                if o not in self.goal and r in self.goal:
                    # you cannot place non-goal object in the goal region
                    continue

                if 'entire' in r:  # and config.domain == 'two_arm_self':
                    continue

                action = Operator('one_arm_pick_one_arm_place',
                                  {'object': self.env.GetKinBody(o), 'region': self.regions[r]})

                actions.append(action)
        return actions
    """
    def get_applicable_ops(self, parent_op=None):
        # used by MCTS
        applicable_ops = []
        op_name = 'two_arm_pick_two_arm_place'

        for region_name in self.region_names:
            if region_name == 'entire_region':
                continue
            for obj_name in self.object_names:
                op = Operator(operator_type=op_name,
                              discrete_parameters={'object': obj_name,
                                                   'region': region_name,
                                                   'one_arm_place_object': obj_name,
                                                   'one_arm_place_region': region_name})
                applicable_ops.append(op)

        return applicable_ops
    """
