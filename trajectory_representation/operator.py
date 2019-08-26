# This class describes an operator, in terms of:
#   type, discrete parameters (represented with entity class instance), continuous parameteres,
#   and the associated low-level motions

from gtamp_utils.utils import two_arm_pick_object, two_arm_place_object, one_arm_pick_object, one_arm_place_object
from gtamp_utils import utils
import openravepy


class Operator:
    def __init__(self, operator_type, discrete_parameters, continuous_parameters=None):
        self.type = operator_type
        assert type(discrete_parameters) is dict, "Discrete parameters of an operator must be a dictionary"
        self.discrete_parameters = discrete_parameters
        if continuous_parameters is None:
            self.continuous_parameters = {'is_feasible': False}
            self.is_skeleton = True
        else:
            self.continuous_parameters = continuous_parameters
            self.is_skeleton = False
        self.low_level_motion = None

    def update_low_level_motion(self, low_level_motion):
        self.low_level_motion = low_level_motion

    def set_continuous_parameters(self, continuous_parameters):
        self.continuous_parameters = continuous_parameters

    def execute(self):
        env = openravepy.RaveGetEnvironments()[0]
        if self.type == 'two_arm_pick':
            obj_to_pick = utils.convert_to_kin_body(self.discrete_parameters['object'])
            if 'q_goal' in self.continuous_parameters and type(self.continuous_parameters['q_goal']) == list and\
                    len(self.continuous_parameters['q_goal']) > 1:
                try:
                    two_arm_pick_object(obj_to_pick, {'q_goal': self.continuous_parameters['q_goal'][0]})
                except:
                    import pdb;pdb.set_trace()
            else:
                two_arm_pick_object(obj_to_pick, self.continuous_parameters)
        elif self.type == 'two_arm_place':
            two_arm_place_object(self.continuous_parameters)
        elif self.type == 'two_arm_pick_two_arm_place':
            obj_to_pick = utils.convert_to_kin_body(self.discrete_parameters['object'])
            two_arm_pick_object(obj_to_pick, self.continuous_parameters['pick'])
            two_arm_place_object(self.continuous_parameters['place'])
        elif self.type == 'one_arm_pick':
            obj_to_pick = utils.convert_to_kin_body(self.discrete_parameters['object'])
            one_arm_pick_object(obj_to_pick, self.continuous_parameters)
        elif self.type == 'one_arm_place':
            one_arm_place_object(self.continuous_parameters)
        elif self.type == 'one_arm_pick_one_arm_place':
            obj_to_pick = utils.convert_to_kin_body(self.discrete_parameters['object'])
            one_arm_pick_object(obj_to_pick, self.continuous_parameters['pick'])
            one_arm_place_object(self.continuous_parameters['place'])
        else:
            raise NotImplementedError

    def execute_pick(self):
        env = openravepy.RaveGetEnvironments()[0]
        if isinstance(self.discrete_parameters['object'], openravepy.KinBody):
            obj_to_pick = self.discrete_parameters['object']
        else:
            obj_to_pick = env.GetKinBody(self.discrete_parameters['object'])

        if self.type == 'two_arm_pick_two_arm_place':
            two_arm_pick_object(obj_to_pick, self.continuous_parameters['pick'])
        else:
            one_arm_pick_object(obj_to_pick, self.continuous_parameters['pick'])

    def is_discrete_parameters_eq_to(self, param):
        if self.type == 'two_arm_pick':
            if type(param) != str:
                param = param.GetName()

            my_obj = self.discrete_parameters['object']
            if type(my_obj) != str:
                my_obj = my_obj.GetName()

            return param == my_obj
        else:
            raise NotImplemented

    def merge_operators(self, operator):
        curr_op_type = self.type
        other_op_type = operator.type
        self.type = curr_op_type + "_" + other_op_type
        for k, v in zip(operator.discrete_parameters.keys(), operator.discrete_parameters.values()):
            self.discrete_parameters[other_op_type+'_'+k] = v

        if 'pick' in curr_op_type:
            self.continuous_parameters['pick'] = {k: v for k, v in self.continuous_parameters.iteritems()}
            self.continuous_parameters['place'] = operator.continuous_parameters

        return self

    def make_pklable(self):
        if 'object' in self.discrete_parameters.keys():
            if isinstance(self.discrete_parameters['object'], openravepy.KinBody):
                self.discrete_parameters['object'] = self.discrete_parameters['object'].GetName()

        if 'region' in self.discrete_parameters.keys():
            if not (isinstance(self.discrete_parameters['region'], str)
                    or isinstance(self.discrete_parameters['region'], unicode)):
                self.discrete_parameters['region'] = self.discrete_parameters['region'].name



