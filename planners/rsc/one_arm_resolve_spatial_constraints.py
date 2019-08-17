from trajectory_representation.operator import Operator
from generators.uniform import UniformGenerator
from planners.subplanners.minimum_constraint_planner import MinimumConstraintPlanner
from trajectory_representation.swept_volume import PickAndPlaceSweptVolume
from manipulation.bodies.bodies import set_color, get_color
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from gtamp_utils import utils

import numpy as np
import time
import pickle
import os


# h_add = Q for unsat goal * number of unsatisfied
# h_max = max over all Q
# h_hsp  = InWayHeuristic * Q

def attach_q_goal_as_low_level_motion(target_op_inst):
    target_op_inst.low_level_motion = {}
    target_op_inst.low_level_motion['pick'] = target_op_inst.continuous_parameters['pick']['q_goal']
    target_op_inst.low_level_motion['place'] = target_op_inst.continuous_parameters['place']['q_goal']
    return target_op_inst


class OneArmResolveSpatialConstraints:
    def __init__(self, problem_env, goal_object_name, goal_region_name):
        self.objects_moved_before = []
        self.plan = []
        self.objects_in_collision = []
        self.objects = objects = {
            o: problem_env.env.GetKinBody(o)
            for o in problem_env.entity_names
            if 'region' not in o
        }
        self.regions = regions = {
            r: problem_env.regions[r]
            for r in problem_env.entity_names
            if 'region' in r
        }
        self.goal_entities = [goal_object_name, goal_region_name]
        self.object_names = [o for o in problem_env.entity_names if 'region' not in o]
        self.region_names = [o for o in problem_env.entity_names if 'region' in o]

        self.problem_env = problem_env
        self.goal_object = self.problem_env.env.GetKinBody(goal_object_name)
        self.goal_region = self.problem_env.regions[goal_region_name]

        self.robot = self.problem_env.robot
        self.sampled_pick_configs_for_objects = {}
        self.env = problem_env.env
        self.recursion_level = 0
        self.number_of_picks = 0
        self.number_of_places = 0
        self.number_of_nodes = 0

        ikcachename = './ikcache.pkl'
        self.iksolutions = {}
        if os.path.isfile(ikcachename):
            self.iksolutions = pickle.load(open(ikcachename, 'r'))

        self.pap_params = {}
        self.pick_params = {}
        self.place_params = {}

        for obj in self.objects:
            self.pick_params[obj] = []
            for r in self.regions:
                self.pap_params[(obj, r)] = []
                self.place_params[(obj, r)] = []

        # self.initialize_pap_pick_place_params(moved_obj=None, parent_state=None)

        self.nocollision_pick_op = {}
        self.collision_pick_op = {}
        # self.determine_collision_and_collision_free_picks()

        self.nocollision_place_op = {}
        self.collision_place_op = {}
        # self.determine_collision_and_collision_free_places()

    def get_num_nodes(self):
        return self.number_of_nodes

    def sample_op_instance(self, curr_obj, region, swept_volume, n_iter):
        op = Operator(operator_type='one_arm_pick_one_arm_place',
                      discrete_parameters={'object': curr_obj, 'region': region})

        # use the following:
        # papg = OneArmPaPUniformGenerator(op_skel, self.problem_env,
        #                                  cached_picks=(self.iksolutions[current_region], self.iksolutions[r]))
        pick_cont_param, place_cont_param = self.get_pap_pick_place_params(curr_obj.GetName(), region.name, swept_volume)

        #generator = OneArmPaPUniformGenerator(op, self.problem_env, swept_volume)
        #pick_cont_param, place_cont_param, status = generator.sample_next_point(max_ik_attempts=n_iter)

        if pick_cont_param is not None:
            status = 'HasSolution'
        else:
            status = 'NoSolution'
        op.continuous_parameters = {'pick': pick_cont_param, 'place': place_cont_param}
        return op, status

    def find_pick_and_place(self, curr_obj, region, swept_volume):

        self.problem_env.enable_objects_in_region('entire_region')
        stime = time.time()
        print "Sampling paps for ", curr_obj
        op, status = self.sample_op_instance(curr_obj, region, swept_volume, 10)
        print "Find PaP time, full", time.time() - stime
        if status == 'HasSolution':
            return op, status
        else:
            self.problem_env.disable_objects_in_region('entire_region')
            curr_obj.Enable(True)
            stime = time.time()
            op, status = self.sample_op_instance(curr_obj, region, swept_volume, 5)
            print "Find PaP, disabled", time.time() - stime
            if status == 'HasSolution':
                return op, status
            else:
                return None, status

    def search(self, object_to_move, parent_swept_volumes, obstacles_to_remove, objects_moved_before, plan,
               stime=None, timelimit=None):
        print objects_moved_before
        print time.time() - stime, timelimit
        if time.time() - stime > timelimit:
            return False, 'NoSolution'

        utils.set_color(plan[-1].discrete_parameters['object'], [0, 0, 1])
        for o in self.problem_env.objects:
            if o in objects_moved_before:
                utils.set_color(o, [0, 0, 0])
            elif o.GetName() in objects_moved_before:
                utils.set_color(o, [0, 0, 0])
            else:
                utils.set_color(o, [0, 1, 0])
        set_color(object_to_move, [1, 0, 0])
        utils.set_color(plan[-1].discrete_parameters['object'], [0, 0, 1])

        # Initialize data necessary for this recursion level
        swept_volumes = PickAndPlaceSweptVolume(self.problem_env, parent_swept_volumes)
        objects_moved_before = [o for o in objects_moved_before]
        plan = [p for p in plan]

        self.number_of_nodes += 1
        if isinstance(object_to_move, unicode):
            object_to_move = self.problem_env.env.GetKinBody(object_to_move)

        # Select the region to move the object to
        if object_to_move == self.goal_object:
            target_region = self.goal_region
        else:
            obj_curr_region = self.problem_env.get_region_containing(object_to_move)
            not_in_box = obj_curr_region.name.find('box') == -1
            if not_in_box:
                # randomly choose one of the shelf regions
                target_region = self.problem_env.shelf_regions.values()[0]
            else:
                target_region = obj_curr_region

        # Get PaP
        self.problem_env.set_exception_objs_when_disabling_objects_in_region(objects_moved_before)

        pap, status = self.find_pick_and_place(object_to_move, target_region, swept_volumes)
        if status != 'HasSolution':
            print "Failed to sample pap, giving up on branch"
            return False, "NoSolution"

        pap = attach_q_goal_as_low_level_motion(pap)
        swept_volumes.add_pap_swept_volume(pap)
        self.problem_env.enable_objects_in_region('entire_region')
        objects_in_collision_for_pap = swept_volumes.get_objects_in_collision_with_last_pap()

        # O_{PAST}
        prev = obstacles_to_remove
        obstacles_to_remove = objects_in_collision_for_pap + [o for o in obstacles_to_remove
                                                              if o not in objects_in_collision_for_pap]

        # O_{FUC} update
        objects_moved_before.append(object_to_move)

        plan.insert(0, pap)

        if len(obstacles_to_remove) == 0:
            return plan, 'HasSolution'

        # enumerate through all object orderings
        print "Obstacles to remove", obstacles_to_remove
        self.problem_env.set_exception_objs_when_disabling_objects_in_region(objects_moved_before)

        for new_obj_to_move in obstacles_to_remove:
            tmp_obstacles_to_remove = set(obstacles_to_remove).difference(set([new_obj_to_move]))
            tmp_obstacles_to_remove = list(tmp_obstacles_to_remove)
            print "tmp obstacles to remove:", tmp_obstacles_to_remove
            print "Recursing on", new_obj_to_move
            branch_plan, status = self.search(new_obj_to_move,
                                              swept_volumes,
                                              tmp_obstacles_to_remove,
                                              objects_moved_before,
                                              plan, stime=stime, timelimit=timelimit)
            is_branch_success = status == 'HasSolution'
            if is_branch_success:
                return branch_plan, status
            else:
                print "Failed on ", new_obj_to_move

        # It should never come down here, as long as the number of nodes have not exceeded the limit
        # but to which level do I back track? To the root node. If this is a root node and
        # the number of nodes have not reached the maximum, keep searching.
        return False, 'NoSolution'

    def reset(self):
        self.problem_env.objects_to_check_collision = None

    def get_pap_pick_place_params(self, obj, region, swept_volume):
        stime = time.time()
        r = region
        print(obj, r)
        current_region = self.problem_env.get_region_containing(obj).name

        # check existing solutions
        pick_op = Operator(operator_type='one_arm_pick', discrete_parameters={'object': obj})

        place_op = Operator(operator_type='one_arm_place', discrete_parameters={'object': obj,
                                                                                'region':
                                                                                 self.problem_env.regions[r]})
        obj_kinbody = self.problem_env.env.GetKinBody(obj)
        if len(self.pap_params[(obj, r)]) > 0:
            for pick_params, place_params in self.pap_params[(obj, r)]:
                pick_op.continuous_parameters = pick_params
                place_op.continuous_parameters = place_params
                if not self.check_collision_in_pap(pick_op, place_op, obj_kinbody, swept_volume):
                    return pick_params, place_params

        op_skel = Operator(operator_type='one_arm_pick_one_arm_place',
                           discrete_parameters={'object': self.problem_env.env.GetKinBody(obj),
                                                'region': self.problem_env.regions[r]})

        papg = OneArmPaPUniformGenerator(op_skel, self.problem_env,
                                         cached_picks=(self.iksolutions[current_region], self.iksolutions[r]))

        num_tries = 200
        pick_params, place_params, status = papg.sample_next_point(num_tries)
        if 'HasSolution' in status:
            self.pap_params[(obj, r)].append((pick_params, place_params))
            self.pick_params[obj].append(pick_params)

            print('success')
            pick_op.continuous_parameters = pick_params
            place_op.continuous_parameters = place_params
            collision = self.check_collision_in_pap(pick_op, place_op, obj_kinbody, swept_volume)

            if not collision:
                print('found nocollision', obj, r)
                return pick_params, place_params

        return None, None

    def check_collision_in_pap(self, pick_op, place_op, obj_object, swept_volume):
        old_tf = obj_object.GetTransform()
        collision = False
        pick_op.execute()
        if self.problem_env.env.CheckCollision(self.problem_env.robot):
            utils.release_obj()
            obj_object.SetTransform(old_tf)
            return True
        place_op.execute()
        if self.problem_env.env.CheckCollision(self.problem_env.robot):
            obj_object.SetTransform(old_tf)
            return True
        if self.problem_env.env.CheckCollision(obj_object):
            obj_object.SetTransform(old_tf)
            return True

        is_object_pose_infeasible = not swept_volume.is_swept_volume_cleared(obj_object)
        if is_object_pose_infeasible:
            obj_object.SetTransform(old_tf)
            return True
        obj_object.SetTransform(old_tf)
        return collision

