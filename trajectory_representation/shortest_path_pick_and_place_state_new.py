from trajectory_representation.operator import Operator
from generators.uniform import UniformPaPGenerator
from gtamp_utils.utils import CustomStateSaver, get_body_xytheta, set_robot_config, set_obj_xytheta

from predicates.is_holding_goal_entity import IsHoldingGoalEntity
from predicates.place_in_way import PlaceInWay
from predicates.pick_in_way import PickInWay
from predicates.in_region import InRegion

from planners.subplanners.motion_planner import BaseMotionPlanner
from gtamp_utils import utils

import copy
from pick_and_place_state import PaPState


class ShortestPathPaPState(PaPState):
    def __init__(self, problem_env, goal_entities, parent_state=None, parent_action=None, planner='greedy', paps_used_in_data=None):
        PaPState.__init__(self, problem_env, goal_entities, parent_state=None, parent_action=None,
                          paps_used_in_data=paps_used_in_data)

        self.parent_state = parent_state
        self.parent_ternary_predicates = {}
        self.parent_binary_predicates = {}
        self.object_names = [str(obj.GetName()) for obj in problem_env.objects]
        if parent_state is not None:
            moved_obj_type = type(parent_action.discrete_parameters['two_arm_place_object'])
            if moved_obj_type == str or moved_obj_type == unicode:
                moved_obj = parent_action.discrete_parameters['two_arm_place_object']
            else:
                moved_obj = parent_action.discrete_parameters['two_arm_place_object'].GetName()
            self.initialize_parent_predicates(moved_obj, parent_state, parent_action)
        else:
            moved_obj = None
        problem_env.enable_objects_in_region('entire_region')
        self.reachable_entities = []
        #self.reachable_regions_while_holding = []
        if paps_used_in_data is not None:
            self.pick_used = copy.deepcopy(paps_used_in_data[0])
            for obj in problem_env.objects:
                if obj.GetName() not in self.pick_used:
                    self.pick_used[obj.GetName()] = self.get_pick_poses(obj, moved_obj, parent_state)

            self.place_used = copy.deepcopy(paps_used_in_data[1])
        else:
            self.pick_used = {
                object.GetName(): self.get_pick_poses(object, moved_obj, parent_state) for object in problem_env.objects
            }
            self.place_used = {}
        if self.use_prm:
            if parent_state is None:
                self.collisions_at_all_obj_pose_pairs, self.collisions_at_current_obj_pose_pairs = self.update_collisions_at_prm_vertices(None)
            else:
                self.collisions_at_all_obj_pose_pairs, self.collisions_at_current_obj_pose_pairs = self.update_collisions_at_prm_vertices(parent_state.collisions_at_all_obj_pose_pairs)

            self.holding_collides = None
            self.current_holding_collides = None
            """
            test_col_1 = set()
            for tmp in self.collisions_at_all_obj_pose_pairs.values():
                test_col_1 = test_col_1.union(tmp)

            test_col_2 = set()
            for tmp in self.collisions_at_current_obj_pose_pairs.values():
                test_col_2 = test_col_2.union(tmp)

            if len(test_col_1) != len(test_col_2.intersection(test_col_1)):
                import pdb;pdb.set_trace()
            """

            # hold an object and check collisions
            """
            if planner == 'mcts':
                self.holding_collides = None
                self.current_holding_collides = None
                saver = utils.CustomStateSaver(self.problem_env.env)
                self.pick_used.values()[0].execute()
                if parent_state is None:
                    self.holding_collides, self.current_holding_collides = self.update_collisions_at_prm_vertices(None)
                else:
                    self.holding_collides, self.current_holding_collides \
                        = self.update_collisions_at_prm_vertices(parent_state.holding_collides)
                saver.Restore()
            """
        else:
            self.holding_collides = None
            self.holding_current_collides = None

        self.cached_pick_paths = {}
        self.cached_place_paths = {}
        self.set_cached_pick_paths(parent_state, moved_obj)
        self.set_cached_place_paths(parent_state, moved_obj)

        # predicates
        self.pick_in_way = PickInWay(self.problem_env,
                                     collides=self.collisions_at_current_obj_pose_pairs,
                                     pick_poses=self.pick_used,
                                     use_shortest_path=True)
        self.place_in_way = PlaceInWay(self.problem_env,
                                       collides=self.collisions_at_current_obj_pose_pairs,
                                       pick_poses=self.pick_used,
                                       use_shortest_path=True)
        self.in_region = InRegion(self.problem_env)
        self.is_holding_goal_entity = IsHoldingGoalEntity(self.problem_env, goal_entities)

        self.nodes = self.get_nodes()
        # note: the ternary and binary edges must be computed in this particular order
        self.ternary_edges = self.get_ternary_edges()
        self.binary_edges = self.get_binary_edges()

    def initialize_parent_predicates(self, moved_obj, parent_state, parent_action):
        assert parent_action is not None

        self.parent_ternary_predicates = {
            (a, b, r): v
            for (a, b, r), v in parent_state.ternary_edges.items()
            if a != moved_obj and b != moved_obj
        }
        self.parent_binary_predicates = {
            (a, b): v
            for (a, b), v in parent_state.binary_edges.items()
            if a != moved_obj and b != moved_obj
        }

    def get_pick_poses(self, object, moved_obj, parent_state):
        if parent_state is not None and moved_obj != object.GetName():
            return parent_state.pick_used[object.GetName()]

        operator_skeleton = Operator('two_arm_pick', {'object': object})
        generator = UniformPaPGenerator(None,
                                        operator_skeleton,
                                        self.problem_env,
                                        None,
                                        n_candidate_params_to_smpl=3,
                                        total_number_of_feasibility_checks=500,
                                        dont_check_motion_existence=True)
        # we should disable objects, because we are getting shortest path that ignors all collisions anyways
        self.problem_env.disable_objects_in_region('entire_region')

        op_cont_params, _ = generator.sample_candidate_params_with_increasing_iteration_limit()
        motion_plan_goals = [op['q_goal'] for op in op_cont_params if op['q_goal'] is not None]
        """
        for n_iter_to_try in n_iters:
            op_cont_params, _ = generator.sample_feasible_op_parameters(operator_skeleton,
                                                                        n_iter=n_iter_to_try,
                                                                        n_parameters_to_try_motion_planning=5)
            # I see. So here, return no op['q_goal'] when it is not feasible.
            motion_plan_goals = [op['q_goal'] for op in op_cont_params if op['q_goal'] is not None]
            if len(motion_plan_goals) > 2:
                break
        """
        self.problem_env.enable_objects_in_region('entire_region')

        assert len(motion_plan_goals) > 0 # if we can't find a pick pose then the object should be treated as unreachable
        operator_skeleton.continuous_parameters['q_goal'] = motion_plan_goals  # to make it consistent with Dpl
        if len(motion_plan_goals) == 0:
            import pdb;pdb.set_trace()
        return operator_skeleton

    def set_cached_pick_paths(self, parent_state, moved_obj):
        motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
        for obj, op_instance in self.pick_used.items():
            motion_plan_goals = op_instance.continuous_parameters['q_goal']
            try:
                assert len(motion_plan_goals) > 0
            except:
                import pdb;pdb.set_trace()
            self.cached_pick_paths[obj] = None

            path, status = motion_planner.get_motion_plan(motion_plan_goals,
                                                          cached_collisions=self.collisions_at_all_obj_pose_pairs)
            if status == 'HasSolution':
                self.reachable_entities.append(obj)
            else:
                path, _ = motion_planner.get_motion_plan(motion_plan_goals, cached_collisions={})
            try:
                assert path is not None
            except:
                import pdb;pdb.set_trace()
            self.cached_pick_paths[obj] = path
            op_instance.low_level_motion = path

    def set_cached_place_paths(self, parent_state, moved_obj):
        motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
        for region_name, region in self.problem_env.regions.items():
            if region.name == 'entire_region':
                continue
            for obj, pick_path in self.cached_pick_paths.items():
                self.cached_place_paths[(obj, region_name)] = None

                saver = CustomStateSaver(self.problem_env.env)
                pick_used = self.pick_used[obj]
                pick_used.execute()
                if region.contains(self.problem_env.env.GetKinBody(obj).ComputeAABB()):
                    path = [get_body_xytheta(self.problem_env.robot).squeeze()]
                    #self.reachable_regions_while_holding.append((obj, region_name))
                else:
                    if self.holding_collides is not None:
                        path, status = motion_planner.get_motion_plan(region, cached_collisions=self.holding_collides)
                    else:
                        # I think the trouble here is that we do not hold the object when checking collisions
                        # So, the best way to solve this problem is to not keep reachable_regions_while_holding
                        # and just use the cached path. But I am wondering how we got a colliding-path in
                        # the first place. It must be from place_in_way? No, we execute this function first,
                        # and then using the cached paths, compute the place_in_way.
                        # Also, there is something wrong with the collision checking too.
                        # I think this has to do with the fact that the collisions are computed using
                        # the robot only, not with the object in hand.
                        # So, here is what I propose:
                        #   Plan motions here, but do not add to reachable regions while holding.
                        # This way, it will plan motions as if it is not holding the object,
                        # but check collisions inside place_in_way
                        path, status = motion_planner.get_motion_plan(region,
                                                                      cached_collisions=self.collisions_at_all_obj_pose_pairs)
                    if status == 'HasSolution':
                        pass
                    else:
                        obj_not_moved = obj != moved_obj
                        parent_state_has_cached_path_for_obj = parent_state is not None \
                                                               and obj in parent_state.cached_place_paths
                        # todo: What is this logic here...?
                        #  it is about re-using the parent place path;
                        #  but this assumes that the object has not moved?
                        if parent_state_has_cached_path_for_obj and obj_not_moved:
                            path = parent_state.cached_place_paths[(obj, region_name)]
                        else:
                            path, _ = motion_planner.get_motion_plan(region, cached_collisions={})
                saver.Restore()
                # assert path is not None
                self.cached_place_paths[(obj, region_name)] = path

    def get_binary_edges(self):
        self.pick_in_way.set_pick_used(self.pick_used)
        edges = {}
        for a in self.problem_env.entity_names:
            for b in self.problem_env.entity_names:
                key = (a, b)
                if key not in edges.keys():
                    pick_edge_features = self.get_binary_edge_features(a, b)  # a = src, b = dest
                    edges[key] = pick_edge_features
        return edges

    def get_ternary_edges(self):
        self.place_in_way.set_pick_used(self.pick_used)
        self.place_in_way.set_place_used(self.place_used)
        self.place_in_way.set_reachable_entities(self.reachable_entities)

        edges = {}
        for a in self.problem_env.entity_names:
            for b in self.problem_env.entity_names:
                #for r in self.problem_env.regions:
                for r in self.problem_env.entity_names:
                    key = (a, b, r)

                    is_r_not_region = r.find('region') == -1
                    if r.find('entire') != -1:
                        edges[key] = [False]
                        continue

                    if is_r_not_region or a == b:
                        edges[key] = [False]
                        continue

                    if key not in edges.keys():
                        place_edge_features = self.get_ternary_edge_features(a, b, r)
                        edges[key] = place_edge_features
        return edges

    def get_nodes(self):
        nodes = {}
        for entity in self.problem_env.entity_names:
            nodes[entity] = self.get_node_features(entity, self.goal_entities)
        return nodes

    def get_node_features(self, entity, goal_entities):
        isobj = entity not in self.problem_env.regions
        obj = self.problem_env.env.GetKinBody(entity) if isobj else None
        pose = get_body_xytheta(obj)[0] if isobj else None

        if isobj:
            is_entity_reachable = entity in self.reachable_entities
        else:
            is_entity_reachable = True

        return [
            0,  # l
            0,  # w
            0,  # h
            pose[0] if isobj else 0,  # x
            pose[1] if isobj else 0,  # y
            pose[2] if isobj else 0,  # theta
            entity not in self.problem_env.regions,  # IsObj
            entity in self.problem_env.regions,  # IsRoom
            entity in self.goal_entities,  # IsGoal
            is_entity_reachable,
            self.is_holding_goal_entity(),
        ]

    def get_ternary_edge_features(self, a, b, r):
        if (a, b, r) in self.parent_ternary_predicates:
            return self.parent_ternary_predicates[(a, b, r)]
        else:
            key = (a, r)
            if key in self.cached_place_paths:
                cached_path = self.cached_place_paths[key]
            else:
                a_is_region = a.find('region') != -1
                try:
                    assert a_is_region
                except:
                    import pdb;pdb.set_trace()
                cached_path = None
            """
            if key in self.reachable_regions_while_holding:
                # if reachable then nothing is in the way
                is_b_in_way_of_reaching_r_while_holding_a = False
            else:
            """
            is_b_in_way_of_reaching_r_while_holding_a = self.place_in_way(a, b, r, cached_path=cached_path)
            return [is_b_in_way_of_reaching_r_while_holding_a]

    def get_binary_edge_features(self, a, b):
        is_place_in_b_reachable_while_holding_a = (a, b) in self.reachable_regions_while_holding
        """
        objs_occluding_moving_a_to_b = [occluding for occluding in self.object_names
                                        if self.ternary_edges[(a, occluding, b)][0]]
        if b in self.object_names or b == 'entire_region':
            is_place_in_b_reachable_while_holding_a = False
        else:
            is_place_in_b_reachable_while_holding_a = len(objs_occluding_moving_a_to_b) == 0
        """

        if 'region' in b:
            cached_path = None
        else:
            cached_path = self.cached_pick_paths[b]

        """
        if (a, b) in self.parent_binary_predicates:
            # we can only do below if the robot configuration didn't change much
            is_a_in_pick_path_of_b = self.parent_binary_predicates[(a, b)][1]
        else:
            is_a_in_pick_path_of_b = self.pick_in_way(a, b, cached_path=cached_path)
        """

        is_a_in_pick_path_of_b = self.pick_in_way(a, b, cached_path=cached_path)

        return [
            self.in_region(a, b),
            is_a_in_pick_path_of_b,
            is_place_in_b_reachable_while_holding_a
        ]
