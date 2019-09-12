from trajectory_representation.state import State
from trajectory_representation.operator import Operator
from gtamp_utils.utils import CustomStateSaver, get_body_xytheta, set_robot_config, set_obj_xytheta

from predicates.is_reachable import IsReachable
from predicates.is_holding_goal_entity import IsHoldingGoalEntity
from predicates.place_in_way import PlaceInWay
from predicates.pick_in_way import PickInWay
from predicates.in_region import InRegion

from planners.subplanners.motion_planner import BaseMotionPlanner

import copy
from gtamp_utils.utils import visualize_path, two_arm_pick_object
from manipulation.bodies.bodies import set_color


class MinimiumConstraintPaPState(State):
    def __init__(self, problem_env, goal_entities, parent_state=None, parent_action=None, paps_used_in_data=None,
                 use_shortest_path=False):
        self.state_saver = CustomStateSaver(problem_env.env)
        self.problem_env = problem_env
        self.parent_state = parent_state
        self.goal_entities = goal_entities

        # raw variables
        self.robot_pose = get_body_xytheta(problem_env.robot)
        self.object_poses = {
            obj.GetName(): get_body_xytheta(obj)
            for obj in problem_env.objects
        }

        # cached info
        self.use_prm = problem_env.name.find('two_arm') != -1
        if self.use_prm:
            self.collides, self.current_collides = self.update_collisions_at_prm_vertices(parent_state)
        else:
            self.collides = None
            self.current_collides = None

        # adopt from parent predicate evaluations
        self.parent_ternary_predicates = {}
        self.parent_binary_predicates = {}
        if parent_state is not None and paps_used_in_data is None:
            assert parent_action is not None
            moved_obj_type = type(parent_action.discrete_parameters['two_arm_place_object'])
            if moved_obj_type == str or moved_obj_type == unicode:
                moved_obj = parent_action.discrete_parameters['two_arm_place_object']
            else:
                moved_obj = parent_action.discrete_parameters['two_arm_place_object'].GetName()

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
        self.use_shortest_path = False

        self.cached_place_paths = {}
        if paps_used_in_data is None:
            self.pick_used = {}
            self.place_used = {}
        else:
            self.pick_used = paps_used_in_data[0]
            self.place_used = paps_used_in_data[1]
        self.mc_pick_path = {}
        self.mc_place_path = {}
        self.reachable_entities = []

        # predicates
        self.pick_in_way = PickInWay(self.problem_env, collides=self.current_collides, pick_poses=self.pick_used,
                                     use_shortest_path=self.use_shortest_path)
        self.place_in_way = PlaceInWay(self.problem_env, collides=self.current_collides, pick_poses=self.pick_used,
                                       use_shortest_path=self.use_shortest_path)
        self.in_region = InRegion(self.problem_env)
        self.is_holding_goal_entity = IsHoldingGoalEntity(self.problem_env, goal_entities)

        self.ternary_edges = self.get_ternary_edges()
        self.binary_edges = self.get_binary_edges()
        self.nodes = self.get_nodes()

    def is_entity_reachable(self, entity):
        return self.nodes[entity][-2]

    def get_entities_in_place_way(self, entity, region):
        inway = []
        for obj_name in self.problem_env.object_names:
            if self.ternary_edges[(entity, obj_name, region)][0]:
                inway.append(obj_name)
        return inway

    def get_entities_in_pick_way(self, entity):
        inway = []
        for obj_name in self.problem_env.object_names:
            if self.binary_edges[(obj_name, entity)][1]:
                inway.append(obj_name)
        return inway

    def visualize_place_inways(self):
        self.problem_env.env.SetViewer('qtcoin')
        for key, val in self.place_in_way.mc_path_to_entity.items():
            hold_obj_name = key[0]
            region_name = key[1]

            objs_in_way = self.place_in_way.mc_to_entity[key]
            if len(objs_in_way) > 0:
                saver = CustomStateSaver(self.problem_env.env)
                self.pick_used[hold_obj_name].execute()
                for tmp in objs_in_way:
                    set_color(self.problem_env.env.GetKinBody(tmp), [0, 0, 0])
                visualize_path(val)
                import pdb;pdb.set_trace()

                for tmp in objs_in_way:
                    set_color(self.problem_env.env.GetKinBody(tmp), [0, 1, 0])
                saver.Restore()

    def set_cached_pick_paths(self, parent_state, moved_obj):
        motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
        for obj, op_instance in self.pick_used.items():
            motion_plan_goals = op_instance.continuous_parameters['q_goal']
            self.cached_pick_paths[obj] = None
            if not self.use_shortest_path:
                continue
            if parent_state is not None and obj in parent_state.cached_pick_paths and obj != moved_obj:
                self.cached_pick_paths[obj] = parent_state.cached_pick_paths[obj]
            else:
                self.cached_pick_paths[obj] = motion_planner.get_motion_plan(motion_plan_goals, cached_collisions={})[0]
                if len(motion_plan_goals) == 0:
                    # This pick path is used for checking the pickinway predicate and to pickup the object in place in way predicate.
                    #
                    assert False
                    self.cached_pick_paths[obj] = None
                else:
                    try:
                        # how can this be, since we disable all the collisions?
                        assert self.cached_pick_paths[obj] is not None
                    except:
                        import pdb;pdb.set_trace()
                    op_instance.continuous_parameters['potential_q_goals'] = motion_plan_goals
                    op_instance.continuous_parameters['q_goal'] = self.cached_pick_paths[obj][-1]

    def set_cached_place_paths(self, parent_state, moved_obj):
        motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
        for region_name, region in self.problem_env.regions.items():
            if region.name == 'entire_region':
                continue
            for obj, pick_path in self.cached_pick_paths.items():
                self.cached_place_paths[(obj, region_name)] = None
                if not self.use_shortest_path:
                    continue
                if pick_path is None:
                    continue
                if parent_state is not None and obj in parent_state.cached_pick_paths and obj != moved_obj:
                    self.cached_place_paths[(obj, region_name)] = parent_state.cached_place_paths[(obj, region_name)]
                else:
                    saver = CustomStateSaver(self.problem_env.env)
                    pick_used = self.pick_used[obj]
                    pick_used.execute()
                    if region.contains(self.problem_env.env.GetKinBody(obj).ComputeAABB()):
                        self.cached_place_paths[(obj, region_name)] = [get_body_xytheta(self.problem_env.robot).squeeze()]
                    else:
                        self.cached_place_paths[(obj, region_name)] = motion_planner.get_motion_plan(region, cached_collisions={})[0]
                        if self.cached_place_paths[(obj, region_name)] is None:
                            import pdb;pdb.set_trace()

                    saver.Restore()
                try:
                    assert self.cached_place_paths[(obj, region_name)] is not None
                except:
                    import pdb;pdb.set_trace()

    def update_cached_data_after_binary(self):
        self.mc_pick_path = self.pick_in_way.mc_path_to_entity
        if not self.use_shortest_path:
            self.reachable_entities = self.pick_in_way.reachable_entities
        self.pick_used = self.pick_in_way.pick_used

    def update_cached_data_after_ternary(self):
        self.place_used = self.place_in_way.place_used
        self.mc_place_path = self.place_in_way.mc_path_to_entity
        self.pick_used = self.place_in_way.pick_used

    def get_binary_edges(self):
        self.pick_in_way.set_pick_used(self.pick_used)
        edges = {}
        for a in self.problem_env.entity_names:
            for b in self.problem_env.entity_names:
                key = (a, b)
                if key not in edges.keys():
                    pick_edge_features = self.get_binary_edge_features(a, b)  # a = src, b = dest
                    edges[key] = pick_edge_features
        self.update_cached_data_after_binary()
        return edges

    def get_ternary_edges(self):
        self.place_in_way.set_pick_used(self.pick_used)
        self.place_in_way.set_place_used(self.place_used)
        self.place_in_way.set_reachable_entities(self.reachable_entities)

        edges = {}
        for a in self.problem_env.entity_names:
            for b in self.problem_env.entity_names:
                for r in self.problem_env.regions:
                    key = (a, b, r)
                    if r.find('region') == -1 or r.find('entire') != -1:
                        continue
                    if key not in edges.keys():
                        place_edge_features = self.get_ternary_edge_features(a, b, r)
                        edges[key] = place_edge_features
        self.update_cached_data_after_ternary()
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
            if self.use_shortest_path:
                motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
                pick_for_obj = self.pick_used[obj.GetName()]
                plan, status = motion_planner.get_motion_plan(pick_for_obj.continuous_parameters['potential_q_goals'], cached_collisions=self.collides)
                pick_for_obj.low_level_motion = plan
                if status == 'HasSolution':
                    pick_for_obj.continuous_parameters['q_goal'] = plan[-1]
                    self.reachable_entities.append(entity)
                    is_entity_reachable = True
                else:
                    is_entity_reachable = False
            else:
                is_entity_reachable = obj.GetName() in self.reachable_entities
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
                # perhaps it is here that we set the mc path?
                cached_path = self.cached_place_paths[key]
            else:
                cached_path = None
            return [self.place_in_way(a, b, r, cached_path=cached_path)]

    def get_binary_edge_features(self, a, b):
        if (a, b) in self.parent_binary_predicates:
            return self.parent_binary_predicates[(a, b)]
        else:
            # todo rename cached_pick_path and cached_place_path as shortest paths
            if self.use_shortest_path:
                # todo this needs to be re-computed too when we are using shortest paths, because
                #   this is true if for all b in (a,b,r), b is not in the way of shortest path to r while holidng a
                #   Since the shortest path plans a path without collision-checking, this is not an accurate computation
                if a in self.problem_env.object_names and b in self.problem_env.region_names and b != 'entire_region':
                    if a not in self.reachable_entities:
                        is_place_in_b_reachable_while_holding_a = False
                    else:
                        saver = CustomStateSaver(self.problem_env.env)
                        self.pick_used[a].execute()  # it should not be in collision
                        motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
                        # note that it doesn't check the collision with the object held
                        plan, status = motion_planner.get_motion_plan(self.problem_env.regions[b], cached_collisions=self.collides)
                        saver.Restore()
                        is_place_in_b_reachable_while_holding_a = status == 'HasSolution'
                else:
                    is_place_in_b_reachable_while_holding_a = False

            else:
                is_place_in_b_reachable_while_holding_a = (a, b) in self.place_in_way.reachable_obj_region_pairs

            if self.use_shortest_path:
                if b.find('region') != -1:
                    cached_path = None
                else:
                    cached_path = self.cached_pick_paths[b]
                is_a_in_pick_path_of_b = self.pick_in_way(a, b, cached_path=cached_path)
            else:
                is_a_in_pick_path_of_b = self.pick_in_way(a, b)

            return [
                self.in_region(a, b),
                is_a_in_pick_path_of_b,
                is_place_in_b_reachable_while_holding_a
            ]

    def make_pklable(self):
        self.problem_env = None
        # self.is_reachable.problem_env = None
        self.in_region.problem_env = None
        self.pick_in_way.problem_env = None
        self.pick_in_way.robot = None
        self.is_holding_goal_entity.problem_env = None
        self.place_in_way.problem_env = None
        self.place_in_way.robot = None

        for operator in self.pick_used.values():
            operator.make_pklable()

        for operator in self.place_used.values():
            operator.make_pklable()

        if self.parent_state is not None:
            self.parent_state.make_pklable()

    def make_plannable(self, problem_env):
        self.problem_env = problem_env
        # self.is_reachable.problem_env = problem_env
        self.in_region.problem_env = problem_env
        self.pick_in_way.problem_env = problem_env
        self.pick_in_way.robot = problem_env.robot
        self.place_in_way.problem_env = problem_env
        self.place_in_way.robot = problem_env.robot
        self.is_holding_goal_entity.problem_env = problem_env
        if self.parent_state is not None:
            self.parent_state.make_plannable(problem_env)

    def print_chosen_entity(self, op):
        is_goal_idx = -3
        is_reachable_idx = -2
        discrete_param = op.discrete_parameters['object']
        discrete_param_node = self.nodes[discrete_param]

        # Node info
        is_reachable = discrete_param_node[is_reachable_idx]
        is_goal_entity = discrete_param_node[is_goal_idx]
        is_in_goal_region = self.binary_edges[(discrete_param, 'home_region')][1]
        literal = "reachable %r goal %r in_goal_region %r" % (is_reachable, is_goal_entity, is_in_goal_region)
        print "Node literal", literal

        # Edge info
        goal_obj = 'rectangular_packing_box2'
        goal_region = 'home_region'
        pick_in_way_idx = 1
        place_in_way_idx = 0
        is_selecte_in_way_of_pick_to_goal_obj = self.binary_edges[(discrete_param, goal_obj)][pick_in_way_idx]
        is_selected_in_way_of_placing_goal_obj_to_goal_region = \
        self.ternary_edges[(goal_obj, discrete_param, goal_region)][place_in_way_idx]

        is_pick_in_way_to_goal_occluding_entity = False
        is_place_in_way_to_goal_occluding_entity = False

        goal_pick_occluding_entities = []
        goal_place_occluding_entities = []
        for obj in self.problem_env.objects:
            obj_name = obj.GetName()
            is_goal_obj_occluding_entity = self.binary_edges[(obj_name, goal_obj)][pick_in_way_idx]
            is_goal_region_occluding_entity = self.ternary_edges[(goal_obj, obj_name, goal_region)][place_in_way_idx]

            if is_goal_obj_occluding_entity:
                goal_pick_occluding_entities.append(obj_name)
            if is_goal_region_occluding_entity:
                goal_place_occluding_entities.append(obj_name)

            if is_goal_obj_occluding_entity or is_goal_region_occluding_entity:
                # is the selected obj pick-in-way to the goal-occluding object?
                if not is_pick_in_way_to_goal_occluding_entity:
                    is_pick_in_way_to_goal_occluding_entity = self.binary_edges[(discrete_param, obj_name)][
                        pick_in_way_idx]
                if not is_place_in_way_to_goal_occluding_entity:
                    is_place_in_way_to_goal_occluding_entity = \
                        self.ternary_edges[(obj_name, discrete_param, 'loading_region')][place_in_way_idx]

        print "Is goal-occluding?"
        literal = "is_selecte_in_way_of_pick_to_goal_obj %r is_selected_in_way_of_placing_goal_obj_to_goal_region %r" \
                  % (is_selecte_in_way_of_pick_to_goal_obj, is_selected_in_way_of_placing_goal_obj_to_goal_region)
        print literal

        print "Is occluding a goal-occluding?"
        literal = "is_blocking_pick_path_of_goal_occluding_entity %r is_blocking_place_path_of_goal_occluding_entity %r" \
                  % (is_pick_in_way_to_goal_occluding_entity, is_place_in_way_to_goal_occluding_entity)
        print literal

        print "goal_pick_occluding_objects", goal_pick_occluding_entities
        print "goal_place_occluding_objects", goal_place_occluding_entities
