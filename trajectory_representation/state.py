from gtamp_utils.utils import get_body_xytheta, CustomStateSaver, set_robot_config, visualize_path
from manipulation.bodies.bodies import set_color

from predicates.is_reachable import IsReachable
from predicates.is_holding_goal_entity import IsHoldingGoalEntity
from predicates.in_way import InWay
from predicates.in_region import InRegion

import copy
import pickle
import sys

prm_vertices = prm_edges = None


def is_object(entity):
    return entity.find('box') != -1


def is_region(entity):
    return entity.find('box') == -1


class AbstractState:
    def __init__(self, problem_env, goal_entities, parent_state=None, parent_action=None):
        self.state_saver = CustomStateSaver(problem_env.env)
        self.problem_env = problem_env
        self.parent = parent_state  # used to update the node features
        # raw variables
        self.robot_pose = get_body_xytheta(problem_env.robot)
        self.object_poses = {
            obj.GetName(): get_body_xytheta(obj)
            for obj in problem_env.objects
        }


class StateWithoutCspacePredicates(AbstractState):
    def __init__(self, problem_env, goal_entities, parent_state=None, parent_action=None):
        AbstractState.__init__(self, problem_env, goal_entities, parent_state=None, parent_action=None)
        # predicates
        self.is_reachable = None
        self.in_way = None
        self.in_region = InRegion(self.problem_env)
        self.is_holding_goal_entity = IsHoldingGoalEntity(self.problem_env, goal_entities)


class State(AbstractState):
    def __init__(self, problem_env, goal_entities, parent_state=None, parent_action=None):
        AbstractState.__init__(problem_env, goal_entities, parent_state, parent_action)

        self.use_prm = problem_env.name.find('two_arm') != -1
        if self.use_prm:
            self.collides, self.current_collides = self.update_collisions_at_prm_vertices(parent_state)
        else:
            self.collides = None
            self.current_collides = None

        # predicates
        self.is_reachable = IsReachable(self.problem_env, collides=self.current_collides)
        self.in_way = InWay(self.problem_env, collides=self.current_collides)
        self.in_region = InRegion(self.problem_env)
        self.is_holding_goal_entity = IsHoldingGoalEntity(self.problem_env, goal_entities)

        # GNN todo write this in a separate function
        self.goal_entities = goal_entities
        if parent_state is not None:
            self.nodes = {}
            self.edges = {}
            if parent_action.type.find('pick') != -1:
                self.update_node_features(parent_state)
                self.update_edge_features(parent_state)
            elif parent_action.type.find('place') != -1:
                assert len(self.problem_env.robot.GetGrabbed()) == 0
                grand_parent_state = parent_state.parent
                self.update_node_features(grand_parent_state, parent_action.discrete_parameters['object'])
                self.update_edge_features(grand_parent_state, parent_action.discrete_parameters['object'])
            else:
                raise NotImplementedError
        else:
            self.nodes = self.get_nodes()
            self.edges = self.get_edges()

        self.update_reachability_based_on_inway()

        # for debugging purpose; to be deleted later
        reachable_idx = 9
        self.reachable_entities = [n for n in self.nodes if self.nodes[n][reachable_idx]]

        for entity in self.nodes:
            if entity.find('region') != -1:
                continue
            if self.nodes[entity][reachable_idx]:
                set_color(self.problem_env.env.GetKinBody(entity), [1, 1, 1])
            else:
                set_color(self.problem_env.env.GetKinBody(entity), [0, 0, 0])
        self.print_reachable_entities()
        self.print_inway_entities()

        # if the goal object is not reachable with naive path, then declare infeasible problem
        goal_object = self.goal_entities[0]
        objects_with_mc_path = self.in_way.minimum_constraint_path_to_entity.keys()
        is_pick_state = len(self.problem_env.robot.GetGrabbed()) == 0
        if is_pick_state and not (
                goal_object in self.is_reachable.reachable_entities or goal_object in objects_with_mc_path):
            print "Infeasible problem instance"
            # todo fix this when there are multiple objects to pack
            # import pdb;pdb.set_trace()
            # sys.exit(-1)

    def update_reachability_based_on_inway(self):
        reachability_idx = 9
        for entity in self.in_way.minimum_constraints_to_entity:
            if len(self.in_way.minimum_constraints_to_entity[entity]) == 0 and \
                    not (entity in self.in_way.reachable_entities):
                self.is_reachable.reachable_entities.append(entity)
                self.in_way.reachable_entities.append(entity)
                self.nodes[entity][reachability_idx] = True
                self.is_reachable.motion_plans[entity] = self.in_way.minimum_constraint_path_to_entity[entity]

    def print_reachable_entities(self):
        print 'Reachable entities:', self.reachable_entities

    def print_inway_entities(self):
        inway_idx = 0
        print "InWay:", [e for e in self.edges if self.edges[e][inway_idx]]

    def get_entities_in_way_to_goal_entities(self):
        inway_idx = 0
        isgoal_idx = -3
        src_dest_pairs_in_way = [e for e in self.edges if self.edges[e][inway_idx]]
        src_dest_pairs_in_way_to_goal = []
        for src_dest_pair in src_dest_pairs_in_way:
            dest = src_dest_pair[1]
            if self.nodes[dest][isgoal_idx]:
                src_dest_pairs_in_way_to_goal.append(src_dest_pair)
        return src_dest_pairs_in_way_to_goal

    def update_collisions_at_prm_vertices(self, parent_state):
        global prm_vertices
        global prm_edges

        if prm_vertices is None or prm_edges is None:
            prm_vertices, prm_edges = pickle.load(open('./prm.pkl', 'rb'))

        holding = len(self.problem_env.robot.GetGrabbed()) > 0
        if holding:
            held = self.problem_env.robot.GetGrabbed()[0]

        def in_collision(q, obj):
            #old_q = get_body_xytheta(self.problem_env.robot)
            set_robot_config(q, self.problem_env.robot)
            col = self.problem_env.env.CheckCollision(self.problem_env.robot, obj)
            #set_robot_config(old_q, self.problem_env.robot)
            return col

        obj_name_to_pose = {
            obj.GetName(): tuple(get_body_xytheta(obj)[0].round(6))
            for obj in self.problem_env.objects
        }

        # if robot is holding an object, all collision information changes
        # but we should still retain previous collision information
        # because collisions at the next state will be similar to collisions at the previous state

        # todo once we placed the object, instead of setting the collides to be empty,
        #  we could use the collision information from state-before-pick, because
        #  the robot shape goes back to whatever it was.
        collides = copy.copy(parent_state.collides) if holding else {}
        old_q = get_body_xytheta(self.problem_env.robot)
        for obj in self.problem_env.objects:
            obj_name_pose_tuple = (obj.GetName(), obj_name_to_pose[obj.GetName()])
            collisions_with_obj_did_not_change = parent_state is not None and \
                                                 obj_name_pose_tuple in parent_state.collides and \
                                                 not holding
            if collisions_with_obj_did_not_change:
                collides[obj_name_pose_tuple] = parent_state.collides[obj_name_pose_tuple]
            else:
                prm_vertices_in_collision_with_obj = {i for i, q in enumerate(prm_vertices) if in_collision(q, obj)}
                collides[obj_name_pose_tuple] = prm_vertices_in_collision_with_obj
        set_robot_config(old_q, self.problem_env.robot)

        current_collides = {
            obj.GetName(): collides[(obj.GetName(), obj_name_to_pose[obj.GetName()])]
            for obj in self.problem_env.objects
        }

        return collides, current_collides

    def update_node_features(self, previous_state, moved_object=None):
        is_obj_holding_state = moved_object is None
        is_holding_idx = -1
        reachability_idx = 9

        for entity in self.problem_env.entity_names:
            self.nodes[entity] = copy.deepcopy(previous_state.nodes[entity])
            if is_obj_holding_state:
                self.nodes[entity][is_holding_idx] = self.is_holding_goal_entity()
                held = str(self.problem_env.robot.GetGrabbed()[0].GetName())
                is_entity_object = not (entity in self.problem_env.regions)
                if is_entity_object:
                    self.nodes[entity][reachability_idx] = False
                else:
                    self.update_node_features_for_entity_when_object_is_held_or_placed(entity, held, previous_state)
            else:
                print entity
                self.nodes[entity][is_holding_idx] = False
                self.update_node_features_for_entity_when_object_is_held_or_placed(entity, moved_object, previous_state)

    def update_edge_features(self, parent_state, moved_object=None):
        is_obj_holding_state = moved_object is None
        in_way_idx = 0
        in_region_idx = 1

        self.in_way.set_reachable_entities(self.is_reachable.reachable_entities)
        self.in_way.set_paths_to_reachable_entities(self.is_reachable.motion_plans)

        for b in self.problem_env.entity_names:
            for a in self.problem_env.entity_names:
                self.edges[(a, b)] = copy.deepcopy(parent_state.edges[(a, b)])
                self.edges[(a, b)][in_region_idx] = self.in_region(a, b)
                if is_obj_holding_state:
                    if is_object(b):
                        # in_way to another object is false when holding an obj
                        self.edges[(a, b)][in_way_idx] = False
                    elif is_region(b):
                        held = str(self.problem_env.robot.GetGrabbed()[0].GetName())
                        was_held_object_in_way = parent_state.edges[(held, b)][in_way_idx]
                        if held == a and was_held_object_in_way:
                            # in_way to region is false when holding the object that was in way
                            self.edges[(a, b)][in_way_idx] = False
                        else:
                            # We make the simplifying assumption that if the object is picked, in_ways don't change.
                            # Note that this may not necessarily be true, because, if we are holding an object, we
                            # are likely to collide with more objects than when we are not holding one.
                            # We make this assumption because when we have an object in our hand, in_way doesn't matter
                            # in the sense that it is not a time to choose which object to clear
                            self.edges[(a, b)][in_way_idx] = parent_state.edges[(a, b)][in_way_idx]
                else:
                    print 'Computing InWay ', a, b
                    # when I place an object, it can be in way to any other entities
                    # I could potentially look at previous minimum constraint path, try to
                    # see if I am still connected to that path, and see if a is in the swept volume
                    self.edges[(a, b)] = self.get_edge_features(a, b)
                    print 'Done '

    def update_node_features_for_entity_when_object_is_held_or_placed(self, entity, manipulated_object, parent_state):
        reachability_idx = 9
        in_way_idx = 0

        previously_reachable = parent_state.nodes[entity][reachability_idx]
        all_objects_in_way_to_entity = [tmp for tmp in self.problem_env.entity_names
                                        if parent_state.edges[(tmp, entity)][in_way_idx]]
        if previously_reachable:
            print "Previously reachable path planning", entity
            self.nodes[entity] = self.get_node_features(entity, self.goal_entities)
            print "Done"
        elif not previously_reachable:
            if len(all_objects_in_way_to_entity) == 1:
                print "There was only one object in way. Did we clear it?", manipulated_object, entity
                self.nodes[entity] = self.get_node_features(entity, self.goal_entities)
                print 'Done'
            else:
                self.nodes[entity][reachability_idx] = False
            print "Done"

    def get_nodes(self, paps_used_in_data=None):
        if paps_used_in_data is not None:
            self.is_reachable.set_pick_used(paps_used_in_data[0])

        nodes = {}
        for entity in self.problem_env.entity_names:
            print 'Getting node features for', entity
            nodes[entity] = self.get_node_features(entity, self.goal_entities)
        return nodes

    def get_edges(self):
        self.in_way.set_reachable_entities(self.is_reachable.reachable_entities)
        print "Reachable entities: ", self.is_reachable.reachable_entities
        self.in_way.set_paths_to_reachable_entities(self.is_reachable.motion_plans)
        edges = {}
        in_way_idx = 0
        for b in self.problem_env.entity_names:
            for a in self.problem_env.entity_names:
                print 'Getting edge features for', (a, b)
                edges[(a, b)] = self.get_edge_features(a, b)  # a = src, b = dest
            """
            if not is_something_in_way_to_b:
                self.nodes[b][reachability_idx] = True
                if b not in self.is_reachable.reachable_entities:
                    self.is_reachable.reachable_entities.append(b)
            """
        return edges

    def get_node_features(self, entity, goal_entities):
        isobj = entity not in self.problem_env.regions
        obj = self.problem_env.env.GetKinBody(entity) if isobj else None
        pose = get_body_xytheta(obj)[0] if isobj else None

        is_entity_reachable = self.is_reachable(entity)
        # todo turn these into one-hot encoding
        # todo I need is_holding predicate for not putting high value on regions
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

    def get_edge_features(self, a, b):
        return [
            # todo bk: I think we need in_way(b,a) here as well, but I forgot why. Comeback later.
            self.in_way(a, b),
            self.in_region(a, b)
        ]

    def get_abstract_state(self):
        # write this function if necessary. returns all predicate evaluations
        self.state_saver.Restore()
        pass

    def make_pklable(self):
        self.problem_env = None
        self.is_reachable.problem_env = None
        self.in_region.problem_env = None
        self.in_way.problem_env = None
        self.in_way.robot = None
        self.is_holding_goal_entity.problem_env = None
        if self.parent is not None:
            self.parent.make_pklable()

    def make_plannable(self, problem_env):
        self.problem_env = problem_env
        self.is_reachable.problem_env = problem_env
        self.in_region.problem_env = problem_env
        self.in_way.problem_env = problem_env
        self.in_way.robot = problem_env.robot
        self.is_holding_goal_entity.problem_env = problem_env
        if self.parent is not None:
            self.parent.make_plannable(problem_env)


class StateForPrediction:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
