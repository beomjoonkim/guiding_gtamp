from trajectory_representation.state import State
from gtamp_utils.utils import CustomStateSaver, get_body_xytheta, set_robot_config, set_obj_xytheta

from gtamp_utils.utils import visualize_path, two_arm_pick_object
from manipulation.bodies.bodies import set_color
import pickle



class PaPState(State):
    def __init__(self, problem_env, goal_entities, parent_state=None, parent_action=None, paps_used_in_data=None):
        self.state_saver = CustomStateSaver(problem_env.env)
        self.problem_env = problem_env
        self.parent_state = parent_state  # used to update the node features
        self.goal_entities = goal_entities
        self.object_names = [str(obj.GetName()) for obj in problem_env.objects]

        # raw variables
        self.robot_pose = get_body_xytheta(problem_env.robot)
        self.object_poses = {
            obj.GetName(): get_body_xytheta(obj)
            for obj in problem_env.objects
        }

        self.use_prm = problem_env.name.find('two_arm') != -1

        if paps_used_in_data is not None:
            self.pick_used = paps_used_in_data[0]
            self.place_used = paps_used_in_data[1]

        self.mc_pick_path = {}
        self.mc_place_path = {}
        self.reachable_entities = []

        self.pick_in_way = None
        self.place_in_way = None
        self.in_region = None
        self.is_holding_goal_entity = None

        self.ternary_edges = None
        self.binary_edges = None
        self.nodes = None

        self.prm_vertices, self.prm_edges = pickle.load(open('./prm.pkl','rb'))

    def update_collisions_at_prm_vertices(self, parent_collides):
        #global prm_vertices
        #global prm_edges

        #if prm_vertices is None or prm_edges is None:
        #    prm_vertices, prm_edges = pickle.load(open('./prm.pkl', 'rb'))

        is_robot_holding = len(self.problem_env.robot.GetGrabbed()) > 0

        def in_collision(q, obj):
            set_robot_config(q, self.problem_env.robot)
            if is_robot_holding:
                # note:
                # openrave bug: when an object is held, it won't check the held_obj and given object collision unless
                #               collision on robot is first checked. So, we have to check it twice
                col = self.problem_env.env.CheckCollision(self.problem_env.robot)
                col = self.problem_env.env.CheckCollision(self.problem_env.robot, obj)
            else:
                col = self.problem_env.env.CheckCollision(self.problem_env.robot, obj)
            return col

        obj_name_to_pose = {
            obj.GetName(): tuple(get_body_xytheta(obj)[0].round(6))
            for obj in self.problem_env.objects
        }

        collisions_at_all_obj_pose_pairs = {}
        old_q = get_body_xytheta(self.problem_env.robot)
        for obj in self.problem_env.objects:
            obj_name_pose_tuple = (obj.GetName(), obj_name_to_pose[obj.GetName()])
            collisions_with_obj_did_not_change = parent_collides is not None and \
                                                 obj_name_pose_tuple in parent_collides
            if collisions_with_obj_did_not_change:
                collisions_at_all_obj_pose_pairs[obj_name_pose_tuple] = parent_collides[obj_name_pose_tuple]
            else:
                prm_vertices_in_collision_with_obj = {i for i, q in enumerate(self.prm_vertices) if in_collision(q, obj)}
                collisions_at_all_obj_pose_pairs[obj_name_pose_tuple] = prm_vertices_in_collision_with_obj
        set_robot_config(old_q, self.problem_env.robot)

        # what's the diff between collides and curr collides?
        # collides include entire set of obj and obj name pose tuple
        collisions_at_current_obj_pose_pairs = {
            obj.GetName(): collisions_at_all_obj_pose_pairs[(obj.GetName(), obj_name_to_pose[obj.GetName()])]
            for obj in self.problem_env.objects
        }

        return collisions_at_all_obj_pose_pairs, collisions_at_current_obj_pose_pairs

    def get_nodes(self):
        nodes = {}
        for entity in self.problem_env.entity_names:
            nodes[entity] = self.get_node_features(entity, self.goal_entities)
        return nodes

    def get_binary_edges(self):
        raise NotImplementedError

    def get_ternary_edges(self):
        raise NotImplementedError

    def get_node_features(self, entity, goal_entities):
        raise NotImplementedError

    def get_ternary_edge_features(self, a, b, r):
        raise NotImplementedError

    def get_binary_edge_features(self, a, b):
        raise NotImplementedError

    def get_entities_in_way_to_goal_entities(self):
        goal_objs = [g for g in self.goal_entities if g.find('region') == -1]
        goal_region = [g for g in self.goal_entities if g.find('region') != -1][0]
        objs_in_way = []
        for obj in goal_objs:
            objs_in_way += self.get_entities_in_pick_way(obj)
            objs_in_way += self.get_entities_in_place_way(obj, goal_region)

        objs_in_way = set(objs_in_way)
        objs_in_way = list(objs_in_way)
        return objs_in_way

    def restore(self, problem_env=None):
        if problem_env is None:
            problem_env = self.problem_env

        for obj_name, obj_pose in self.object_poses.items():
            set_obj_xytheta(obj_pose, problem_env.env.GetKinBody(obj_name))
        set_robot_config(self.robot_pose, problem_env.robot)

    def update_cached_data_after_binary(self):
        self.mc_pick_path = self.pick_in_way.mc_path_to_entity
        self.reachable_entities = self.pick_in_way.reachable_entities
        self.pick_used = self.pick_in_way.pick_used

    def update_cached_data_after_ternary(self):
        self.place_used = self.place_in_way.place_used
        self.mc_place_path = self.place_in_way.mc_path_to_entity
        self.pick_used = self.place_in_way.pick_used

    def is_entity_reachable(self, entity):
        return self.nodes[entity][-2]

    def get_entities_occluded_by(self, entity):
        return self.pick_entities_occluded_by(entity) + self.place_entities_occluded_by(entity)

    def is_goal_entity(self, entity):
        return entity in self.goal_entities

    def pick_entities_occluded_by(self, entity):
        inway = []
        entity_names = self.problem_env.object_names + self.problem_env.region_names
        for obj_name in entity_names:
            if 'entire' in obj_name:
                continue
            if (entity, obj_name) not in self.binary_edges:
                continue
            if self.binary_edges[(entity, obj_name)][1]:
                inway.append(obj_name)
        return inway

    def place_entities_occluded_by(self, entity):
        inway = []
        for region_name in self.problem_env.region_names:
            if region_name == 'entire_region':
                continue
            for obj_name in self.problem_env.object_names:
                if (obj_name, entity, region_name) not in self.ternary_edges:
                    continue
                if self.ternary_edges[(obj_name, entity, region_name)][0]:
                    inway.append((obj_name, region_name))
        return inway

    def get_entities_in_place_way(self, entity, region):
        inway = []
        for obj_name in self.object_names:
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
                import pdb;
                pdb.set_trace()

                for tmp in objs_in_way:
                    set_color(self.problem_env.env.GetKinBody(tmp), [0, 1, 0])
                saver.Restore()

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

    def get_predicate_evaluations(self):
        return {'nodes': self.nodes, 'binary': self.binary_edges, 'ternary': self.ternary_edges}
