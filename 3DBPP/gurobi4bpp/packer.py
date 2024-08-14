from typing import List, Dict

from box import Box, BoxRelation
from container import Container
from gurobipy import Model, GRB, quicksum


class Packer:
    def __init__(self, boxes, container):
        self.boxes: List[Box] = boxes
        self.container: Container = container
        self.box_relations: List[BoxRelation] = []
        self.box_map: Dict[int, Box] = {}
        self.box_relation_map: Dict[(int, int), BoxRelation] = {}
        self.loaded_boxes = []
        self.unloaded_boxes = []
        self._init_model()
        self.model = Model("Packer")

    def _init_model(self):
        for box in self.boxes:
            self.box_map[box.index] = box
            for other_box in self.boxes:
                if box.index != other_box.index:
                    box_relation = BoxRelation(box, other_box)
                    self.box_relations.append(box_relation)
                    self.box_relation_map[(box.index, other_box.index)] = box_relation

    def set_box_vars(self):
        for box in self.boxes:
            box.var_x = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{box.index}")
            box.var_y = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{box.index}")
            box.var_z = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{box.index}")
            box.var_loaded = self.model.addVar(vtype=GRB.BINARY, name=f"loaded_{box.index}")
            for orientation in box.orientations:
                orientation.var_selection = self.model.addVar(
                    vtype=GRB.BINARY, name=f"selection_{box.index}_{orientation.idx}")

    def set_box_relations_vars(self):
        for box_relation in self.box_relations:
            box_relation.var_relation_x = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"relation_x_{box_relation.first_box.index}_{box_relation.second_box.index}")
            box_relation.var_relation_y = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"relation_y_{box_relation.first_box.index}_{box_relation.second_box.index}")
            box_relation.var_relation_z = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"relation_z_{box_relation.first_box.index}_{box_relation.second_box.index}")

    def orientation_cons(self):
        for box in self.boxes:
            self.model.addConstr(
                quicksum([orientation.var_selection for orientation in box.orientations]) == box.var_loaded)

    def geometry_cons(self):
        for box in self.boxes:
            self.model.addConstr(box.var_x + quicksum([orientation.var_selection * orientation.x for orientation in
                                                       box.orientations]) <= self.container.length * box.var_loaded)
            self.model.addConstr(box.var_y + quicksum([orientation.var_selection * orientation.y for orientation in
                                                       box.orientations]) <= self.container.width * box.var_loaded)
            self.model.addConstr(box.var_z + quicksum([orientation.var_selection * orientation.z for orientation in
                                                       box.orientations]) <= self.container.height * box.var_loaded)

    def relation_cons(self):
        index = list(self.box_map.keys())
        for i in index:
            for j in index:
                if i < j:
                    self.model.addConstr(
                        self.box_relation_map.get((i, j)).var_relation_x +
                        self.box_relation_map.get((j, i)).var_relation_x <= self.box_map.get(i).var_loaded)
                    self.model.addConstr(
                        self.box_relation_map.get((i, j)).var_relation_y +
                        self.box_relation_map.get((j, i)).var_relation_y <= self.box_map.get(i).var_loaded)
                    self.model.addConstr(
                        self.box_relation_map.get((i, j)).var_relation_z +
                        self.box_relation_map.get((j, i)).var_relation_z <= self.box_map.get(i).var_loaded)
                    #
                    self.model.addConstr(self.box_relation_map.get((i, j)).var_relation_x +
                                         self.box_relation_map.get((i, j)).var_relation_y +
                                         self.box_relation_map.get((i, j)).var_relation_z +
                                         self.box_relation_map.get((j, i)).var_relation_x +
                                         self.box_relation_map.get((j, i)).var_relation_y +
                                         self.box_relation_map.get((j, i)).var_relation_z >=
                                         self.box_map.get(i).var_loaded + self.box_map.get(j).var_loaded - 1)
                if i != j:
                    box1 = self.box_map.get(i)
                    box2 = self.box_map.get(j)
                    self.model.addConstr(
                        box1.var_x +
                        quicksum([orientation.var_selection * orientation.x for orientation in box1.orientations]) <=
                        box2.var_x + self.container.length * (1 - self.box_relation_map.get((i, j)).var_relation_x))
                    self.model.addConstr(
                        box1.var_y +
                        quicksum([orientation.var_selection * orientation.y for orientation in box1.orientations]) <=
                        box2.var_y + self.container.width * (1 - self.box_relation_map.get((i, j)).var_relation_y))
                    self.model.addConstr(
                        box1.var_z +
                        quicksum([orientation.var_selection * orientation.z for orientation in box1.orientations]) <=
                        box2.var_z + self.container.height * (1 - self.box_relation_map.get((i, j)).var_relation_z))

    def set_objective(self):
        self.model.setObjective(quicksum([box.var_loaded * box.length * box.width * box.height for box in self.boxes]),
                                GRB.MAXIMIZE)
        # self.model.setObjective(quicksum([box.var_loaded for box in self.boxes]),
        #                         GRB.MAXIMIZE)

    def optimize(self):
        self.set_box_vars()
        self.set_box_relations_vars()
        self.orientation_cons()
        self.geometry_cons()
        self.relation_cons()
        self.set_objective()
        self.model.optimize()
        self.get_solution()

    def get_solution(self):
        if self.model.status == GRB.OPTIMAL:
            for box in self.boxes:
                if box.var_loaded.x > 0.5:
                    for orientation in box.orientations:
                        if orientation.var_selection.x > 0.5:
                            self.loaded_boxes.append([box.index, orientation.idx])
                else:
                    self.unloaded_boxes.append(box.index)
        print("loaded: %s" % str(self.loaded_boxes))
        print("unloaded: %s" % str(self.unloaded_boxes))
