from typing import List
from itertools import permutations


class Box:
    def __init__(self, index, length, width, height):
        self.index = index
        self.length = length
        self.width = width
        self.height = height
        # if loaded
        self.var_loaded = None
        # bottom-left corner
        self.var_x = None
        self.var_y = None
        self.var_z = None
        self.orientations: List[Orientation] = []
        self._add_orientation()

    def _add_orientation(self):
        permutations_list = list(permutations([self.length, self.width, self.height]))
        for idx, permutation in enumerate(permutations_list):
            self.orientations.append(Orientation(idx, permutation[0], permutation[1], permutation[2]))


class Orientation:
    def __init__(self, idx, x, y, z):
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z
        self.var_selection = None


class BoxRelation:
    def __init__(self, box1, box2):
        self.first_box: Box = box1
        self.second_box: Box = box2
        self.var_relation_x = None
        self.var_relation_y = None
        self.var_relation_z = None