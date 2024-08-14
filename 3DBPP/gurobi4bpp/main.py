import random

from packer import Packer
from container import Container
from box import Box


def solve():
    boxes = []
    box_number = 60
    # generate random boxes:
    for i in range(box_number):
        boxes.append(Box(i, random.randint(1, 300), random.randint(1, 300), random.randint(1, 300)))
    # boxes.extend(
    #     [Box(0, 270, 110, 15),
    #      Box(1, 25.3, 25.3, 79.5),
    #      Box(2, 16, 25, 86),
    #      Box(3, 34, 22, 87.8),
    #      Box(4, 34, 22, 87.8),
    #      Box(5, 34, 22, 87.8),
    #      Box(6, 34, 22, 87.8),
    #      Box(7, 34, 22, 87.8),
    #      Box(8, 34, 22, 87.8),
    #      Box(9, 29.1, 45.15, 97.15),
    #      Box(10, 33, 54, 113),
    #      Box(11, 49, 49, 114.5),
    #      Box(12, 49, 49, 114.5),
    #      Box(13, 61, 40, 170),
    #      Box(14, 61, 40, 170),
    #      ]
    # )
    print([[box.length, box.width, box.height] for box in boxes])
    # container = Container(270, 324, 104)
    container = Container(500, 1000, 500)
    packer = Packer(boxes, container)
    packer.optimize()
    # packer.model.computeIIS()
    # packer.model.write("model.ilp")


if __name__ == '__main__':
    solve()