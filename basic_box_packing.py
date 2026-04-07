import numpy as np
import itertools

class Box:
    def __init__(self, size, name=None):
        self.size = np.array(size)  # (dx, dy, dz)
        self.name = name

class PlacedBox:
    def __init__(self, box, position):
        self.box = box
        self.position = np.array(position)  # bottom-left-front corner

    def get_bounds(self):
        min_corner = self.position
        max_corner = self.position + self.box.size
        return min_corner, max_corner
    
def overlap(a: PlacedBox, b: PlacedBox):
    min1, max1 = a.get_bounds()
    min2, max2 = b.get_bounds()

    return np.all(max1 > min2) and np.all(max2 > min1)

class Container:
    def __init__(self, size):
        self.size = np.array(size)
        self.placed = []

    def fits_inside(self, placed_box):
        min_c, max_c = placed_box.get_bounds()
        return np.all(min_c >= 0) and np.all(max_c <= self.size)

    def is_valid(self, placed_box):
        if not self.fits_inside(placed_box):
            return False
        for other in self.placed:
            if overlap(placed_box, other):
                return False
        return True
    
def generate_layer_positions(container, box, step=1.0):
    x_max, y_max, z_max = container.size
    dx, dy, dz = box.size

    # possible z levels (bottom-up)
    zs = np.arange(0, z_max - dz + step, step)

    for z in zs:
        xs = np.arange(0, x_max - dx + step, step)
        ys = np.arange(0, y_max - dy + step, step)

        for x in xs:
            for y in ys:
                yield np.array([x, y, z])

import itertools

def get_rotations(size):
    return list(set(itertools.permutations(size)))

def pack_boxes_layered(container, boxes, step=1.0):
    boxes = sorted(boxes, key=lambda b: np.prod(b.size), reverse=True)

    for box in boxes:
        print(f"Placing {box.name}...")
        placed = False

        rotations = get_rotations(box.size)

        for rot in rotations:
            rotated_box = Box(rot, name=box.name)

            for pos in generate_layer_positions(container, rotated_box, step):
                candidate = PlacedBox(rotated_box, pos)

                if container.is_valid(candidate):
                    container.placed.append(candidate)
                    print(f"  → Placed at {pos} with rotation {rot}")
                    placed = True
                    break

            if placed:
                break

        if not placed:
            print(f"  → Could not place {box.name}")

    return container

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_boxes(container):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw_box(min_corner, max_corner):
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner

        vertices = [
            [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0)],
            [(x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1)],
            [(x0,y0,z0), (x1,y0,z0), (x1,y0,z1), (x0,y0,z1)],
            [(x1,y0,z0), (x1,y1,z0), (x1,y1,z1), (x1,y0,z1)],
            [(x1,y1,z0), (x0,y1,z0), (x0,y1,z1), (x1,y1,z1)],
            [(x0,y1,z0), (x0,y0,z0), (x0,y0,z1), (x0,y1,z1)],
        ]

        ax.add_collection3d(
            Poly3DCollection(vertices, edgecolor='black', alpha=0.4)
        )

    # Draw each placed box
    for pb in container.placed:
        min_c, max_c = pb.get_bounds()
        draw_box(min_c, max_c)

    # Set limits
    ax.set_xlim([0, container.size[0]])
    ax.set_ylim([0, container.size[1]])
    ax.set_zlim([0, container.size[2]])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

if __name__ == "__main__":
    container = Container(size=[4, 8, 10])

    boxes = [
        Box([4, 4, 4], "A"),
        Box([3, 3, 3], "B"),
        Box([2, 2, 2], "C"),
        Box([1, 5, 2], "D"),
    ]

    pack_boxes_layered(container, boxes, step=1.0)

    plot_boxes(container)
