import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors


class Box:
    def __init__(self, size, name=None):
        self.size = np.array(size)
        self.name = name

class PlacedBox:
    def __init__(self, box, position):
        self.box = box
        self.position = np.array(position)

    def get_bounds(self):
        return self.position, self.position + self.box.size

class Container:
    def __init__(self, size):
        self.size = np.array(size)
        self.placed = []
        self.points = [np.array([0, 0, 0])]

    def fits_inside(self, placed_box):
        min_c, max_c = placed_box.get_bounds()
        return np.all(min_c >= 0) and np.all(max_c <= self.size)

    def overlaps(self, placed_box):
        for other in self.placed:
            min1, max1 = placed_box.get_bounds()
            min2, max2 = other.get_bounds()
            if np.all(max1 > min2) and np.all(max2 > min1):
                return True
        return False

    def add_box(self, box, position):
        new_pb = PlacedBox(box, position)
        self.placed.append(new_pb)
        
        p = position
        s = box.size
        new_points = [
            p + [s[0], 0, 0], 
            p + [0, s[1], 0], 
            p + [0, 0, s[2]]  
        ]
        
        for pt in new_points:
            if np.all(pt < self.size) and not any(np.array_equal(pt, existing) for existing in self.points):
                self.points.append(pt)
        
        
        self.points.sort(key=lambda pt: (pt[2], pt[1], pt[0]))

def get_unique_rotations(size):
    return list(set(itertools.permutations(size)))

def pack_with_corners(container, boxes):
    boxes = sorted(boxes, key=lambda b: np.prod(b.size), reverse=True)

    for box in boxes:
        placed = False
        rotations = get_unique_rotations(box.size)

        for pt in container.points[:]:  
            for rot in rotations:
                candidate_box = Box(rot, name=box.name)
                candidate_pb = PlacedBox(candidate_box, pt)

                if container.fits_inside(candidate_pb) and not container.overlaps(candidate_pb):
                    container.add_box(candidate_box, pt)
                    container.points = [p for p in container.points if not np.array_equal(p, pt)]
                    placed = True
                    break
            if placed: break
        
        if not placed:
            print(f"Could not place {box.name}")

def plot_boxes_improved(container, title="3D Bin Packing (Corner Points)"):
    """Visualizes the container and its contents."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)

    # Use Tableau Colors for clear distinction, cycle if too many boxes
    color_cycle = list(mcolors.TABLEAU_COLORS.values())

    def draw_box(placed_box, color, index):
        min_corner, max_corner = placed_box.get_bounds()
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner

        # Define the 6 faces using 3D vertices
        faces = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], # Floor (Zmin)
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], # Ceiling (Zmax)
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], # Front (Ymin)
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], # Right (Xmax)
            [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], # Back (Ymax)
            [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], # Left (Xmin)
        ]

        # Draw the faces
        face_collection = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.6)
        ax.add_collection3d(face_collection)

        # Optional: Label the box (centered)
        # label_pos = min_corner + (placed_box.box.size / 2.0)
        # ax.text(label_pos[0], label_pos[1], label_pos[2], f"{placed_box.box.name}", color='black', ha='center', va='center', fontsize=8)

    # 1. Draw all placed boxes
    for i, pb in enumerate(container.placed):
        color = color_cycle[i % len(color_cycle)]
        draw_box(pb, color, i)

    # 2. Configure axes and boundaries
    ax.set_xlim([0, container.size[0]])
    ax.set_ylim([0, container.size[1]])
    ax.set_zlim([0, container.size[2]])

    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Length)')
    ax.set_zlabel('Z (Height)')

    # Maintain the aspect ratio properly
    ax.set_box_aspect(container.size) 

    print(f"Visualization complete. Displaying plot.")
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    bin_size = [10, 6, 8]
    cont = Container(bin_size)
    items = [
            Box([6, 6, 6], "Big A"),
            Box([3, 3, 3], "Small B"),
            Box([5, 2, 2], "Long C"),
            Box([1, 1, 1], "Tiny D"),
            Box([4, 4, 2], "Flat E"),
            Box([2, 5, 2], "Tall F"),
            Box([3, 2, 4], "Brick G"),
        ]    
    
    
    pack_with_corners(cont, items)


    if cont.placed:
        plot_boxes_improved(cont, title=f"Packing Simulation for Container {bin_size}")
    else:
        print("No items could be packed. Skipping visualization.")