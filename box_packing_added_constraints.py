"""Corner-point packing with weight, crushability, and temperature rules.

Stacking is a relative heuristic: each upper box must be no heavier and
at least as crushable as every box directly supporting it. No cumulative
load capacity or container weight limit is modeled. All dimensions and the
hot/frozen gap use the same units; weights must use one consistent unit.
Valid placements are ranked to reduce empty space and increase face contact.
Hot/frozen separation is a scored preference by default, so it does not
force an empty lane. Set strict_temperature_gap=True to require it.
"""
from collections import Counter
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EPS = 1e-9
COLORS = {"hot": "tomato", "ambient": "goldenrod", "frozen": "deepskyblue"}


def dimensions(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Size must contain three finite positive dimensions.")
    return result


class Box:
    def __init__(self, size, name=None, weight=1.0, crushability=1, temperature="ambient", upright_only=False):
        self.size = dimensions(size)
        self.name = name
        self.weight = float(weight)
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Weight must be finite and positive.")
        if isinstance(crushability, bool) or not isinstance(crushability, (int, np.integer)) or crushability not in range(1, 6):
            raise ValueError("Crushability must be an integer from 1 to 5.")
        self.crushability = int(crushability)
        if temperature not in COLORS:
            raise ValueError("Temperature must be hot, ambient, or frozen.")
        self.temperature = temperature
        self.upright_only = bool(upright_only)

    def rotated(self, size):
        return Box(size, self.name, self.weight, self.crushability, self.temperature, self.upright_only)


class PlacedBox:
    def __init__(self, box, position):
        self.box = box
        self.position = np.asarray(position, dtype=float)
        if self.position.shape != (3,) or not np.all(np.isfinite(self.position)):
            raise ValueError("Position must contain three finite coordinates.")

    def get_bounds(self):
        return self.position, self.position + self.box.size


class Container:
    def __init__(self, size, temperature_gap=1.0, strict_temperature_gap=False, temperature_weight=0.15):
        self.size = dimensions(size)
        self.temperature_gap = float(temperature_gap)
        if not np.isfinite(self.temperature_gap) or self.temperature_gap < 0:
            raise ValueError("Temperature gap must be finite and nonnegative.")
        self.strict_temperature_gap = bool(strict_temperature_gap)
        self.temperature_weight = float(temperature_weight)
        if not np.isfinite(self.temperature_weight) or self.temperature_weight < 0:
            raise ValueError("Temperature weight must be finite and nonnegative.")
        self.placed = []
        self.points = [np.zeros(3)]
        self.unplaced = []

    def fits_inside(self, candidate):
        low, high = candidate.get_bounds()
        return np.all(low >= -EPS) and np.all(high <= self.size + EPS)

    def overlaps(self, candidate):
        low, high = candidate.get_bounds()
        return any(np.all(high > other.position + EPS) and
                   np.all(other.position + other.box.size > low + EPS)
                   for other in self.placed)

    def support_contacts(self, candidate):
        """Return supporting boxes and their contact areas (no overlapping interiors)."""
        low, high = candidate.get_bounds()
        contacts = []
        for other in self.placed:
            bottom, top = other.get_bounds()
            if abs(top[2] - low[2]) > EPS:
                continue
            extent = np.minimum(high[:2], top[:2]) - np.maximum(low[:2], bottom[:2])
            if np.all(extent > EPS):
                contacts.append((other, float(np.prod(extent))))
        return contacts

    def placement_errors(self, candidate):
        errors = []
        if not self.fits_inside(candidate):
            errors.append("outside container")
        if self.overlaps(candidate):
            errors.append("overlap")
        low, high = candidate.get_bounds()
        if low[2] > EPS:
            contacts = self.support_contacts(candidate)
            base_area = float(np.prod(candidate.box.size[:2]))
            if not np.isclose(sum(area for _, area in contacts), base_area, rtol=1e-9, atol=EPS):
                errors.append("base not fully supported")
            for support, _ in contacts:
                if candidate.box.weight > support.box.weight + EPS:
                    errors.append("heavier than supporting box")
                if candidate.box.crushability < support.box.crushability:
                    errors.append("less crushable than supporting box")
        for other in self.placed:
            if {candidate.box.temperature, other.box.temperature} == {"hot", "frozen"}:
                other_low, other_high = other.get_bounds()
                separation = np.maximum(0, np.maximum(other_low - high, low - other_high))
                if self.strict_temperature_gap and np.linalg.norm(separation) + EPS < self.temperature_gap:
                    errors.append("hot/frozen separation too small")
        return sorted(set(errors))

    def add_box(self, box, position):
        candidate = PlacedBox(box, position)
        errors = self.placement_errors(candidate)
        if errors:
            raise ValueError(f"Cannot place {box.name}: {', '.join(errors)}")
        self.placed.append(candidate)
        # Combine face coordinates to include corners formed by multiple boxes.
        # Gap-offset faces give hot/frozen boxes a chance to satisfy separation.
        axes = [{0.0} for _ in range(3)]
        for placed in self.placed:
            low, high = placed.get_bounds()
            for axis in range(3):
                axes[axis].update((float(low[axis]), float(high[axis]),
                                   float(high[axis] + self.temperature_gap)))
        self.points = [np.array(point) for point in itertools.product(*map(sorted, axes))
                       if np.all(np.array(point) < self.size - EPS)]
        self.points.sort(key=lambda point: (point[2], point[1], point[0]))


def get_unique_rotations(size):
    return sorted(set(itertools.permutations(size)))


def packing_metrics(placed):
    """Empty volume inside the occupied envelope, and touching face area.

    A smaller empty envelope discourages gaps; more face contact breaks ties
    in favor of bracing. This is a geometric heuristic, not a motion model.
    """
    if not placed:
        return {"empty_volume": 0.0, "contact_area": 0.0, "envelope_volume": 0.0}
    lows = np.array([item.position for item in placed])
    highs = np.array([item.position + item.box.size for item in placed])
    envelope = float(np.prod(highs.max(axis=0) - lows.min(axis=0)))
    contact = 0.0
    for first, second in itertools.combinations(placed, 2):
        a0, a1 = first.get_bounds()
        b0, b1 = second.get_bounds()
        for axis in range(3):
            if abs(a1[axis] - b0[axis]) <= EPS or abs(b1[axis] - a0[axis]) <= EPS:
                other_axes = [i for i in range(3) if i != axis]
                overlap = np.minimum(a1, b1)[other_axes] - np.maximum(a0, b0)[other_axes]
                if np.all(overlap > EPS):
                    contact += float(np.prod(overlap))
    return {"empty_volume": max(0.0, envelope - sum(float(np.prod(p.box.size)) for p in placed)),
            "contact_area": contact, "envelope_volume": envelope}


def temperature_shortfall(placed, preferred_gap):
    """Average squared shortfall from the desired hot/frozen distance, 0..1.

    An ambient item can occupy the separating space. This measures geometric
    distance only, not insulation or heat transfer.
    """
    penalties = []
    if preferred_gap <= EPS:
        return 0.0
    for first, second in itertools.combinations(placed, 2):
        if {first.box.temperature, second.box.temperature} != {"hot", "frozen"}:
            continue
        a0, a1 = first.get_bounds()
        b0, b1 = second.get_bounds()
        distance = float(np.linalg.norm(np.maximum(0, np.maximum(a0 - b1, b0 - a1))))
        penalties.append(max(0.0, 1 - distance / preferred_gap) ** 2)
    return sum(penalties) / len(penalties) if penalties else 0.0


def compactness_score(placed, container):
    metrics = packing_metrics(placed)
    item_volume = sum(float(np.prod(p.box.size)) for p in placed)
    # Normalize by item volume so scores do not depend on inches vs centimeters.
    # A 0.15 weight means a complete temperature shortfall must save more than
    # 15% of item volume in empty space to be worthwhile.
    score = (metrics["empty_volume"] / max(item_volume, EPS)
             + container.temperature_weight * temperature_shortfall(placed, container.temperature_gap))
    height = max((p.position[2] + p.box.size[2] for p in placed), default=0)
    return (round(score, 8), -round(metrics["contact_area"], 8), height)


def pack_with_corners(container, boxes):
    """Rank all valid candidate positions by compactness instead of first fit.

    For up to five new items, try every insertion order. Larger inputs try
    four deterministic orders to bound runtime. This searches useful layouts
    but does not guarantee a globally optimal packing. All
    support, weight, and crushability checks remain mandatory. Temperature
    separation is a preference unless strict_temperature_gap is enabled.
    """
    boxes = list(boxes)
    preferred = sorted(boxes, key=lambda b: (b.crushability, -b.weight, -np.prod(b.size)))
    if len(boxes) <= 5:
        orders = itertools.permutations(preferred)
    else:
        orders = [preferred, sorted(boxes, key=lambda b: -b.weight),
                  sorted(boxes, key=lambda b: -np.prod(b.size)),
                  sorted(boxes, key=lambda b: (b.temperature != "ambient", b.crushability, -b.weight))]
    initial = list(container.placed)
    best = None
    best_key = None
    for order in orders:
        trial = Container(container.size, container.temperature_gap,
                          container.strict_temperature_gap, container.temperature_weight)
        for placed in initial:
            trial.add_box(placed.box, placed.position)
        rejected = []
        for box in order:
            candidates = []
            failures = Counter()
            for point in trial.points:
                for rotation in get_unique_rotations(box.size):
                    if box.upright_only and abs(rotation[2] - box.size[2]) > EPS:
                        continue
                    candidate = PlacedBox(box.rotated(rotation), point)
                    errors = trial.placement_errors(candidate)
                    if errors:
                        failures.update(errors)
                        continue
                    score = compactness_score(trial.placed + [candidate], trial)
                    candidates.append((score, candidate))
            if candidates:
                _, chosen = min(candidates, key=lambda entry: entry[0])
                trial.add_box(chosen.box, chosen.position)
            else:
                rejected.append((box, failures))
        # Packing every item takes priority over compactness.
        key = (len(rejected), *compactness_score(trial.placed, trial))
        if best_key is None or key < best_key:
            best, best_key = (trial, rejected), key
    trial, rejected = best
    container.placed = trial.placed
    container.points = trial.points
    container.unplaced = [box for box, _ in rejected]
    for box, failures in rejected:
        print(f"Could not place {box.name}. Rejected candidates: {dict(failures)}")
    return container.unplaced


def plot_boxes_improved(container, title="3D Packing: Weight, Crushability & Temperature"):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    shortfall = temperature_shortfall(container.placed, container.temperature_gap)
    if shortfall > EPS:
        title += "\nCompact mode: hot/frozen spacing preference relaxed"
    ax.set_title(title)
    for placed in container.placed:
        low, high = placed.get_bounds()
        vertices = list(itertools.product(*zip(low, high)))
        faces = [[vertices[i] for i in indices] for indices in
                 [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                  (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]]
        box = placed.box
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=COLORS[box.temperature], edgecolors="black", alpha=0.5))
        ax.text(*(low + box.size / 2),
                f"{box.name}\nw={box.weight:g}, crush={box.crushability}",
                ha="center", va="center", fontsize=8)
    ax.set(xlim=(0, container.size[0]), ylim=(0, container.size[1]),
           zlim=(0, container.size[2]), xlabel="X (Width)",
           ylabel="Y (Length)", zlabel="Z (Height)")
    ax.set_box_aspect(container.size)
    ax.legend(handles=[Patch(color=color, label=name.title()) for name, color in COLORS.items()])
    plt.tight_layout()
    plt.show()
    return fig


def run_original_demo():
    cont = Container([10, 6, 8], temperature_gap=1.0)
    items = [
        Box([6, 6, 3], "Stable base", weight=12, crushability=1),
        Box([3, 3, 3], "Frozen food", weight=4, crushability=2, temperature="frozen"),
        Box([3, 2, 2], "Hot food", weight=3, crushability=2, temperature="hot"),
        Box([4, 4, 2], "Soft produce", weight=2, crushability=4),
        Box([2, 2, 2], "Delicate", weight=0.5, crushability=5),
        Box([2, 5, 2], "Sturdy goods", weight=6, crushability=1),
        Box([3, 2, 4], "Packaged goods", weight=3, crushability=3),
    ]
    pack_with_corners(cont, items)
    for placed in cont.placed:
        box = placed.box
        print(f"{box.name}: position={placed.position.tolist()}, size={box.size.tolist()}, "
              f"weight={box.weight:g}, crushability={box.crushability}, temperature={box.temperature}")
    print(f"Placed {len(cont.placed)}/{len(items)} boxes.")
    if cont.placed:
        plot_boxes_improved(cont)


def grocery_test_cases():
    """Illustrative package sizes in inches and weights in pounds.

    Weights approximate the packaged item (including the milk jug).
    Crushability scores are assumptions, not measured material strengths.
    Chilled milk, eggs, and meat use 'ambient' because this model only
    distinguishes hot/frozen conflicts; this does not mean shelf-stable.
    Grocery packages stay upright but may turn around the vertical axis.
    """
    def grocery(size, name, weight, crushability, temperature="ambient"):
        return Box(size, name, weight, crushability, temperature, upright_only=True)

    return {
        1: ("Cereal, flour, and milk", Container([18, 9, 18]), [
            grocery([8, 3, 12], "Box of cereal", 1.1, 3),
            grocery([12, 8, 5], "10 lb bag of flour", 10.1, 1),
            grocery([6, 6, 10], "Gallon of milk", 8.6, 1),
        ]),
        2: ("Pasta, eggs, and deli meat", Container([13, 9, 8]), [
            grocery([8, 3, 2], "Box of pasta", 1.05, 3),
            grocery([12, 4, 3], "Carton of eggs (dozen)", 1.6, 5),
            grocery([12, 6, 2], "Box of deli meat (2 lb)", 2.1, 2),
        ]),
        3: ("Bread, ground meat, frozen vegetables, and hot bar", Container([18, 7, 9], temperature_gap=2), [
            grocery([8, 4, 4], "Loaf of bread (1 lb)", 1.0, 5),
            grocery([7, 5, 2], "Pound of ground meat", 1.05, 3),
            grocery([8, 6, 2], "Box of frozen vegetables (1 lb)", 1.1, 2, "frozen"),
            grocery([8, 6, 3], "1 lb box of hot bar food", 1.1, 3, "hot"),
        ]),
    }


def run_grocery_test(case_number, show_plot=True):
    title, container, items = grocery_test_cases()[case_number]
    print(f"\nGrocery case {case_number}: {title}")
    print(f"Container: {container.size.tolist()} inches; weights in pounds.")
    unplaced = pack_with_corners(container, items)
    assert not unplaced, f"Case {case_number}: some groceries did not fit."
    # Recheck each placement against all other boxes, independently of insertion order.
    for placed in container.placed:
        others = Container(container.size, container.temperature_gap,
                           container.strict_temperature_gap, container.temperature_weight)
        others.placed = [other for other in container.placed if other is not placed]
        assert not others.placement_errors(placed), f"Invalid placement: {placed.box.name}"
        box = placed.box
        print(f"  {box.name}: size={box.size.tolist()}, position={placed.position.tolist()}, "
              f"weight={box.weight:g}, crushability={box.crushability}, temperature={box.temperature}")
    assert any(placed.position[2] > EPS for placed in container.placed), "Expected stacking."
    metrics = packing_metrics(container.placed)
    print(f"Occupied envelope: {metrics['envelope_volume']:g} cubic inches; "
          f"internal empty space: {metrics['empty_volume']:g} cubic inches; "
          f"touching face area: {metrics['contact_area']:g} square inches.")
    shortfall = temperature_shortfall(container.placed, container.temperature_gap)
    if shortfall > EPS:
        print("Temperature preference relaxed for a tighter packing (strict mode is off).")
    print(f"PASS: all {len(items)} items fit, with stacking and all hard constraints satisfied.")
    if show_plot:
        plot_boxes_improved(container, title=f"Grocery case {case_number}: {title}")
    return container


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", "1", "2", "3", "original"], default="all",
                        help="Grocery scenario to run (default: all), or original demo.")
    parser.add_argument("--no-plot", action="store_true", help="Run grocery checks without a plot.")
    args = parser.parse_args()
    if args.case == "original":
        run_original_demo()
    else:
        for number in ([1, 2, 3] if args.case == "all" else [int(args.case)]):
            run_grocery_test(number, show_plot=not args.no_plot)
