"""Compare five packing loss profiles across five ten-grocery baskets.

Run with the robo virtual environment. Lower losses are better. Coordinates
are one-inch voxels and weights are pounds. Package dimensions and weights
are illustrative estimates for common grocery formats.

Within each basket, all profiles share a container, starting layout and search budget.
The search removes a suffix of placements and rebuilds it using different
item orders, positions and orientations. It occasionally accepts worse moves
to escape local minima, while retaining the best complete layout.
"""
import argparse
from dataclasses import dataclass
from math import exp
from pathlib import Path

import numpy as np


COMPONENTS = ("crush", "space", "temperature", "stability")
# Keep crushing at 6 and vary the secondary terms against the same reference.
PROFILES = {
    "Crush first": (6, 1, 1, 1),
    "More compactness": (6, 3, 1, 1),
    "More temperature buffer": (6, 1, 3, 1),
    "More stability": (6, 1, 1, 3),
    "More of all three": (6, 2, 2, 2),
}
# Keep the original starting-layout score so this sweep shares its baseline
# with the earlier experiment.
BASELINE_WEIGHTS = (1, 1, 1, 1)
SIZE = (18, 12, 15)
EPS = 1e-9


@dataclass(frozen=True)
class Grocery:
    name: str
    size: tuple
    weight: float
    crushability: int
    shape: str = "box"
    temperature: str = "ambient"


@dataclass(frozen=True)
class Placement:
    item: int
    origin: tuple
    turn: int = 0


GROCERIES = (
    Grocery("10 lb flour bag", (11, 8, 5), 10.1, 2, "rounded"),
    Grocery("Gallon of milk", (6, 6, 10), 8.6, 1, "jug"),
    Grocery("Canned tomatoes", (4, 4, 5), 1.9, 1, "oval"),
    Grocery("Pasta box", (8, 3, 2), 1.05, 2),
    Grocery("Cereal box", (8, 3, 12), 1.15, 3),
    Grocery("Egg carton", (11, 4, 3), 1.6, 5, "rounded"),
    Grocery("Sandwich bread", (9, 5, 4), 1.1, 5, "rounded"),
    Grocery("Peanut butter jar", (3, 3, 5), 1.2, 1, "oval"),
    Grocery("Frozen vegetables", (8, 6, 2), 1.1, 2, temperature="frozen"),
    Grocery("Hot bar clamshell", (8, 7, 3), 1.15, 3, "rounded", "hot"),
)


def shape_mask(item):
    """Rasterize upright boxes, cylinders, rounded packs and a tapered jug."""
    x, y, z = np.indices(item.size, dtype=float)
    u = abs(2 * (x + 0.5) / item.size[0] - 1)
    v = abs(2 * (y + 0.5) / item.size[1] - 1)
    if item.shape == "oval":
        return u**2 + v**2 <= 1
    if item.shape == "rounded":
        return u**4 + v**4 <= 1
    if item.shape == "jug":
        limit = np.where(z < item.size[2] - 2, 1, np.where(z < item.size[2] - 1, 0.68, 0.36))
        return u**4 + v**4 <= limit**4
    return np.ones(item.size, dtype=bool)


SPINACH = Grocery("Bag of baby spinach", (9, 7, 3), 0.4, 5, "rounded")
BERRIES = Grocery("Raspberry punnet", (6, 4, 2), 0.45, 5, "rounded")
TOMATOES = Grocery("Grape tomato clamshell", (7, 4, 3), 0.8, 4, "rounded")
CHIPS = Grocery("Bag of potato chips", (10, 7, 4), 0.55, 5, "rounded")
BUNS = Grocery("Bag of brioche buns", (9, 6, 4), 0.85, 5, "rounded")
PEACHES = Grocery("Two peaches in a tray", (7, 4, 3), 0.85, 4, "rounded")
YOGURT = Grocery("Tub of Greek yogurt", (5, 5, 5), 2.1, 2, "oval")
OIL = Grocery("Bottle of olive oil", (4, 4, 10), 2.5, 1, "oval")
RICE = Grocery("5 lb bag of rice", (10, 7, 3), 5.1, 2, "rounded")
SODA = Grocery("Six-pack of soda", (8, 6, 5), 5.0, 1)
SOUP = Grocery("Hot soup tub", (5, 5, 5), 1.4, 3, "oval", "hot")
BURRITO = Grocery("Hot burrito in a box", (8, 4, 3), 1.0, 3, temperature="hot")
CHICKEN = Grocery("Hot rotisserie chicken", (10, 8, 5), 2.5, 3, "rounded", "hot")
WAFFLES = Grocery("Frozen waffles", (8, 5, 3), 0.9, 2, temperature="frozen")
ICE_CREAM = Grocery("Pint of ice cream", (4, 4, 5), 1.1, 3, "oval", "frozen")
FROZEN_FRUIT = Grocery("Bag of frozen fruit", (8, 6, 3), 1.2, 3, "rounded", "frozen")

# Keep the original basket as a reference. Others stress different tradeoffs.
BASKETS = {
    "reference": ("Original mixed basket", GROCERIES),
    "produce": ("Fragile produce and eggs", (
        SPINACH, BERRIES, TOMATOES, PEACHES, GROCERIES[5], GROCERIES[6],
        GROCERIES[2], YOGURT, FROZEN_FRUIT, SOUP,
    )),
    "heavy": ("Heavy staples with delicate groceries", (
        GROCERIES[0], GROCERIES[1], OIL, RICE, GROCERIES[5], SPINACH,
        BERRIES, GROCERIES[3], WAFFLES, BURRITO,
    )),
    "bulky": ("Bulky snacks and bakery", (
        CHIPS, BUNS, GROCERIES[4], GROCERIES[6], SPINACH, GROCERIES[5],
        GROCERIES[7], SODA, ICE_CREAM, BURRITO,
    )),
    "dinner": ("Hot dinner and frozen dessert", (
        CHICKEN, ICE_CREAM, SPINACH, GROCERIES[5], BUNS, TOMATOES,
        GROCERIES[2], GROCERIES[3], OIL, YOGURT,
    )),
}
COLORS = ("#b18c65", "#6c8ebf", "#a75045", "#e7b44d", "#6c9a66",
          "#998ac0", "#d09763", "#8e5e43", "#55a5a1", "#d188ab")


class Basket:
    def __init__(self, title, groceries):
        self.title = title
        self.groceries = groceries
        self.masks = tuple((mask, np.rot90(mask, axes=(0, 1)))
                           for mask in map(shape_mask, groceries))
        self.turns = tuple((0,) if np.array_equal(*masks) else (0, 1)
                           for masks in self.masks)
        self.volumes = np.array([masks[0].sum() for masks in self.masks])
        self.weights = np.array([item.weight for item in groceries])
        self.densities = self.weights / self.volumes
        self.hot = tuple(i for i, item in enumerate(groceries) if item.temperature == "hot")
        self.frozen = tuple(i for i, item in enumerate(groceries) if item.temperature == "frozen")


class Layout:
    def __init__(self, basket):
        self.basket = basket
        self.grid = np.full(SIZE, -1, dtype=np.int8)
        self.placements = []
        self.stack_loss = 0.0
        self.worst_stack_loss = 0.0
        self.balance_loss = 0.0
        self.volume = 0
        self.low = np.array(SIZE)
        self.high = np.zeros(3, dtype=int)

    def inspect(self, placement):
        """Check bounds, collisions and minimum support. Return scoring data."""
        mask = self.basket.masks[placement.item][placement.turn]
        x, y, z = placement.origin
        dx, dy, dz = mask.shape
        if min(placement.origin) < 0 or x + dx > SIZE[0] or y + dy > SIZE[1] or z + dz > SIZE[2]:
            return None
        region = self.grid[x:x+dx, y:y+dy, z:z+dz]
        if np.any(region[mask] >= 0):
            return None
        if z == 0:
            return 0.0, 0.0
        base = mask[:, :, 0]
        below = self.grid[x:x+dx, y:y+dy, z-1]
        support = base & (below >= 0)
        coverage = support.sum() / base.sum()
        if coverage < 0.5:
            return None
        center = np.argwhere(mask).mean(axis=0)[:2]
        supported_cells = np.argwhere(support)
        # At least one support cell must be under the central region. The
        # support fraction and eccentricity then distinguish valid choices.
        if np.min(np.max(abs(supported_cells - center), axis=1)) > 1.5:
            return None
        imbalance = np.linalg.norm((supported_cells.mean(axis=0) - center) / np.array([dx, dy]))
        balance = min(1.0, (1 - coverage) + 2 * imbalance)
        item = self.basket.groceries[placement.item]
        supporters, counts = np.unique(below[support], return_counts=True)
        stacking = 0.0
        for supporter, count in zip(supporters, counts):
            lower = self.basket.groceries[supporter]
            share = count / support.sum()
            crush = max(0, lower.crushability - item.crushability) / 4
            load = min(1, item.weight * share / lower.weight)
            # Load and package density amplify pressure on fragile groceries.
            density = min(1, self.basket.densities[placement.item] / (2 * self.basket.densities[supporter]))
            susceptibility = (lower.crushability - 1) / 4
            stacking += share * (0.55 * crush + 0.45 * susceptibility * load * density)
        return stacking, balance

    def add(self, placement, costs):
        item = placement.item
        mask = self.basket.masks[item][placement.turn]
        origin = np.array(placement.origin)
        end = origin + mask.shape
        region = tuple(slice(a, b) for a, b in zip(origin, end))
        self.grid[region][mask] = item
        self.placements.append(placement)
        self.stack_loss += costs[0]
        self.worst_stack_loss = max(self.worst_stack_loss, costs[0])
        center_height = origin[2] + np.argwhere(mask)[:, 2].mean() + 0.5
        self.balance_loss += self.basket.weights[item] * (0.8 * costs[1] + 0.2 * center_height / SIZE[2])
        self.volume += self.basket.volumes[item]
        self.low = np.minimum(self.low, origin)
        self.high = np.maximum(self.high, end)

    def copy(self):
        other = Layout(self.basket)
        other.grid = self.grid.copy()
        other.placements = self.placements.copy()
        other.stack_loss = self.stack_loss
        other.worst_stack_loss = self.worst_stack_loss
        other.balance_loss = self.balance_loss
        other.volume = self.volume
        other.low = self.low.copy()
        other.high = self.high.copy()
        return other

    def components(self):
        occupied = self.grid >= 0
        if not self.placements:
            return np.zeros(4)
        envelope = np.prod(self.high - self.low)
        # Side contacts between packages reduce room for lateral movement.
        faces = 0
        contacts = 0
        for a, b in ((self.grid[1:], self.grid[:-1]), (self.grid[:, 1:], self.grid[:, :-1])):
            faces += np.count_nonzero((a >= 0) != (b >= 0))
            contacts += np.count_nonzero((a >= 0) & (b >= 0) & (a != b))
        side_air = faces / max(1, faces + 2 * contacts)
        space = 0.75 * (1 - self.volume / envelope) + 0.25 * side_air
        total_weight = sum(self.basket.weights[p.item] for p in self.placements)
        # Keep one risky stack from disappearing into the basket average.
        crush = 0.65 * self.worst_stack_loss + 0.35 * self.stack_loss / len(self.placements)
        return np.array([crush, space,
                         temperature_loss(self.grid, occupied, self.basket), self.balance_loss / total_weight])


def temperature_loss(grid, occupied, basket):
    """Average the exposure score across every hot/frozen pair present."""
    hot_items = [np.argwhere(grid == i) for i in basket.hot]
    cold_items = [np.argwhere(grid == i) for i in basket.frozen]
    losses = [pair_temperature_loss(hot, cold, occupied)
              for hot in hot_items for cold in cold_items if len(hot) and len(cold)]
    return float(np.mean(losses)) if losses else 0.0


def pair_temperature_loss(hot, cold, occupied):
    """Score exposed hot/frozen proximity. Intervening packages earn credit."""
    best_buffer = 0.0
    for axis in range(3):
        left, right = (hot, cold) if hot[:, axis].min() <= cold[:, axis].min() else (cold, hot)
        lo = left[:, axis].max() + 1
        hi = right[:, axis].min()
        if lo >= hi:
            continue
        projection = [i for i in range(3) if i != axis]
        left_face = np.zeros(tuple(SIZE[i] for i in projection), dtype=bool)
        right_face = left_face.copy()
        left_face[tuple(left[:, projection].T)] = True
        right_face[tuple(right[:, projection].T)] = True
        facing = left_face & right_face
        if not facing.any():
            continue
        between = [slice(None)] * 3
        between[axis] = slice(lo, hi)
        screened = occupied[tuple(between)].any(axis=axis)
        best_buffer = max(best_buffer, np.count_nonzero(facing & screened) / facing.sum())
    # Distance gives a small benefit even when no package fits between them.
    # Coverage is the main benefit and direct contact has the largest loss.
    differences = abs(hot[:, None, :] - cold[None, :, :])
    surface_gap = np.linalg.norm(np.maximum(0, differences - 1), axis=2).min()
    return (1 - best_buffer) * (0.75 + 0.25 / (1 + surface_gap))


def candidate_origins(layout, item, turn, rng):
    """Try face-aligned origins and a few off-grid-of-corners alternatives."""
    dx, dy, dz = layout.basket.masks[item][turn].shape
    xmax, ymax = SIZE[0] - dx, SIZE[1] - dy
    xs, ys, zs = {0, xmax}, {0, ymax}, {0}
    anchors = set()
    for other in layout.placements:
        x, y, z = other.origin
        wx, wy, wz = layout.basket.masks[other.item][other.turn].shape
        xs.update((x, x + wx, x - dx, x + wx - dx))
        ys.update((y, y + wy, y - dy, y + wy - dy))
        zs.add(z + wz)
        anchors.update(((x + wx, y, 0), (x, y + wy, 0), (x - dx, y, 0),
                        (x, y - dy, 0), (x, y, z + wz),
                        (x + (wx - dx) // 2, y + (wy - dy) // 2, z + wz)))
    anchors.add((0, 0, 0))
    xs = sorted(x for x in xs if 0 <= x <= xmax)
    ys = sorted(y for y in ys if 0 <= y <= ymax)
    zs = sorted(z for z in zs if z + dz <= SIZE[2])
    for _ in range(24):
        anchors.add((int(rng.choice(xs)), int(rng.choice(ys)), int(rng.choice(zs))))
    for _ in range(8):
        anchors.add((int(rng.integers(xmax + 1)), int(rng.integers(ymax + 1)), 0))
    return sorted(anchors)


def options(layout, item, weights, rng):
    ranked = []
    for turn in layout.basket.turns[item]:
        for origin in candidate_origins(layout, item, turn, rng):
            placement = Placement(item, origin, turn)
            costs = layout.inspect(placement)
            if costs is None:
                continue
            trial = layout.copy()
            trial.add(placement, costs)
            value = float(trial.components() @ weights)
            ranked.append((value, placement, costs))
    return sorted(ranked, key=lambda entry: entry[0])


def rebuild(prefix, remaining, weights, rng, budget, explore=False):
    """Bounded depth-first backtracking through the top three placements."""
    visited = 0

    def visit(layout, depth):
        nonlocal visited
        if depth == len(remaining):
            return layout
        if visited >= budget:
            return None
        visited += 1
        candidates = options(layout, remaining[depth], weights, rng)[:3]
        if explore and len(candidates) > 1 and rng.random() < 0.35:
            rng.shuffle(candidates)
        for _, placement, costs in candidates:
            trial = layout.copy()
            trial.add(placement, costs)
            result = visit(trial, depth + 1)
            if result is not None:
                return result
        return None

    return visit(prefix, 0)


def replay(basket, placements):
    layout = Layout(basket)
    for placement in placements:
        costs = layout.inspect(placement)
        if costs is None:
            raise ValueError(f"Invalid placement for {basket.groceries[placement.item].name}")
        layout.add(placement, costs)
    return layout


def normalize(weights):
    values = np.asarray(weights, dtype=float)
    return values / values.sum()


def baseline(basket, seed):
    order = sorted(range(len(basket.groceries)),
                   key=lambda i: (basket.groceries[i].crushability, -basket.weights[i]))
    rng = np.random.default_rng(seed)
    weights = normalize(BASELINE_WEIGHTS)
    for attempt in range(8):
        result = rebuild(Layout(basket), order, weights, rng, budget=30)
        if result is not None:
            return result
        rng.shuffle(order)
    raise RuntimeError("Could not find a complete baseline packing.")


def optimize(initial, weights, seed, iterations):
    rng = np.random.default_rng(seed)
    current = best = initial
    current_score = best_score = float(initial.components() @ weights)
    complete_moves = 0
    for step in range(iterations):
        # Regularly reopen the entire stack so early greedy choices can change.
        cut = 0 if step % 8 == 0 else int(rng.integers(0, len(current.placements) - 1))
        prefix = replay(current.basket, current.placements[:cut])
        remaining = [p.item for p in current.placements[cut:]]
        rng.shuffle(remaining)
        proposal = rebuild(prefix, remaining, weights, rng, budget=len(remaining) + 8, explore=True)
        if proposal is None:
            continue
        complete_moves += 1
        score = float(proposal.components() @ weights)
        if score < best_score:
            best, best_score = proposal, score
        cooling = 0.025 * (1 - step / iterations) + 0.002
        if score <= current_score or rng.random() < exp((current_score - score) / cooling):
            current, current_score = proposal, score
    return best, complete_moves


def plot_results(basket, results, destination):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(17, 10), layout="constrained")
    fig.suptitle(basket.title, fontsize=20)
    palette = [("#ee684d" if g.temperature == "hot" else "#47b3d7"
                if g.temperature == "frozen" else COLORS[i])
               for i, g in enumerate(basket.groceries)]
    for index, (name, layout, _, _) in enumerate(results, 1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        colors = np.empty(SIZE, dtype=object)
        colors[:] = "white"
        for item, color in enumerate(palette):
            colors[layout.grid == item] = color
        ax.voxels(layout.grid >= 0, facecolors=colors, edgecolor="#444444", linewidth=0.12)
        ax.set(xlim=(0, SIZE[0]), ylim=(0, SIZE[1]), zlim=(0, SIZE[2]),
               xlabel="Width", ylabel="Depth", zlabel="Height")
        ax.set_box_aspect(SIZE)
        ax.view_init(elev=27, azim=-65)
        terms = layout.components()
        ax.set_title(f"{name}\nCrush {terms[0]:.3f} · space {terms[1]:.3f} · "
                     f"temp {terms[2]:.3f} · stability {terms[3]:.3f}", fontsize=10)
    legend = fig.add_subplot(2, 3, 6)
    legend.axis("off")
    legend.legend(handles=[Patch(color=c, label=g.name) for c, g in zip(palette, basket.groceries)],
                  loc="center", title="Groceries · axis units in inches", frameon=False)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_basket(basket, seed, iterations):
    print(f"\n{basket.title} · container {SIZE} in · density in lb per occupied cubic inch")
    for index, grocery in enumerate(basket.groceries):
        print(f"  {grocery.name:25s} {str(grocery.size):13s} {grocery.shape:8s} "
              f"{grocery.weight:5.2f} lb  density {basket.densities[index]:.3f}  "
              f"crush {grocery.crushability}  {grocery.temperature}")
    start = baseline(basket, seed)
    results = []
    for name, raw in PROFILES.items():
        weights = normalize(raw)
        layout, completed = optimize(start, weights, seed + 1, iterations)
        # Validate the complete placement sequence through the ordinary geometry checks.
        assert len({p.item for p in layout.placements}) == len(basket.groceries)
        assert np.array_equal(replay(basket, layout.placements).grid, layout.grid)
        terms = layout.components()
        stacked = sum(p.origin[2] > 0 for p in layout.placements)
        results.append((name, layout, weights, completed))
        print(f"\n{name}  weights={raw}  complete rebuilds={completed}/{iterations}  "
              f"stacked={stacked}/{len(basket.groceries)}")
        print(f"  Weighted loss {start.components() @ weights:.4f} -> {terms @ weights:.4f}")
        print("  " + "  ".join(f"{key}={value:.3f}" for key, value in zip(COMPONENTS, terms)))
        for p in layout.placements:
            print(f"  {basket.groceries[p.item].name:23s} {str(p.origin):14s} turn={90*p.turn}°")
    print("\nCross-score each layout under all profiles (columns in profile order):")
    for name, layout, _, _ in results:
        print(f"  {name:20s} " + "  ".join(f"{layout.components() @ weights:.4f}"
                                                 for _, _, weights, _ in results))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=64, help="Rebuild attempts per profile.")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--basket", choices=["all", *BASKETS], default="all")
    parser.add_argument("--plot", type=Path,
                        help="Optional image path. A basket suffix is added when running all five.")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    selected = BASKETS if args.basket == "all" else {args.basket: BASKETS[args.basket]}
    by_basket = {}
    for key, (title, groceries) in selected.items():
        basket = Basket(title, groceries)
        results = run_basket(basket, args.seed, args.iterations)
        by_basket[key] = {name: layout.components() for name, layout, _, _ in results}
        if args.plot:
            destination = args.plot
            if len(selected) > 1:
                destination = destination.with_name(f"{destination.stem}_{key}{destination.suffix}")
            plot_results(basket, results, destination)

    if len(by_basket) > 1:
        print("\nMean component losses across baskets (lower is better in each column):")
        print(f"  {'Profile':21s} " + "  ".join(f"{term:>11s}" for term in COMPONENTS))
        for name in PROFILES:
            mean = np.mean([scores[name] for scores in by_basket.values()], axis=0)
            print(f"  {name:21s} " + "  ".join(f"{value:11.3f}" for value in mean))


if __name__ == "__main__":
    main()
