"""
Sub-Goal Imputation for MiniGrid RL trajectories.

Algorithm
---------
1. Derive a sub-goal *vocabulary* from the environment semantics:
       NAVIGATE_TO(obj)  – agent is adjacent to an object
       PICKUP(key_color) – key of that color is in the agent's inventory
       OPEN(door_color)  – door of that color is open
       REACH_GOAL        – agent is standing on the goal cell
       EXPLORE           – agent visited a cell not yet visited

2. Scan each trajectory for *completion events*: timesteps where a sub-goal
   transitions from False → True for the first time.

3. Segment the trajectory by those events; every timestep in a segment
   is labelled with that segment's completing sub-goal.

4. Validate each segment with a BFS efficiency score:
       efficiency = BFS_shortest_path_length / segment_length
   Values close to 1 indicate near-optimal behavior within that segment;
   values << 1 indicate detours.

Visualisations
--------------
* Horizontal bar (Gantt) chart of sub-goal segments over time.
* DAG of sub-goal prerequisite relations.

MiniGrid grid encoding (3D array, shape (W, H, 3)):
    channel 0 – OBJECT_TO_IDX
    channel 1 – COLOR_TO_IDX
    channel 2 – state (door: 0=open,1=closed,2=locked; key: 0=none)

Object indices (minigrid.core.constants):
    OBJECT_TO_IDX = {
        'unseen': 0, 'empty': 1, 'wall': 2, 'floor': 3,
        'door': 4, 'key': 5, 'ball': 6, 'box': 7, 'goal': 8, 'lava': 9, 'agent': 10
    }
COLOR_TO_IDX = {'red':0,'green':1,'blue':2,'purple':3,'yellow':4,'grey':5}
"""

from __future__ import annotations

import collections
import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# MiniGrid constants
try:
    from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, IDX_TO_COLOR, IDX_TO_OBJECT
except ImportError:
    # Fallback definitions matching minigrid>=2.3
    OBJECT_TO_IDX: Dict[str, int] = {
        "unseen": 0, "empty": 1, "wall": 2, "floor": 3,
        "door": 4, "key": 5, "ball": 6, "box": 7,
        "goal": 8, "lava": 9, "agent": 10,
    }
    COLOR_TO_IDX: Dict[str, int] = {
        "red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5,
    }
    IDX_TO_COLOR: Dict[int, str] = {v: k for k, v in COLOR_TO_IDX.items()}
    IDX_TO_OBJECT: Dict[int, str] = {v: k for k, v in OBJECT_TO_IDX.items()}

# Convenience aliases
_OBJ = OBJECT_TO_IDX
_COL = COLOR_TO_IDX

# Door state values (minigrid convention)
DOOR_OPEN   = 0
DOOR_CLOSED = 1
DOOR_LOCKED = 2

# Action indices (MiniGrid default)
ACTION_LEFT    = 0
ACTION_RIGHT   = 1
ACTION_FORWARD = 2
ACTION_PICKUP  = 3
ACTION_DROP    = 4
ACTION_TOGGLE  = 5
ACTION_DONE    = 6


# ---------------------------------------------------------------------------
# SubGoal dataclass
# ---------------------------------------------------------------------------

@dataclass
class SubGoal:
    """
    A symbolic sub-goal with an associated predicate function.

    Attributes
    ----------
    name : str
        Human-readable label, e.g. "PICKUP(yellow)" or "OPEN(blue)".
    predicate : Callable[[np.ndarray, Tuple[int,int]], bool]
        Function(grid_state, agent_pos) → bool.
        grid_state : np.ndarray  shape (W, H, 3)
        agent_pos  : (col, row) tuple (MiniGrid convention: x=col, y=row)
    category : str
        One of: NAVIGATE_TO | PICKUP | OPEN | REACH_GOAL | EXPLORE
    color : Optional[str]
        Relevant color for PICKUP/OPEN sub-goals.
    obj_type : Optional[str]
        Relevant object type for NAVIGATE_TO.
    """

    name: str
    predicate: Callable[[np.ndarray, Tuple[int, int]], bool]
    category: str
    color: Optional[str] = None
    obj_type: Optional[str] = None

    def check(self, grid_state: np.ndarray, agent_pos: Tuple[int, int]) -> bool:
        """Evaluate the predicate for a given grid state and agent position."""
        return self.predicate(grid_state, agent_pos)

    def __repr__(self) -> str:
        return f"SubGoal({self.name})"

    # Make SubGoal hashable so it can be used in sets / as graph nodes
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SubGoal):
            return self.name == other.name
        return NotImplemented


# ---------------------------------------------------------------------------
# Grid helper functions
# ---------------------------------------------------------------------------

def _find_objects(
    grid: np.ndarray, obj_type: str, color: Optional[str] = None
) -> List[Tuple[int, int]]:
    """
    Find all cells of a given object type (and optionally color).

    Parameters
    ----------
    grid : np.ndarray  shape (W, H, 3)
    obj_type : str
    color : str, optional

    Returns
    -------
    List of (col, row) positions.
    """
    type_idx = _OBJ.get(obj_type, -1)
    mask = grid[:, :, 0] == type_idx
    if color is not None:
        col_idx = _COL.get(color, -1)
        mask = mask & (grid[:, :, 1] == col_idx)
    cols, rows = np.where(mask)
    return list(zip(cols.tolist(), rows.tolist()))


def _agent_adjacent(
    agent_pos: Tuple[int, int], target_pos: Tuple[int, int]
) -> bool:
    """True when agent_pos is orthogonally adjacent to target_pos."""
    dx = abs(agent_pos[0] - target_pos[0])
    dy = abs(agent_pos[1] - target_pos[1])
    return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)


def _agent_carrying_key(
    grid: np.ndarray, agent_pos: Tuple[int, int], color: str
) -> bool:
    """
    Infer whether the agent is carrying a key of the given color.

    In fully-observed MiniGrid, the agent's inventory is not directly encoded
    in the grid array.  We proxy this by checking that NO key of the given
    color exists anywhere on the grid (it was picked up) AND at least one
    door of that color is still present (so the key is relevant).

    More precisely: key disappears from the grid when picked up; if the key
    cell no longer appears AND the mission requires that color, the agent
    is carrying it.  Since we lack inventory info directly, we use absence
    of the key on the grid as the signal.
    """
    col_idx = _COL.get(color, -1)
    key_positions = _find_objects(grid, "key", color=color)
    return len(key_positions) == 0


def _door_is_open(grid: np.ndarray, color: str) -> bool:
    """True when all doors of the given color are open (state==0)."""
    col_idx = _COL.get(color, -1)
    type_idx = _OBJ["door"]
    mask = (grid[:, :, 0] == type_idx) & (grid[:, :, 1] == col_idx)
    if not np.any(mask):
        return False  # No such door exists
    return np.all(grid[:, :, 2][mask] == DOOR_OPEN)


def _agent_at_goal(grid: np.ndarray, agent_pos: Tuple[int, int]) -> bool:
    """True when a goal cell exists and the agent is on it."""
    goal_positions = _find_objects(grid, "goal")
    return agent_pos in goal_positions


def _extract_agent_pos(grid: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Extract agent position from the encoded grid.
    The agent cell has object type 10 (OBJECT_TO_IDX['agent']).
    Returns (col, row) or None if not found.
    """
    agent_type = _OBJ.get("agent", 10)
    cols, rows = np.where(grid[:, :, 0] == agent_type)
    if len(cols) == 0:
        return None
    return (int(cols[0]), int(rows[0]))


# ---------------------------------------------------------------------------
# BFS helper
# ---------------------------------------------------------------------------

def _bfs_distance(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal_fn: Callable[[Tuple[int, int]], bool],
) -> int:
    """
    BFS on the grid to find the shortest path from *start* to any cell
    satisfying *goal_fn*, treating walls and closed/locked doors as impassable.

    Parameters
    ----------
    grid : np.ndarray  shape (W, H, 3)
    start : (col, row)
    goal_fn : callable  position -> bool

    Returns
    -------
    int  – number of steps, or -1 if unreachable.
    """
    W, H, _ = grid.shape
    wall_idx = _OBJ["wall"]
    door_idx = _OBJ["door"]

    visited = set()
    queue: collections.deque[Tuple[Tuple[int, int], int]] = collections.deque()
    queue.append((start, 0))
    visited.add(start)

    while queue:
        pos, dist = queue.popleft()
        if goal_fn(pos):
            return dist
        col, row = pos
        for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nc, nr = col + dc, row + dr
            if not (0 <= nc < W and 0 <= nr < H):
                continue
            if (nc, nr) in visited:
                continue
            cell_obj = grid[nc, nr, 0]
            cell_state = grid[nc, nr, 2]
            # Walls are impassable
            if cell_obj == wall_idx:
                continue
            # Closed or locked doors are impassable
            if cell_obj == door_idx and cell_state != DOOR_OPEN:
                continue
            visited.add((nc, nr))
            queue.append(((nc, nr), dist + 1))

    return -1  # Unreachable


# ---------------------------------------------------------------------------
# SubGoalImputation
# ---------------------------------------------------------------------------

class SubGoalImputation:
    """
    Sub-Goal Imputation: derives a vocabulary of symbolic sub-goals from
    a MiniGrid environment, detects completion events along a trajectory,
    segments the trajectory, and validates each segment with BFS efficiency.

    Parameters
    ----------
    env_name : str
        MiniGrid environment ID (used to infer the relevant sub-goal types).
    """

    def __init__(self, env_name: str) -> None:
        self.env_name = env_name
        self.subgoal_vocab: List[SubGoal] = []

        # Color palette for plotting
        self._palette: Dict[str, str] = {
            "NAVIGATE_TO": "#4C9BE8",
            "PICKUP":      "#F4A259",
            "OPEN":        "#6ABF69",
            "REACH_GOAL":  "#E06C75",
            "EXPLORE":     "#C678DD",
        }

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def extract_subgoal_vocab(self, env) -> List[SubGoal]:
        """
        Instantiate the environment, run a reset, and inspect the initial
        grid to build the environment-specific sub-goal vocabulary.

        Parameters
        ----------
        env : gymnasium.Env
            A FullyObsWrapper + ImgObsWrapper-wrapped MiniGrid env.
            The environment should be reset before calling this method,
            or we reset it here.

        Returns
        -------
        List[SubGoal]
        """
        # Reset to get an initial grid
        obs, _ = env.reset()
        grid = env.unwrapped.grid.encode()  # (W, H, 3)
        vocab: List[SubGoal] = []

        # -- REACH_GOAL (universal) --------------------------------------
        vocab.append(SubGoal(
            name="REACH_GOAL",
            predicate=lambda g, pos: _agent_at_goal(g, pos),
            category="REACH_GOAL",
        ))

        # -- EXPLORE (universal) ----------------------------------------
        # Maintained externally by the imputer; predicate always False here
        # (completion is detected when a new cell is first visited).
        vocab.append(SubGoal(
            name="EXPLORE",
            predicate=lambda g, pos: False,  # handled specially
            category="EXPLORE",
        ))

        # -- PICKUP sub-goals: one per key color present in the map ------
        key_colors_present = set()
        for col, row in _find_objects(grid, "key"):
            col_idx = int(grid[col, row, 1])
            color = IDX_TO_COLOR.get(col_idx, "unknown")
            key_colors_present.add(color)

        for color in sorted(key_colors_present):
            c = color  # capture for lambda
            vocab.append(SubGoal(
                name=f"PICKUP({c})",
                predicate=lambda g, pos, _c=c: _agent_carrying_key(g, pos, _c),
                category="PICKUP",
                color=c,
            ))

        # -- OPEN sub-goals: one per door color present in the map -------
        door_colors_present = set()
        for col, row in _find_objects(grid, "door"):
            col_idx = int(grid[col, row, 1])
            color = IDX_TO_COLOR.get(col_idx, "unknown")
            door_colors_present.add(color)

        for color in sorted(door_colors_present):
            c = color
            vocab.append(SubGoal(
                name=f"OPEN({c})",
                predicate=lambda g, pos, _c=c: _door_is_open(g, _c),
                category="OPEN",
                color=c,
            ))

        # -- NAVIGATE_TO sub-goals: one per interesting object type -------
        interesting_types = ["key", "door", "goal", "box", "ball"]
        nav_pairs: set = set()

        for obj_type in interesting_types:
            positions = _find_objects(grid, obj_type)
            for obj_col, obj_row in positions:
                col_idx = int(grid[obj_col, obj_row, 1])
                color = IDX_TO_COLOR.get(col_idx, "")
                label = f"{obj_type}_{color}" if color else obj_type
                if label in nav_pairs:
                    continue
                nav_pairs.add(label)

                oc, or_ = obj_col, obj_row  # capture
                vocab.append(SubGoal(
                    name=f"NAVIGATE_TO({label})",
                    predicate=lambda g, pos, _c=oc, _r=or_: _agent_adjacent(pos, (_c, _r)),
                    category="NAVIGATE_TO",
                    obj_type=obj_type,
                    color=color if color else None,
                ))

        self.subgoal_vocab = vocab
        return vocab

    # ------------------------------------------------------------------
    # Core imputation algorithm
    # ------------------------------------------------------------------

    def impute(
        self, trajectory: Dict[str, Any]
    ) -> Tuple[
        List[Optional[SubGoal]],          # goal_assignments
        List[Tuple[int, int, SubGoal]],   # segments: (t_start, t_end, subgoal)
        List[float],                       # efficiencies per segment
    ]:
        """
        Run the sub-goal imputation algorithm on a single trajectory.

        Parameters
        ----------
        trajectory : dict
            Expected keys:
                grid_state_seq : List[np.ndarray]  shape (W, H, 3) per step
                obs_seq        : List[np.ndarray]  raw observations
                action_seq     : List[int]

        Returns
        -------
        goal_assignments : List[Optional[SubGoal]]
            Sub-goal label per timestep (None if no sub-goal completed in segment).
        segments : List[(t_start, t_end, SubGoal)]
            Each element represents a contiguous segment ending with a
            sub-goal completion event.
        efficiencies : List[float]
            BFS efficiency score per segment in [0, 1] (or -1 if BFS fails).
        """
        grid_states: List[np.ndarray] = trajectory["grid_state_seq"]
        T = len(grid_states)

        if T == 0:
            return [], [], []

        # ------------------------------------------------------------------
        # Phase 1: Detect completion events
        # Each element: (t, SubGoal)
        # ------------------------------------------------------------------
        completion_events: List[Tuple[int, SubGoal]] = []

        # Track which sub-goals have already fired to avoid re-triggering
        fired: set = set()

        # Separate tracking for EXPLORE (fires on each new cell visit)
        visited_cells: set = set()
        explore_sg = next((sg for sg in self.subgoal_vocab if sg.category == "EXPLORE"), None)
        non_explore_vocab = [sg for sg in self.subgoal_vocab if sg.category != "EXPLORE"]

        # Previous state of each sub-goal predicate
        prev_state: Dict[str, bool] = {}
        for sg in non_explore_vocab:
            agent_pos = _extract_agent_pos(grid_states[0])
            prev_state[sg.name] = sg.check(grid_states[0], agent_pos) if agent_pos else False

        for t in range(T):
            grid = grid_states[t]
            agent_pos = _extract_agent_pos(grid)
            if agent_pos is None:
                continue

            # EXPLORE: fires on first visit to new cell
            if explore_sg is not None and agent_pos not in visited_cells:
                visited_cells.add(agent_pos)
                if t > 0:  # Skip t=0 (initial position is not "explored")
                    completion_events.append((t, explore_sg))

            # Check all other sub-goals for False -> True transitions
            for sg in non_explore_vocab:
                if sg.name in fired:
                    continue
                curr = sg.check(grid, agent_pos)
                if curr and not prev_state.get(sg.name, False):
                    completion_events.append((t, sg))
                    fired.add(sg.name)
                prev_state[sg.name] = curr

        # ------------------------------------------------------------------
        # Phase 2: Build segments
        # ------------------------------------------------------------------
        # Sort events by time; multiple events at same t are all recorded
        completion_events.sort(key=lambda x: x[0])

        segments: List[Tuple[int, int, SubGoal]] = []
        goal_assignments: List[Optional[SubGoal]] = [None] * T

        seg_start = 0
        for t_event, sg in completion_events:
            t_end = t_event
            if t_end >= T:
                t_end = T - 1
            segments.append((seg_start, t_end, sg))
            for t in range(seg_start, t_end + 1):
                goal_assignments[t] = sg
            seg_start = t_end + 1

        # Final segment (after last event, if any steps remain)
        if seg_start < T:
            # Label with REACH_GOAL if that's in vocab, else None
            reach_sg = next(
                (sg for sg in self.subgoal_vocab if sg.category == "REACH_GOAL"), None
            )
            last_sg = reach_sg
            segments.append((seg_start, T - 1, last_sg))
            for t in range(seg_start, T):
                goal_assignments[t] = last_sg

        # ------------------------------------------------------------------
        # Phase 3: BFS efficiency per segment
        # ------------------------------------------------------------------
        efficiencies: List[float] = []
        for t_start, t_end, sg in segments:
            seg_len = t_end - t_start + 1
            if seg_len <= 0 or sg is None:
                efficiencies.append(1.0)
                continue

            start_grid = grid_states[t_start]
            agent_start = _extract_agent_pos(start_grid)
            if agent_start is None:
                efficiencies.append(-1.0)
                continue

            # Define goal function based on sub-goal category
            end_grid = grid_states[t_end]
            if sg.category == "REACH_GOAL":
                goal_positions = _find_objects(start_grid, "goal")
                if not goal_positions:
                    efficiencies.append(-1.0)
                    continue
                gp = goal_positions[0]
                goal_fn: Callable[[Tuple[int, int]], bool] = lambda pos, _gp=gp: pos == _gp
            elif sg.category == "NAVIGATE_TO":
                # Navigate adjacent to the target object
                if sg.obj_type is None:
                    efficiencies.append(1.0)
                    continue
                obj_positions = _find_objects(start_grid, sg.obj_type, color=sg.color)
                if not obj_positions:
                    efficiencies.append(-1.0)
                    continue
                op = obj_positions[0]
                goal_fn = lambda pos, _op=op: _agent_adjacent(pos, _op)
            elif sg.category == "PICKUP":
                # Navigate to the key
                key_positions = _find_objects(start_grid, "key", color=sg.color)
                if not key_positions:
                    efficiencies.append(-1.0)
                    continue
                kp = key_positions[0]
                goal_fn = lambda pos, _kp=kp: _agent_adjacent(pos, _kp)
            elif sg.category == "OPEN":
                # Navigate to the door
                door_positions = _find_objects(start_grid, "door", color=sg.color)
                if not door_positions:
                    efficiencies.append(-1.0)
                    continue
                dp = door_positions[0]
                goal_fn = lambda pos, _dp=dp: _agent_adjacent(pos, _dp)
            elif sg.category == "EXPLORE":
                # BFS to nearest unvisited cell
                unvisited = set()
                W, H, _ = start_grid.shape
                for c in range(W):
                    for r in range(H):
                        obj = start_grid[c, r, 0]
                        if obj not in (_OBJ["wall"], _OBJ["unseen"]) and (c, r) not in visited_cells:
                            unvisited.add((c, r))
                if not unvisited:
                    efficiencies.append(1.0)
                    continue
                goal_fn = lambda pos, _uv=unvisited: pos in _uv
            else:
                efficiencies.append(1.0)
                continue

            bfs_dist = _bfs_distance(start_grid, agent_start, goal_fn)
            if bfs_dist < 0:
                efficiencies.append(-1.0)
            else:
                eff = min(1.0, bfs_dist / max(seg_len, 1))
                efficiencies.append(eff)

        return goal_assignments, segments, efficiencies

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def visualize_timeline(
        self,
        trajectory: Dict[str, Any],
        goal_assignments: List[Optional[SubGoal]],
        segments: List[Tuple[int, int, SubGoal]],
        efficiencies: List[float],
        save_path: str,
    ) -> None:
        """
        Horizontal bar (Gantt) chart of sub-goal segments over time.

        Each segment is drawn as a horizontal bar coloured by sub-goal
        category, annotated with the sub-goal name and BFS efficiency.

        Parameters
        ----------
        trajectory : dict
            Original trajectory dict (used for length).
        goal_assignments : list
            Per-timestep sub-goal labels.
        segments : list of (t_start, t_end, SubGoal)
        efficiencies : list of float
        save_path : str
            Output PNG path.
        """
        T = len(goal_assignments)
        if T == 0 or not segments:
            print("[visualize_timeline] Empty trajectory or no segments; skipping.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 5),
                                  gridspec_kw={"height_ratios": [3, 1]})
        ax_gantt, ax_eff = axes

        # --- Gantt chart -------------------------------------------------
        y_tick_labels: List[str] = []
        y_positions: List[float] = []
        category_colors: Dict[str, str] = {}

        for seg_idx, (t_start, t_end, sg) in enumerate(segments):
            cat = sg.category if sg else "UNKNOWN"
            color = self._palette.get(cat, "#888888")
            category_colors[cat] = color

            bar_width = t_end - t_start + 1
            ax_gantt.barh(
                y=0,
                width=bar_width,
                left=t_start,
                height=0.8,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
            # Label inside bar (if wide enough)
            if bar_width > 3:
                label = sg.name if sg else "?"
                ax_gantt.text(
                    t_start + bar_width / 2,
                    0,
                    label,
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                    clip_on=True,
                )

        ax_gantt.set_xlim(0, T)
        ax_gantt.set_ylim(-0.6, 0.6)
        ax_gantt.set_yticks([])
        ax_gantt.set_xlabel("Timestep", fontsize=9)
        ax_gantt.set_title(
            f"Sub-Goal Segments — {self.env_name}", fontsize=11, fontweight="bold"
        )

        # Legend
        patches = [
            mpatches.Patch(facecolor=col, label=cat)
            for cat, col in sorted(category_colors.items())
        ]
        ax_gantt.legend(handles=patches, loc="upper right", fontsize=7, ncol=3)

        # --- Efficiency timeline ----------------------------------------
        t_mids = [(t_start + t_end) / 2 for t_start, t_end, _ in segments]
        effs = [e if e >= 0 else 0.0 for e in efficiencies]
        colors_eff = [self._palette.get(sg.category if sg else "", "#888888")
                      for _, _, sg in segments]

        ax_eff.bar(
            t_mids,
            effs,
            width=[(t_end - t_start + 1) * 0.8 for t_start, t_end, _ in segments],
            color=colors_eff,
            alpha=0.75,
            edgecolor="white",
        )
        ax_eff.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_eff.set_xlim(0, T)
        ax_eff.set_ylim(0, 1.2)
        ax_eff.set_xlabel("Timestep", fontsize=9)
        ax_eff.set_ylabel("BFS Efficiency", fontsize=9)
        ax_eff.set_title("BFS Efficiency per Segment", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize_timeline] Saved -> {save_path}")

    def visualize_dependency_graph(
        self,
        segments: List[Tuple[int, int, SubGoal]],
        save_path: str,
    ) -> None:
        """
        Directed acyclic graph (DAG) of sub-goal prerequisite relations.

        Edges are drawn from sub-goal A to sub-goal B when A appears as
        a segment *before* B in the trajectory (temporal precedence = proxy
        for prerequisite).  Edge weight reflects how consistently A precedes B.

        Parameters
        ----------
        segments : list of (t_start, t_end, SubGoal)
        save_path : str
            Output PNG path.
        """
        if not segments:
            print("[visualize_dependency_graph] No segments; skipping.")
            return

        # Build ordered list of unique sub-goals
        seen: List[str] = []
        sg_by_name: Dict[str, SubGoal] = {}
        for _, _, sg in segments:
            if sg is None:
                continue
            if sg.name not in seen:
                seen.append(sg.name)
                sg_by_name[sg.name] = sg

        # Create directed graph: edge from preceding to following sub-goals
        G = nx.DiGraph()
        for name in seen:
            G.add_node(name)

        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                G.add_edge(seen[i], seen[j])

        # Layout
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            try:
                pos = nx.planar_layout(G)
            except Exception:
                pos = nx.spring_layout(G, seed=42)

        # Node colors by category
        node_colors = []
        for name in G.nodes():
            sg = sg_by_name.get(name)
            cat = sg.category if sg else "UNKNOWN"
            node_colors.append(self._palette.get(cat, "#888888"))

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=1800, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                                font_color="white", font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555",
                               arrows=True, arrowsize=20,
                               connectionstyle="arc3,rad=0.1",
                               width=1.5)

        # Legend
        patches = [
            mpatches.Patch(facecolor=col, label=cat)
            for cat, col in sorted(self._palette.items())
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=8)

        ax.set_title(
            f"Sub-Goal Dependency DAG — {self.env_name}", fontsize=12, fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize_dependency_graph] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Convenience: run imputation on a batch of trajectories
# ---------------------------------------------------------------------------

def batch_impute(
    imputer: SubGoalImputation,
    trajectories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run imputation on every trajectory in the list and attach results in-place.

    Each trajectory dict gains three new keys:
        goal_assignments : List[Optional[SubGoal]]
        segments         : List[(t_start, t_end, SubGoal)]
        efficiencies     : List[float]

    Parameters
    ----------
    imputer : SubGoalImputation
    trajectories : list of trajectory dicts

    Returns
    -------
    The same list with updated dicts.
    """
    for traj in trajectories:
        ga, segs, effs = imputer.impute(traj)
        traj["goal_assignments"] = ga
        traj["segments"] = segs
        traj["efficiencies"] = effs
    return trajectories


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import gymnasium as gym
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    env_name = "MiniGrid-DoorKey-8x8-v0"
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    imputer = SubGoalImputation(env_name)
    vocab = imputer.extract_subgoal_vocab(env)
    print(f"Sub-goal vocabulary ({len(vocab)} goals):")
    for sg in vocab:
        print(f"  {sg}")

    # Collect a short random trajectory
    obs, _ = env.reset()
    grid_states, obs_seq, action_seq = [], [], []
    done = False
    for _ in range(50):
        grid_states.append(env.unwrapped.grid.encode())
        obs_seq.append(obs.copy())
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        action_seq.append(action)
        if done or truncated:
            break

    traj = {
        "grid_state_seq": grid_states,
        "obs_seq": obs_seq,
        "action_seq": action_seq,
    }
    ga, segs, effs = imputer.impute(traj)
    print(f"\nTrajectory length: {len(grid_states)}")
    print(f"Segments ({len(segs)}):")
    for i, (ts, te, sg) in enumerate(segs):
        name = sg.name if sg else "None"
        print(f"  [{ts:3d} -> {te:3d}]  {name}  (eff={effs[i]:.3f})")

    env.close()
