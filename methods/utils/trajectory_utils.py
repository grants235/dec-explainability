"""
Utilities for RL trajectory processing on MiniGrid environments.

Functions
---------
- bfs_shortest_path          : BFS on a MiniGrid full grid state
- extract_grid_state         : Extract the full grid state array from env
- compute_optimal_subgoal_seq : Topological-sort-based optimal sub-goal sequence
- segment_trajectory_by_subgoal : Segment a trajectory into per-subgoal spans
- compute_gn_score           : Goal Necessity score for a single timestep
- load_trajectory_npz        : Load a saved .npz trajectory dict
- save_trajectory_npz        : Save a trajectory dict to .npz
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Grid representation constants (MiniGrid)
# ---------------------------------------------------------------------------

# MiniGrid tile encoding index
TILE_TYPE_IDX = 0
TILE_COLOR_IDX = 1
TILE_STATE_IDX = 2

# Object type IDs used by MiniGrid
OBJECT_TO_IDX: Dict[str, int] = {
    "unseen":      0,
    "empty":       1,
    "wall":        2,
    "floor":       3,
    "door":        4,
    "key":         5,
    "ball":        6,
    "box":         7,
    "goal":        8,
    "lava":        9,
    "agent":       10,
}
IDX_TO_OBJECT: Dict[int, str] = {v: k for k, v in OBJECT_TO_IDX.items()}

# Door states
DOOR_OPEN   = 0
DOOR_CLOSED = 1
DOOR_LOCKED = 2

# Action IDs
ACTION_LEFT   = 0
ACTION_RIGHT  = 1
ACTION_FORWARD = 2
ACTION_PICKUP  = 3
ACTION_DROP    = 4
ACTION_TOGGLE  = 5
ACTION_DONE    = 6

# Direction vectors: (dy, dx) for dir_id 0..3 (right, down, left, up)
DIR_VECTORS: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]


# ---------------------------------------------------------------------------
# BFS on MiniGrid grid state
# ---------------------------------------------------------------------------

def bfs_shortest_path(
    grid: np.ndarray,
    start: Tuple[int, int, int],
    goal_pos: Tuple[int, int],
    has_key: bool = False,
) -> Optional[List[Tuple[int, int, int]]]:
    """
    BFS to find the shortest path from ``start`` to ``goal_pos`` on a
    MiniGrid grid.

    State representation: ``(row, col, direction)`` where direction ∈ {0,1,2,3}.

    Parameters
    ----------
    grid : np.ndarray
        Shape ``(height, width, 3)`` – the full grid encoding with
        [type, colour, state] per cell.
    start : (row, col, dir)
        Agent's starting state.
    goal_pos : (goal_row, goal_col)
        Target cell (the agent must *be in front of* this cell to reach it,
        or step onto it if it is the goal tile).
    has_key : bool
        Whether the agent currently holds a key (needed to pass locked doors).

    Returns
    -------
    list of (row, col, dir) tuples from start to goal (inclusive), or None
    if no path exists.
    """
    height, width = grid.shape[:2]

    def _is_passable(row: int, col: int, carrying_key: bool) -> bool:
        if not (0 <= row < height and 0 <= col < width):
            return False
        tile_type = int(grid[row, col, TILE_TYPE_IDX])
        tile_state = int(grid[row, col, TILE_STATE_IDX])
        if tile_type == OBJECT_TO_IDX["wall"]:
            return False
        if tile_type == OBJECT_TO_IDX["lava"]:
            return False
        if tile_type == OBJECT_TO_IDX["door"]:
            if tile_state == DOOR_LOCKED:
                return carrying_key
            if tile_state == DOOR_CLOSED:
                return True  # can toggle open
        return True

    goal_row, goal_col = goal_pos
    init_state = (start[0], start[1], start[2], has_key)

    visited: Set[Tuple[int, int, int, bool]] = set()
    parent: Dict[Tuple[int, int, int, bool], Optional[Tuple[int, int, int, bool]]] = {}
    queue: deque = deque()

    queue.append(init_state)
    visited.add(init_state)
    parent[init_state] = None

    while queue:
        row, col, direction, key = queue.popleft()

        # Check if we have reached the goal
        if (row, col) == (goal_row, goal_col):
            # Reconstruct path
            path = []
            cur = (row, col, direction, key)
            while cur is not None:
                path.append((cur[0], cur[1], cur[2]))
                cur = parent[cur]
            path.reverse()
            return path

        # Generate successors
        # Turn left
        new_dir_l = (direction - 1) % 4
        ns_l = (row, col, new_dir_l, key)
        if ns_l not in visited:
            visited.add(ns_l)
            parent[ns_l] = (row, col, direction, key)
            queue.append(ns_l)

        # Turn right
        new_dir_r = (direction + 1) % 4
        ns_r = (row, col, new_dir_r, key)
        if ns_r not in visited:
            visited.add(ns_r)
            parent[ns_r] = (row, col, direction, key)
            queue.append(ns_r)

        # Move forward
        dy, dx = DIR_VECTORS[direction]
        new_row, new_col = row + dy, col + dx
        if _is_passable(new_row, new_col, key):
            tile_type = int(grid[new_row, new_col, TILE_TYPE_IDX])
            new_key = key or (tile_type == OBJECT_TO_IDX["key"])
            ns_f = (new_row, new_col, direction, new_key)
            if ns_f not in visited:
                visited.add(ns_f)
                parent[ns_f] = (row, col, direction, key)
                queue.append(ns_f)

    return None  # No path found


def bfs_path_length(
    grid: np.ndarray,
    start: Tuple[int, int, int],
    goal_pos: Tuple[int, int],
    has_key: bool = False,
) -> int:
    """
    Return the BFS shortest path length (in steps) from start to goal.

    Returns -1 if unreachable.
    """
    path = bfs_shortest_path(grid, start, goal_pos, has_key)
    return len(path) - 1 if path is not None else -1


# ---------------------------------------------------------------------------
# Extract full grid state from a MiniGrid env instance
# ---------------------------------------------------------------------------

def extract_grid_state(env: Any) -> np.ndarray:
    """
    Extract the full grid encoding from a live MiniGrid environment.

    Parameters
    ----------
    env : gymnasium.Env
        A MiniGrid environment (or wrapped version).  Attempts to access
        ``env.unwrapped.grid`` (MiniGrid convention).

    Returns
    -------
    np.ndarray
        Shape ``(height, width, 3)``.  Each cell is ``[type, colour, state]``.

    Raises
    ------
    AttributeError
        If the environment does not expose a MiniGrid grid.
    """
    # Unwrap until we find the MiniGrid env
    inner = env
    for _ in range(10):
        if hasattr(inner, "grid"):
            break
        if hasattr(inner, "env"):
            inner = inner.env
        elif hasattr(inner, "unwrapped"):
            inner = inner.unwrapped
            break
        else:
            break

    grid_obj = inner.grid
    height = grid_obj.height
    width = grid_obj.width

    encoding = np.zeros((height, width, 3), dtype=np.int32)
    for row in range(height):
        for col in range(width):
            cell = grid_obj.get(col, row)  # MiniGrid uses (x, y) = (col, row)
            if cell is None:
                encoding[row, col] = [OBJECT_TO_IDX["empty"], 0, 0]
            else:
                encoding[row, col] = cell.encode()

    return encoding


def get_agent_state(env: Any) -> Tuple[int, int, int]:
    """
    Return the agent's (row, col, direction) from a live MiniGrid env.
    """
    inner = env
    for _ in range(10):
        if hasattr(inner, "agent_pos") and hasattr(inner, "agent_dir"):
            col, row = inner.agent_pos  # MiniGrid stores (x=col, y=row)
            return (row, col, inner.agent_dir)
        if hasattr(inner, "env"):
            inner = inner.env
        elif hasattr(inner, "unwrapped"):
            inner = inner.unwrapped
            break
    raise AttributeError("Cannot find agent_pos / agent_dir on environment.")


# ---------------------------------------------------------------------------
# Sub-goal identification in grid
# ---------------------------------------------------------------------------

def find_subgoal_positions(
    grid: np.ndarray,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Identify positions of all interactable objects in the grid.

    Returns a dict mapping object type name -> list of (row, col).
    """
    positions: Dict[str, List[Tuple[int, int]]] = {
        name: [] for name in ["key", "door", "goal", "box", "ball"]
    }
    height, width = grid.shape[:2]
    for row in range(height):
        for col in range(width):
            tile_type = IDX_TO_OBJECT.get(int(grid[row, col, TILE_TYPE_IDX]), "unknown")
            if tile_type in positions:
                positions[tile_type].append((row, col))
    return positions


# ---------------------------------------------------------------------------
# Optimal sub-goal sequence via topological sort
# ---------------------------------------------------------------------------

def compute_optimal_subgoal_sequence(
    grid: np.ndarray,
    agent_state: Tuple[int, int, int],
    subgoal_positions: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Compute the optimal ordering of sub-goals via a greedy nearest-neighbour
    heuristic seeded by BFS distances, falling back to a topological sort
    when dependency constraints are present (e.g. key before locked door).

    Parameters
    ----------
    grid : np.ndarray
        Full grid state ``(H, W, 3)``.
    agent_state : (row, col, dir)
        Current agent position and direction.
    subgoal_positions : list of (row, col) or None
        If provided, only these positions are considered as sub-goals.
        Otherwise all interactable objects are used.

    Returns
    -------
    list of (row, col)
        Ordered list of sub-goal positions to visit.
    """
    if subgoal_positions is None:
        sg_dict = find_subgoal_positions(grid)
        all_sgs: List[Tuple[int, int]] = []
        for positions in sg_dict.values():
            all_sgs.extend(positions)
        subgoal_positions = all_sgs

    if not subgoal_positions:
        return []

    # Build dependency graph: key must come before any locked door
    key_positions = find_subgoal_positions(grid).get("key", [])
    door_positions = [
        (r, c)
        for r, c in find_subgoal_positions(grid).get("door", [])
        if int(grid[r, c, TILE_STATE_IDX]) == DOOR_LOCKED
    ]

    # Greedy BFS-distance ordering with dependency check
    has_key = False
    ordered: List[Tuple[int, int]] = []
    remaining = list(subgoal_positions)
    current_state = agent_state

    while remaining:
        # Determine which sub-goals are currently reachable
        reachable = []
        for sg in remaining:
            # Locked door requires key
            tile_type = IDX_TO_OBJECT.get(int(grid[sg[0], sg[1], TILE_TYPE_IDX]), "empty")
            tile_state = int(grid[sg[0], sg[1], TILE_STATE_IDX])
            if tile_type == "door" and tile_state == DOOR_LOCKED and not has_key:
                continue
            dist = bfs_path_length(grid, current_state, sg, has_key)
            if dist >= 0:
                reachable.append((dist, sg))

        if not reachable:
            # No reachable sub-goal; force-add all remaining in original order
            ordered.extend(remaining)
            break

        # Pick nearest
        reachable.sort(key=lambda x: x[0])
        _, next_sg = reachable[0]

        ordered.append(next_sg)
        remaining.remove(next_sg)

        # Update has_key
        tile_type = IDX_TO_OBJECT.get(int(grid[next_sg[0], next_sg[1], TILE_TYPE_IDX]), "empty")
        if tile_type == "key":
            has_key = True

        # Approximate new current position as next_sg
        current_state = (next_sg[0], next_sg[1], current_state[2])

    return ordered


# ---------------------------------------------------------------------------
# Trajectory segmentation
# ---------------------------------------------------------------------------

def segment_trajectory_by_subgoal(
    traj_obs: np.ndarray,
    traj_actions: np.ndarray,
    subgoal_positions: List[Tuple[int, int]],
    agent_positions: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int]]:
    """
    Segment a trajectory into sub-goal spans.

    Each segment runs from the timestep after the previous sub-goal was
    reached until the current sub-goal is reached.

    Parameters
    ----------
    traj_obs : np.ndarray
        Shape ``(T, obs_dim)``.
    traj_actions : np.ndarray
        Shape ``(T,)`` int.
    subgoal_positions : list of (row, col)
        Ordered sequence of sub-goal positions.
    agent_positions : np.ndarray or None
        Shape ``(T, 2)`` – per-timestep (row, col) agent positions.
        If None the positions are estimated from actions (approximate).

    Returns
    -------
    list of (subgoal_idx, start_t, end_t)
        Each tuple gives the sub-goal index (0-based), start timestep, and
        end timestep (inclusive) for that segment.
    """
    T = len(traj_actions)
    segments: List[Tuple[int, int, int]] = []
    seg_start = 0

    for sg_idx, sg_pos in enumerate(subgoal_positions):
        # Find the first timestep >= seg_start where agent is at sg_pos
        if agent_positions is not None:
            for t in range(seg_start, T):
                if (int(agent_positions[t, 0]), int(agent_positions[t, 1])) == sg_pos:
                    segments.append((sg_idx, seg_start, t))
                    seg_start = t + 1
                    break
            else:
                # Sub-goal not reached; run to end of trajectory
                segments.append((sg_idx, seg_start, T - 1))
                seg_start = T
                break
        else:
            # Without position data, partition evenly
            seg_end = min(seg_start + max(1, T // max(len(subgoal_positions), 1)) - 1, T - 1)
            segments.append((sg_idx, seg_start, seg_end))
            seg_start = seg_end + 1

    # Catch any remaining timesteps in a final segment
    if seg_start < T and segments:
        last_sg_idx = segments[-1][0]
        segments.append((last_sg_idx, seg_start, T - 1))

    return segments


# ---------------------------------------------------------------------------
# Goal Necessity (GN) score
# ---------------------------------------------------------------------------

def compute_gn_score(
    action_dist_with_goal: np.ndarray,
    action_dist_without_goal: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute the Goal Necessity (GN) score for a single timestep.

    GN measures how much the goal specification changes the agent's action
    distribution:

        GN = KL( pi(a|o,g) || pi(a|o) )  normalised to [0, 1]

    using a soft upper bound of log(n_actions).

    Parameters
    ----------
    action_dist_with_goal : np.ndarray
        Probability vector ``pi(a | o, g)``, shape ``(n_actions,)``.
    action_dist_without_goal : np.ndarray
        Probability vector ``pi(a | o)``, shape ``(n_actions,)``.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    float
        GN score in [0, 1].
    """
    p = np.clip(action_dist_with_goal, epsilon, 1.0)
    q = np.clip(action_dist_without_goal, epsilon, 1.0)

    # Normalise
    p = p / p.sum()
    q = q / q.sum()

    kl = float(np.sum(p * np.log(p / q)))
    n_actions = len(p)
    kl_max = float(np.log(n_actions)) if n_actions > 1 else 1.0

    return float(np.clip(kl / kl_max, 0.0, 1.0))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_trajectory_npz(trajectory: Dict[str, Any], path: str) -> None:
    """
    Save a trajectory dict to a compressed .npz file.

    Numeric arrays are stored natively; other values are stored as
    object arrays.
    """
    arrays: Dict[str, Any] = {}
    for key, val in trajectory.items():
        if isinstance(val, np.ndarray):
            arrays[key] = val
        elif isinstance(val, list):
            try:
                arrays[key] = np.array(val)
            except Exception:  # noqa: BLE001
                arrays[key] = np.array(val, dtype=object)
        else:
            arrays[key] = np.array([val], dtype=object)

    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_trajectory_npz(path: str) -> Dict[str, Any]:
    """
    Load a trajectory dict from a .npz file.

    Returns a plain Python dict with numpy arrays as values.
    """
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_trajectories_from_dir(traj_dir: str) -> List[Dict[str, Any]]:
    """
    Load all .npz trajectory files from a directory.

    Returns a list of trajectory dicts, sorted by filename.
    """
    import os
    trajectories: List[Dict[str, Any]] = []
    if not os.path.isdir(traj_dir):
        return trajectories
    for fname in sorted(os.listdir(traj_dir)):
        if fname.endswith(".npz"):
            path = os.path.join(traj_dir, fname)
            trajectories.append(load_trajectory_npz(path))
    return trajectories
