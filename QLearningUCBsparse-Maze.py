# -*- coding: utf-8 -*-


from __future__ import annotations


import math


import random


import time


import os


import heapq


from collections import deque


from dataclasses import dataclass


from typing import List, Sequence, Tuple, Optional, Dict





import numpy as np


import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt


from matplotlib import colors


from matplotlib import animation


from matplotlib.lines import Line2D





Coordinate = Tuple[int, int]








class MazeEnv:


    """Maze with deterministic transitions and terminal rewards."""





    # ? + 


    ACTIONS: Sequence[Coordinate] = (


        (-1, 0),  # up


        (0, 1),  # right


        (1, 0),  # down


        (0, -1),  # left


        (0, 0),  # stay ()


    )





    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]





    def __init__(self, size: int = 15, horizon: int = 500, seed: int = 42,


                 grid: np.ndarray | None = None, start: Coordinate | None = None,


                 goal: Coordinate | None = None, wall_penalty: float = -1,


                 stay_penalty: float = 0.0,


                 move_penalty: float = 0.0) -> None:


        self.size = size


        self.horizon = horizon


        # 


        self.random = random.Random(seed)


        self.start: Coordinate = start if start is not None else (0, 0)


        self.goal: Coordinate = goal if goal is not None else (size - 1, size - 1)


        self.grid = self._generate_maze(seed) if grid is None else grid.copy()


        self.state: Coordinate = self.start


        self.step_count = 0


        self.wall_penalty = wall_penalty  # 


        self.stay_penalty = stay_penalty  # 


        self.move_penalty = move_penalty  # ?


        self.seed = seed  # 





    @property


    def n_states(self) -> int:


        return self.size * self.size





    @property


    def n_actions(self) -> int:


        return len(self.ACTIONS)





    def _generate_maze(self, seed: int) -> np.ndarray:


        """Creates a maze with multiple carved corridors plus a guaranteed start-goal path."""


        rng = random.Random(seed)


        grid = np.ones((self.size, self.size), dtype=np.int8)


        safe_path: set[Coordinate] = set()





        def carve_path(a: Coordinate, b: Coordinate) -> None:


            r, c = a


            safe_path.add((r, c))


            while c != b[1]:


                c += 1 if b[1] > c else -1


                safe_path.add((r, c))


            while r != b[0]:


                r += 1 if b[0] > r else -1


                safe_path.add((r, c))





        # Guarantee connections between notable waypoints.


        carve_path((0, 0), (0, self.size - 1))


        carve_path((0, self.size - 1), (self.size - 1, self.size - 1))


        carve_path(self.start, self.goal)


        carve_path((0, 0), self.start)





        # Add additional diagonal corridor for variety.


        for idx in range(self.size):


            safe_path.add((idx, idx))





        for r in range(self.size):


            for c in range(self.size):


                if (r, c) in safe_path or (r, c) == self.goal:


                    grid[r, c] = 0


                elif rng.random() < 0.3:


                    grid[r, c] = 0


        return grid





    def reset(self) -> Coordinate:


        self.state = self.start


        self.step_count = 0


        return self.state





    def step(self, action: int) -> Tuple[Coordinate, float, bool, dict]:


        reward = 0.0


        hit_wall = False


        is_stay = False





        # 


        dr, dc = self.ACTIONS[action]


        target_r = self.state[0] + dr


        target_c = self.state[1] + dc





        next_coord = self.state


        # 


        if dr == 0 and dc == 0:


            is_stay = True


            # ?


        else:


            # ?


            if not (0 <= target_r < self.size and 0 <= target_c < self.size):


                hit_wall = True


            else:


                # 


                if self.grid[target_r, target_c] == 0:


                    next_coord = (target_r, target_c)


                    self.state = next_coord


                else:


                    hit_wall = True





        self.step_count += 1


        reached_goal = next_coord == self.goal


        done = self.step_count >= self.horizon





        if reached_goal:


            reward = 1.0


        elif hit_wall:


            reward = self.wall_penalty


        elif is_stay:


            reward = self.stay_penalty


        else:


            reward = self.move_penalty





        return next_coord, reward, done, {"hit_wall": hit_wall, "is_stay": is_stay}





    def coord_to_state(self, coord: Coordinate) -> int:


        return coord[0] * self.size + coord[1]





    def state_to_coord(self, state: int) -> Coordinate:


        return divmod(state, self.size)





    def render(self, path: Sequence[Coordinate] | None = None) -> str:


        path_set = set(path) if path else set()


        rows: List[str] = []


        for r in range(self.size):


            chars: List[str] = []


            for c in range(self.size):


                coord = (r, c)


                if coord == self.start:


                    ch = 'S'


                elif coord == self.goal:


                    ch = 'G'


                elif self.grid[r, c] == 1:


                    ch = '#'


                else:


                    ch = '.'


                if coord in path_set and ch == '.':


                    ch = '*'


                chars.append(ch)


            rows.append(''.join(chars))


        return '\n'.join(rows)








def build_corridor_maze(size: int = 15, seed: int = 7) -> np.ndarray:


    """Generates a maze with explicit walls using a DFS-backtracking algorithm."""


    # 7


    rng = random.Random(seed)


    grid = np.ones((size, size), dtype=np.int8)





    def carve_passage(cell: Coordinate, nxt: Coordinate) -> None:


        r1, c1 = cell


        r2, c2 = nxt


        grid[r1, c1] = 0


        grid[r2, c2] = 0


        grid[(r1 + r2) // 2, (c1 + c2) // 2] = 0





    start = (0, 0)


    stack: List[Coordinate] = [start]


    visited = {start}





    def cell_neighbors(coord: Coordinate) -> List[Coordinate]:


        r, c = coord


        dirs = [(-2, 0), (0, 2), (2, 0), (0, -2)]


        result: List[Coordinate] = []


        for dr, dc in dirs:


            nr, nc = r + dr, c + dc


            if 0 <= nr < size and 0 <= nc < size:


                result.append((nr, nc))


        rng.shuffle(result)


        return result





    grid[start] = 0


    while stack:


        current = stack[-1]


        neighbors = [nbr for nbr in cell_neighbors(current) if nbr not in visited]


        if not neighbors:


            stack.pop()


            continue


        nxt = neighbors[0]


        carve_passage(current, nxt)


        visited.add(nxt)


        stack.append(nxt)





    # Ensure goal region is open and connected.


    grid[size - 1, size - 1] = 0


    grid[size - 2, size - 1] = 0


    grid[size - 1, size - 2] = 0





    # Add a limited number of random shortcuts to create loops.


    for _ in range(size):


        r = rng.randrange(1, size - 1)


        c = rng.randrange(1, size - 1)


        grid[r, c] = 0





    # Ensure cells adjacent to the start are free so the agent can move initially.


    for dr, dc in [(1, 0), (0, 1)]:


        nr, nc = dr, dc


        if 0 <= nr < size and 0 <= nc < size:


            grid[nr, nc] = 0





    return grid








@dataclass


class TrainingStats:


    episodes: List[int]


    rewards: List[float]


    successes: List[bool]  # episode


    success_rate: float


    episode_paths: List[List[Coordinate]] | None = None








@dataclass


class QPathSelection:


    path: List[Coordinate]


    selection: str


    reaches_goal: bool


    is_strict_max_q: bool


    is_plotted: bool


    q_score: float








def sparse_goal_reward(next_coord: Coordinate, goal: Coordinate) -> float:


    """Returns a pure sparse reward: 1 on goal, otherwise 0."""


    return 1.0 if next_coord == goal else 0.0








def select_best_path(stats: TrainingStats, goal: Coordinate) -> List[Coordinate] | None:


    """Selects the shortest successful training path, breaking ties by reward."""


    if not stats.episode_paths or not stats.rewards or not stats.successes:


        return None





    best_idx: int | None = None


    best_length = float("inf")


    best_reward = float("-inf")





    for idx, (path, reward, success) in enumerate(zip(stats.episode_paths, stats.rewards, stats.successes)):


        if not success or not path or path[-1] != goal:


            continue


        path_length = len(path)


        if path_length < best_length or (path_length == best_length and reward > best_reward):


            best_idx = idx


            best_length = path_length


            best_reward = reward





    if best_idx is None:


        return None


    return stats.episode_paths[best_idx]








def best_q_actions(q_table: np.ndarray, state: int) -> np.ndarray:


    """Returns all actions tied for the highest Q-value in a state."""


    q_values = q_table[state]


    max_value = np.max(q_values)


    return np.flatnonzero(np.isclose(q_values, max_value))








def transition_without_side_effect(env: MazeEnv, coord: Coordinate, action: int) -> Coordinate:


    """Computes the next coordinate for an action without mutating the environment."""


    dr, dc = env.ACTIONS[action]


    if dr == 0 and dc == 0:


        return coord





    target_r = coord[0] + dr


    target_c = coord[1] + dc


    if not (0 <= target_r < env.size and 0 <= target_c < env.size):


        return coord


    if env.grid[target_r, target_c] != 0:


        return coord


    return (target_r, target_c)








def path_uses_only_actual_max_q_moves(q_table: np.ndarray, env: MazeEnv,


                                      path: Sequence[Coordinate]) -> bool:


    """Checks whether each path step follows one of the state's max-Q actions."""


    if len(path) < 2:


        return False





    for coord, next_coord in zip(path, path[1:]):


        state = env.coord_to_state(coord)


        valid_next_coords = {


            transition_without_side_effect(env, coord, int(action))


            for action in best_q_actions(q_table, state)


        }


        if next_coord == coord or next_coord not in valid_next_coords:


            return False


    return True








def actions_matching_transition(env: MazeEnv, coord: Coordinate,


                                next_coord: Coordinate) -> List[int]:


    """Returns actions that can explain a coordinate transition."""


    if coord == env.goal and next_coord == env.goal:


        return list(range(env.n_actions))


    if coord == next_coord:


        return [env.n_actions - 1]


    return [


        action


        for action in range(env.n_actions)


        if transition_without_side_effect(env, coord, action) == next_coord


    ]








def path_q_score(q_table: np.ndarray, env: MazeEnv,


                 path: Sequence[Coordinate]) -> float:


    """Scores a plotted path by average Q-value consistent with each shown step."""


    if len(path) < 2:


        return 0.0





    score = 0.0


    step_count = 0


    for coord, next_coord in zip(path, path[1:]):


        state = env.coord_to_state(coord)


        matching_actions = actions_matching_transition(env, coord, next_coord)


        if not matching_actions:


            continue


        score += float(np.max(q_table[state, matching_actions]))


        step_count += 1


    if step_count == 0:


        return 0.0


    return score / step_count








def path_has_actual_moves(path: Sequence[Coordinate]) -> bool:


    """Checks whether the path contains at least one visible move."""


    return any(


        path[i] != path[i + 1]


        for i in range(max(0, len(path) - 1))


    )








def path_reaches_goal(path: Sequence[Coordinate], goal: Coordinate) -> bool:


    """Checks whether the path reaches the goal at any point."""


    return any(coord == goal for coord in path)








def truncate_path_at_goal(path: Sequence[Coordinate], goal: Coordinate) -> List[Coordinate]:


    """Returns the path up to and including the first goal visit."""


    for idx, coord in enumerate(path):


        if coord == goal:


            return list(path[:idx + 1])


    return list(path)








def loop_erased_path(path: Sequence[Coordinate]) -> List[Coordinate]:


    """Removes stays and loops while preserving a route that actually occurred."""


    erased: List[Coordinate] = []


    positions: Dict[Coordinate, int] = {}





    for coord in path:


        if erased and coord == erased[-1]:


            continue


        if coord in positions:


            keep_until = positions[coord]


            for removed in erased[keep_until + 1:]:


                positions.pop(removed, None)


            erased = erased[:keep_until + 1]


            continue


        positions[coord] = len(erased)


        erased.append(coord)





    return erased








def remove_consecutive_stays(path: Sequence[Coordinate]) -> List[Coordinate]:


    """Removes repeated consecutive coordinates while keeping the actual walked route."""


    cleaned: List[Coordinate] = []


    for coord in path:


        if cleaned and coord == cleaned[-1]:


            continue


        cleaned.append(coord)


    return cleaned








def select_training_path_by_q(q_table: np.ndarray, env: MazeEnv,


                              stats: TrainingStats | None,


                              require_goal: bool) -> Tuple[List[Coordinate], float] | None:


    """Selects a short loop-erased path that actually occurred during training."""


    if stats is None or not stats.episode_paths:


        return None





    best_path: List[Coordinate] | None = None


    best_score = float("-inf")


    best_length = float("inf")


    best_visible_steps = -1





    for path in stats.episode_paths:


        if not path:


            continue


        reaches_goal = path_reaches_goal(path, env.goal)


        if require_goal and not reaches_goal:


            continue


        truncated_path = truncate_path_at_goal(path, env.goal)


        if require_goal:


            display_path = loop_erased_path(truncated_path)


        else:


            display_path = remove_consecutive_stays(truncated_path)


        if require_goal and not path_reaches_goal(display_path, env.goal):


            continue


        has_moves = path_has_actual_moves(display_path)


        if not has_moves:


            continue





        score = path_q_score(q_table, env, display_path)


        path_length = len(display_path)


        visible_steps = path_length - 1





        if require_goal:


            is_better = (


                path_length < best_length


                or (path_length == best_length and score > best_score)


            )


        else:


            is_better = (


                visible_steps > best_visible_steps


                or (visible_steps == best_visible_steps and score > best_score)


            )





        if is_better:


            best_path = display_path


            best_score = score


            best_length = path_length


            best_visible_steps = visible_steps





    if best_path is None:


        return None


    return best_path, best_score








def strict_max_q_path_to_goal(q_table: np.ndarray, env: MazeEnv,


                              horizon: int) -> List[Coordinate] | None:


    """Finds a goal path using only actual max-Q moves."""


    start = env.start


    goal = env.goal


    if start == goal:


        return [start]





    queue = deque([(start, [start])])


    visited = {start}





    while queue:


        coord, path = queue.popleft()


        if coord == goal:


            return path


        if len(path) - 1 >= horizon:


            continue





        state = env.coord_to_state(coord)


        candidates: List[Tuple[int, Coordinate]] = []


        for action in best_q_actions(q_table, state):


            next_coord = transition_without_side_effect(env, coord, int(action))


            if next_coord == coord:


                continue


            distance = abs(next_coord[0] - goal[0]) + abs(next_coord[1] - goal[1])


            candidates.append((distance, next_coord))





        for _, next_coord in sorted(candidates):


            if next_coord in visited:


                continue


            visited.add(next_coord)


            queue.append((next_coord, path + [next_coord]))





    return None








def high_q_real_path_to_goal(q_table: np.ndarray, env: MazeEnv,


                             horizon: int) -> List[Coordinate] | None:


    """Finds a real goal path with minimal Q loss relative to local max-Q actions."""


    start = env.start


    goal = env.goal


    if start == goal:


        return [start]





    counter = 0


    queue: List[Tuple[float, int, int, int, Coordinate, List[Coordinate]]] = []


    start_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])


    heapq.heappush(queue, (0.0, 0, start_distance, counter, start, [start]))


    best_seen: Dict[Coordinate, Tuple[float, int]] = {start: (0.0, 0)}





    while queue:


        regret, steps, _, _, coord, path = heapq.heappop(queue)


        if coord == goal:


            return path


        if steps >= horizon:


            continue


        if (regret, steps) != best_seen.get(coord):


            continue





        state = env.coord_to_state(coord)


        best_local_q = float(np.max(q_table[state]))


        moving_candidates: List[Tuple[float, int, Coordinate]] = []


        for action in range(env.n_actions):


            next_coord = transition_without_side_effect(env, coord, action)


            if next_coord == coord:


                continue


            moving_candidates.append((float(q_table[state, action]), action, next_coord))





        for action_q, action, next_coord in moving_candidates:


            next_steps = steps + 1


            next_regret = regret + max(0.0, best_local_q - action_q)


            previous = best_seen.get(next_coord)


            if previous is not None and previous <= (next_regret, next_steps):


                continue





            best_seen[next_coord] = (next_regret, next_steps)


            distance = abs(next_coord[0] - goal[0]) + abs(next_coord[1] - goal[1])


            counter += 1


            heapq.heappush(


                queue,


                (next_regret, next_steps, distance, counter, next_coord, path + [next_coord])


            )





    return None








def true_max_q_failed_rollout(q_table: np.ndarray, env: MazeEnv,


                              horizon: int) -> List[Coordinate]:


    """Rolls out the actual deterministic max-Q policy for diagnostics."""


    goal = env.goal


    coord = env.reset()


    path = [coord]


    state = env.coord_to_state(coord)


    visited_states = {state}





    for _ in range(horizon):


        candidates = []


        for action in best_q_actions(q_table, state):


            next_coord = transition_without_side_effect(env, coord, int(action))


            distance = abs(next_coord[0] - goal[0]) + abs(next_coord[1] - goal[1])


            no_move = 1 if next_coord == coord else 0


            candidates.append((no_move, distance, int(action), next_coord))





        if not candidates:


            break





        _, _, action, _ = min(candidates)


        coord, _, done, _ = env.step(action)


        path.append(coord)


        state = env.coord_to_state(coord)


        if coord == goal or done or state in visited_states:


            break


        visited_states.add(state)





    return path








def deterministic_max_q_rollout(q_table: np.ndarray, env: MazeEnv,


                                horizon: int) -> List[Coordinate]:


    """Rolls out the fixed argmax-Q policy without searching among tied actions."""


    coord = env.reset()


    path = [coord]


    state = env.coord_to_state(coord)


    visited_states = {state}





    for _ in range(horizon):


        action = int(np.argmax(q_table[state]))


        coord, _, done, _ = env.step(action)


        path.append(coord)


        state = env.coord_to_state(coord)


        if coord == env.goal or done or state in visited_states:


            break


        visited_states.add(state)





    return path








def select_q_path_for_plot(q_table: np.ndarray, env: MazeEnv,


                           horizon: int,


                           stats: TrainingStats | None = None) -> QPathSelection:


    """Selects a truthful route for plots without inventing paths.





    Priority:


    1. Fixed argmax-Q rollout, if it reaches the goal.


    2. Shortest loop-erased training path that actually reached the goal.


    3. Longest visible training path with consecutive stays removed.


    4. Fixed argmax-Q rollout, even if it fails.


    """


    greedy_path = deterministic_max_q_rollout(q_table, env, horizon)


    greedy_reaches_goal = path_reaches_goal(greedy_path, env.goal)


    greedy_score = path_q_score(q_table, env, greedy_path)


    if greedy_reaches_goal and path_has_actual_moves(greedy_path):


        return QPathSelection(


            path=greedy_path,


            selection="deterministic_max_q_policy_rollout",


            reaches_goal=True,


            is_strict_max_q=True,


            is_plotted=True,


            q_score=greedy_score,


        )





    successful_training_path = select_training_path_by_q(


        q_table, env, stats, require_goal=True


    )


    if successful_training_path is not None:


        path, score = successful_training_path


        return QPathSelection(


            path=path,


            selection="actual_training_success_path_shortest_loop_erased_not_policy",


            reaches_goal=True,


            is_strict_max_q=path_uses_only_actual_max_q_moves(q_table, env, path),


            is_plotted=True,


            q_score=score,


        )





    moving_training_path = select_training_path_by_q(


        q_table, env, stats, require_goal=False


    )


    if moving_training_path is not None:


        path, score = moving_training_path


        return QPathSelection(


            path=path,


            selection="actual_training_moving_path_visible_no_goal",


            reaches_goal=path_reaches_goal(path, env.goal),


            is_strict_max_q=path_uses_only_actual_max_q_moves(q_table, env, path),


            is_plotted=True,


            q_score=score,


        )





    return QPathSelection(


        path=greedy_path,


        selection="failed_deterministic_max_q_policy_rollout",


        reaches_goal=greedy_reaches_goal,


        is_strict_max_q=True,


        is_plotted=bool(greedy_path),


        q_score=greedy_score,


    )








def q_greedy_path_to_goal(q_table: np.ndarray, env: MazeEnv, horizon: int) -> List[Coordinate]:


    """Returns the selected Q-value path used by route visualizations."""


    return select_q_path_for_plot(q_table, env, horizon).path








def compute_reward_axis(reward_series_list: Sequence[Sequence[float]]) -> Tuple[int, int, int]:


    """Computes a shared y-axis range and tick step for reward plots."""


    # Use fixed, pre-determined plotting range for consistent comparison across experiments.


    # Top (upper) = 3000, Bottom (lower) = -15000, tick step = 1000


    y_lower = -10000


    y_upper = 3000


    y_tick_step = 1000


    return y_lower, y_upper, y_tick_step








def normalize_compare_label(label: str) -> str:


    """Keeps comparison legends consistent across old and new call sites."""


    if "UCB-H" in label and "[4]" not in label:


        return "UCB-H [4]"


    if "HER" in label and "[17]" not in label:


        return "HER [17]"


    if "greedy" in label.lower():


        return "Epsilon-greedy"


    return label








def get_compare_series_color(label: str) -> str:

    """Maps comparison labels to consistent plotting colors."""

    normalized = normalize_compare_label(label).lower()

    if "algorithm 1" in normalized:

        return "black"

    if "proposed" in normalized:

        return "red"

    if "ucb-h" in normalized:

        return "#2FEA17FE"

    if "greedy" in normalized:

        return "blue"

    if "her" in normalized:

        return "purple"

    return "black"


def plot_reward_trace(episodes: Sequence[int], rewards: Sequence[float],


                      out_path: str = "reward_vs_step.png", window: int = 50,


                      y_lower: int | None = None, y_upper: int | None = None,


                      y_tick_step: int | None = None) -> str:


    """Plots reward vs. step (episode index) and saves to disk."""


    plt.figure(figsize=(6, 6))


    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.3)


    plt.xlabel("Episode", fontsize=14, fontweight="bold")


    plt.ylabel("Cumulative Reward for an Episode", fontsize=14, fontweight="bold")


    ax = plt.gca()


    ax.set_box_aspect(1)


    ax.yaxis.set_label_coords(-0.14, 0.5)


    # No title per request.


    plt.xlim(0,500)


    if y_lower is None or y_upper is None or y_tick_step is None:


        y_lower, y_upper, y_tick_step = compute_reward_axis([rewards])


    plt.ylim(y_lower, y_upper)


    ax.set_xticks(np.arange(0, 501, 50))


    ax.set_yticks(np.arange(y_lower, y_upper + y_tick_step, y_tick_step))


    plt.setp(ax.get_xticklabels(), fontweight="bold")


    plt.setp(ax.get_yticklabels(), fontweight="bold")


    plt.grid(True, linestyle="--", alpha=0.3)


    plt.legend(loc="lower right", fontsize=12)


    ax.set_frame_on(True)


    ax.spines["left"].set_bounds(*ax.get_ylim())


    ax.spines["bottom"].set_bounds(*ax.get_xlim())


    ax.spines["right"].set_visible(True)


    ax.spines["top"].set_visible(True)


    plt.tight_layout()


    # 


    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)


    plt.close()


    return out_path








def plot_reward_comparison_three(stats_a: TrainingStats, label_a: str,


                                 stats_b: TrainingStats, label_b: str,


                                 stats_c: TrainingStats, label_c: str,


                                 out_path: str = "reward_vs_step_compare_three.png",


                                 y_lower: int | None = None, y_upper: int | None = None,


                                 y_tick_step: int | None = None) -> str:


    """Plots reward vs. episode for three runs on the same figure."""


    label_a = normalize_compare_label(label_a)


    label_b = normalize_compare_label(label_b)


    label_c = normalize_compare_label(label_c)


    plt.figure(figsize=(6, 6))


    plt.plot(stats_a.episodes, stats_a.rewards, color=get_compare_series_color(label_a), alpha=0.6, label=label_a)


    plt.plot(stats_b.episodes, stats_b.rewards, color=get_compare_series_color(label_b), alpha=0.6, label=label_b)


    plt.plot(stats_c.episodes, stats_c.rewards, color=get_compare_series_color(label_c), alpha=0.6, label=label_c)


    plt.xlabel("Episode", fontsize=14, fontweight="bold")


    plt.ylabel("Cumulative Reward for an Episode", fontsize=14, fontweight="bold")


    ax = plt.gca()


    ax.set_box_aspect(1)


    ax.yaxis.set_label_coords(-0.14, 0.5)


    plt.grid(True, linestyle="--", alpha=0.3)


    plt.xlim(0, 500)


    if y_lower is None or y_upper is None or y_tick_step is None:


        y_lower, y_upper, y_tick_step = compute_reward_axis(


            [stats_a.rewards, stats_b.rewards, stats_c.rewards]


        )


    plt.ylim(y_lower, y_upper)


    ax.set_xticks(np.arange(0, 501, 50))


    ax.set_yticks(np.arange(y_lower, y_upper + y_tick_step, y_tick_step))


    plt.setp(ax.get_xticklabels(), fontweight="bold")


    plt.setp(ax.get_yticklabels(), fontweight="bold")


    plt.legend(loc="lower right", fontsize=12)


    ax.set_frame_on(True)


    ax.spines["left"].set_bounds(*ax.get_ylim())


    ax.spines["bottom"].set_bounds(*ax.get_xlim())


    ax.spines["right"].set_visible(True)


    ax.spines["top"].set_visible(True)


    plt.tight_layout()


    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)


    plt.close()


    return out_path








def plot_reward_comparison_four(stats_a: TrainingStats, label_a: str,


                                stats_b: TrainingStats, label_b: str,


                                stats_c: TrainingStats, label_c: str,


                                stats_d: TrainingStats, label_d: str,


                                out_path: str = "reward_vs_step_compare_four.png",


                                y_lower: int | None = None, y_upper: int | None = None,


                                y_tick_step: int | None = None) -> str:


    """Plots reward vs. episode for four runs on the same figure."""


    label_a = normalize_compare_label(label_a)


    label_b = normalize_compare_label(label_b)


    label_c = normalize_compare_label(label_c)


    label_d = normalize_compare_label(label_d)


    plt.figure(figsize=(6, 6))


    plt.plot(stats_a.episodes, stats_a.rewards, color=get_compare_series_color(label_a), alpha=0.6, label=label_a)


    plt.plot(stats_b.episodes, stats_b.rewards, color=get_compare_series_color(label_b), alpha=0.6, label=label_b)


    plt.plot(stats_c.episodes, stats_c.rewards, color=get_compare_series_color(label_c), alpha=0.6, label=label_c)


    plt.plot(stats_d.episodes, stats_d.rewards, color=get_compare_series_color(label_d), alpha=0.6, label=label_d)


    plt.xlabel("Episode", fontsize=14, fontweight="bold")


    plt.ylabel("Cumulative Reward for an Episode", fontsize=14, fontweight="bold")


    ax = plt.gca()


    ax.set_box_aspect(1)


    ax.yaxis.set_label_coords(-0.14, 0.5)


    plt.grid(True, linestyle="--", alpha=0.3)


    plt.xlim(0, 500)


    if y_lower is None or y_upper is None or y_tick_step is None:


        y_lower, y_upper, y_tick_step = compute_reward_axis(


            [stats_a.rewards, stats_b.rewards, stats_c.rewards, stats_d.rewards]


        )


    plt.ylim(y_lower, y_upper)


    ax.set_xticks(np.arange(0, 501, 50))


    ax.set_yticks(np.arange(y_lower, y_upper + y_tick_step, y_tick_step))


    plt.setp(ax.get_xticklabels(), fontweight="bold")


    plt.setp(ax.get_yticklabels(), fontweight="bold")


    plt.legend(loc="lower right", fontsize=12)


    ax.set_frame_on(True)


    ax.spines["left"].set_bounds(*ax.get_ylim())


    ax.spines["bottom"].set_bounds(*ax.get_xlim())


    ax.spines["right"].set_visible(True)


    ax.spines["top"].set_visible(True)


    plt.tight_layout()


    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)


    plt.close()


    return out_path








def plot_reward_comparison_five(stats_a: TrainingStats, label_a: str,

                                stats_b: TrainingStats, label_b: str,

                                stats_c: TrainingStats, label_c: str,

                                stats_d: TrainingStats, label_d: str,

                                stats_e: TrainingStats, label_e: str,

                                out_path: str = "reward_vs_step_compare_five.png",

                                y_lower: int | None = None, y_upper: int | None = None,

                                y_tick_step: int | None = None) -> str:

    """Plots reward vs. episode for five runs on the same figure."""

    label_a = normalize_compare_label(label_a)

    label_b = normalize_compare_label(label_b)

    label_c = normalize_compare_label(label_c)

    label_d = normalize_compare_label(label_d)

    label_e = normalize_compare_label(label_e)

    plt.figure(figsize=(6, 6))

    plt.plot(stats_a.episodes, stats_a.rewards, color=get_compare_series_color(label_a), alpha=0.6, label=label_a)

    plt.plot(stats_b.episodes, stats_b.rewards, color=get_compare_series_color(label_b), alpha=0.6, label=label_b)

    plt.plot(stats_c.episodes, stats_c.rewards, color=get_compare_series_color(label_c), alpha=0.6, label=label_c)

    plt.plot(stats_d.episodes, stats_d.rewards, color=get_compare_series_color(label_d), alpha=0.6, label=label_d)

    plt.plot(stats_e.episodes, stats_e.rewards, color=get_compare_series_color(label_e), alpha=0.6, label=label_e)

    plt.xlabel("Episode", fontsize=14, fontweight="bold")

    plt.ylabel("Cumulative Reward for an Episode", fontsize=14, fontweight="bold")

    ax = plt.gca()

    ax.set_box_aspect(1)

    ax.yaxis.set_label_coords(-0.14, 0.5)

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.xlim(0, 500)

    if y_lower is None or y_upper is None or y_tick_step is None:

        y_lower, y_upper, y_tick_step = compute_reward_axis(

            [stats_a.rewards, stats_b.rewards, stats_c.rewards, stats_d.rewards, stats_e.rewards]

        )

    plt.ylim(y_lower, y_upper)

    ax.set_xticks(np.arange(0, 501, 50))

    ax.set_yticks(np.arange(y_lower, y_upper + y_tick_step, y_tick_step))

    plt.setp(ax.get_xticklabels(), fontweight="bold")

    plt.setp(ax.get_yticklabels(), fontweight="bold")

    plt.legend(loc="lower right", fontsize=12)

    ax.set_frame_on(True)

    ax.spines["left"].set_bounds(*ax.get_ylim())

    ax.spines["bottom"].set_bounds(*ax.get_xlim())

    ax.spines["right"].set_visible(True)

    ax.spines["top"].set_visible(True)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)

    plt.close()

    return out_path


def save_maze_visualization(grid: np.ndarray, start: Coordinate, goal: Coordinate,


                            path: Sequence[Coordinate] | None = None,


                            out_path: str = "maze_layout.pdf",


                            marker_style: str = "line") -> str:


    """Saves a color visualization of the maze with optional path overlay."""


    # 


    os.makedirs(os.path.dirname(out_path), exist_ok=True)





    fig, ax = plt.subplots(figsize=(6, 6))


    cmap = colors.ListedColormap(["white", "black"])


    ax.imshow(grid, cmap=cmap, origin="upper")


    ax.scatter(start[1], start[0], c="black", marker="o", s=60, label="Start")


    ax.scatter(goal[1], goal[0], c="red", marker="*", s=80, label="Goal")


    if path:


        rows = [coord[0] for coord in path]


        cols = [coord[1] for coord in path]


        if marker_style == "hollow_blue":


            ax.plot(cols, rows, linestyle="None", marker="o", markersize=3,


                    markerfacecolor="none", markeredgecolor="blue",


                    color="blue", markevery=2, label="Path")


        elif marker_style == "black_dashed":


            ax.plot(cols, rows, color="#2FEA17FE", linewidth=1.2, linestyle="-",


                    marker="^", markersize=3, markerfacecolor="none",


                    markeredgecolor="#2FEA17FE", markevery=2, label="Path")


        elif marker_style == "purple_dashed":


            ax.plot(cols, rows, color="purple", linewidth=1.2, linestyle="-",


                    marker="s", markersize=3, markerfacecolor="none",


                    markeredgecolor="purple", markevery=3, label="Path")


        else:


            ax.plot(cols, rows, color="blue", linewidth=1.5, label="Path")


    ax.set_xticks([])


    ax.set_yticks([])


    # No title per request.


    ax.legend(loc="lower left")


    plt.tight_layout()


    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)


    plt.close(fig)


    return out_path








def save_maze_visualization_compare_three(grid: np.ndarray, start: Coordinate, goal: Coordinate,


                                          path_a: Sequence[Coordinate], label_a: str,


                                          path_b: Sequence[Coordinate], label_b: str,


                                          path_c: Sequence[Coordinate], label_c: str,


                                          out_path: str = "maze_path_compare_three.pdf") -> str:


    """Saves a comparison visualization for three paths."""


    label_a = normalize_compare_label(label_a)


    label_b = normalize_compare_label(label_b)


    label_c = normalize_compare_label(label_c)


    os.makedirs(os.path.dirname(out_path), exist_ok=True)





    fig, ax = plt.subplots(figsize=(6, 6))


    cmap = colors.ListedColormap(["white", "black"])


    ax.imshow(grid, cmap=cmap, origin="upper")


    ax.scatter(start[1], start[0], c="black", marker="o", s=60)


    ax.scatter(goal[1], goal[0], c="red", marker="*", s=80)





    if path_a:


        a_rows = [coord[0] for coord in path_a]


        a_cols = [coord[1] for coord in path_a]


        ax.plot(a_cols, a_rows, color="red", linewidth=1.5, linestyle="--")





    if path_b:


        b_rows = [coord[0] for coord in path_b]


        b_cols = [coord[1] for coord in path_b]


        ax.plot(b_cols, b_rows, color="#2FEA17FE", linewidth=1.2, linestyle="-")





    if path_c:


        c_rows = [coord[0] for coord in path_c]


        c_cols = [coord[1] for coord in path_c]


        ax.plot(c_cols, c_rows, linestyle="None", marker="o", markersize=3,


                markerfacecolor="none", markeredgecolor="blue",


                color="blue", markevery=2)





    ax.set_xticks([])


    ax.set_yticks([])


    legend_handles = [


        Line2D([0], [0], linestyle="None", marker="o", markersize=6,


               markerfacecolor="black", markeredgecolor="black", color="black", label="Start"),


        Line2D([0], [0], linestyle="None", marker="*", markersize=8,


               markerfacecolor="red", markeredgecolor="red", color="red", label="Goal"),


        Line2D([0], [0], color="red", linewidth=1.5, linestyle="--", label=label_a),


        Line2D([0], [0], color="#2FEA17FE", linewidth=1.2, linestyle="-", label=label_b),


        Line2D([0], [0], linestyle="None", marker="o", markersize=4,


               markerfacecolor="none", markeredgecolor="blue", color="blue", label=label_c),


    ]


    ax.legend(handles=legend_handles, loc="lower left")


    plt.tight_layout()


    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)


    plt.close(fig)


    return out_path








def save_maze_visualization_compare_four(grid: np.ndarray, start: Coordinate, goal: Coordinate,


                                         path_a: Sequence[Coordinate], label_a: str,


                                         path_b: Sequence[Coordinate], label_b: str,


                                         path_c: Sequence[Coordinate], label_c: str,


                                         path_d: Sequence[Coordinate], label_d: str,


                                         out_path: str = "maze_path_compare_four.pdf") -> str:


    """Saves a comparison visualization for four paths."""


    label_a = normalize_compare_label(label_a)


    label_b = normalize_compare_label(label_b)


    label_c = normalize_compare_label(label_c)


    label_d = normalize_compare_label(label_d)


    os.makedirs(os.path.dirname(out_path), exist_ok=True)





    fig, ax = plt.subplots(figsize=(6, 6))


    cmap = colors.ListedColormap(["white", "black"])


    ax.imshow(grid, cmap=cmap, origin="upper")


    ax.scatter(start[1], start[0], c="black", marker="o", s=60)


    ax.scatter(goal[1], goal[0], c="red", marker="*", s=80)





    if path_a:


        a_rows = [coord[0] for coord in path_a]


        a_cols = [coord[1] for coord in path_a]


        ax.plot(a_cols, a_rows, color="red", linewidth=1.5, linestyle="--")





    if path_d:


        d_rows = [coord[0] for coord in path_d]


        d_cols = [coord[1] for coord in path_d]


        ax.plot(d_cols, d_rows, color="purple", linewidth=1.2, linestyle="-",


                marker="s", markersize=3, markerfacecolor="none",


                markeredgecolor="purple", markevery=3, alpha=0.65)





    if path_b:


        b_rows = [coord[0] for coord in path_b]


        b_cols = [coord[1] for coord in path_b]


        ax.plot(b_cols, b_rows, color="#2FEA17FE", linewidth=1.5, linestyle="-", zorder=4)





    if path_c:


        c_rows = [coord[0] for coord in path_c]


        c_cols = [coord[1] for coord in path_c]


        ax.plot(c_cols, c_rows, linestyle="None", marker="o", markersize=3,


                markerfacecolor="none", markeredgecolor="blue",


                color="blue", markevery=2)





    ax.set_xticks([])


    ax.set_yticks([])


    legend_handles = [


        Line2D([0], [0], linestyle="None", marker="o", markersize=6,


               markerfacecolor="black", markeredgecolor="black", color="black", label="Start"),


        Line2D([0], [0], linestyle="None", marker="*", markersize=8,


               markerfacecolor="red", markeredgecolor="red", color="red", label="Goal"),


        Line2D([0], [0], color="red", linewidth=1.5, linestyle="--", label=label_a),


        Line2D([0], [0], color="#2FEA17FE", linewidth=1.2, linestyle="-", label=label_b),


        Line2D([0], [0], linestyle="None", marker="o", markersize=4,


               markerfacecolor="none", markeredgecolor="blue", color="blue", label=label_c),


        Line2D([0], [0], color="purple", linewidth=1.2, linestyle="-",


               marker="s", markersize=4, markerfacecolor="none",


               markeredgecolor="purple", label=label_d),


    ]


    ax.legend(handles=legend_handles, loc="lower left")


    plt.tight_layout()


    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)


    plt.close(fig)


    return out_path








def save_path_coordinates(path: Sequence[Coordinate], out_path: str) -> str:


    """Saves a path as step,row,col records for checking plotted Q-greedy routes."""


    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    with open(out_path, "w", encoding="utf-8") as f:


        f.write("step,row,col\n")


        for step, (row, col) in enumerate(path):


            f.write(f"{step},{row},{col}\n")


    return out_path








def save_episode_animation(grid: np.ndarray, start: Coordinate, goal: Coordinate,


                           episode_paths: Sequence[Sequence[Coordinate]],


                           rewards: Sequence[float] | None = None,


                           out_path: str = "episode_paths.gif",


                           interval_ms: int = 120) -> str:


    """Saves an animation that shows each episode's path in sequence."""


    # 


    os.makedirs(os.path.dirname(out_path), exist_ok=True)





    fig, ax = plt.subplots(figsize=(6, 6))


    cmap = colors.ListedColormap(["white", "black"])


    ax.imshow(grid, cmap=cmap, origin="upper")


    ax.scatter(start[1], start[0], c="black", marker="o", s=60, label="Start")


    ax.scatter(goal[1], goal[0], c="red", marker="*", s=80, label="Goal")


    path_line, = ax.plot([], [], color="blue", linewidth=1.5)


    ax.set_xticks([])


    ax.set_yticks([])


    ax.set_title("Episode Paths")


    ax.legend(loc="lower left")





    def init() -> tuple:


        path_line.set_data([], [])


        return (path_line,)





    def update(frame_idx: int) -> tuple:


        path = episode_paths[frame_idx]


        rows = [coord[0] for coord in path]


        cols = [coord[1] for coord in path]


        path_line.set_data(cols, rows)


        if rewards is not None and frame_idx < len(rewards):


            ax.set_title(f"Episode {frame_idx + 1} | reward={rewards[frame_idx]:.2f}")


        else:


            ax.set_title(f"Episode {frame_idx + 1}")


        return (path_line,)





    anim = animation.FuncAnimation(


        fig,


        update,


        init_func=init,


        frames=len(episode_paths),


        interval=interval_ms,


        blit=True,


        repeat=False,


    )


    anim.save(out_path, writer="pillow")


    plt.close(fig)


    return out_path








class RandomNumberPool:


    """Pre-generated random number pool for deterministic sampling."""





    def __init__(self, pool_size: int = 1000000, seed: int = 42):


        """Initializes the random number pool.





        Args:


            pool_size: Number of random values to pre-generate.


            seed: Random generator seed.


        """


        self.pool_size = pool_size


        self.pool = None


        self.index = 0


        self.seed = seed


        self._generate_pool()





    def _generate_pool(self):


        """Generates the random number pool."""


        np.random.seed(self.seed)


        #  [0, 1)


        self.pool = np.random.random(self.pool_size)





    def get_random(self) -> float:


        """Returns one random value from the pool."""


        if self.index >= self.pool_size:


            # ?


            self._refill_pool()





        value = self.pool[self.index]


        self.index += 1


        return value





    def get_random_int(self, low: int, high: int) -> int:


        """Returns a random integer in [low, high).





        Args:


            low: Inclusive lower bound.


            high: Exclusive upper bound.


        """


        rand_float = self.get_random()


        return int(low + rand_float * (high - low))





    def get_random_choice(self, array):


        """Returns a random element from an array."""


        if len(array) == 0:


            return None


        idx = self.get_random_int(0, len(array))


        return array[idx]





    def _refill_pool(self):


        """Refills the random number pool."""


        # 


        new_seed = int(self.seed + self.index)


        np.random.seed(new_seed)


        self.pool = np.random.random(self.pool_size)


        self.index = 0





    def reset(self):


        """Resets the pool index."""


        self.index = 0








class EpisodeRandomPool:


    """Per-episode random number pool."""





    def __init__(self, base_seed: int, pool_size: int = 10000):


        """Initializes the episode random pool.





        Args:


            base_seed: Base random seed.


            pool_size: Random pool size for each episode.


        """


        self.base_seed = base_seed


        self.pool_size = pool_size


        self.episode_pools = {}





    def get_episode_pool(self, episode_idx: int) -> RandomNumberPool:


        """Returns or creates the pool for one episode."""


        if episode_idx not in self.episode_pools:


            # episode?


            episode_seed = self.base_seed + episode_idx * 1000


            self.episode_pools[episode_idx] = RandomNumberPool(


                pool_size=self.pool_size,


                seed=episode_seed


            )


        return self.episode_pools[episode_idx]





    def get_step_pool(self, episode_idx: int, step_idx: int) -> RandomNumberPool:


        """Returns or creates the pool for one episode step."""


        pool_key = (episode_idx, step_idx)


        if pool_key not in self.episode_pools:


            # episodetep?


            step_seed = self.base_seed + episode_idx * 10000 + step_idx * 100


            self.episode_pools[pool_key] = RandomNumberPool(


                pool_size=1000,  # step pool?


                seed=step_seed


            )


        return self.episode_pools[pool_key]








class QLearningUCBHoeffdingSparse:


    """Algorithm 1 implementation with sparse reward awareness."""





    def __init__(self, env: MazeEnv, horizon: int, episodes: int,


                 failure_prob: float = 0.1, bonus_constant: float = 1.0,


                 sparse_fraction: float = 0.1, seed: Optional[int] = None,


                 use_sparse_reward_only: bool = False) -> None:


        self.env = env


        self.H = horizon


        self.K = episodes


        self.S = env.n_states


        self.A = env.n_actions


        self.sparse_fraction = sparse_fraction


        self.sparse_steps = self.sparse_fraction * self.H  # s in sparse reward description.


        self.failure_prob = failure_prob


        self.use_sparse_reward_only = use_sparse_reward_only


        # 


        if seed is None:


            seed = random.randint(0, 1000000)


        self.base_seed = seed





        # ?


        self.episode_random_pool = EpisodeRandomPool(base_seed=seed, pool_size=10000)


        self.eval_random_pool = RandomNumberPool(pool_size=50000, seed=seed + 1000000)


        self.global_random_pool = RandomNumberPool(pool_size=100000, seed=seed + 2000000)





        total_steps = self.H * self.K


        self.iota = math.log(max(self.S * self.A * total_steps / self.failure_prob, 1.0001))


        # Scale c so that b_t = c * s * sqrt(H iota / t) stays well below 1.


        self.c = bonus_constant / (self.sparse_steps * math.sqrt(self.H * self.iota))





        # ?s


        self.q_init_value = self.sparse_steps





        # Q?s


        self.Q = np.full((self.S, self.A), self.q_init_value, dtype=float)





        # V?0


        self.V = np.zeros(self.S, dtype=float)





        self.N = np.zeros((self.S, self.A), dtype=int)





        # Q?


        self.initial_q_min = np.min(self.Q)


        self.initial_q_max = np.max(self.Q)


        self.initial_q_mean = np.mean(self.Q)


        self.initial_q_std = np.std(self.Q)





        # V?


        self.initial_v_min = np.min(self.V)


        self.initial_v_max = np.max(self.V)


        self.initial_v_mean = np.mean(self.V)


        self.initial_v_std = np.std(self.V)





    def train(self, log_interval: int = 50, record_paths: bool = False) -> TrainingStats:


        rewards: List[float] = []


        successes: List[bool] = []  # episode


        episodes_axis: List[int] = []


        episode_paths: List[List[Coordinate]] | None = [] if record_paths else None





        # 


        exploration_stats = {


            'unique_state_actions': []  # episode??


        }





        for episode in range(1, self.K + 1):


            # episode?


            episode_pool = self.episode_random_pool.get_episode_pool(episode - 1)  # ??





            state_coord = self.env.reset()


            if episode_paths is not None:


                episode_coords: List[Coordinate] = [state_coord]


            state = self.env.coord_to_state(state_coord)


            episode_total_reward = 0.0


            episode_success = False  # episode





            # pisode?


            episode_unique_sa = set()





            for h in range(self.H):


                # step?


                step_pool = self.episode_random_pool.get_step_pool(episode - 1, h)





                action = self._greedy_action_with_pool(state, step_pool)





                episode_unique_sa.add((state, action))





                next_coord, reward, done, info = self.env.step(action)


                if self.use_sparse_reward_only:


                    reward = sparse_goal_reward(next_coord, self.env.goal)





                if episode_paths is not None:


                    episode_coords.append(next_coord)


                next_state = self.env.coord_to_state(next_coord)


                t = self.N[state, action] + 1


                self.N[state, action] = t





                # ?


                episode_total_reward += reward





                # ?


                if next_coord == self.env.goal:


                    episode_success = True





                alpha = (self.H + 1) / (self.H + t)


                bonus = self.c * self.sparse_steps * math.sqrt((self.H * self.iota) / max(t, 1))


                next_value = 0.0 if h == self.H - 1 or done else self.V[next_state]


                target = reward + next_value + bonus


                self.Q[state, action] = (1 - alpha) * self.Q[state, action] + alpha * target


                self.V[state] = min(self.sparse_steps, np.max(self.Q[state]))


                state = next_state


                if done:


                    break





            rewards.append(episode_total_reward)


            successes.append(episode_success)  # 


            episodes_axis.append(episode)


            if episode_paths is not None:


                episode_paths.append(episode_coords)





            # 


            exploration_stats['unique_state_actions'].append(len(episode_unique_sa))





            if episode % log_interval == 0:


                recent_rewards = rewards[-log_interval:]


                recent_successes = successes[-log_interval:]





                avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0


                success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0





                recent_unique = exploration_stats['unique_state_actions'][-log_interval:]


                avg_unique = sum(recent_unique) / len(recent_unique) if recent_unique else 0





                print(f"Episode {episode:4d}/{self.K}: "


                      f"avg reward={avg_reward:.2f}, "


                      f"success rate={success_rate:.2f}, "


                      f"unique SA={avg_unique:.1f}")





                # Q?


                if episode % (log_interval * 5) == 0:


                    self._print_value_statistics(episode)





        # ?


        final_successes = successes[-min(100, len(successes)):]


        success_rate = sum(final_successes) / max(1, len(final_successes))





        # ?


        self._print_exploration_summary(exploration_stats)





        return TrainingStats(


            episodes=episodes_axis,


            rewards=rewards,


            successes=successes,


            success_rate=success_rate,


            episode_paths=episode_paths,


        )





    def _print_value_statistics(self, episode: int):


        """Prints Q and V value statistics."""


        negative_q_count = np.sum(self.Q < 0)


        positive_q_count = np.sum(self.Q > 0)





        print(f"  Value-statistics at episode {episode}:")


        print(f"    Negative Q-values: {negative_q_count} ({negative_q_count / (self.S * self.A) * 100:.1f}%)")


        print(f"    Positive Q-values: {positive_q_count} ({positive_q_count / (self.S * self.A) * 100:.1f}%)")


        print(f"    Q min/max: {np.min(self.Q):.2f}/{np.max(self.Q):.2f}")


        print(f"    Q mean/std: {np.mean(self.Q):.2f}/{np.std(self.Q):.2f}")


        print(f"    V min/max: {np.min(self.V):.2f}/{np.max(self.V):.2f}")


        print(f"    V mean/std: {np.mean(self.V):.2f}/{np.std(self.V):.2f}")





    def _print_exploration_summary(self, exploration_stats: dict):


        """Prints exploration statistics."""


        print("\n" + "=" * 60)


        print(":")


        print("=" * 60)





        total_unique = sum(exploration_stats['unique_state_actions'])


        avg_unique = total_unique / len(exploration_stats['unique_state_actions'])


        max_unique = max(exploration_stats['unique_state_actions'])





        print(f"pisode?? {avg_unique:.1f} (? {max_unique})")


        print(f"?? {total_unique}")


        print(f"?? {self.S * self.A}")


        print(f"? {total_unique / (self.S * self.A) * 100:.1f}%")





    def _greedy_action_with_pool(self, state: int, random_pool: RandomNumberPool) -> int:


        """Selects a greedy action using the random pool for tie-breaking."""


        q_values = self.Q[state]


        max_value = np.max(q_values)


        max_actions = np.flatnonzero(np.isclose(q_values, max_value))





        # 


        if len(max_actions) == 1:


            return int(max_actions[0])


        else:


            # ?


            return int(max_actions[random_pool.get_random_int(0, len(max_actions))])





    def greedy_path_from_q(self, env: MazeEnv, deterministic: bool = True,


                           episode_seed: Optional[int] = None) -> List[Coordinate]:


        """Follows the greedy policy derived from the Q-table to get a path."""


        return q_greedy_path_to_goal(self.Q, env, self.H)





    def rollout(self, env: MazeEnv, episode_seed: Optional[int] = None) -> Tuple[List[Coordinate], float, bool]:


        """Follows the greedy policy for a single episode and returns the positions, total reward, and success status."""


        # 


        if episode_seed is None:


            episode_seed = int(self.eval_random_pool.get_random() * 1000000)


        eval_pool = RandomNumberPool(pool_size=1000, seed=episode_seed)





        coord = env.reset()


        coords = [coord]


        state = env.coord_to_state(coord)


        total_reward = 0.0


        success = False  # 





        for h in range(self.H):


            # tep?


            step_pool = RandomNumberPool(pool_size=100, seed=episode_seed + h * 100)





            q_values = self.Q[state]


            max_value = np.max(q_values)


            max_actions = np.flatnonzero(np.isclose(q_values, max_value))





            # step


            if len(max_actions) == 1:


                action = int(max_actions[0])


            else:


                action = int(max_actions[step_pool.get_random_int(0, len(max_actions))])





            coord, reward, done, _ = env.step(action)


            if self.use_sparse_reward_only:


                reward = sparse_goal_reward(coord, env.goal)


            coords.append(coord)


            total_reward += reward





            # ?


            if coord == env.goal:


                success = True





            state = env.coord_to_state(coord)


            if done:


                break


        return coords, total_reward, success





    def evaluate(self, env: MazeEnv, episodes: int = 50) -> Tuple[float, float]:


        """Estimates the success rate and average reward of the learned policy."""


        successes = 0


        total_rewards = 0.0





        for eval_episode in range(episodes):


            # 


            eval_pool = self.eval_random_pool





            coord = env.reset()


            state = env.coord_to_state(coord)


            episode_reward = 0.0


            episode_success = False





            for h in range(self.H):


                # step?


                step_pool = RandomNumberPool(


                    pool_size=100,


                    seed=int(self.global_random_pool.get_random() * 1000000 + h * 100)


                )





                q_values = self.Q[state]


                max_value = np.max(q_values)


                max_actions = np.flatnonzero(np.isclose(q_values, max_value))





                # step


                if len(max_actions) == 1:


                    action = int(max_actions[0])


                else:


                    action = int(max_actions[step_pool.get_random_int(0, len(max_actions))])





                coord, reward, done, _ = env.step(action)


                if self.use_sparse_reward_only:


                    reward = sparse_goal_reward(coord, env.goal)


                episode_reward += reward





                # ?


                if coord == env.goal:


                    episode_success = True





                state = env.coord_to_state(coord)


                if done:


                    if episode_success:


                        successes += 1


                    break





            total_rewards += episode_reward





        success_rate = successes / episodes


        avg_reward = total_rewards / episodes


        return success_rate, avg_reward





    def save_q_table_with_coordinates(self, filepath: str, env: MazeEnv) -> None:


        """Saves the Q table with coordinates and action names."""


        # 


        os.makedirs(os.path.dirname(filepath), exist_ok=True)





        with open(filepath, 'w') as f:


            # 


            f.write("# Q-Table with State Coordinates and Action Descriptions\n")


            f.write("# ======================================================\n")


            f.write(f"# Maze Size: {env.size}x{env.size}\n")


            f.write(f"# Number of States: {self.S}\n")


            f.write(f"# Number of Actions: {self.A}\n")


            f.write(f"# Start: {env.start}\n")


            f.write(f"# Goal: {env.goal}\n")


            f.write(f"# Training Episodes: {self.K}\n")


            f.write(f"# Horizon: {self.H}\n")


            f.write(f"# Initial Q Value: {self.q_init_value}\n")


            f.write("# ======================================================\n\n")





            # 


            f.write("# Action Mapping:\n")


            for i, (dr, dc) in enumerate(env.ACTIONS):


                action_name = env.ACTION_NAMES[i] if i < len(env.ACTION_NAMES) else f"ACTION_{i}"


                f.write(f"#   Action {i}: {action_name} (move: {dr}, {dc})\n")


            f.write("\n")





            # Q?


            f.write("# State Information and Q-Values:\n")


            f.write(


                "# Format: StateID (row, col) | IsStart | IsGoal | IsWall | UP | RIGHT | DOWN | LEFT | STAY | BestAction\n")


            f.write(


                "# ---------------------------------------------------------------------------------------------------\n")





            for state in range(self.S):


                coord = env.state_to_coord(state)


                row, col = coord





                # ?


                is_start = coord == env.start


                is_goal = coord == env.goal


                is_wall = env.grid[row, col] == 1 if 0 <= row < env.size and 0 <= col < env.size else True





                # ?


                q_values = self.Q[state]


                best_action_idx = int(np.argmax(q_values))


                best_action_name = env.ACTION_NAMES[best_action_idx] if best_action_idx < len(


                    env.ACTION_NAMES) else f"ACTION_{best_action_idx}"





                # ?


                f.write(f"State {state:3d} ({row:2d}, {col:2d}): ")


                f.write(f"Start={1 if is_start else 0}, ")


                f.write(f"Goal={1 if is_goal else 0}, ")


                f.write(f"Wall={1 if is_wall else 0}")


                f.write(" | ")





                # Q?


                for action in range(self.A):


                    q_value = self.Q[state, action]


                    f.write(f"{q_value:8.4f} ")





                # ?


                f.write(f"| Best: {best_action_name} (Action {best_action_idx})\n")





            # 


            f.write("\n# Best Policy Summary:\n")


            f.write("# --------------------\n")





            for state in range(self.S):


                coord = env.state_to_coord(state)


                row, col = coord


                is_wall = env.grid[row, col] == 1 if 0 <= row < env.size and 0 <= col < env.size else True





                if is_wall:


                    continue  # ?





                q_values = self.Q[state]


                best_action_idx = int(np.argmax(q_values))


                best_action_name = env.ACTION_NAMES[best_action_idx] if best_action_idx < len(


                    env.ACTION_NAMES) else f"ACTION_{best_action_idx}"


                f.write(f"  ({row:2d}, {col:2d}) -> {best_action_name}\n")





    def save_visit_counts_with_coordinates(self, filepath: str, env: MazeEnv) -> None:


        """Saves visit counts with coordinates and action names."""


        # 


        os.makedirs(os.path.dirname(filepath), exist_ok=True)





        with open(filepath, 'w') as f:


            # 


            f.write("# Visit Counts with State Coordinates and Action Descriptions\n")


            f.write("# ===========================================================\n")


            f.write(f"# Maze Size: {env.size}x{env.size}\n")


            f.write(f"# Number of States: {self.S}\n")


            f.write(f"# Number of Actions: {self.A}\n")


            f.write("# ===========================================================\n\n")





            # 


            f.write("# Action Mapping:\n")


            for i, (dr, dc) in enumerate(env.ACTIONS):


                action_name = env.ACTION_NAMES[i] if i < len(env.ACTION_NAMES) else f"ACTION_{i}"


                f.write(f"#   Action {i}: {action_name} (move: {dr}, {dc})\n")


            f.write("\n")





            # 


            f.write("# State Information and Visit Counts:\n")


            f.write("# Format: StateID (row, col) | TotalVisits | UP | RIGHT | DOWN | LEFT | STAY\n")


            f.write("# -------------------------------------------------------------------------\n")





            for state in range(self.S):


                coord = env.state_to_coord(state)


                row, col = coord





                # ?


                total_visits = np.sum(self.N[state, :])





                # ?


                f.write(f"State {state:3d} ({row:2d}, {col:2d}): ")


                f.write(f"Total={total_visits:6d} | ")





                # 


                for action in range(self.A):


                    visit_count = self.N[state, action]


                    f.write(f"{visit_count:6d} ")


                f.write("\n")





            # 


            f.write("\n# Visit Counts Statistics:\n")


            f.write("# -----------------------\n")


            total_all_visits = np.sum(self.N)


            f.write(f"Total visits across all states and actions: {total_all_visits}\n")





            # 


            if total_all_visits > 0:


                max_visit_state = np.unravel_index(np.argmax(self.N), self.N.shape)[0]


                max_visit_action = np.unravel_index(np.argmax(self.N), self.N.shape)[1]


                max_visit_coord = env.state_to_coord(max_visit_state)


                max_visit_count = np.max(self.N)


                max_action_name = env.ACTION_NAMES[max_visit_action] if max_visit_action < len(


                    env.ACTION_NAMES) else f"ACTION_{max_visit_action}"





                f.write(


                    f"Most visited state-action pair: State {max_visit_state} ({max_visit_coord[0]}, {max_visit_coord[1]}) ")


                f.write(f"Action {max_visit_action} ({max_action_name}) with {max_visit_count} visits\n")


            else:


                f.write("No visits recorded yet.\n")





    def save_q_table_summary(self, filepath: str, env: MazeEnv) -> None:


        """Saves summary statistics for the Q table."""


        # 


        os.makedirs(os.path.dirname(filepath), exist_ok=True)





        with open(filepath, 'w') as f:


            f.write("# Q-Table and V-Table Summary Statistics\n")


            f.write("# ======================================\n")


            f.write(f"# Maze Size: {env.size}x{env.size}\n")


            f.write(f"# Q-table Shape: {self.Q.shape}\n")


            f.write(f"# V-table Shape: {self.V.shape}\n")


            f.write(f"# Initial Q Value: {self.q_init_value}\n")


            f.write("# ======================================\n\n")





            # Q?


            f.write("## Initial Q-values Statistics:\n")


            f.write(f"  Initial Min: {self.initial_q_min}\n")


            f.write(f"  Initial Max: {self.initial_q_max}\n")


            f.write(f"  Initial Mean: {self.initial_q_mean}\n")


            f.write(f"  Initial Std: {self.initial_q_std}\n\n")





            # V?


            f.write("## Initial V-values Statistics:\n")


            f.write(f"  Initial Min: {self.initial_v_min}\n")


            f.write(f"  Initial Max: {self.initial_v_max}\n")


            f.write(f"  Initial Mean: {self.initial_v_mean}\n")


            f.write(f"  Initial Std: {self.initial_v_std}\n\n")





            # ?


            f.write("## Final Q-values Statistics:\n")


            f.write(f"  Min Q-value: {np.min(self.Q):.6f}\n")


            f.write(f"  Max Q-value: {np.max(self.Q):.6f}\n")


            f.write(f"  Mean Q-value: {np.mean(self.Q):.6f}\n")


            f.write(f"  Std Q-value: {np.std(self.Q):.6f}\n")





            # ?


            f.write("\n## Final V-values Statistics:\n")


            f.write(f"  Min V-value: {np.min(self.V):.6f}\n")


            f.write(f"  Max V-value: {np.max(self.V):.6f}\n")


            f.write(f"  Mean V-value: {np.mean(self.V):.6f}\n")


            f.write(f"  Std V-value: {np.std(self.V):.6f}\n")





            # Q?


            positive_q = np.sum(self.Q > 0)


            negative_q = np.sum(self.Q < 0)


            zero_q = np.sum(self.Q == 0)


            total_q = self.Q.size





            f.write(f"\n## Q-value Distribution:\n")


            f.write(f"  Positive Q-values: {positive_q} ({positive_q / total_q * 100:.2f}%)\n")


            f.write(f"  Negative Q-values: {negative_q} ({negative_q / total_q * 100:.2f}%)\n")


            f.write(f"  Zero Q-values: {zero_q} ({zero_q / total_q * 100:.2f}%)\n")





            # ?


            f.write(f"\n## Best Policy Analysis:\n")


            best_actions = np.argmax(self.Q, axis=1)


            action_counts = {i: np.sum(best_actions == i) for i in range(self.A)}





            for action_idx, count in action_counts.items():


                action_name = env.ACTION_NAMES[action_idx] if action_idx < len(


                    env.ACTION_NAMES) else f"ACTION_{action_idx}"


                percentage = count / self.S * 100


                f.write(f"  {action_name}: {count} states ({percentage:.2f}%)\n")











class QLearningEpsilonGreedy:


    """Independent epsilon-greedy Q-learning baseline."""





    def __init__(self, env: MazeEnv, horizon: int, episodes: int,


                 alpha: float = 0.2, gamma: float = 0.99,


                 epsilon: float = 0.2, epsilon_min: float = 0,


                 epsilon_decay: float = 1.0, seed: Optional[int] = None,


                 use_sparse_reward_only: bool = False) -> None:


        self.env = env


        self.H = horizon


        self.K = episodes


        self.S = env.n_states


        self.A = env.n_actions


        self.alpha = alpha


        self.gamma = gamma


        self.epsilon = epsilon


        self.epsilon_min = epsilon_min


        self.epsilon_decay = epsilon_decay


        self.use_sparse_reward_only = use_sparse_reward_only


        self.rng = random.Random(seed)





        self.Q = np.zeros((self.S, self.A), dtype=float)


        self.N = np.zeros((self.S, self.A), dtype=int)





    def _select_action(self, state: int, epsilon: float) -> int:


        if self.rng.random() < epsilon:


            return self.rng.randrange(self.A)


        q_values = self.Q[state]


        max_q = np.max(q_values)


        best_actions = np.flatnonzero(q_values == max_q)


        return int(self.rng.choice(best_actions))





    def train(self, log_interval: int = 50, record_paths: bool = False) -> TrainingStats:


        rewards: List[float] = []


        successes: List[bool] = []


        episodes_axis: List[int] = []


        episode_paths: List[List[Coordinate]] | None = [] if record_paths else None





        eps = self.epsilon


        for episode in range(1, self.K + 1):


            state_coord = self.env.reset()


            state = self.env.coord_to_state(state_coord)


            if episode_paths is not None:


                episode_coords: List[Coordinate] = [state_coord]





            episode_total_reward = 0.0


            episode_success = False





            for _ in range(self.H):


                action = self._select_action(state, eps)


                next_coord, reward, done, info = self.env.step(action)


                if self.use_sparse_reward_only:


                    reward = sparse_goal_reward(next_coord, self.env.goal)


                next_state = self.env.coord_to_state(next_coord)





                td_target = reward + self.gamma * np.max(self.Q[next_state])


                td_error = td_target - self.Q[state, action]


                self.Q[state, action] += self.alpha * td_error


                self.N[state, action] += 1





                episode_total_reward += reward


                if next_coord == self.env.goal:


                    episode_success = True





                state = next_state


                if episode_paths is not None:


                    episode_coords.append(next_coord)


                if done:


                    break





            if episode_paths is not None:


                episode_paths.append(episode_coords)





            rewards.append(episode_total_reward)


            successes.append(episode_success)


            episodes_axis.append(episode)





            if log_interval and episode % log_interval == 0:


                success_rate = sum(successes[-log_interval:]) / log_interval


                print(f"[EpsilonGreedy] Episode {episode:4d} | eps={eps:.4f} | "


                      f"reward={episode_total_reward:.2f} | success={success_rate:.2f}")





        success_rate = sum(successes) / max(1, len(successes))


        return TrainingStats(


            episodes=episodes_axis,


            rewards=rewards,


            successes=successes,


            success_rate=success_rate,


            episode_paths=episode_paths,


        )





    def evaluate(self, env: MazeEnv, episodes: int = 100) -> Tuple[float, float]:


        success_count = 0


        rewards: List[float] = []





        for _ in range(episodes):


            state_coord = env.reset()


            state = env.coord_to_state(state_coord)


            episode_reward = 0.0


            episode_success = False





            for _ in range(self.H):


                action = int(np.argmax(self.Q[state]))


                next_coord, reward, done, info = env.step(action)


                if self.use_sparse_reward_only:


                    reward = sparse_goal_reward(next_coord, env.goal)


                next_state = env.coord_to_state(next_coord)


                episode_reward += reward


                if next_coord == env.goal:


                    episode_success = True


                state = next_state


                if done:


                    break





            if episode_success:


                success_count += 1


            rewards.append(episode_reward)





        success_rate = success_count / max(1, episodes)


        avg_reward = float(np.mean(rewards)) if rewards else 0.0


        return success_rate, avg_reward





    def rollout(self, env: MazeEnv) -> Tuple[List[Coordinate], float, bool]:


        state_coord = env.reset()


        state = env.coord_to_state(state_coord)


        path: List[Coordinate] = [state_coord]


        total_reward = 0.0


        success = False





        for _ in range(self.H):


            action = int(np.argmax(self.Q[state]))


            next_coord, reward, done, info = env.step(action)


            if self.use_sparse_reward_only:


                reward = sparse_goal_reward(next_coord, env.goal)


            next_state = env.coord_to_state(next_coord)


            total_reward += reward


            path.append(next_coord)


            if next_coord == env.goal:


                success = True


            state = next_state


            if done:


                break





        return path, total_reward, success





    def greedy_path_from_q(self, env: MazeEnv, deterministic: bool = True) -> List[Coordinate]:


        """Follows the greedy policy derived from the Q-table to get a path."""


        return q_greedy_path_to_goal(self.Q, env, self.H)








class QLearningHER:


    """Tabular goal-conditioned Q-learning with hindsight experience replay."""





    def __init__(self, env: MazeEnv, horizon: int, episodes: int,


                 alpha: float = 0.2, gamma: float = 0.99,


                 epsilon: float = 0.2, her_k: int = 4,


                 seed: Optional[int] = None) -> None:


        self.env = env


        self.H = horizon


        self.K = episodes


        self.S = env.n_states


        self.A = env.n_actions


        self.alpha = alpha


        self.gamma = gamma


        self.epsilon = epsilon


        self.her_k = her_k


        self.rng = random.Random(seed)





        self.goal_Q = np.zeros((self.S, self.S, self.A), dtype=float)


        self.N = np.zeros((self.S, self.S, self.A), dtype=int)


        self.Q = self._q_for_goal(env.goal).copy()





    def _q_for_goal(self, goal: Coordinate) -> np.ndarray:


        goal_state = self.env.coord_to_state(goal)


        return self.goal_Q[:, goal_state, :]





    def _select_action(self, state: int, goal_state: int, epsilon: float) -> int:


        if self.rng.random() < epsilon:


            return self.rng.randrange(self.A)


        q_values = self.goal_Q[state, goal_state]


        max_q = np.max(q_values)


        best_actions = np.flatnonzero(q_values == max_q)


        return int(self.rng.choice(best_actions))





    def _goal_reward(self, next_coord: Coordinate, goal_state: int,


                     env_reward: float) -> float:


        goal_coord = self.env.state_to_coord(goal_state)


        if next_coord == goal_coord:


            return 1.0


        if env_reward < 0:


            return env_reward


        return 0.0





    def _update(self, state: int, action: int, env_reward: float,


                next_state: int, next_coord: Coordinate,


                goal_state: int) -> None:


        reward = self._goal_reward(next_coord, goal_state, env_reward)


        td_target = reward + self.gamma * np.max(self.goal_Q[next_state, goal_state])


        td_error = td_target - self.goal_Q[state, goal_state, action]


        self.goal_Q[state, goal_state, action] += self.alpha * td_error


        self.N[state, goal_state, action] += 1





    def train(self, log_interval: int = 50, record_paths: bool = False) -> TrainingStats:


        rewards: List[float] = []


        successes: List[bool] = []


        episodes_axis: List[int] = []


        episode_paths: List[List[Coordinate]] | None = [] if record_paths else None


        true_goal_state = self.env.coord_to_state(self.env.goal)





        for episode in range(1, self.K + 1):


            state_coord = self.env.reset()


            state = self.env.coord_to_state(state_coord)


            episode_total_reward = 0.0


            episode_success = False


            episode_coords: List[Coordinate] = [state_coord]


            transitions: List[Tuple[int, int, float, int, Coordinate]] = []





            for _ in range(self.H):


                action = self._select_action(state, true_goal_state, self.epsilon)


                next_coord, reward, done, _ = self.env.step(action)


                next_state = self.env.coord_to_state(next_coord)





                self._update(state, action, reward, next_state, next_coord, true_goal_state)


                transitions.append((state, action, reward, next_state, next_coord))





                episode_total_reward += reward


                episode_coords.append(next_coord)


                if next_coord == self.env.goal:


                    episode_success = True





                state = next_state


                if done:


                    break





            for idx, (state, action, reward, next_state, next_coord) in enumerate(transitions):


                future_coords = episode_coords[idx + 1:]


                if not future_coords:


                    continue


                sample_count = min(self.her_k, len(future_coords))


                for hindsight_goal in self.rng.sample(future_coords, sample_count):


                    goal_state = self.env.coord_to_state(hindsight_goal)


                    self._update(state, action, reward, next_state, next_coord, goal_state)





            if episode_paths is not None:


                episode_paths.append(episode_coords)


            rewards.append(episode_total_reward)


            successes.append(episode_success)


            episodes_axis.append(episode)





            if log_interval and episode % log_interval == 0:


                success_rate = sum(successes[-log_interval:]) / log_interval


                print(f"[HER] Episode {episode:4d} | "


                      f"reward={episode_total_reward:.2f} | success={success_rate:.2f}")





        self.Q = self._q_for_goal(self.env.goal).copy()


        success_rate = sum(successes) / max(1, len(successes))


        return TrainingStats(


            episodes=episodes_axis,


            rewards=rewards,


            successes=successes,


            success_rate=success_rate,


            episode_paths=episode_paths,


        )





    def evaluate(self, env: MazeEnv, episodes: int = 100) -> Tuple[float, float]:


        success_count = 0


        rewards: List[float] = []


        goal_state = env.coord_to_state(env.goal)





        for _ in range(episodes):


            state_coord = env.reset()


            state = env.coord_to_state(state_coord)


            episode_reward = 0.0


            episode_success = False





            for _ in range(self.H):


                action = int(np.argmax(self.goal_Q[state, goal_state]))


                next_coord, reward, done, _ = env.step(action)


                next_state = env.coord_to_state(next_coord)


                episode_reward += reward


                if next_coord == env.goal:


                    episode_success = True


                state = next_state


                if done:


                    break





            if episode_success:


                success_count += 1


            rewards.append(episode_reward)





        self.Q = self._q_for_goal(env.goal).copy()


        return success_count / max(1, episodes), float(np.mean(rewards)) if rewards else 0.0





    def rollout(self, env: MazeEnv) -> Tuple[List[Coordinate], float, bool]:


        state_coord = env.reset()


        state = env.coord_to_state(state_coord)


        goal_state = env.coord_to_state(env.goal)


        path: List[Coordinate] = [state_coord]


        total_reward = 0.0


        success = False





        for _ in range(self.H):


            action = int(np.argmax(self.goal_Q[state, goal_state]))


            next_coord, reward, done, _ = env.step(action)


            next_state = env.coord_to_state(next_coord)


            total_reward += reward


            path.append(next_coord)


            if next_coord == env.goal:


                success = True


            state = next_state


            if done:


                break





        self.Q = self._q_for_goal(env.goal).copy()


        return path, total_reward, success





    def greedy_path_from_q(self, env: MazeEnv, deterministic: bool = True) -> List[Coordinate]:


        self.Q = self._q_for_goal(env.goal).copy()


        return q_greedy_path_to_goal(self.Q, env, self.H)








def ensure_output_dir() -> str:


    """Ensures that the output directory exists and returns its path."""


    timestamp = time.strftime("%Y%m%d_%H%M%S")


    output_dir = timestamp





    # 


    os.makedirs(output_dir, exist_ok=True)





    # ?


    subdirs = ["plots", "visualizations", "animations", "data", "tables"]


    for subdir in subdirs:


        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)





    return output_dir








def main(**kwargs) -> None:


    """Runs the configured experiment batch."""





    # ?


    config = {


        "experiment_name": "maze_experiment",


        "base_seed": None,


        "output_dir": None,


        "maze_size": 15,


        "maze_seed": 1,


        "env_seed": 42,


        "episodes": 500,


        "horizon": 2000,


        "failure_prob": 0.1,


        "bonus_constant": 1.0,


        "sparse_fraction": 0.01,


        "wall_penalty": -100.0,


        "stay_penalty": 0.0,


        "move_penalty": 0.0,


        "log_interval": 100,


        "eval_episodes": 500,


        "record_paths": True,


    }





    # ?


    seed = kwargs.get('seed', config.get('base_seed'))


    output_dir = kwargs.get('output_dir', config.get('output_dir'))


    wall_penalty = kwargs.get('wall_penalty', config.get('wall_penalty'))


    stay_penalty = kwargs.get('stay_penalty', config.get('stay_penalty'))


    reward_shaping_move_penalty = kwargs.get('move_penalty', config.get('move_penalty'))


    episodes = kwargs.get('episodes', config.get('episodes'))


    horizon = kwargs.get('horizon', config.get('horizon'))





    # 


    failure_prob = config.get('failure_prob', 0.1)


    bonus_constant = config.get('bonus_constant', 0.1)


    sparse_fraction = config.get('sparse_fraction', 0.01)


    log_interval = config.get('log_interval', 100)


    eval_episodes = config.get('eval_episodes', 500)


    record_paths = config.get('record_paths', True)

    # If caller requested a quick Algorithm 1 comparison, run it and exit.
    if kwargs.get('run_algo1_compare', False):
        out = kwargs.get('output_dir', None) or ensure_output_dir()
        run_algorithm1_comparison(
            output_dir=out,
            maze_size=kwargs.get('maze_size', config.get('maze_size')),
            maze_seed=kwargs.get('maze_seed', config.get('maze_seed')),
            episodes=kwargs.get('episodes', config.get('episodes')),
            horizon=kwargs.get('horizon', config.get('horizon')),
            seed=kwargs.get('seed', seed),
        )
        return





    if seed is None:


        seed = random.randint(0, 1000000)





    # Run experiment batch and exit.


    if output_dir is None:


        output_dir = ensure_output_dir()


    else:


        os.makedirs(output_dir, exist_ok=True)





    base_sparse_fraction = sparse_fraction


    sparse_one = 1.0


    experiments = []


    scenario_templates = [


        {


            "scenario_key": "maze_seed_1_size_20",


            "scenario_label": "maze1_20x20_a",


            "maze_seed": 1,


            "maze_size": 20,


            "proposed_bonus_constant": bonus_constant,


            "proposed_sparse_fraction": base_sparse_fraction,


            "proposed_use_sparse_reward_only": True,


            "proposed_use_reward_shaping": False,


            "compare_plot_filename": "Fig2.pdf",


            "compare_path_filename": "Fig1.pdf",


        },


        {


            "scenario_key": "maze_seed_1_size_20_b",


            "scenario_label": "maze1_20x20_b",


            "maze_seed": 1,


            "maze_size": 20,


            "proposed_bonus_constant": bonus_constant,


            "proposed_sparse_fraction": 1.0,


            "proposed_use_sparse_reward_only": False,


            "proposed_use_reward_shaping": True,


            "compare_plot_filename": "Fig4.pdf",


            "compare_path_filename": "Fig3.pdf",


        },


        {


            "scenario_key": "maze_seed_1_size_20_c",


            "scenario_label": "maze1_20x20_c",


            "maze_seed": 1,


            "maze_size": 20,


            "proposed_bonus_constant": bonus_constant,


            "proposed_sparse_fraction": 1.0 / 2000.0,


            "proposed_use_sparse_reward_only": False,


            "proposed_use_reward_shaping": True,


            "compare_plot_filename": "Fig6.pdf",


            "compare_path_filename": "Fig5.pdf",


        },


        {


            "scenario_key": "maze_seed_1_size_20_d",


            "scenario_label": "maze1_20x20_d",


            "maze_seed": 1,


            "maze_size": 20,


            "proposed_bonus_constant": bonus_constant,


            "proposed_sparse_fraction": 0.01,


            "proposed_use_sparse_reward_only": False,


            "proposed_use_reward_shaping": True,


            "compare_plot_filename": "Fig8.pdf",


            "compare_path_filename": "Fig7.pdf",


        },


    ]


    algorithm_templates = [


        {


            "algo": "ucb",


            "variant": "base",


            "output_name": "ucb",


            "label": "UCB",


            "sparse_fraction": base_sparse_fraction,


        },


        {


            "algo": "ucb",


            "variant": "algo1",


            "output_name": "algo1",


            "label": "Algorithm 1",


            "sparse_fraction": sparse_one,


        },


        {


            "algo": "ucb",


            "variant": "sparse1",


            "output_name": "ucb_h",


            "label": "UCB-H [4]",


            "sparse_fraction": sparse_one,


        },


        {


            "algo": "eps",


            "variant": "eps",


            "output_name": "eps",


            "label": "Epsilon-greedy",


            "sparse_fraction": base_sparse_fraction,


        },


        {


            "algo": "her",


            "variant": "her",


            "output_name": "her",


            "label": "HER [17]",


            "sparse_fraction": base_sparse_fraction,


        },


    ]


    experiments = []


    for scenario in scenario_templates:


        for algo in algorithm_templates:


            exp = {**scenario, **algo}


            if algo["variant"] == "base":


                exp["bonus_constant"] = scenario.get("proposed_bonus_constant", bonus_constant)


                exp["sparse_fraction"] = scenario.get("proposed_sparse_fraction", base_sparse_fraction)


                exp["use_sparse_reward_only"] = scenario.get("proposed_use_sparse_reward_only", False)


                exp["use_reward_shaping"] = scenario.get("proposed_use_reward_shaping", False)


            else:


                exp["use_reward_shaping"] = False


            experiments.append(exp)





    reward_axis_series: List[Sequence[float]] = []


    group_results_all: Dict[str, Dict[str, TrainingStats]] = {}


    q_optimal_paths_all: Dict[str, Dict[str, List[Coordinate]]] = {}


    scenario_meta: Dict[str, Dict[str, object]] = {}





    for exp in experiments:


        maze_size = int(exp.get("maze_size", config.get('maze_size', 15)))


        scenario_key = exp.get("scenario_key", f"maze_seed_{exp['maze_seed']}_size_{maze_size}")


        scenario_label = exp.get("scenario_label", f"maze{exp['maze_seed']}_{maze_size}x{maze_size}")


        compare_plot_filename = exp.get(


            "compare_plot_filename",


            "Fig4.pdf" if exp["maze_seed"] == 1 else "Fig3.pdf" if exp["maze_seed"] == 12


            else f"reward_vs_step_compare_three_{scenario_key}.pdf"


        )


        compare_path_filename = exp.get(


            "compare_path_filename",


            "Fig2.pdf" if exp["maze_seed"] == 1 else "Fig1.pdf" if exp["maze_seed"] == 12


            else f"q_greedy_path_compare_three_{scenario_key}.pdf"


        )


        scenario_meta[scenario_key] = {


            "maze_seed": exp["maze_seed"],


            "maze_size": maze_size,


            "compare_plot_filename": compare_plot_filename,


            "compare_path_filename": compare_path_filename,


        }


        group_dir = os.path.join(output_dir, scenario_key)


        exp_output_dir = group_dir


        os.makedirs(exp_output_dir, exist_ok=True)


        subdirs = ["plots", "visualizations", "animations", "data", "tables"]


        for subdir in subdirs:


            os.makedirs(os.path.join(exp_output_dir, subdir), exist_ok=True)





        print("=" * 60)


        print(f"Experiment: {exp['label']}")


        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")


        print(f"Base seed: {seed}")


        print(f"Maze seed: {exp['maze_seed']}")


        print(f"Scenario: {scenario_label}")


        print(f"Algorithm: {exp['algo']}")


        print(f"Sparse fraction: {exp['sparse_fraction']}")


        if exp["variant"] == "base":


            print(f"Proposed bonus constant: {exp['bonus_constant']}")


            print(f"Proposed sparse reward only: {exp.get('use_sparse_reward_only', False)}")


            print(f"Proposed reward shaping: {exp.get('use_reward_shaping', False)}")


        # Reward shaping: when enabled for this experiment, set both move and stay penalties to -0.2


        default_move_penalty = config.get('move_penalty', 0.0)


        default_stay_penalty = config.get('stay_penalty', 0.0)


        if exp.get("use_reward_shaping", False):


            exp_move_penalty = -0.2


            exp_stay_penalty = -0.2


        else:


            exp_move_penalty = default_move_penalty


            exp_stay_penalty = default_stay_penalty


        print(f"Maze size: {maze_size}x{maze_size}")


        print(f"Wall penalty: {wall_penalty}")


        print(f"Stay penalty: {exp_stay_penalty}")


        print(f"Move penalty: {exp_move_penalty}")


        print(f"Episodes: {episodes}")


        print(f"Horizon: {horizon}")


        print(f"Output dir: {exp_output_dir}")


        print("=" * 60)





        grid = build_corridor_maze(size=maze_size, seed=exp["maze_seed"])





        env = MazeEnv(


            size=maze_size,


            horizon=horizon,


            seed=config.get('env_seed', 42),


            grid=grid,


            start=(0, 0),


            goal=(maze_size - 1, maze_size - 1),


            wall_penalty=wall_penalty,


            stay_penalty=exp_stay_penalty,


            move_penalty=exp_move_penalty


        )





        layout_path = os.path.join(exp_output_dir, "visualizations", "maze_layout.pdf")


        save_maze_visualization(grid, env.start, env.goal, out_path=layout_path)





        if exp["algo"] == "ucb":


            agent = QLearningUCBHoeffdingSparse(


                env=env,


                horizon=horizon,


                episodes=episodes,


                failure_prob=failure_prob,


                bonus_constant=exp.get("bonus_constant", bonus_constant),


                sparse_fraction=exp["sparse_fraction"],


                seed=seed,


                use_sparse_reward_only=exp.get("use_sparse_reward_only", False)


            )


        elif exp["algo"] == "her":


            agent = QLearningHER(


                env=env,


                horizon=horizon,


                episodes=episodes,


                seed=seed


            )


        else:


            agent = QLearningEpsilonGreedy(


                env=env,


                horizon=horizon,


                episodes=episodes,


                seed=seed,


                use_sparse_reward_only=False


            )





        stats = agent.train(log_interval=log_interval, record_paths=record_paths)


        reward_axis_series.append(stats.rewards)


        overall_success_rate = sum(stats.successes) / max(1, len(stats.successes))


        group_results_all.setdefault(scenario_key, {})[exp["output_name"]] = stats





        success_rate, avg_reward = agent.evaluate(env, episodes=eval_episodes)





        demo_path, demo_reward, success = agent.rollout(env)


        stay_count = sum(1 for i in range(len(demo_path) - 1) if demo_path[i] == demo_path[i + 1])


        print(f"Demo path length: {len(demo_path)} | reward={demo_reward:.4f} | success={success}")





        if stats.episode_paths and stats.rewards:


            best_reward_idx = int(np.argmax(stats.rewards))


            best_reward_path = stats.episode_paths[best_reward_idx]


            best_reward_value = stats.rewards[best_reward_idx]


            best_reward_path_vis = os.path.join(


                exp_output_dir, "visualizations", f"best_reward_path_{exp['output_name']}.pdf"


            )


            save_maze_visualization(


                env.grid, env.start, env.goal, best_reward_path, best_reward_path_vis,


                marker_style="line"


            )


            print(


                f"Best reward path episode: {best_reward_idx + 1} | "


                f"reward={best_reward_value:.4f}"


            )





        q_path_result = select_q_path_for_plot(agent.Q, env, horizon, stats)


        q_path_for_plot = q_path_result.path


        q_path_selection = q_path_result.selection


        q_path_reaches_goal = q_path_result.reaches_goal


        q_path_is_strict = q_path_result.is_strict_max_q


        q_path_is_valid_for_plot = q_path_result.is_plotted


        q_path_score = q_path_result.q_score


        training_reached_goal = any(stats.successes)


        q_path_vis = os.path.join(


            exp_output_dir, "visualizations", f"q_greedy_path_{exp['output_name']}.pdf"


        )


        if exp["algo"] == "eps":


            marker_style = "hollow_blue"


        elif exp["algo"] == "her":


            marker_style = "purple_dashed"


        elif exp["variant"] == "sparse1":


            marker_style = "black_dashed"


        else:


            marker_style = "line"


        save_maze_visualization(


            env.grid, env.start, env.goal, q_path_for_plot, q_path_vis,


            marker_style=marker_style


        )


        q_path_csv = os.path.join(


            exp_output_dir, "data", f"q_optimal_path_{exp['output_name']}.csv"


        )


        save_path_coordinates(q_path_for_plot, q_path_csv)


        print(f"Q-value path saved to: {q_path_vis} ({q_path_selection})")


        print(f"Q-value optimal path coordinates saved to: {q_path_csv}")


        q_optimal_paths_all.setdefault(scenario_key, {})[exp["output_name"]] = q_path_for_plot





        if record_paths and stats.episode_paths:


            anim_path = os.path.join(


                exp_output_dir, "animations", f"episode_paths_{exp['output_name']}.gif"


            )


            save_episode_animation(


                grid, env.start, env.goal,


                episode_paths=stats.episode_paths,


                rewards=stats.rewards,


                out_path=anim_path,


                interval_ms=120


            )





        if exp["algo"] == "ucb":


            q_table_path = os.path.join(


                exp_output_dir, "tables", f"q_table_detailed_{exp['output_name']}.txt"


            )


            agent.save_q_table_with_coordinates(q_table_path, env)





            visit_counts_path = os.path.join(


                exp_output_dir, "tables", f"visit_counts_{exp['output_name']}.txt"


            )


            agent.save_visit_counts_with_coordinates(visit_counts_path, env)





            q_summary_path = os.path.join(


                exp_output_dir, "tables", f"q_table_summary_{exp['output_name']}.txt"


            )


            agent.save_q_table_summary(q_summary_path, env)





        config_path = os.path.join(


            exp_output_dir, "data", f"experiment_config_{exp['output_name']}.json"


        )


        os.makedirs(os.path.dirname(config_path), exist_ok=True)





        saved_config = {


            "experiment_info": {


                "name": config.get('experiment_name', 'maze_experiment'),


                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),


                "base_seed": seed,


                "scenario_key": scenario_key,


                "scenario_label": scenario_label,


                "maze_seed": exp["maze_seed"],


                "env_seed": config.get('env_seed', 42),


                "algorithm": exp["algo"],


                "variant": exp["variant"],


            },


            "environment_params": {


                "maze_size": env.size,


                "horizon": horizon,


                "wall_penalty": wall_penalty,


                "stay_penalty": exp_stay_penalty,


                "move_penalty": exp_move_penalty,


            },


            "algorithm_params": {


                "episodes": episodes,


                "failure_prob": failure_prob,


                "bonus_constant": bonus_constant,


                "sparse_fraction": exp["sparse_fraction"],


                "use_reward_shaping": exp.get("use_reward_shaping", False),


            },


            "training_params": {


                "log_interval": log_interval,


                "eval_episodes": eval_episodes,


                "record_paths": record_paths,


            },


            "results": {


                "final_success_rate": stats.success_rate,


                "overall_success_rate": overall_success_rate,


                "evaluation_success_rate": success_rate,


                "evaluation_avg_reward": avg_reward,


                "demo_success": success,


                "demo_stay_count": stay_count,


                "q_optimal_path_length": len(q_path_for_plot),


                "q_optimal_path_reaches_goal": q_path_reaches_goal,


                "q_optimal_path_selection": q_path_selection,


                "q_optimal_path_q_score": q_path_score,


                "q_optimal_path_is_strict_max_q": q_path_is_strict,


                "q_optimal_path_training_reached_goal": training_reached_goal,


                "q_optimal_path_is_plotted": q_path_is_valid_for_plot,


                "q_optimal_path_has_actual_moves": any(


                    q_path_for_plot[i] != q_path_for_plot[i + 1]


                    for i in range(max(0, len(q_path_for_plot) - 1))


                ),


                "q_optimal_path_all_moves_are_actual": all(


                    q_path_for_plot[i] != q_path_for_plot[i + 1]


                    for i in range(max(0, len(q_path_for_plot) - 1))


                ),


            }


        }





        import json


        with open(config_path, "w", encoding="utf-8") as f:


            json.dump(saved_config, f, indent=2, ensure_ascii=False)





    shared_y_lower, shared_y_upper, shared_y_tick_step = compute_reward_axis(reward_axis_series)





    for exp in experiments:


        maze_size = int(exp.get("maze_size", config.get('maze_size', 15)))


        scenario_key = exp.get("scenario_key", f"maze_seed_{exp['maze_seed']}_size_{maze_size}")


        stats = group_results_all.get(scenario_key, {}).get(exp["output_name"])


        if stats is None:


            continue


        reward_plot_path = os.path.join(


            output_dir, scenario_key, "plots", f"reward_vs_step_{exp['output_name']}.pdf"


        )


        plot_reward_trace(


            stats.episodes,


            stats.rewards,


            out_path=reward_plot_path,


            y_lower=shared_y_lower,


            y_upper=shared_y_upper,


            y_tick_step=shared_y_tick_step


        )





    for scenario_key, results in group_results_all.items():


        if "ucb" not in results or "eps" not in results:


            continue


        compare_plot_filename = str(scenario_meta[scenario_key]["compare_plot_filename"])

        if compare_plot_filename.endswith(".pdf"):

            compare_plot_filename = compare_plot_filename.replace(".pdf", "_five_series.pdf")

        elif compare_plot_filename.endswith(".png"):

            compare_plot_filename = compare_plot_filename.replace(".png", "_five_series.png")


        compare_plot_path = os.path.join(output_dir, compare_plot_filename)


        if "algo1" in results and "ucb_h" in results and "her" in results:

            plot_reward_comparison_five(

                results["ucb"], "Proposed",

                results["algo1"], "Algorithm 1",

                results["ucb_h"], "UCB-H [4]",

                results["eps"], "Epsilon-greedy",

                results["her"], "HER [17]",

                out_path=compare_plot_path,

                y_lower=shared_y_lower,

                y_upper=shared_y_upper,

                y_tick_step=shared_y_tick_step


            )

            continue

        if "algo1" in results:


            if "her" in results:


                plot_reward_comparison_four(


                    results["ucb"], "Proposed",


                    results["algo1"], "Algorithm 1",


                    results["eps"], "Epsilon-greedy",


                    results["her"], "HER [17]",


                    out_path=compare_plot_path,


                    y_lower=shared_y_lower,


                    y_upper=shared_y_upper,


                    y_tick_step=shared_y_tick_step


                )


            else:


                plot_reward_comparison_three(


                    results["ucb"], "Proposed",


                    results["algo1"], "Algorithm 1",


                    results["eps"], "Epsilon-greedy",


                    out_path=compare_plot_path,


                    y_lower=shared_y_lower,


                    y_upper=shared_y_upper,


                    y_tick_step=shared_y_tick_step


                )


            continue


        if "ucb_h" not in results:


            continue


        if "her" in results:


            plot_reward_comparison_four(


                results["ucb"], "Proposed",


                results["ucb_h"], "UCB-H [4]",


                results["eps"], "Epsilon-greedy",


                results["her"], "HER [17]",


                out_path=compare_plot_path,


                y_lower=shared_y_lower,


                y_upper=shared_y_upper,


                y_tick_step=shared_y_tick_step


            )


            continue


        plot_reward_comparison_three(


            results["ucb"], "Proposed",


            results["ucb_h"], "UCB-H [4]",


            results["eps"], "Epsilon-greedy",


            out_path=compare_plot_path,


            y_lower=shared_y_lower,


            y_upper=shared_y_upper,


            y_tick_step=shared_y_tick_step


        )





    for scenario_key, paths in q_optimal_paths_all.items():


        if "ucb" not in paths or "eps" not in paths:


            continue


        maze_seed = int(scenario_meta[scenario_key]["maze_seed"])


        maze_size = int(scenario_meta[scenario_key]["maze_size"])


        grid = build_corridor_maze(size=maze_size, seed=maze_seed)


        start = (0, 0)


        goal = (maze_size - 1, maze_size - 1)


        compare_path_filename = str(scenario_meta[scenario_key]["compare_path_filename"])


        compare_path_out = os.path.join(output_dir, compare_path_filename)


        if False and "algo1" in paths:


            if "her" in paths:


                save_maze_visualization_compare_four(


                    grid, start, goal,


                    paths["ucb"], "Proposed",


                    paths["algo1"], "Algorithm 1",


                    paths["eps"], "Epsilon-greedy",


                    paths["her"], "HER [17]",


                    out_path=compare_path_out


                )


            else:


                save_maze_visualization_compare_three(


                    grid, start, goal,


                    paths["ucb"], "Proposed",


                    paths["algo1"], "Algorithm 1",


                    paths["eps"], "Epsilon-greedy",


                    out_path=compare_path_out


                )


            continue


        if "ucb_h" not in paths:


            continue


        if "her" in paths:


            save_maze_visualization_compare_four(


                grid, start, goal,


                paths["ucb"], "Proposed",


                paths["ucb_h"], "UCB-H [4]",


                paths["eps"], "Epsilon-greedy",


                paths["her"], "HER [17]",


                out_path=compare_path_out


            )


            continue


        save_maze_visualization_compare_three(


            grid, start, goal,


            paths["ucb"], "Proposed",


            paths["ucb_h"], "UCB-H [4]",


            paths["eps"], "Epsilon-greedy",


            out_path=compare_path_out


        )





    return





    # 


    if output_dir is None:


        output_dir = ensure_output_dir()


    else:


        os.makedirs(output_dir, exist_ok=True)


        subdirs = ["plots", "visualizations", "animations", "data", "tables"]


        for subdir in subdirs:


            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)





    print("=" * 60)


    print("Maze reinforcement learning experiment")


    print(f"Experiment time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


    print(f"Base random seed: {seed}")


    print(f"Wall penalty: {wall_penalty}")


    print(f"Stay penalty: {stay_penalty}")


    print(f"Move penalty: {move_penalty}")


    print(f"Episodes: {episodes}")


    print(f"Horizon: {horizon}")


    print(f"Output directory: {output_dir}")


    print("Action space: UP, RIGHT, DOWN, LEFT, STAY")


    print("Random numbers: pre-generated pools")


    s_init = sparse_fraction * horizon


    print(f"Q: {s_init} (?s)")


    print("V: 0.0")


    print("=" * 60)





    # ??


    grid = build_corridor_maze(size=config.get('maze_size', 15), seed=config.get('maze_seed', 7))





    # 42


    env = MazeEnv(


        size=config.get('maze_size', 15),


        horizon=horizon,


        seed=config.get('env_seed', 42),


        grid=grid,


        start=(0, 0),


        goal=(config.get('maze_size', 15) - 1, config.get('maze_size', 15) - 1),


        wall_penalty=wall_penalty,


        stay_penalty=stay_penalty,


        move_penalty=move_penalty


    )





    print(f" {env.size}x{env.size}  (S=, G=):")


    print(env.render())


    print(f": {env.start}, : {env.goal}")





    # 


    layout_path = os.path.join(output_dir, "visualizations", "maze_layout.pdf")


    save_maze_visualization(grid, env.start, env.goal, out_path=layout_path)


    print(f": {layout_path}")





    # 


    agent = QLearningUCBHoeffdingSparse(


        env=env,


        horizon=horizon,


        episodes=episodes,


        failure_prob=failure_prob,


        bonus_constant=bonus_constant,


        sparse_fraction=sparse_fraction,


        seed=seed


    )





    print(f": {episodes}")


    print(f"Q? {agent.q_init_value}")


    print("V? 0.0")


    print("-" * 40)





    stats = agent.train(log_interval=log_interval, record_paths=record_paths)





    # ?


    overall_success_rate = sum(stats.successes) / max(1, len(stats.successes))





    print("\n!")


    print(f": {stats.success_rate:.4f}")


    print(f"? {overall_success_rate:.4f}")





    # ?


    reward_plot_path = os.path.join(output_dir, "plots", "reward_vs_step.pdf")


    plot_reward_trace(stats.episodes, stats.rewards, out_path=reward_plot_path)


    print(f": {reward_plot_path}")





    # 


    print("\n?..")


    success_rate, avg_reward = agent.evaluate(env, episodes=eval_episodes)


    print(f" - ? {success_rate:.4f}, : {avg_reward:.4f}")





    # ?


    print("\n?..")


    demo_path, demo_reward, success = agent.rollout(env)


    stay_count = sum(1 for i in range(len(demo_path) - 1) if demo_path[i] == demo_path[i + 1])


    print(f": {len(demo_path)}, ? {demo_reward:.4f}")


    print(f"Reached goal: {'YES' if success else 'NO'}")


    print(f": {stay_count}")


    print(":")


    print(env.render(demo_path))





    # Q?


    print("\nQ?..")


    q_path = agent.greedy_path_from_q(env, deterministic=True)


    print(env.render(q_path))


    q_path_vis = os.path.join(output_dir, "visualizations", "q_greedy_path.pdf")


    save_maze_visualization(env.grid, env.start, env.goal, q_path, q_path_vis)


    print(f"Q? {q_path_vis}")





    # 


    if record_paths and stats.episode_paths:


        anim_path = os.path.join(output_dir, "animations", "episode_paths.gif")


        save_episode_animation(


            grid, env.start, env.goal,


            episode_paths=stats.episode_paths,


            rewards=stats.rewards,


            out_path=anim_path,


            interval_ms=120


        )


        print(f"? {anim_path}")





    # Q


    q_table_path = os.path.join(output_dir, "tables", "q_table_detailed.txt")


    agent.save_q_table_with_coordinates(q_table_path, env)


    print(f"Q: {q_table_path}")





    visit_counts_path = os.path.join(output_dir, "tables", "visit_counts.txt")


    agent.save_visit_counts_with_coordinates(visit_counts_path, env)


    print(f"? {visit_counts_path}")





    q_summary_path = os.path.join(output_dir, "tables", "q_table_summary.txt")


    agent.save_q_table_summary(q_summary_path, env)


    print(f"Q: {q_summary_path}")





    # 


    config_path = os.path.join(output_dir, "data", "experiment_config.json")


    os.makedirs(os.path.dirname(config_path), exist_ok=True)





    # ?


    saved_config = {


        "experiment_info": {


            "name": config.get('experiment_name', 'maze_experiment'),


            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),


            "base_seed": seed,


            "maze_seed": config.get('maze_seed', 7),


            "env_seed": config.get('env_seed', 42),


        },


        "environment_params": {


            "maze_size": env.size,


            "horizon": horizon,


            "wall_penalty": wall_penalty,


            "stay_penalty": stay_penalty,


            "move_penalty": move_penalty,


        },


        "algorithm_params": {


            "episodes": episodes,


            "failure_prob": failure_prob,


            "bonus_constant": bonus_constant,


            "sparse_fraction": sparse_fraction,


        },


        "training_params": {


            "log_interval": log_interval,


            "eval_episodes": eval_episodes,


            "record_paths": record_paths,


        },


        "results": {


            "final_success_rate": stats.success_rate,


            "overall_success_rate": overall_success_rate,


            "evaluation_success_rate": success_rate,


            "evaluation_avg_reward": avg_reward,


            "demo_success": success,


            "demo_stay_count": stay_count,


        }


    }





    import json


    with open(config_path, "w", encoding="utf-8") as f:


        json.dump(saved_config, f, indent=2, ensure_ascii=False)





    print(f": {config_path}")








# main?


if __name__ == "__main__":


    import argparse





    parser = argparse.ArgumentParser(description="Maze reinforcement learning experiment")





    # 


    parser.add_argument("--seed", type=int, default=None,


                        help="Base random seed")


    parser.add_argument("--output_dir", type=str, default=None,


                        help="Output directory")


    parser.add_argument("--wall_penalty", type=float, default=None,


                        help="Wall penalty")


    parser.add_argument("--stay_penalty", type=float, default=None,


                        help="Stay penalty")


    parser.add_argument("--move_penalty", type=float, default=None,


                        help="Move penalty")


    parser.add_argument("--episodes", type=int, default=500,


                        help="Number of training episodes")


    parser.add_argument("--horizon", type=int, default=2000,


                        help="Maximum steps per episode")





    args = parser.parse_args()





    # one


    kwargs = {k: v for k, v in vars(args).items() if v is not None}





    main(**kwargs)



def run_algorithm1_comparison(output_dir: str, maze_size: int = 20, maze_seed: int = 1,
                              episodes: int = 500, horizon: int = 2000, seed: int | None = None) -> None:
    """Runs a focused comparison: Algorithm 1 (UCB-H) vs Epsilon-greedy and HER.

    This function uses the same `episodes` and `horizon` settings as the rest of
    the codebase and explicitly disables reward shaping (move/stay penalties = 0).
    Results are saved into `output_dir` under a timestamped subfolder.
    """
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(output_dir, f'algo1_compare_{stamp}')
    os.makedirs(outdir, exist_ok=True)

    if seed is None:
        seed = random.randint(0, 1000000)

    grid = build_corridor_maze(size=maze_size, seed=maze_seed)
    env = MazeEnv(size=maze_size, horizon=horizon, seed=seed, grid=grid,
                  start=(0, 0), goal=(maze_size - 1, maze_size - 1),
                  wall_penalty=-100.0, stay_penalty=0.0, move_penalty=0.0)

    # Algorithm 1 (UCB-H): use sparse1 variant behaviour via sparse_fraction=1.0
    ucb = QLearningUCBHoeffdingSparse(env=env, horizon=horizon, episodes=episodes,
                                      failure_prob=0.1, bonus_constant=1.0,
                                      sparse_fraction=1.0, seed=seed,
                                      use_sparse_reward_only=False)

    eps = QLearningEpsilonGreedy(env=env, horizon=horizon, episodes=episodes, seed=seed)
    her = QLearningHER(env=env, horizon=horizon, episodes=episodes, seed=seed)

    print('Running Algorithm 1 (UCB-H) comparison:')
    print(f' Out dir: {outdir} | maze {maze_size}x{maze_size} | episodes={episodes} | horizon={horizon}')

    stats_ucb = ucb.train(log_interval=max(1, episodes // 10), record_paths=True)
    stats_eps = eps.train(log_interval=max(1, episodes // 10), record_paths=True)
    stats_her = her.train(log_interval=max(1, episodes // 10), record_paths=True)

    # Save combined reward comparison
    plot_path = os.path.join(outdir, 'reward_compare_algo1.png')
    plot_reward_comparison_three(stats_ucb, 'UCB-H (Algorithm 1)',
                                 stats_eps, 'Epsilon-greedy',
                                 stats_her, 'HER', out_path=plot_path)

    # Save maze layout and Q-tables
    save_maze_visualization(grid, env.start, env.goal, out_path=os.path.join(outdir, 'maze_layout.pdf'))
    if hasattr(ucb, 'save_q_table_with_coordinates'):
        ucb.save_q_table_with_coordinates(os.path.join(outdir, 'ucb_q_table.txt'), env)
    if hasattr(eps, 'save_q_table_with_coordinates'):
        eps.save_q_table_with_coordinates(os.path.join(outdir, 'eps_q_table.txt'), env)
    if hasattr(her, 'save_q_table_with_coordinates'):
        her.save_q_table_with_coordinates(os.path.join(outdir, 'her_q_table.txt'), env)

    # Evaluate
    s_ucb, r_ucb = ucb.evaluate(env, episodes=100)
    s_eps, r_eps = eps.evaluate(env, episodes=100)
    s_her, r_her = her.evaluate(env, episodes=100)

    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write(f'UCB-H success_rate={s_ucb:.4f}, avg_reward={r_ucb:.4f}\n')
        f.write(f'EPS success_rate={s_eps:.4f}, avg_reward={r_eps:.4f}\n')
        f.write(f'HER success_rate={s_her:.4f}, avg_reward={r_her:.4f}\n')

    print('Algorithm 1 comparison finished. Results saved in', outdir)





