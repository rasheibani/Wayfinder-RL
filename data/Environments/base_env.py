"""
A clean stand-alone re-implementation of TextWorldEnv with:
 • 8-sector default grammar
 • destination one-hot appended to observation
"""

from __future__ import annotations
import random, re, math
from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Direction conventions (must match the z8 world files)
# ---------------------------------------------------------------------------
GRAMMAR_DIRECTIONS = {
    4: ['north', 'south', 'east', 'west'],
    6: ['north', 'south', 'east', 'west', 'northeast', 'southwest'],
    8: ['north', 'south', 'east', 'west',
        'northeast', 'northwest', 'southeast', 'southwest']
}
DIR2IDX_8 = {d: i for i, d in enumerate(GRAMMAR_DIRECTIONS[8])}


def text_to_index(txt: str) -> int:
    """
    'go north' -> 0 , etc.
    Returns -1 for 'look' or un-recognised.
    """
    txt = txt.lower().strip()
    for d, i in DIR2IDX_8.items():
        if d in txt:
            return i
    return -1


def index_to_text(idx: int) -> str:
    if 0 <= idx < len(GRAMMAR_DIRECTIONS[8]):
        return f"go {GRAMMAR_DIRECTIONS[8][idx]}"
    return "look"


# ---------------------------------------------------------------------------
# Helper – observation normalisation
# ---------------------------------------------------------------------------
def _normalise(vec: np.ndarray, max_val: float = 8.0) -> np.ndarray:
    """Simple min-max into [0,1] ; sentinel 8 stays 0."""
    out = vec.astype(np.float32).copy()
    sentinel = 8.0
    mask = out != sentinel
    out[mask] /= max_val
    out[~mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# The base environment used everywhere below
# ---------------------------------------------------------------------------
class TextWorldEnv(gym.Env):
    """
    Generic single-origin TextWorld navigation.
    We expose admissible-actions flags, route instructions,
    (instruction_index , current_instruction) and finally
    a one-hot *destination tag* so the agent knows which goal
    the current episode belongs to.
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            game_dict: Dict[str, Dict[str, str]],
            room_positions: Dict[str, Tuple[float, float]],
            *,
            grammar: int = 8,
            reward_type: str = "sparse",
            route_instructions: List[int] | None = None,
            x_destination: float | None = None,
            y_destination: float | None = None
    ):
        super().__init__()
        # Graph + positions ---------------------------------------------------
        self.game_dict = game_dict
        self.room_positions = room_positions
        self.room_ids = list(room_positions)
        self.n_rooms = len(self.room_ids)

        # Navigation settings -------------------------------------------------
        self.grammar = grammar
        self.directions = GRAMMAR_DIRECTIONS[self.grammar]
        self.max_directions = max(len(v) for v in GRAMMAR_DIRECTIONS.values())
        self.reward_type = reward_type

        # External episode wiring --------------------------------------------
        self.route_instructions = route_instructions or []
        self.x_destination = x_destination
        self.y_destination = y_destination
        self.dest_room_id = None  # set during reset
        self.current_room_id = None

        # Internal counters ---------------------------------------------------
        self.instruction_index = 0
        self.counter = 0
        self.visited_states_actions = set()

        # Action / observation spaces ----------------------------------------
        # (+1) for "look"
        self.action_space = spaces.Discrete(self.max_directions + 1)

        # admissible(max_dir) + instr(15) + idx(2) + dest_one_hot(n_rooms)
        obs_len = self.max_directions + 15 + 2 + self.n_rooms
        self.observation_space = spaces.Box(
            low=0, high=8, shape=(obs_len,), dtype=np.float32
        )

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def reset(self, *, route_instructions: List[int] | None = None,
              origin_room_id: str | None = None,
              dest_room_id: str | None = None,
              seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        self.instruction_index = 0
        self.visited_states_actions.clear()

        # origin / destination ------------------------------------------------
        self.current_room_id = origin_room_id or random.choice(self.room_ids)
        self.dest_room_id = dest_room_id or random.choice(self.room_ids)
        while self.dest_room_id == self.current_room_id:
            self.dest_room_id = random.choice(self.room_ids)

        self.x_destination, self.y_destination = self.room_positions[self.dest_room_id]
        self.x_origin, self.y_origin = self.room_positions[self.current_room_id]

        # instructions --------------------------------------------------------
        if route_instructions:
            self.route_instructions = route_instructions
        elif not self.route_instructions:
            # trivial single-step fallback (used only when BaseEnv is run
            # stand-alone – MultiOrigin wrapper passes its own list)
            self.route_instructions = [0]

        # initial obs ---------------------------------------------------------
        obs = self._construct_observation(self._get_admissible_actions())
        return obs, {}

    # -----------------------------------------------------------------------
    # Core STEP
    # -----------------------------------------------------------------------
    def step(self, action: int):
        done = False
        truncated = False

        look_idx = self.max_directions
        sentence = index_to_text(action)

        # invalid or "look" ---------------------------------------------------
        admissible = self._get_admissible_actions()
        if action == look_idx or sentence not in admissible:
            reward = -1 if self.reward_type == "step_cost" else 0
            obs = self._construct_observation(admissible)
            return obs, reward, done, truncated, {}

        # execute ------------------------------------------------------------
        next_room = self.game_dict[self.current_room_id][sentence]
        self.current_room_id = next_room
        self.counter += 1
        self.instruction_index = min(self.instruction_index + 1,
                                     len(self.route_instructions) - 1)
        self.visited_states_actions.add((self.current_room_id, action))

        # reward --------------------------------------------------------------
        dest_x, dest_y = self.room_positions[self.dest_room_id]
        cur_x, cur_y = self.room_positions[self.current_room_id]
        if math.isclose(cur_x, dest_x, abs_tol=1e-2) and \
           math.isclose(cur_y, dest_y, abs_tol=1e-2):
            reward = 25
            done = True
        else:
            reward = -1 if self.reward_type == "step_cost" else 0

        # step-limit truncation (simple) -------------------------------------
        if self.counter > 60:
            truncated = True

        obs = self._construct_observation(self._get_admissible_actions())
        return obs, reward, done, truncated, {}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _get_admissible_actions(self) -> List[str]:
        return list(self.game_dict[self.current_room_id].keys())

    def _admissible_to_vec(self, admissible: List[str]) -> np.ndarray:
        vec = np.zeros(self.max_directions, dtype=np.float32)
        for i, d in enumerate(self.directions):
            if f"go {d}" in admissible:
                vec[i] = 1.0
        return vec

    def _pad_instructions(self) -> np.ndarray:
        pad = np.full(15, 8, dtype=np.float32)
        upto = min(15, len(self.route_instructions))
        pad[:upto] = self.route_instructions[:upto]
        return pad

    # dest id -> index into one-hot
    def _dest_index(self) -> int:
        return self.room_ids.index(self.dest_room_id)

    def _construct_observation(self, admissible: List[str]) -> np.ndarray:
        admissible_vec = self._admissible_to_vec(admissible)
        instr_pad = self._pad_instructions()
        idx_arr = np.array([self.instruction_index,
                            self.route_instructions[self.instruction_index]],
                           dtype=np.float32)
        dest_vec = np.zeros(self.n_rooms, dtype=np.float32)
        dest_vec[self._dest_index()] = 1.0

        raw = np.concatenate([admissible_vec, instr_pad, idx_arr, dest_vec])
        return _normalise(raw)

    # -----------------------------------------------------------------------
    def render(self):
        pass
