"""
Claude-powered ARC-AGI-3 agent using Anthropic's vision + tool_use API.

Single-file agent: ClaudeVision (rendering), ClaudeReasoner (hypotheses),
ClaudeAgent (main loop extending Agent base class).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from collections import Counter
from typing import Any, Optional

import anthropic
import numpy as np
from arcengine import FrameData, GameAction, GameState
from PIL import Image, ImageDraw, ImageFont

from ..agent import Agent

logger = logging.getLogger()

# ── 16-color ARC palette (RGBA) ──────────────────────────────────────────────
PALETTE: list[tuple[int, int, int, int]] = [
    (0xFF, 0xFF, 0xFF, 0xFF),  # 0  White
    (0xCC, 0xCC, 0xCC, 0xFF),  # 1  Off-white
    (0x99, 0x99, 0x99, 0xFF),  # 2  Neutral light
    (0x66, 0x66, 0x66, 0xFF),  # 3  Neutral
    (0x33, 0x33, 0x33, 0xFF),  # 4  Off-black
    (0x00, 0x00, 0x00, 0xFF),  # 5  Black
    (0xE5, 0x3A, 0xA3, 0xFF),  # 6  Magenta
    (0xFF, 0x7B, 0xCC, 0xFF),  # 7  Magenta light
    (0xF9, 0x3C, 0x31, 0xFF),  # 8  Red
    (0x1E, 0x93, 0xFF, 0xFF),  # 9  Blue
    (0x88, 0xD8, 0xF1, 0xFF),  # 10 Blue light
    (0xFF, 0xDC, 0x00, 0xFF),  # 11 Yellow
    (0xFF, 0x85, 0x1B, 0xFF),  # 12 Orange
    (0x92, 0x12, 0x31, 0xFF),  # 13 Maroon
    (0x4F, 0xCC, 0x30, 0xFF),  # 14 Green
    (0xA3, 0x56, 0xD6, 0xFF),  # 15 Purple
]

COLOR_NAMES = [
    "white", "off-white", "light-gray", "gray", "dark-gray", "black",
    "magenta", "pink", "red", "blue", "light-blue", "yellow",
    "orange", "maroon", "green", "purple",
]

# ── All tool schemas ─────────────────────────────────────────────────────────
# Indexed by GameAction enum value for dynamic filtering

ALL_TOOL_SCHEMAS: dict[int, dict[str, Any]] = {
    0: {
        "name": "reset",
        "description": "Reset the current level. Use only when completely stuck.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why you are resetting."},
            },
            "required": ["reasoning"],
        },
    },
    1: {
        "name": "move_up",
        "description": "Move up on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why this move."},
            },
            "required": ["reasoning"],
        },
    },
    2: {
        "name": "move_down",
        "description": "Move down on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why this move."},
            },
            "required": ["reasoning"],
        },
    },
    3: {
        "name": "move_left",
        "description": "Move left on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why this move."},
            },
            "required": ["reasoning"],
        },
    },
    4: {
        "name": "move_right",
        "description": "Move right on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why this move."},
            },
            "required": ["reasoning"],
        },
    },
    5: {
        "name": "confirm",
        "description": "Confirm / perform action (submit answer, activate).",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why confirming."},
            },
            "required": ["reasoning"],
        },
    },
    6: {
        "name": "click",
        "description": "Click a specific cell on the 64x64 grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coord (0-63).", "minimum": 0, "maximum": 63},
                "y": {"type": "integer", "description": "Y coord (0-63).", "minimum": 0, "maximum": 63},
                "reasoning": {"type": "string", "description": "Why clicking here."},
            },
            "required": ["x", "y", "reasoning"],
        },
    },
    7: {
        "name": "undo",
        "description": "Undo the last action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why undoing."},
            },
            "required": ["reasoning"],
        },
    },
}

# Virtual tool: navigate_to is not a real GameAction but generates a movement queue
NAVIGATE_TO_SCHEMA: dict[str, Any] = {
    "name": "navigate_to",
    "description": (
        "Navigate to a specific grid position. The agent will take multiple movement steps "
        "to reach (x,y), then return control to you. Use this for efficient multi-step navigation "
        "instead of individual move commands. Provide the target coordinates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Target X coord (0-63).", "minimum": 0, "maximum": 63},
            "y": {"type": "integer", "description": "Target Y coord (0-63).", "minimum": 0, "maximum": 63},
            "reasoning": {"type": "string", "description": "Why navigating here."},
        },
        "required": ["x", "y", "reasoning"],
    },
}

TOOL_TO_ACTION: dict[str, GameAction] = {
    "reset": GameAction.RESET,
    "move_up": GameAction.ACTION1,
    "move_down": GameAction.ACTION2,
    "move_left": GameAction.ACTION3,
    "move_right": GameAction.ACTION4,
    "confirm": GameAction.ACTION5,
    "click": GameAction.ACTION6,
    "undo": GameAction.ACTION7,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ClaudeVision — Grid rendering, diffing, object detection
# ═══════════════════════════════════════════════════════════════════════════════


class ClaudeVision:
    """Renders 64x64 ARC grids as annotated PNGs and extracts spatial info."""

    SCALE = 8  # 64 * 8 = 512
    IMG_SIZE = 64 * SCALE  # 512

    @staticmethod
    def _render_grid(grid: list[list[int]], label: str = "") -> Image.Image:
        """Render a single 64x64 grid to a PIL Image."""
        raw = bytearray()
        for row in grid:
            for idx in row:
                c = idx if 0 <= idx < 16 else 0
                raw.extend(PALETTE[c])

        img = Image.frombytes("RGBA", (64, 64), bytes(raw))
        img = img.resize(
            (ClaudeVision.IMG_SIZE, ClaudeVision.IMG_SIZE), Image.NEAREST
        )

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except (OSError, IOError):
            font = ImageFont.load_default()

        scale = ClaudeVision.SCALE
        for i in range(0, 65, 8):
            pos = i * scale
            if pos < ClaudeVision.IMG_SIZE:
                draw.line([(pos, 0), (pos, ClaudeVision.IMG_SIZE - 1)], fill=(255, 255, 255, 60), width=1)
                draw.line([(0, pos), (ClaudeVision.IMG_SIZE - 1, pos)], fill=(255, 255, 255, 60), width=1)
                if i < 64:
                    draw.text((pos + 2, 1), str(i), fill=(255, 255, 0, 180), font=font)
                    draw.text((1, pos + 2), str(i), fill=(255, 255, 0, 180), font=font)

        if label:
            draw.text((5, ClaudeVision.IMG_SIZE - 16), label, fill=(255, 255, 0, 255), font=font)

        return img

    @staticmethod
    def render_frame(frame_3d: list[list[list[int]]]) -> str:
        """Render frame layers. Multi-layer: tile layers side-by-side."""
        if not frame_3d:
            grid = [[0] * 64 for _ in range(64)]
            return ClaudeVision._img_to_base64(ClaudeVision._render_grid(grid))

        if len(frame_3d) == 1:
            return ClaudeVision._img_to_base64(ClaudeVision._render_grid(frame_3d[0]))

        # Multi-layer: render unique layers side-by-side (skip duplicates)
        unique_layers: list[tuple[int, list[list[int]]]] = []
        seen_hashes: set[str] = set()
        for i, layer in enumerate(frame_3d):
            h = hashlib.md5(json.dumps(layer, separators=(",", ":")).encode()).hexdigest()[:12]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_layers.append((i, layer))

        n = len(unique_layers)
        tile_size = max(128, ClaudeVision.IMG_SIZE // max(n, 1))
        canvas_w = tile_size * n
        canvas = Image.new("RGBA", (canvas_w, tile_size), (0, 0, 0, 255))

        for j, (idx, layer) in enumerate(unique_layers):
            img = ClaudeVision._render_grid(layer, f"L{idx}")
            img = img.resize((tile_size, tile_size), Image.NEAREST)
            canvas.paste(img, (j * tile_size, 0))

        return ClaudeVision._img_to_base64(canvas)

    @staticmethod
    def render_diff(frame_a: list[list[list[int]]], frame_b: list[list[list[int]]]) -> str:
        """Render a red-on-black diff between two frames."""
        grid_a = frame_a[-1] if frame_a else [[0] * 64 for _ in range(64)]
        grid_b = frame_b[-1] if frame_b else [[0] * 64 for _ in range(64)]

        arr_a = np.array(grid_a, dtype=np.uint8)
        arr_b = np.array(grid_b, dtype=np.uint8)
        diff_mask = arr_a != arr_b

        diff_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        diff_rgb[diff_mask] = [255, 0, 0]

        img = Image.fromarray(diff_rgb, "RGB")
        img = img.resize(
            (ClaudeVision.IMG_SIZE, ClaudeVision.IMG_SIZE), Image.NEAREST
        )
        return ClaudeVision._img_to_base64(img)

    @staticmethod
    def frame_hash(frame_3d: list[list[list[int]]]) -> str:
        """Quick hash for change detection."""
        grid = frame_3d[-1] if frame_3d else []
        raw = json.dumps(grid, separators=(",", ":")).encode()
        return hashlib.md5(raw).hexdigest()[:12]

    @staticmethod
    def content_hash(frame_3d: list[list[list[int]]]) -> str:
        """Hash ignoring timer bars (first row + last 2 rows) for stuck detection.

        Timer bars can appear at the top (row 0, e.g. vc33) or bottom (rows 62-63, e.g. ft09/ls20).
        Strip both to ensure stuck detection works across all game types.
        """
        grid = frame_3d[-1] if frame_3d else []
        content = grid[1:-2] if len(grid) > 3 else grid
        raw = json.dumps(content, separators=(",", ":")).encode()
        return hashlib.md5(raw).hexdigest()[:12]

    @staticmethod
    def content_change_magnitude(frame_a: list[list[list[int]]], frame_b: list[list[list[int]]]) -> int:
        """Count pixels that differ between two frames, ignoring timer bar (last 2 rows).

        Player movement changes ~2 pixels (old pos -> bg, new pos -> player color).
        Puzzle events (switch activation, pattern rotation) change many more pixels (10+).
        """
        grid_a = frame_a[-1] if frame_a else []
        grid_b = frame_b[-1] if frame_b else []
        rows = min(len(grid_a), len(grid_b)) - 2  # skip timer bar
        if rows <= 0:
            return 0
        changes = 0
        for y in range(rows):
            row_a = grid_a[y]
            row_b = grid_b[y]
            cols = min(len(row_a), len(row_b))
            for x in range(cols):
                if row_a[x] != row_b[x]:
                    changes += 1
        return changes

    @staticmethod
    def extract_objects(frame_3d: list[list[list[int]]]) -> str:
        """Extract colored object positions as structured text.

        Identifies non-background objects, their bounding boxes, and centroids.
        This gives Claude precise spatial data without relying on vision alone.
        """
        grid = frame_3d[-1] if frame_3d else [[0] * 64 for _ in range(64)]
        arr = np.array(grid, dtype=np.uint8)

        # Determine background color (most common)
        unique, counts = np.unique(arr, return_counts=True)
        bg_color = unique[np.argmax(counts)]

        # Also skip the second most common (likely walls/border)
        skip_colors = set()
        for c, cnt in zip(unique, counts):
            if cnt > 500:  # more than ~12% of grid
                skip_colors.add(int(c))

        objects = []
        for c in unique:
            c = int(c)
            if c in skip_colors:
                continue
            cnt = int(np.sum(arr == c))
            if cnt == 0:
                continue
            ys, xs = np.where(arr == c)
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            objects.append({
                "color": c,
                "name": COLOR_NAMES[c] if c < 16 else str(c),
                "pixels": cnt,
                "bbox": f"x=[{int(xs.min())}-{int(xs.max())}] y=[{int(ys.min())}-{int(ys.max())}]",
                "center": f"({cx},{cy})",
            })

        if not objects:
            return "No distinct objects detected."

        lines = []
        for o in sorted(objects, key=lambda x: -x["pixels"]):
            lines.append(
                f"- {o['name']} ({o['pixels']}px): {o['bbox']}, center {o['center']}"
            )
        return "\n".join(lines)

    @staticmethod
    def detect_changes(frame_a: list[list[list[int]]], frame_b: list[list[list[int]]]) -> str:
        """Describe what changed between two frames in text."""
        grid_a = frame_a[-1] if frame_a else [[0] * 64 for _ in range(64)]
        grid_b = frame_b[-1] if frame_b else [[0] * 64 for _ in range(64)]

        arr_a = np.array(grid_a, dtype=np.uint8)
        arr_b = np.array(grid_b, dtype=np.uint8)

        if np.array_equal(arr_a, arr_b):
            return "No pixels changed."

        diff_mask = arr_a != arr_b
        changed_count = int(np.sum(diff_mask))
        ys, xs = np.where(diff_mask)

        # What colors appeared/disappeared
        old_colors = set(arr_a[diff_mask].tolist())
        new_colors = set(arr_b[diff_mask].tolist())

        parts = [f"{changed_count} pixels changed in region x=[{xs.min()}-{xs.max()}] y=[{ys.min()}-{ys.max()}]."]

        old_names = [COLOR_NAMES[c] for c in old_colors if c < 16]
        new_names = [COLOR_NAMES[c] for c in new_colors if c < 16]
        parts.append(f"Old colors: {', '.join(old_names)}. New colors: {', '.join(new_names)}.")

        return " ".join(parts)

    @staticmethod
    def _img_to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ═══════════════════════════════════════════════════════════════════════════════
# ClaudeReasoner — Hypothesis & planning engine
# ═══════════════════════════════════════════════════════════════════════════════


class ClaudeReasoner:
    """Tracks hypotheses, confirmed rules, action outcomes, and spatial memory."""

    # Direction names mapped to action names
    DIR_NAMES = {"move_up": "up", "move_down": "down", "move_left": "left", "move_right": "right"}

    def __init__(self) -> None:
        self.hypotheses: list[dict[str, Any]] = []
        self.confirmed_rules: list[str] = []
        self.action_log: list[dict[str, Any]] = []
        self.stuck_counter: int = 0
        self.no_progress_counter: int = 0
        self._last_hash: str = ""
        # Spatial memory
        self.position_history: list[tuple[int, int]] = []
        self.goal_position: Optional[tuple[int, int]] = None
        self.visited_positions: set[tuple[int, int]] = set()
        self.directions_tried: dict[tuple[int, int], set[str]] = {}  # pos -> set of dirs tried
        self.walls: set[tuple[tuple[int, int], str]] = set()  # (pos, direction) = wall

    def on_level_complete(self) -> None:
        """Clear level-specific state, preserve general rules."""
        self.hypotheses.clear()
        self.action_log.clear()
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self._last_hash = ""
        self.position_history.clear()
        self.goal_position = None
        self.visited_positions.clear()
        self.directions_tried.clear()
        self.walls.clear()

    def update_stuck(self, current_hash: str) -> None:
        if current_hash == self._last_hash:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self._last_hash = current_hash

    def record_outcome(
        self,
        action_name: str,
        tool_input: dict[str, Any],
        frame_changed: bool,
        player_pos: Optional[tuple[int, int]] = None,
    ) -> None:
        entry = {
            "action": action_name,
            "reasoning": tool_input.get("reasoning", ""),
            "frame_changed": frame_changed,
            "pos": player_pos,
        }
        self.action_log.append(entry)

        # Track position
        if player_pos is not None:
            self.position_history.append(player_pos)
            self.visited_positions.add(player_pos)

        # Track directions tried at previous position
        if len(self.position_history) >= 2 and action_name in self.DIR_NAMES:
            prev_pos = self.position_history[-2] if len(self.position_history) >= 2 else None
            if prev_pos is not None:
                if prev_pos not in self.directions_tried:
                    self.directions_tried[prev_pos] = set()
                self.directions_tried[prev_pos].add(self.DIR_NAMES[action_name])

                # Detect walls: tried a direction but didn't move
                if player_pos == prev_pos:
                    self.walls.add((prev_pos, self.DIR_NAMES[action_name]))

        if frame_changed:
            for h in self.hypotheses:
                if h["confidence"] >= 0.7:
                    rule = h["text"]
                    if rule not in self.confirmed_rules:
                        self.confirmed_rules.append(rule)

    def detect_loop(self, window: int = 12) -> Optional[str]:
        """Detect if player is oscillating between positions."""
        if len(self.position_history) < 6:
            return None

        recent = self.position_history[-window:]
        pos_counts = Counter(recent)

        # Check for oscillation: any position visited 3+ times in last N steps
        repeated = [(pos, cnt) for pos, cnt in pos_counts.items() if cnt >= 3]
        if repeated:
            positions = [f"({p[0]},{p[1]})" for p, _ in sorted(repeated, key=lambda x: -x[1])]
            return f"LOOP DETECTED: You keep returning to {', '.join(positions[:3])}. You are going in circles!"

        # Check for 2-position ping-pong
        if len(recent) >= 4:
            last4 = recent[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                return f"PING-PONG: Oscillating between ({last4[0][0]},{last4[0][1]}) and ({last4[1][0]},{last4[1][1]}). Break the pattern!"

        return None

    def get_untried_directions(self, pos: tuple[int, int]) -> list[str]:
        """Get directions not yet tried at this position."""
        all_dirs = {"up", "down", "left", "right"}
        tried = self.directions_tried.get(pos, set())
        return sorted(all_dirs - tried)

    def get_distance_to_goal(self, pos: tuple[int, int]) -> Optional[int]:
        """Manhattan distance to goal."""
        if self.goal_position is None:
            return None
        return abs(pos[0] - self.goal_position[0]) + abs(pos[1] - self.goal_position[1])

    def get_progress_text(self) -> str:
        """Summary of spatial progress."""
        if len(self.position_history) < 2:
            return ""
        cur = self.position_history[-1]
        parts = [f"Position: ({cur[0]},{cur[1]})"]

        if self.goal_position:
            dist = self.get_distance_to_goal(cur)
            parts.append(f"Goal: ({self.goal_position[0]},{self.goal_position[1]}), distance={dist}")

            # Show progress trend
            if len(self.position_history) >= 5:
                old_dist = self.get_distance_to_goal(self.position_history[-5])
                if old_dist is not None and dist is not None:
                    if dist < old_dist:
                        parts.append(f"Progress: GOOD (distance {old_dist}->{dist})")
                    elif dist > old_dist:
                        parts.append(f"Progress: REGRESSING (distance {old_dist}->{dist})")
                    else:
                        parts.append(f"Progress: STALLED (distance unchanged at {dist})")

        untried = self.get_untried_directions(cur)
        if untried:
            parts.append(f"Untried directions here: {', '.join(untried)}")

        # Nearby walls
        nearby_walls = [(p, d) for (p, d) in self.walls
                        if abs(p[0] - cur[0]) <= 3 and abs(p[1] - cur[1]) <= 3]
        if nearby_walls:
            wall_strs = [f"({p[0]},{p[1]})->{d}" for p, d in sorted(nearby_walls)[:6]]
            parts.append(f"Nearby walls: {', '.join(wall_strs)}")

        return " | ".join(parts)

    def get_rules_text(self) -> str:
        if not self.confirmed_rules:
            return "None yet."
        return "\n".join(f"- {r}" for r in self.confirmed_rules[-8:])

    def get_hypotheses_text(self) -> str:
        if not self.hypotheses:
            return "None yet."
        lines = []
        for h in sorted(self.hypotheses, key=lambda x: -x["confidence"])[:5]:
            lines.append(f"- [{h['confidence']:.0%}] {h['text']}")
        return "\n".join(lines)

    def get_recent_actions_text(self, n: int = 10) -> str:
        if not self.action_log:
            return "None yet."
        recent = self.action_log[-n:]
        lines = []
        for a in recent:
            changed = "Y" if a["frame_changed"] else "N"
            pos_str = f"@({a['pos'][0]},{a['pos'][1]})" if a.get("pos") else ""
            lines.append(f"  {a['action']:12s} [{changed}] {pos_str:10s} {a['reasoning'][:50]}")
        return "\n".join(lines)

    def get_stuck_guidance(self) -> str:
        # Loop detection takes priority
        loop = self.detect_loop()
        if loop:
            cur = self.position_history[-1] if self.position_history else None
            untried = self.get_untried_directions(cur) if cur else []
            msg = f"CRITICAL: {loop}"
            if untried:
                msg += f" Try an UNTRIED direction: {', '.join(untried)}."
            else:
                msg += " All directions tried here. Backtrack to a junction you passed earlier."
            return msg

        if self.stuck_counter >= 8:
            return (
                "CRITICAL: 8+ actions with no frame change. You are blocked by a wall. "
                "Try the OPPOSITE direction or a completely different approach."
            )
        if self.stuck_counter >= 4:
            return (
                "WARNING: 4+ actions with no change. Your current approach isn't working. "
                "Try a different direction."
            )
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# MazeNavigator — Programmatic DFS exploration for movement-only games
# ═══════════════════════════════════════════════════════════════════════════════


class MazeNavigator:
    """Persistent maze explorer that builds an adjacency graph across timer resets.

    For movement-only maze games (available_actions=[1,2,3,4]), this replaces
    Claude calls with efficient programmatic DFS/BFS exploration. The graph
    persists across timer resets so each attempt starts with accumulated knowledge
    and navigates directly to the exploration frontier.
    """

    DIR_TO_ACTION = {"up": GameAction.ACTION1, "down": GameAction.ACTION2,
                     "left": GameAction.ACTION3, "right": GameAction.ACTION4}
    ACTION_TO_DIR = {v: k for k, v in DIR_TO_ACTION.items()}
    REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}

    def __init__(self) -> None:
        self.graph: dict[tuple[int, int], dict[str, tuple[int, int]]] = {}
        self.walls: set[tuple[tuple[int, int], str]] = set()
        self.visited: set[tuple[int, int]] = set()
        self.goal: Optional[tuple[int, int]] = None
        self.start_pos: Optional[tuple[int, int]] = None
        self._pending_action_dir: Optional[str] = None
        self._pending_from: Optional[tuple[int, int]] = None
        self.actions_taken: int = 0

    def full_reset(self) -> None:
        """Clear everything (game-level reset)."""
        self.graph.clear()
        self.walls.clear()
        self.visited.clear()
        self._pending_action_dir = None
        self._pending_from = None
        self.actions_taken = 0
        self.start_pos = None

    def timer_reset(self) -> None:
        """Timer expired — keep the graph but clear pending state."""
        self._pending_action_dir = None
        self._pending_from = None

    def set_goal(self, goal: tuple[int, int]) -> None:
        self.goal = goal

    def is_timer_teleport(self, old_pos: Optional[tuple[int, int]], new_pos: tuple[int, int]) -> bool:
        """Detect if position jumped due to timer reset (not a real move)."""
        if old_pos is None:
            return False
        dx = abs(new_pos[0] - old_pos[0])
        dy = abs(new_pos[1] - old_pos[1])
        # Normal moves are exactly 5 pixels in one axis
        return dx > 5 or dy > 5 or (dx > 0 and dy > 0)

    def record_move(self, old_pos: tuple[int, int], direction: str, new_pos: tuple[int, int]) -> None:
        """Record the result of a movement action."""
        # Reject moves that look like timer teleports
        if self.is_timer_teleport(old_pos, new_pos):
            return

        if old_pos == new_pos:
            self.walls.add((old_pos, direction))
        else:
            if old_pos not in self.graph:
                self.graph[old_pos] = {}
            self.graph[old_pos][direction] = new_pos
            rev = self.REVERSE[direction]
            if new_pos not in self.graph:
                self.graph[new_pos] = {}
            self.graph[new_pos][rev] = old_pos

        self.visited.add(old_pos)
        self.visited.add(new_pos)

    def update(self, new_pos: tuple[int, int]) -> None:
        """Called after each action with the resulting position."""
        if self._pending_action_dir is not None and self._pending_from is not None:
            self.record_move(self._pending_from, self._pending_action_dir, new_pos)
        self._pending_action_dir = None
        self._pending_from = None

    def get_path_to_goal(self, from_pos: tuple[int, int]) -> Optional[list[str]]:
        """BFS shortest path from current position to goal through known graph."""
        if self.goal is None:
            return None
        return self._bfs_path(from_pos, self.goal)

    def _bfs_path(self, start: tuple[int, int], target: tuple[int, int]) -> Optional[list[str]]:
        """BFS through explored graph to find path."""
        if start == target:
            return []
        queue: list[tuple[tuple[int, int], list[str]]] = [(start, [])]
        seen: set[tuple[int, int]] = {start}
        while queue:
            pos, path = queue.pop(0)
            for direction, neighbor in self.graph.get(pos, {}).items():
                if neighbor in seen:
                    continue
                new_path = path + [direction]
                if neighbor == target:
                    return new_path
                seen.add(neighbor)
                queue.append((neighbor, new_path))
        return None

    def _get_unexplored_dirs(self, pos: tuple[int, int]) -> list[str]:
        """Get directions not yet tried at this position, sorted toward goal."""
        all_dirs = ["down", "right", "up", "left"]
        if self.goal:
            gx, gy = self.goal
            px, py = pos
            def goal_score(d: str) -> int:
                dx = {"left": -5, "right": 5}.get(d, 0)
                dy = {"up": -5, "down": 5}.get(d, 0)
                return abs(px + dx - gx) + abs(py + dy - gy)
            all_dirs.sort(key=goal_score)

        result = []
        neighbors = self.graph.get(pos, {})
        for d in all_dirs:
            if (pos, d) not in self.walls and d not in neighbors:
                result.append(d)
        return result

    def _find_nearest_unexplored(self, pos: tuple[int, int]) -> Optional[list[str]]:
        """BFS to find shortest path to a position with unexplored directions."""
        queue: list[tuple[tuple[int, int], list[str]]] = [(pos, [])]
        seen: set[tuple[int, int]] = {pos}
        while queue:
            curr, path = queue.pop(0)
            for d, neighbor in self.graph.get(curr, {}).items():
                if neighbor in seen:
                    continue
                new_path = path + [d]
                if self._get_unexplored_dirs(neighbor):
                    return new_path
                seen.add(neighbor)
                queue.append((neighbor, new_path))
        return None

    def suggest_action(self, pos: tuple[int, int]) -> Optional[GameAction]:
        """Always return an action: follow path to goal, explore, or navigate to frontier."""
        if pos is None:
            return None

        self.visited.add(pos)
        if self.start_pos is None:
            self.start_pos = pos

        # 1. If goal known and path exists, follow it
        if self.goal is not None:
            path = self.get_path_to_goal(pos)
            if path:
                return self._emit(path[0], pos)

        # 2. Try unexplored directions at current position
        unexplored = self._get_unexplored_dirs(pos)
        if unexplored:
            return self._emit(unexplored[0], pos)

        # 3. Navigate to nearest position with unexplored directions
        path_to_frontier = self._find_nearest_unexplored(pos)
        if path_to_frontier:
            return self._emit(path_to_frontier[0], pos)

        # 4. All explored — navigate toward the position closest to goal
        if self.goal is not None:
            best_pos = min(
                self.visited,
                key=lambda p: abs(p[0] - self.goal[0]) + abs(p[1] - self.goal[1]),  # type: ignore
            )
            if best_pos != pos:
                path = self._bfs_path(pos, best_pos)
                if path:
                    return self._emit(path[0], pos)

        # 5. True dead end — try any non-wall direction
        neighbors = self.graph.get(pos, {})
        for d in ["down", "right", "up", "left"]:
            if (pos, d) not in self.walls:
                return self._emit(d, pos)

        return None

    def _emit(self, direction: str, pos: tuple[int, int]) -> GameAction:
        """Set pending state and return the corresponding action."""
        self._pending_action_dir = direction
        self._pending_from = pos
        self.actions_taken += 1
        return self.DIR_TO_ACTION[direction]


# ═══════════════════════════════════════════════════════════════════════════════
# ClaudeAgent — Main agent class
# ═══════════════════════════════════════════════════════════════════════════════


class ClaudeAgent(Agent):
    """ARC-AGI-3 agent powered by Claude Sonnet 4.5 with vision + tool use."""

    MODEL = "claude-sonnet-4-5-20250929"
    MAX_ACTIONS = 200
    MAX_HISTORY_TURNS = 6  # more context for better reasoning
    MAX_TOKENS = 512  # tool calls are short

    # Send image every N actions (text-only in between saves ~7k tokens/turn)
    IMAGE_EVERY_N = 3

    # Maze navigation: use programmatic DFS after N initial Claude calls
    MAZE_MODE_AFTER = 999  # disabled — puzzles are interactive, not simple mazes
    CLAUDE_EVERY_N_MAZE = 40  # call Claude infrequently during maze nav

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        self.vision = ClaudeVision()
        self.reasoner = ClaudeReasoner()
        self.navigator = MazeNavigator()
        self.message_history: list[dict[str, Any]] = []
        self._prev_frame_hash: str = ""
        self._prev_levels_completed: int = 0
        self._prev_frame: Optional[list[list[list[int]]]] = None
        self._prev_player_pos: Optional[tuple[int, int]] = None
        self._last_tool_name: str = ""
        self._last_tool_input: dict[str, Any] = {}
        self._available_actions: list[int] = []
        self._turns_since_image: int = 0
        self._claude_calls_this_level: int = 0
        self._maze_steps_since_claude: int = 0
        self._is_movement_only: bool = False
        self._position_stuck_counter: int = 0
        self._consecutive_confirms: int = 0
        self._nav_queue: list[GameAction] = []  # queued moves from navigate_to
        self._nav_target: Optional[tuple[int, int]] = None  # current navigation target
        self._nav_wall_retries: int = 0  # wall hits during navigation
        self._nav_steps_taken: int = 0  # total steps in current navigate_to
        self._nav_best_dist: int = 999  # closest Manhattan distance achieved
        self._game_type: str = "unknown"
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # ── 1. Handle NOT_PLAYED / GAME_OVER → RESET ────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.reasoner.on_level_complete()
            self.navigator.full_reset()
            self.message_history.clear()
            self._prev_frame = None
            self._prev_frame_hash = ""
            self._prev_player_pos = None
            self._turns_since_image = 0
            self._claude_calls_this_level = 0
            self._maze_steps_since_claude = 0
            self._is_movement_only = False
            self._position_stuck_counter = 0
            self._consecutive_confirms = 0
            self._nav_queue.clear()
            self._nav_target = None
            self._nav_wall_retries = 0
            self._nav_steps_taken = 0
            self._nav_best_dist = 999
            # Keep _game_type across resets - game type doesn't change
            return GameAction.RESET

        # ── 2. Detect level transition ───────────────────────────────────
        if latest_frame.levels_completed > self._prev_levels_completed:
            logger.info(
                f"Level complete! {self._prev_levels_completed} → {latest_frame.levels_completed}"
            )
            self.reasoner.on_level_complete()
            self.navigator.full_reset()
            self.message_history.clear()
            self._turns_since_image = 0
            self._claude_calls_this_level = 0
            self._maze_steps_since_claude = 0
            self._is_movement_only = False
            self._prev_player_pos = None
            self._position_stuck_counter = 0
            self._consecutive_confirms = 0
            self._nav_queue.clear()
            self._nav_target = None
            self._nav_wall_retries = 0
            self._nav_steps_taken = 0
            self._nav_best_dist = 999
        self._prev_levels_completed = latest_frame.levels_completed

        # ── 3. Track available actions ───────────────────────────────────
        self._available_actions = latest_frame.available_actions
        movement_actions = {1, 2, 3, 4}
        if set(self._available_actions).issubset(movement_actions):
            self._is_movement_only = True

        # Detect game type on first observation
        if self._game_type == "unknown" and self._available_actions:
            actions = set(self._available_actions)
            if actions == {6}:
                self._game_type = "click_only"
            elif 5 in actions and 6 in actions:
                self._game_type = "arc_puzzle"
            elif actions.issubset({1, 2, 3, 4}):
                self._game_type = "movement"
            else:
                self._game_type = "mixed"
            logger.info(f"Game type: {self._game_type} (actions: {self._available_actions})")

        # ── 4. Extract positions + detect timer resets ───────────────────
        current_hash = self.vision.frame_hash(latest_frame.frame)
        content_hash = self.vision.content_hash(latest_frame.frame)  # ignores timer bar
        frame_changed = current_hash != self._prev_frame_hash

        # Player/goal detection only for movement games
        player_pos: Optional[tuple[int, int]] = None
        goal_pos: Optional[tuple[int, int]] = None
        if self._game_type in ("movement", "unknown"):
            player_pos = self._find_player_position(latest_frame.frame)
            goal_pos = self._find_goal_position(latest_frame.frame)
            if goal_pos and self.reasoner.goal_position is None:
                self.reasoner.goal_position = goal_pos
            if goal_pos:
                self.navigator.set_goal(goal_pos)

            # Detect timer-based reset (position teleports back to start)
            if player_pos is not None and self._prev_player_pos is not None:
                if self.navigator.is_timer_teleport(self._prev_player_pos, player_pos):
                    logger.info(
                        f"Timer reset detected: {self._prev_player_pos} -> {player_pos}, "
                        f"preserving graph ({len(self.navigator.graph)} nodes, "
                        f"{len(self.navigator.walls)} walls)"
                    )
                    self.navigator.timer_reset()
                    self.message_history.clear()
                    self._claude_calls_this_level = 0
                    self._maze_steps_since_claude = 0
                    self._position_stuck_counter = 0
                    self._nav_queue.clear()
                    self._nav_target = None
                    self._nav_wall_retries = 0
                    self._nav_steps_taken = 0
                    self._nav_best_dist = 999
                else:
                    # Normal move — update navigator
                    self.navigator.update(player_pos)
            elif player_pos is not None:
                self.navigator.update(player_pos)

            # Position-based stuck detection (immune to timer pixel changes)
            if player_pos is not None and player_pos == self._prev_player_pos:
                self._position_stuck_counter += 1
            else:
                self._position_stuck_counter = 0

        if self._last_tool_name and self._prev_frame_hash:
            self.reasoner.record_outcome(
                self._last_tool_name, self._last_tool_input, frame_changed,
                player_pos=player_pos,
            )

        old_player_pos = self._prev_player_pos  # save before update for nav wall detection
        self._prev_player_pos = player_pos
        self.reasoner.update_stuck(content_hash)  # use content hash (ignores timer bar)

        # ── 5. Navigate-to queue execution ───────────────────────────────
        # If Claude previously issued navigate_to, drain the queue step by step
        if self._nav_queue and player_pos is not None:
            # Check if navigation target was reached (within 3 pixels)
            if self._nav_target:
                dist = abs(player_pos[0] - self._nav_target[0]) + abs(player_pos[1] - self._nav_target[1])
                if dist <= 3:
                    logger.info(f"NavTo: reached target {self._nav_target} at {player_pos}")
                    self._nav_queue.clear()
                    self._nav_target = None
                    self._nav_wall_retries = 0
                    self._nav_steps_taken = 0
                    self._nav_best_dist = 999
                    # Fall through to call Claude

                # Track best distance for progress detection
                if dist < self._nav_best_dist:
                    self._nav_best_dist = dist

            # Detect wall hit: position didn't change after a move
            if (self._nav_queue and player_pos == old_player_pos
                    and self._last_tool_name.startswith("move_")):
                self._nav_wall_retries += 1
                if self._nav_wall_retries >= 3:
                    logger.info(f"NavTo: stuck at wall after {self._nav_wall_retries} retries, aborting nav to {self._nav_target}")
                    self._nav_queue.clear()
                    self._nav_target = None
                    self._nav_wall_retries = 0
                    self._nav_steps_taken = 0
                    self._nav_best_dist = 999
                    # Fall through to call Claude
                else:
                    # Recompute path from current position avoiding the wall direction
                    self._recompute_nav_queue(player_pos)
            elif player_pos != old_player_pos:
                self._nav_wall_retries = 0

            # Progress-based abort: if we've taken many steps without getting closer, give up
            if self._nav_queue and self._nav_target and self._nav_steps_taken > 12:
                curr_dist = abs(player_pos[0] - self._nav_target[0]) + abs(player_pos[1] - self._nav_target[1])
                if curr_dist >= self._nav_best_dist:
                    logger.info(
                        f"NavTo: no progress after {self._nav_steps_taken} steps "
                        f"(best dist={self._nav_best_dist}, current={curr_dist}), aborting"
                    )
                    self._nav_queue.clear()
                    self._nav_target = None
                    self._nav_wall_retries = 0
                    self._nav_steps_taken = 0
                    self._nav_best_dist = 999

        if self._nav_queue and player_pos is not None:
            action = self._nav_queue.pop(0)
            self._nav_steps_taken += 1
            dir_name = MazeNavigator.ACTION_TO_DIR.get(action, "?")
            action.reasoning = {
                "tool": f"nav_to_{dir_name}",
                "reasoning": f"navigate_to {self._nav_target} via {dir_name}",
                "stuck": self._position_stuck_counter,
                "tokens": [self.input_tokens, self.output_tokens],
            }
            self._last_tool_name = f"move_{dir_name}"
            self._last_tool_input = {"reasoning": f"nav_to {self._nav_target}"}
            self._prev_frame = latest_frame.frame
            self._prev_frame_hash = current_hash
            logger.info(
                f"NavTo: {dir_name} toward {self._nav_target} from ({player_pos[0]},{player_pos[1]}) | "
                f"queue={len(self._nav_queue)} remaining | step {self._nav_steps_taken}"
            )
            return action

        # ── 6. Build tools (filtered to available_actions) ───────────────
        tools = self._get_available_tools()

        # ── 6. Decide whether to include image this turn ─────────────────
        send_image = (
            self._game_type in ("arc_puzzle", "click_only")  # always for puzzles
            or self._turns_since_image >= self.IMAGE_EVERY_N
            or self._prev_frame is None  # first frame
            or frame_changed  # something changed, worth seeing
            or self.reasoner.stuck_counter >= 3  # stuck, need fresh look
        )

        # ── 7. Render visuals ────────────────────────────────────────────
        frame_b64: Optional[str] = None
        diff_b64: Optional[str] = None
        if send_image:
            frame_b64 = self.vision.render_frame(latest_frame.frame)
            if self._prev_frame is not None and frame_changed:
                diff_b64 = self.vision.render_diff(self._prev_frame, latest_frame.frame)
            self._turns_since_image = 0
        else:
            self._turns_since_image += 1

        # ── 8. Extract object positions (always — cheap text) ────────────
        objects_text = self.vision.extract_objects(latest_frame.frame)
        changes_text = ""
        if self._prev_frame is not None:
            changes_text = self.vision.detect_changes(self._prev_frame, latest_frame.frame)

        # ── 9. Build system prompt ───────────────────────────────────────
        system_prompt = self._build_system_prompt()

        # ── 10. Build user message ───────────────────────────────────────
        user_content = self._build_user_message(
            latest_frame, frame_b64, diff_b64, frame_changed,
            objects_text, changes_text,
        )

        # ── 11. Call Claude ──────────────────────────────────────────────
        messages = [*self.message_history, {"role": "user", "content": user_content}]

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=tools,
                tool_choice={"type": "any"},
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            messages = [{"role": "user", "content": user_content}]
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=[{"type": "text", "text": system_prompt}],
                tools=tools,
                tool_choice={"type": "any"},
                messages=messages,
            )

        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

        # ── 12. Extract tool call ────────────────────────────────────────
        tool_name, tool_input = self._extract_tool_call(response)

        # ── 13. Map to GameAction ────────────────────────────────────────
        # Handle navigate_to virtual tool
        if tool_name == "navigate_to" and player_pos is not None:
            tx = max(0, min(63, tool_input.get("x", 32)))
            ty = max(0, min(63, tool_input.get("y", 32)))
            self._nav_target = (tx, ty)
            self._nav_queue = self._compute_nav_queue(player_pos, (tx, ty))
            self._nav_wall_retries = 0
            self._nav_steps_taken = 0
            self._nav_best_dist = abs(player_pos[0] - tx) + abs(player_pos[1] - ty)
            logger.info(
                f"NavTo: planning path from ({player_pos[0]},{player_pos[1]}) to ({tx},{ty}), "
                f"{len(self._nav_queue)} steps"
            )
            if self._nav_queue:
                action = self._nav_queue.pop(0)
                dir_name = MazeNavigator.ACTION_TO_DIR.get(action, "?")
                action.reasoning = {
                    "tool": f"nav_to_{dir_name}",
                    "reasoning": tool_input.get("reasoning", f"navigate_to ({tx},{ty})"),
                    "stuck": self.reasoner.stuck_counter,
                    "tokens": [self.input_tokens, self.output_tokens],
                }
                self._last_tool_name = f"move_{dir_name}"
                self._last_tool_input = tool_input
                self._prev_frame = latest_frame.frame
                self._prev_frame_hash = current_hash
                self._claude_calls_this_level += 1
                # Don't update message history for navigate_to — next Claude call gets fresh context
                return action
            # If no path computed, fall through to regular action

        action = TOOL_TO_ACTION.get(tool_name, GameAction.RESET)

        # Track confirm spam — 3+ confirms without level advance = stuck
        if tool_name == "confirm":
            self._consecutive_confirms += 1
            if self._consecutive_confirms >= 3:
                logger.info(
                    f"Confirm spam detected ({self._consecutive_confirms} consecutive), "
                    f"forcing reset and clearing history"
                )
                action = GameAction.RESET
                tool_name = "reset"
                tool_input = {"reasoning": "forced reset: 3+ confirms without progress"}
                self._consecutive_confirms = 0
                self.message_history.clear()
        else:
            self._consecutive_confirms = 0

        # Reset stuck counter when Claude explicitly resets
        if tool_name == "reset":
            self.reasoner.stuck_counter = 0
            self.message_history.clear()

        # Validate action is actually available
        if action.value not in self._available_actions and action != GameAction.RESET:
            logger.warning(f"Claude chose unavailable action {tool_name}, falling back to first available")
            if self._available_actions:
                action = GameAction.from_id(self._available_actions[0])
            else:
                action = GameAction.RESET

        if action == GameAction.ACTION6:  # click
            x = max(0, min(63, tool_input.get("x", 32)))
            y = max(0, min(63, tool_input.get("y", 32)))
            action.set_data({"x": x, "y": y})
            logger.info(f"Click at ({x}, {y})")

        action.reasoning = {
            "tool": tool_name,
            "reasoning": tool_input.get("reasoning", ""),
            "stuck": self.reasoner.stuck_counter,
            "tokens": [self.input_tokens, self.output_tokens],
        }

        # Track this move in navigator for graph building
        if player_pos is not None and action in MazeNavigator.ACTION_TO_DIR:
            dir_name = MazeNavigator.ACTION_TO_DIR[action]
            self.navigator._pending_action_dir = dir_name
            self.navigator._pending_from = player_pos

        self._claude_calls_this_level += 1
        self._maze_steps_since_claude = 0

        # ── 14. Update message history ───────────────────────────────────
        self.message_history.append({"role": "user", "content": user_content})

        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        self.message_history.append({"role": "assistant", "content": assistant_content})

        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use_block:
            self.message_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": "OK. Next frame follows.",
                    }
                ],
            })

        self._evict_history()

        # ── 15. Save state ───────────────────────────────────────────────
        self._prev_frame = latest_frame.frame
        self._prev_frame_hash = current_hash
        self._last_tool_name = tool_name
        self._last_tool_input = tool_input

        logger.info(
            f"Claude chose {tool_name} | stuck={self.reasoner.stuck_counter} "
            f"| tokens=[{self.input_tokens},{self.output_tokens}]"
        )

        return action

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_available_tools(self) -> list[dict[str, Any]]:
        """Return only tools matching the game's available_actions."""
        if not self._available_actions:
            # Fallback: offer all tools
            return list(ALL_TOOL_SCHEMAS.values())

        tools = []
        for action_id in self._available_actions:
            if action_id in ALL_TOOL_SCHEMAS:
                tools.append(ALL_TOOL_SCHEMAS[action_id])

        # Always include reset as escape hatch
        if 0 not in self._available_actions:
            tools.append(ALL_TOOL_SCHEMAS[0])

        # For movement games, add navigate_to virtual tool
        if self._game_type in ("movement", "unknown"):
            tools.append(NAVIGATE_TO_SCHEMA)

        return tools if tools else list(ALL_TOOL_SCHEMAS.values())

    def _find_player_position(self, frame_3d: list[list[list[int]]]) -> Optional[tuple[int, int]]:
        """Find the player object's centroid. Looks for small colored objects (orange, green, blue)."""
        grid = frame_3d[-1] if frame_3d else [[0] * 64 for _ in range(64)]
        arr = np.array(grid, dtype=np.uint8)

        unique, counts = np.unique(arr, return_counts=True)
        # Background + large structural colors (walls, floor)
        skip = {int(c) for c, n in zip(unique, counts) if n > 200}

        # Player is typically small (4-50 pixels), often orange(12), green(14), or blue(9)
        # Prefer orange as player indicator
        candidates = []
        for c, n in zip(unique, counts):
            c, n = int(c), int(n)
            if c in skip or n < 2 or n > 60:
                continue
            ys, xs = np.where(arr == c)
            candidates.append((c, n, int(np.mean(xs)), int(np.mean(ys))))

        if not candidates:
            return None

        # Prefer orange (12) as likely player
        for c, n, cx, cy in candidates:
            if c == 12:  # orange
                return (cx, cy)
        # Otherwise smallest distinct object
        candidates.sort(key=lambda x: x[1])
        return (candidates[0][2], candidates[0][3])

    def _find_goal_position(self, frame_3d: list[list[list[int]]]) -> Optional[tuple[int, int]]:
        """Find the goal object's centroid. Typically red(8) or green(14)."""
        grid = frame_3d[-1] if frame_3d else [[0] * 64 for _ in range(64)]
        arr = np.array(grid, dtype=np.uint8)

        unique, counts = np.unique(arr, return_counts=True)
        skip = {int(c) for c, n in zip(unique, counts) if n > 200}

        # Goal is typically red(8) and small-medium sized
        for c, n in zip(unique, counts):
            c, n = int(c), int(n)
            if c in skip or n < 2:
                continue
            if c == 8:  # red
                ys, xs = np.where(arr == c)
                return (int(np.mean(xs)), int(np.mean(ys)))

        return None

    def _compute_nav_queue(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> list[GameAction]:
        """Compute a simple greedy path from from_pos to to_pos.

        Each step moves ~5 pixels. Alternates between x and y to handle L-shaped paths.
        """
        STEP = 5
        queue: list[GameAction] = []
        cx, cy = from_pos
        tx, ty = to_pos
        max_steps = 30  # safety limit

        while max_steps > 0 and (abs(cx - tx) > 2 or abs(cy - ty) > 2):
            dx = tx - cx
            dy = ty - cy

            # Move in the axis with larger remaining distance
            if abs(dx) >= abs(dy):
                if dx > 0:
                    queue.append(GameAction.ACTION4)  # right
                    cx += STEP
                else:
                    queue.append(GameAction.ACTION3)  # left
                    cx -= STEP
            else:
                if dy > 0:
                    queue.append(GameAction.ACTION2)  # down
                    cy += STEP
                else:
                    queue.append(GameAction.ACTION1)  # up
                    cy -= STEP
            max_steps -= 1

        return queue

    def _recompute_nav_queue(self, from_pos: tuple[int, int]) -> None:
        """Recompute nav queue after hitting a wall. Try perpendicular approach."""
        if self._nav_target is None:
            self._nav_queue.clear()
            return

        # Get the last direction that failed (wall hit)
        blocked_dir: Optional[str] = None
        if self._last_tool_name.startswith("move_"):
            blocked_dir = self._last_tool_name.replace("move_", "")

        tx, ty = self._nav_target
        cx, cy = from_pos
        STEP = 5

        # Try perpendicular direction first, then resume toward target
        queue: list[GameAction] = []
        if blocked_dir in ("left", "right"):
            # Try going up or down to get around the wall
            if ty > cy:
                queue.append(GameAction.ACTION2)  # down
            else:
                queue.append(GameAction.ACTION1)  # up
        elif blocked_dir in ("up", "down"):
            if tx > cx:
                queue.append(GameAction.ACTION4)  # right
            else:
                queue.append(GameAction.ACTION3)  # left

        # Then continue toward target
        queue.extend(self._compute_nav_queue(from_pos, self._nav_target))
        self._nav_queue = queue[:20]  # limit re-computed path

    def _build_system_prompt(self) -> str:
        if self._game_type == "arc_puzzle":
            return self._build_system_prompt_arc()
        elif self._game_type == "click_only":
            return self._build_system_prompt_click()
        else:
            return self._build_system_prompt_movement()

    def _build_system_prompt_arc(self) -> str:
        return f"""You are an expert ARC puzzle solver. The 64x64 grid shows an ARC-style transformation puzzle.

LAYOUT (4 quadrants):
- TOP-LEFT: Example INPUT (3x3 grid of colored cells)
- TOP-RIGHT: Example OUTPUT (3x3 grid showing the transformation result)
- BOTTOM-LEFT: Test INPUT (3x3 grid you must transform)
- BOTTOM-RIGHT: Test OUTPUT (EDITABLE - bordered area you must fill in correctly)

Each "cell" in the 3x3 grids is a 6x6 pixel block. Cells are separated by black lines.
The test output grid has a red/dark-gray border marking it as editable.
A progress bar at the bottom row shows remaining time.

YOUR TASK:
1. ANALYZE: Compare the example Input to the example Output. What transformation was applied?
   - Look for: symmetry operations, color swaps, rotations, reflections, pattern completion
   - The output may make the pattern symmetric, fill in missing elements, or apply a rule
2. PREDICT: Apply the same transformation to the Test Input to determine the correct Test Output
3. MODIFY: Click on cells in the Test Output that need to change color
   - Each click cycles the cell to the next color
   - Click the CENTER of the 6x6 pixel cell you want to change
4. SUBMIT: Use CONFIRM (action 5) when you believe the output is correct

IMPORTANT:
- The Test Output starts with a DEFAULT fill (usually all one color) - you must change it
- Movement actions (up/down/left/right) select or navigate within the grid
- Use CLICK to modify cell colors in the test output
- Use CONFIRM to submit your answer ONLY when you are confident it is correct
- You have limited time (progress bar) - work efficiently
- After confirming, a new level with a different puzzle will appear
- DO NOT spam confirm! If confirm doesn't advance to the next level, your answer is WRONG.
  Stop confirming, click cells to fix your answer, then try confirm again.
- After 2 failed confirms, consider using RESET to start the level fresh with a new strategy

ACTION #{self.action_counter}/{self.MAX_ACTIONS} | Stuck: {self.reasoner.stuck_counter} | Tokens: [{self.input_tokens},{self.output_tokens}]"""

    def _build_system_prompt_click(self) -> str:
        stuck = self.reasoner.get_stuck_guidance()
        stuck_section = f"\n\n**{stuck}**" if stuck else ""
        return f"""You are solving a visual click-based puzzle on a 64x64 grid.

ONLY ACTION AVAILABLE: Click (provide x,y coordinates 0-63).

STRATEGY:
1. OBSERVE: Study the grid image AND the objects list. Identify distinct colored regions and small objects.
2. CLICK SMALL OBJECTS: Interactive elements are typically small colored objects (4-25 pixels).
   Large uniform regions (hundreds of pixels) are background, NOT interactive.
3. WATCH: After each click, check if the grid changed. If nothing changed, that click target was wrong.
4. SYSTEMATIC: If your clicks aren't working, try DIFFERENT coordinates. Use the objects list to find
   small distinct-colored objects you haven't tried yet.
5. NEVER repeat the same click coordinates if nothing changed.

COORDINATE SYSTEM:
- x=0 left, x=63 right. y=0 top, y=63 bottom.
- Click coordinates are PIXEL positions on the 64x64 grid.
- Use the objects list center coordinates for precise clicking.

WHAT TO CLICK:
- Small colored objects (maroon, teal, cyan, yellow blocks of 4-25 pixels)
- NOT large background regions (green, black areas of 500+ pixels)
- NOT the timer bar (row 0 or rows 62-63)
- Try clicking the CENTER of each small object from the objects list
- There may be multiple levels — after solving one, a new puzzle appears

ACTION #{self.action_counter}/{self.MAX_ACTIONS} | Stuck: {self.reasoner.stuck_counter} | Tokens: [{self.input_tokens},{self.output_tokens}]{stuck_section}"""

    def _build_system_prompt_movement(self) -> str:
        stuck = self.reasoner.get_stuck_guidance()
        stuck_section = f"\n\n**{stuck}**" if stuck else ""

        progress = self.reasoner.get_progress_text()
        progress_section = f"\n\nSPATIAL STATUS: {progress}" if progress else ""

        return f"""You are playing a 64x64 grid puzzle game using movement (up/down/left/right).

YOUR TASK: Touch a ROTATION SWITCH to make a pattern match, then enter the TARGET BOX.

GRID LAYOUT:
- PLAYER: Orange+maroon sprite (~5x5px). You start at approximately (x=39, y=45) in the right room.
- BOX 1 (TARGET): Upper area, rows 8-16, cols 32-40. Contains the FIXED target pattern. Enter here AFTER matching.
- BOX 2 (DISPLAY): Bottom-left, rows 53-62, cols 1-10. Contains a ROTATABLE pattern.
- ROTATION SWITCH: Small blue+black object at approximately (x=20-22, y=31-33). Move onto position (x=19, y=30) to activate it.
- TIMER: ~41 moves per life, 3 lives total. Teal bar at rows 61-62.
- GREEN (3) = walls (impassable). YELLOW (4) = walkable floor. GRAY (5) = walkable.

HOW IT WORKS:
1. Box 2's pattern starts in State A (does NOT match Box 1).
2. Each time you walk onto the SWITCH at (x=19, y=30), Box 2's pattern rotates 90° clockwise.
3. After exactly 1 touch, Box 2 rotates to State B which MATCHES Box 1's target pattern.
4. After matching, navigate to BOX 1 (upper area, y~15, x~34) and walk INTO it to complete the level.
5. WARNING: Touching the switch again rotates AWAY from the match. Touch it exactly ONCE.

OPTIMAL STRATEGY (13 moves):
1. navigate_to(19, 30) — move LEFT and UP to reach the switch. This rotates Box 2 to match.
2. navigate_to(34, 15) — move UP and RIGHT to enter Box 1. Level complete!

COORDINATE SYSTEM:
- x=0 left, x=63 right. y=0 top, y=63 bottom.
- move_up: y-=5. move_down: y+=5. move_left: x-=5. move_right: x+=5.

NAVIGATION:
- Use navigate_to(x, y) for efficient multi-step movement.
- If navigate_to gets stuck at walls, use individual moves to find openings.
- The switch is in the LEFT room. Box 1 is in the UPPER area connected by a corridor.

CRITICAL RULES:
- Touch the switch EXACTLY ONCE, then go to Box 1.
- Do NOT touch the switch more than once (it rotates away from the match).
- Do NOT wander — every move costs timer. Follow the optimal path.
- You have ~41 moves per life. The solution needs ~13 moves. Be efficient.

RULES LEARNED:
{self.reasoner.get_rules_text()}{progress_section}{stuck_section}"""

    def _build_user_message(
        self,
        frame: FrameData,
        frame_b64: Optional[str],
        diff_b64: Optional[str],
        frame_changed: bool,
        objects_text: str,
        changes_text: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        # Image (only when sending)
        if frame_b64 is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": frame_b64},
            })

        if diff_b64 is not None:
            content.append({"type": "text", "text": "Diff (red = changed):"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": diff_b64},
            })

        # Build compact text context
        parts = []
        parts.append(f"Level {frame.levels_completed}/{frame.win_levels} | Action #{self.action_counter}/{self.MAX_ACTIONS}")

        if self._last_tool_name:
            status = "CHANGED" if frame_changed else "NO CHANGE"
            parts.append(f"Last: {self._last_tool_name} → {status}")

        if changes_text and changes_text != "No pixels changed.":
            parts.append(f"Changes: {changes_text}")

        if self._game_type in ("movement", "unknown"):
            # Position + progress (critical for navigation)
            progress = self.reasoner.get_progress_text()
            if progress:
                parts.append(f"\n{progress}")
            parts.append(f"\nObjects on grid (sorted by size):\n{objects_text}")
            parts.append(f"\nRecent actions (Y=moved, N=wall):\n{self.reasoner.get_recent_actions_text()}")
            if self.action_counter <= 2:
                parts.append("\nSTEP 1: Navigate to the SWITCH at (x=19, y=30). Use navigate_to(19, 30).")
                parts.append("This will rotate Box 2's pattern to match the target.")
            elif self.action_counter <= 8:
                parts.append("\nSTEP 2: Now navigate to BOX 1 (TARGET) at (x=34, y=15). Use navigate_to(34, 15).")
                parts.append("Entering Box 1 with matching pattern completes the level.")
            else:
                parts.append("\nIf stuck, try individual moves to find wall openings. Goal: reach Box 1 at upper area.")
        elif self._game_type == "arc_puzzle":
            parts.append(f"\nObjects on grid:\n{objects_text}")
            if self.reasoner.stuck_counter >= 3:
                parts.append(f"\n** STUCK ({self.reasoner.stuck_counter} actions with no grid change). "
                             f"Your clicks are not landing on editable cells or confirm is failing. "
                             f"Try clicking DIFFERENT cell centers or RESET to start fresh. **")
            if self.action_counter <= 2:
                parts.append("\nAnalyze the example input→output pair. Identify the transformation rule.")
                parts.append("Then click on cells in the test output (bottom-right bordered area) to set the correct colors.")
                parts.append("Use confirm when done.")
            else:
                parts.append(f"\nRecent actions:\n{self.reasoner.get_recent_actions_text()}")
                parts.append("\nContinue modifying the test output or confirm if complete.")
        else:  # click_only
            parts.append(f"\nObjects on grid:\n{objects_text}")
            if self.reasoner.stuck_counter >= 3:
                parts.append(f"\n** STUCK ({self.reasoner.stuck_counter} actions with no grid change). "
                             f"Your clicks are NOT hitting interactive targets! "
                             f"Try clicking the CENTER of SMALL objects from the objects list. "
                             f"Avoid large background regions and the timer bar. **")
            parts.append(f"\nRecent actions:\n{self.reasoner.get_recent_actions_text()}")
            if self.action_counter <= 2:
                parts.append("\nStudy the image and objects list. Click the center of the SMALLEST distinct colored objects first.")
            else:
                parts.append("\nClick a DIFFERENT target. Use object center coordinates from the list above.")

        content.append({"type": "text", "text": "\n".join(parts)})

        return content

    def _extract_tool_call(self, response: Any) -> tuple[str, dict[str, Any]]:
        for block in response.content:
            if block.type == "tool_use":
                return block.name, block.input
        logger.warning("No tool call in response, defaulting to first available action")
        if self._available_actions:
            action = GameAction.from_id(self._available_actions[0])
            name = next(k for k, v in TOOL_TO_ACTION.items() if v == action)
            return name, {"reasoning": "fallback"}
        return "reset", {"reasoning": "fallback"}

    def _evict_history(self) -> None:
        """Sliding window: keep first turn + last N turns."""
        msgs_per_turn = 3
        max_msgs = self.MAX_HISTORY_TURNS * msgs_per_turn

        if len(self.message_history) <= max_msgs:
            return

        first_turn = self.message_history[:msgs_per_turn]
        remaining = self.message_history[msgs_per_turn:]
        keep_msgs = (self.MAX_HISTORY_TURNS - 1) * msgs_per_turn
        self.message_history = first_turn + remaining[-keep_msgs:]
