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
    def render_frame(frame_3d: list[list[list[int]]]) -> str:
        """Render a 3D frame to a base64 PNG (512x512 with grid lines)."""
        grid = frame_3d[-1] if frame_3d else [[0] * 64 for _ in range(64)]

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

        return ClaudeVision._img_to_base64(img)

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
    """Tracks hypotheses, confirmed rules, and action outcomes."""

    def __init__(self) -> None:
        self.hypotheses: list[dict[str, Any]] = []
        self.confirmed_rules: list[str] = []
        self.action_log: list[dict[str, Any]] = []
        self.stuck_counter: int = 0
        self.no_progress_counter: int = 0  # tracks no level progress across resets
        self._last_hash: str = ""

    def on_level_complete(self) -> None:
        """Clear level-specific state, preserve general rules."""
        self.hypotheses.clear()
        self.action_log.clear()
        self.stuck_counter = 0
        self.no_progress_counter = 0
        self._last_hash = ""

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
    ) -> None:
        entry = {
            "action": action_name,
            "reasoning": tool_input.get("reasoning", ""),
            "frame_changed": frame_changed,
        }
        self.action_log.append(entry)

        if frame_changed:
            for h in self.hypotheses:
                if h["confidence"] >= 0.7:
                    rule = h["text"]
                    if rule not in self.confirmed_rules:
                        self.confirmed_rules.append(rule)

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
            lines.append(f"  {a['action']:12s} [{changed}] {a['reasoning'][:60]}")
        return "\n".join(lines)

    def get_stuck_guidance(self) -> str:
        if self.stuck_counter >= 8:
            return (
                "CRITICAL: 8+ actions with no frame change. You are blocked. "
                "Try the OPPOSITE direction or a completely different approach."
            )
        if self.stuck_counter >= 4:
            return (
                "WARNING: 4+ actions with no change. Your current approach isn't working. "
                "Try a different direction."
            )
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# ClaudeAgent — Main agent class
# ═══════════════════════════════════════════════════════════════════════════════


class ClaudeAgent(Agent):
    """ARC-AGI-3 agent powered by Claude Sonnet 4.5 with vision + tool use."""

    MODEL = "claude-sonnet-4-5-20250929"
    MAX_ACTIONS = 200
    MAX_HISTORY_TURNS = 4  # keep context tight for efficiency
    MAX_TOKENS = 512  # tool calls are short

    # Send image every N actions (text-only in between saves ~7k tokens/turn)
    IMAGE_EVERY_N = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        self.vision = ClaudeVision()
        self.reasoner = ClaudeReasoner()
        self.message_history: list[dict[str, Any]] = []
        self._prev_frame_hash: str = ""
        self._prev_levels_completed: int = 0
        self._prev_frame: Optional[list[list[list[int]]]] = None
        self._last_tool_name: str = ""
        self._last_tool_input: dict[str, Any] = {}
        self._available_actions: list[int] = []
        self._turns_since_image: int = 0
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
            self.message_history.clear()
            self._prev_frame = None
            self._prev_frame_hash = ""
            self._turns_since_image = 0
            return GameAction.RESET

        # ── 2. Detect level transition ───────────────────────────────────
        if latest_frame.levels_completed > self._prev_levels_completed:
            logger.info(
                f"Level complete! {self._prev_levels_completed} → {latest_frame.levels_completed}"
            )
            self.reasoner.on_level_complete()
            self.message_history.clear()
            self._turns_since_image = 0
        self._prev_levels_completed = latest_frame.levels_completed

        # ── 3. Track available actions ───────────────────────────────────
        self._available_actions = latest_frame.available_actions

        # ── 4. Update stuck counter ──────────────────────────────────────
        current_hash = self.vision.frame_hash(latest_frame.frame)
        frame_changed = current_hash != self._prev_frame_hash

        if self._last_tool_name and self._prev_frame_hash:
            self.reasoner.record_outcome(
                self._last_tool_name, self._last_tool_input, frame_changed
            )

        self.reasoner.update_stuck(current_hash)

        # Force reset if stuck too long
        if self.reasoner.stuck_counter >= 10:
            logger.info("Stuck for 10+ actions, forcing RESET")
            self.reasoner.stuck_counter = 0
            self.reasoner.no_progress_counter += 1
            self.message_history.clear()
            self._turns_since_image = 0
            return GameAction.RESET

        # ── 5. Build tools (filtered to available_actions) ───────────────
        tools = self._get_available_tools()

        # ── 6. Decide whether to include image this turn ─────────────────
        send_image = (
            self._turns_since_image >= self.IMAGE_EVERY_N
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
        action = TOOL_TO_ACTION.get(tool_name, GameAction.RESET)

        # Validate action is actually available
        if action.value not in self._available_actions and action != GameAction.RESET:
            logger.warning(f"Claude chose unavailable action {tool_name}, falling back to first available")
            if self._available_actions:
                action = GameAction.from_id(self._available_actions[0])
            else:
                action = GameAction.RESET

        if action == GameAction.ACTION6:  # click
            x = max(0, min(63, tool_input.get("x", 0)))
            y = max(0, min(63, tool_input.get("y", 0)))
            action.set_data({"x": x, "y": y})

        action.reasoning = {
            "tool": tool_name,
            "reasoning": tool_input.get("reasoning", ""),
            "stuck": self.reasoner.stuck_counter,
            "tokens": [self.input_tokens, self.output_tokens],
        }

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

        return tools if tools else list(ALL_TOOL_SCHEMAS.values())

    def _build_system_prompt(self) -> str:
        stuck = self.reasoner.get_stuck_guidance()
        stuck_section = f"\n\n**{stuck}**" if stuck else ""

        return f"""You are an expert ARC-AGI-3 puzzle agent navigating a 64x64 grid environment.

GOAL: Solve each level as efficiently as possible. Minimize wasted moves.

STRATEGY:
1. On first seeing a level, identify: your avatar/player object, the goal/target, obstacles, and any UI elements (timers, score bars).
2. Plan a path from your current position toward the goal. Move deliberately — each move should bring you closer to the objective.
3. After each move, check: did the frame change? Did you move in the expected direction? Are there walls blocking you?
4. If blocked (wall/obstacle), immediately try a different direction to route around it.
5. Look for patterns: colored objects often have meaning (goals, keys, doors, switches).
6. A shrinking bar usually means limited moves/time — be efficient!

RULES LEARNED:
{self.reasoner.get_rules_text()}

HYPOTHESES:
{self.reasoner.get_hypotheses_text()}

SPATIAL REASONING:
- Grid is 64x64. x=0 is left, x=63 is right. y=0 is top, y=63 is bottom.
- Objects are described by bounding box and center coordinates.
- "Move up" decreases y. "Move down" increases y. "Move left" decreases x. "Move right" increases x.
- To reach a target at lower-right from upper-left, you need move_down and move_right.

BE CONCISE: Just pick the best action. Don't overthink.{stuck_section}"""

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
            status = "CHANGED" if frame_changed else "NO EFFECT"
            parts.append(f"Last: {self._last_tool_name} → {status}")

        if changes_text and changes_text != "No pixels changed.":
            parts.append(f"Changes: {changes_text}")

        parts.append(f"\nObjects:\n{objects_text}")
        parts.append(f"\nRecent actions (Y=changed, N=no effect):\n{self.reasoner.get_recent_actions_text()}")
        parts.append("\nChoose your next move.")

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
