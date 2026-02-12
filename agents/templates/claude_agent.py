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

# ── Tool schemas (Anthropic tool_use format) ─────────────────────────────────
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "reset",
        "description": "Reset the current level. Use when stuck or game over.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why you are resetting.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "move_up",
        "description": "Move up one cell on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why this move.",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "What you expect to happen.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "move_down",
        "description": "Move down one cell on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why this move.",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "What you expect to happen.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "move_left",
        "description": "Move left one cell on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why this move.",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "What you expect to happen.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "move_right",
        "description": "Move right one cell on the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why this move.",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "What you expect to happen.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "confirm",
        "description": "Confirm / perform action (submit answer, activate, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why you are confirming.",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "What you expect to happen.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "click",
        "description": "Click a specific cell on the 64x64 grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "X coordinate (0-63, left to right).",
                    "minimum": 0,
                    "maximum": 63,
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate (0-63, top to bottom).",
                    "minimum": 0,
                    "maximum": 63,
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why you are clicking here.",
                },
                "target_object": {
                    "type": "string",
                    "description": "What object you are targeting.",
                },
            },
            "required": ["x", "y", "reasoning"],
        },
    },
    {
        "name": "undo",
        "description": "Undo the last action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why you are undoing.",
                },
            },
            "required": ["reasoning"],
        },
    },
]

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
# ClaudeVision — Grid rendering & diffing
# ═══════════════════════════════════════════════════════════════════════════════


class ClaudeVision:
    """Renders 64x64 ARC grids as annotated PNGs and computes diffs."""

    SCALE = 8  # 64 * 8 = 512
    IMG_SIZE = 64 * SCALE  # 512

    @staticmethod
    def render_frame(frame_3d: list[list[list[int]]]) -> str:
        """Render a 3D frame (list of 64x64 grids) to a base64 PNG.

        Takes the last layer as the visible frame, scales to 512x512,
        and adds axis labels every 8 cells.
        """
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

        # Add light grid lines every 8 cells and axis labels
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except (OSError, IOError):
            font = ImageFont.load_default()

        scale = ClaudeVision.SCALE
        for i in range(0, 65, 8):
            pos = i * scale
            if pos < ClaudeVision.IMG_SIZE:
                # vertical grid line
                draw.line(
                    [(pos, 0), (pos, ClaudeVision.IMG_SIZE - 1)],
                    fill=(255, 255, 255, 60),
                    width=1,
                )
                # horizontal grid line
                draw.line(
                    [(0, pos), (ClaudeVision.IMG_SIZE - 1, pos)],
                    fill=(255, 255, 255, 60),
                    width=1,
                )
                # axis labels
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

        # Build diff image: red where changed, black elsewhere
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
    def compress_grid_text(grid: list[list[int]]) -> str:
        """Run-length encode a 2D grid for supplementary text context."""
        lines = []
        for y, row in enumerate(grid):
            if y % 8 != 0:
                continue
            runs: list[str] = []
            prev, count = row[0], 1
            for cell in row[1:]:
                if cell == prev:
                    count += 1
                else:
                    name = COLOR_NAMES[prev] if 0 <= prev < 16 else str(prev)
                    runs.append(f"{name}x{count}" if count > 1 else name)
                    prev, count = cell, 1
            name = COLOR_NAMES[prev] if 0 <= prev < 16 else str(prev)
            runs.append(f"{name}x{count}" if count > 1 else name)
            lines.append(f"y{y}: {' '.join(runs)}")
        return "\n".join(lines)

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
        self._last_hash: str = ""

    def on_level_complete(self) -> None:
        """Clear level-specific state, preserve general rules."""
        self.hypotheses.clear()
        self.action_log.clear()
        self.stuck_counter = 0
        self._last_hash = ""

    def update_stuck(self, current_hash: str) -> None:
        """Update stuck counter based on frame hash."""
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
        """Record action and whether it had an effect."""
        entry = {
            "action": action_name,
            "reasoning": tool_input.get("reasoning", ""),
            "expected": tool_input.get("expected_outcome", ""),
            "frame_changed": frame_changed,
        }
        self.action_log.append(entry)

        # Auto-promote: if we predicted something and it happened
        if frame_changed and tool_input.get("expected_outcome"):
            for h in self.hypotheses:
                if h["confidence"] >= 0.7:
                    rule = h["text"]
                    if rule not in self.confirmed_rules:
                        self.confirmed_rules.append(rule)

    def add_hypothesis(self, text: str, confidence: float, evidence: str) -> None:
        # Avoid exact duplicates
        for h in self.hypotheses:
            if h["text"] == text:
                h["confidence"] = max(h["confidence"], confidence)
                return
        self.hypotheses.append(
            {"text": text, "confidence": confidence, "evidence": evidence}
        )

    def get_rules_text(self) -> str:
        if not self.confirmed_rules:
            return "No confirmed rules yet."
        return "\n".join(f"- {r}" for r in self.confirmed_rules)

    def get_hypotheses_text(self) -> str:
        if not self.hypotheses:
            return "No active hypotheses."
        lines = []
        for h in sorted(self.hypotheses, key=lambda x: -x["confidence"]):
            lines.append(
                f"- [{h['confidence']:.0%}] {h['text']} (evidence: {h['evidence']})"
            )
        return "\n".join(lines)

    def get_recent_actions_text(self, n: int = 8) -> str:
        if not self.action_log:
            return "No actions taken yet."
        recent = self.action_log[-n:]
        lines = []
        for a in recent:
            changed = "changed" if a["frame_changed"] else "NO CHANGE"
            lines.append(f"- {a['action']}: {a['reasoning'][:80]} [{changed}]")
        return "\n".join(lines)

    def get_stuck_guidance(self) -> str:
        if self.stuck_counter >= 12:
            return (
                "CRITICAL: You have been stuck for 12+ actions with no frame change. "
                "You MUST call reset to restart this level with your accumulated knowledge."
            )
        if self.stuck_counter >= 5:
            return (
                "WARNING: 5+ actions with no change. Do a full re-analysis of the grid. "
                "Try completely different actions — click on objects, try confirm, "
                "or move in directions you haven't tried."
            )
        if self.stuck_counter >= 3:
            return (
                "HINT: 3 actions with no frame change. Try an action you haven't "
                "attempted yet. Consider clicking on colored objects or using confirm."
            )
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# ClaudeAgent — Main agent class
# ═══════════════════════════════════════════════════════════════════════════════


class ClaudeAgent(Agent):
    """ARC-AGI-3 agent powered by Claude Sonnet 4.5 with vision + tool use."""

    MODEL = "claude-sonnet-4-5-20250929"
    MAX_ACTIONS = 200
    MAX_HISTORY_TURNS = 6  # 12 messages in sliding window
    MAX_TOKENS = 1024

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
            return GameAction.RESET

        # ── 2. Detect level transition ───────────────────────────────────
        if latest_frame.levels_completed > self._prev_levels_completed:
            logger.info(
                f"Level complete! {self._prev_levels_completed} → {latest_frame.levels_completed}"
            )
            self.reasoner.on_level_complete()
            self.message_history.clear()
        self._prev_levels_completed = latest_frame.levels_completed

        # ── 3. Update stuck counter ──────────────────────────────────────
        current_hash = self.vision.frame_hash(latest_frame.frame)
        frame_changed = current_hash != self._prev_frame_hash

        # Record outcome of previous action
        if self._last_tool_name and self._prev_frame_hash:
            self.reasoner.record_outcome(
                self._last_tool_name, self._last_tool_input, frame_changed
            )

        self.reasoner.update_stuck(current_hash)

        # Force reset if stuck too long
        if self.reasoner.stuck_counter >= 12:
            logger.info("Stuck for 12+ actions, forcing RESET")
            self.reasoner.stuck_counter = 0
            return GameAction.RESET

        # ── 4. Render current frame ──────────────────────────────────────
        frame_b64 = self.vision.render_frame(latest_frame.frame)

        # ── 5. Render diff if previous frame exists ──────────────────────
        diff_b64: Optional[str] = None
        if self._prev_frame is not None:
            diff_b64 = self.vision.render_diff(self._prev_frame, latest_frame.frame)

        # ── 6. Build system prompt ───────────────────────────────────────
        system_prompt = self._build_system_prompt()

        # ── 7. Build user message ────────────────────────────────────────
        user_content = self._build_user_message(
            latest_frame, frame_b64, diff_b64, frame_changed
        )

        # ── 8. Call Claude ───────────────────────────────────────────────
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
                tools=TOOL_SCHEMAS,
                tool_choice={"type": "any"},
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            # Fallback: try with fewer messages
            messages = [{"role": "user", "content": user_content}]
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=[{"type": "text", "text": system_prompt}],
                tools=TOOL_SCHEMAS,
                tool_choice={"type": "any"},
                messages=messages,
            )

        # Track tokens
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

        # ── 9. Extract tool call ─────────────────────────────────────────
        tool_name, tool_input = self._extract_tool_call(response)

        # ── 10. Map to GameAction ────────────────────────────────────────
        action = TOOL_TO_ACTION.get(tool_name, GameAction.RESET)

        if action == GameAction.ACTION6:  # click
            x = max(0, min(63, tool_input.get("x", 0)))
            y = max(0, min(63, tool_input.get("y", 0)))
            action.set_data({"x": x, "y": y})

        # Attach reasoning metadata
        action.reasoning = {
            "tool": tool_name,
            "reasoning": tool_input.get("reasoning", ""),
            "expected_outcome": tool_input.get("expected_outcome", ""),
            "stuck_counter": self.reasoner.stuck_counter,
            "tokens": [self.input_tokens, self.output_tokens],
        }

        # ── 11. Update message history ───────────────────────────────────
        # Add user message
        self.message_history.append({"role": "user", "content": user_content})

        # Add assistant response (reconstruct from tool use)
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        self.message_history.append({"role": "assistant", "content": assistant_content})

        # Add tool result stub so conversation stays valid
        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use_block:
            self.message_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": "Action submitted. Observe the next frame to see the result.",
                        }
                    ],
                }
            )

        # Sliding window eviction (keep first observation + last N turns)
        self._evict_history()

        # ── 12. Save state for next iteration ────────────────────────────
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

    def _build_system_prompt(self) -> str:
        stuck_guidance = self.reasoner.get_stuck_guidance()
        stuck_section = f"\n\n## STUCK ALERT\n{stuck_guidance}" if stuck_guidance else ""

        return f"""You are an expert ARC-AGI-3 puzzle agent. You interact with a 64x64 grid-based \
environment by choosing actions (move, click, confirm, undo, reset) to solve multi-level puzzles.

## Methodology
1. OBSERVE: Carefully examine the current frame image. Note object positions, colors, patterns.
2. COMPARE: If a diff image is shown, identify exactly what changed from your last action.
3. HYPOTHESIZE: Form theories about game rules based on observations and action outcomes.
4. PLAN: Decide what to test or accomplish next. Think 2-3 steps ahead.
5. ACT: Choose ONE action using the available tools. Explain your reasoning.

## Confirmed Rules
{self.reasoner.get_rules_text()}

## Active Hypotheses
{self.reasoner.get_hypotheses_text()}

## Key Guidelines
- The grid is 64x64. Coordinates: x=0-63 (left to right), y=0-63 (top to bottom).
- Images have grid lines every 8 cells with axis labels in yellow.
- Colors: {', '.join(f'{i}={COLOR_NAMES[i]}' for i in range(16))}
- Prefer movement and confirm before clicking. Click is for targeting specific objects.
- If an action has no effect, try something different — don't repeat failed actions.
- Each level may have different rules. Learn by experimenting systematically.{stuck_section}"""

    def _build_user_message(
        self,
        frame: FrameData,
        frame_b64: str,
        diff_b64: Optional[str],
        frame_changed: bool,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        # Current frame image
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": frame_b64,
                },
            }
        )

        # Diff image if available
        if diff_b64 is not None:
            content.append({"type": "text", "text": "Diff (red = changed pixels):"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": diff_b64,
                    },
                }
            )

        # Text context
        state_text = f"""## Current State
- Game state: {frame.state.value}
- Levels completed: {frame.levels_completed} / {frame.win_levels}
- Action #{self.action_counter} of {self.MAX_ACTIONS}
- Available actions: {[GameAction.from_id(a).name for a in frame.available_actions]}"""

        if self._last_tool_name:
            change_text = "Frame CHANGED" if frame_changed else "Frame UNCHANGED (no effect)"
            state_text += f"""

## Last Action Result
- Action: {self._last_tool_name}
- Result: {change_text}"""

        recent = self.reasoner.get_recent_actions_text(6)
        state_text += f"""

## Recent Action History
{recent}"""

        state_text += "\n\nAnalyze the frame and choose your next action."

        content.append({"type": "text", "text": state_text})

        return content

    def _extract_tool_call(
        self, response: Any
    ) -> tuple[str, dict[str, Any]]:
        """Extract tool name and input from Claude's response."""
        for block in response.content:
            if block.type == "tool_use":
                return block.name, block.input
        # Fallback if no tool call found
        logger.warning("No tool call in Claude response, defaulting to move_right")
        return "move_right", {"reasoning": "fallback — no tool call returned"}

    def _evict_history(self) -> None:
        """Keep message history within sliding window bounds.

        Each 'turn' is 3 messages: user, assistant, tool_result.
        Keep the first turn (initial observation) and the last MAX_HISTORY_TURNS turns.
        """
        msgs_per_turn = 3
        max_msgs = self.MAX_HISTORY_TURNS * msgs_per_turn

        if len(self.message_history) <= max_msgs:
            return

        # Keep first turn + last (MAX_HISTORY_TURNS - 1) turns
        first_turn = self.message_history[:msgs_per_turn]
        remaining = self.message_history[msgs_per_turn:]
        keep_msgs = (self.MAX_HISTORY_TURNS - 1) * msgs_per_turn
        self.message_history = first_turn + remaining[-keep_msgs:]
