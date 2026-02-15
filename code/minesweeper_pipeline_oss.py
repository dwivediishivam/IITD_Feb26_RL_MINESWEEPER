# ################################################################
# CELL 0: SETUP - Imports & Model Discovery
# ################################################################
import os
import glob
import shutil

# Point HuggingFace to the pre-downloaded model cache
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"

# Discover available models and their snapshot paths
print("=" * 60)
print("AVAILABLE MODELS IN CACHE:")
print("=" * 60)
cache_dir = "/root/.cache/huggingface"
model_dirs = sorted(glob.glob(os.path.join(cache_dir, "models--*")))
if model_dirs:
    for d in model_dirs:
        name = os.path.basename(d).replace("models--", "").replace("--", "/")
        snapshots = sorted(glob.glob(os.path.join(d, "snapshots", "*")))
        print(f"  {name}")
        for s in snapshots:
            print(f"    snapshot: {s}")
else:
    print("  No models found in cache - will try HF download")

import json
import random
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
from datasets import Dataset

print("\nAll imports ready!")


# ################################################################
# CELL 1: LOAD GPT-OSS-20B MODEL WITH UNSLOTH
# ################################################################
# gpt-oss-20b: MoE model with 20B total params, 3.6B active
# NOT instruction-tuned - needs more SFT training
# Use BF16 (no 4-bit quantization for MoE models)
# ################################################################
from unsloth import FastLanguageModel
import torch

max_seq_length = 4096
lora_rank = 64  # Higher rank for non-instruct model

# Force gpt-oss-20b
model_dir = "/root/.cache/huggingface/models--unsloth--gpt-oss-20b-BF16"
model_name = None
if os.path.exists(model_dir):
    snapshots = sorted(glob.glob(os.path.join(model_dir, "snapshots", "*")))
    if snapshots:
        model_name = snapshots[-1]
        print(f"Found gpt-oss-20b: {model_name}")
if model_name is None:
    model_name = "unsloth/gpt-oss-20b-BF16"
    print(f"Will download: {model_name}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)

# Ensure tokenizer has chat template (gpt-oss may not have one)
if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    print("Added chat template for non-instruct model")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {model_name}")
print(f"Device: {model.device}")


# ################################################################
# CELL 2: ADD LoRA ADAPTERS
# ################################################################
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print(f"LoRA adapters added (rank={lora_rank}, alpha={lora_rank * 2})")


# ################################################################
# CELL 3: MINESWEEPER GAME CLASS (DO NOT MODIFY)
# ################################################################
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import random

@dataclass
class MinesweeperGame:
    rows: int
    cols: int
    num_mines: int
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _revealed: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _flagged: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.num_mines >= self.rows * self.cols:
            raise ValueError("Too many mines for board size")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._place_mines()
        self._calculate_numbers()

    def _place_mines(self):
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_positions = self._rng.sample(positions, self.num_mines)
        for r, c in mine_positions:
            self._board[r][c] = -1

    def _calculate_numbers(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self._board[nr][nc] == -1:
                                count += 1
                self._board[r][c] = count

    def _reveal_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed or (row, col) in self._flagged:
            return False
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in self._revealed:
                continue
            self._revealed.add((r, c))
            if self._board[r][c] == -1:
                self._state = "failed"
                return True
            if self._board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols
                                and (nr, nc) not in self._revealed
                                and (nr, nc) not in self._flagged):
                            stack.append((nr, nc))
        return True

    def _flag_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed:
            return False
        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action: dict) -> str:
        if self._state != "ongoing":
            return "game_over"
        if not isinstance(action, dict):
            self._state = "failed"
            return "invalid_format"
        action_type = action.get("type")
        row = action.get("row")
        col = action.get("col")
        if action_type not in ["reveal", "flag"] or row is None or col is None:
            self._state = "failed"
            return "invalid_format"
        try:
            row, col = int(row), int(col)
        except (ValueError, TypeError):
            self._state = "failed"
            return "invalid_format"
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self._state = "failed"
            return "out_of_bounds"
        if action_type == "reveal":
            if (row, col) in self._revealed:
                self._state = "failed"
                return "already_revealed"
            if (row, col) in self._flagged:
                self._state = "failed"
                return "flagged_cell"
            valid = self._reveal_cell(row, col)
        else:
            if (row, col) in self._revealed:
                self._state = "failed"
                return "invalid_flag"
            valid = self._flag_cell(row, col)
        if not valid:
            self._state = "failed"
            return "invalid_format"
        self._check_win()
        if self._state == "failed":
            return "mine"
        if self._state == "success":
            return "win"
        return "ok"

    def _check_win(self):
        total_cells = self.rows * self.cols
        safe_cells = total_cells - self.num_mines
        if len(self._revealed) == safe_cells:
            self._state = "success"

    def get_visible_board(self) -> List[List[str]]:
        visible = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self._flagged:
                    row.append('F')
                elif (r, c) in self._revealed:
                    val = self._board[r][c]
                    row.append('*' if val == -1 else str(val))
                else:
                    row.append('.')
            visible.append(row)
        return visible

    def state(self) -> str:
        return self._state

    def pretty_print(self) -> str:
        visible = self.get_visible_board()
        lines = []
        header = "   " + " ".join(f"{i:2d}" for i in range(self.cols))
        lines.append(header)
        lines.append("  " + "\u2500" * (self.cols * 3 + 1))
        for r, row in enumerate(visible):
            line = f"{r:2d}\u2502 " + "  ".join(row)
            lines.append(line)
        return "\n".join(lines)

# Quick test
game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=42)
game.do_action({"type": "reveal", "row": 3, "col": 3})
print(game.pretty_print())
print(f"State: {game.state()}")


# ################################################################
# CELL 4: EXPERT SOLVER + COMPACT PROMPT + PARSER + DEDUCTION
# ################################################################
# Solver: Constraint propagation + coupled subset analysis
# Tested: 56-80% win rates across board sizes
# Compact prompt: 90% token savings vs JSON (essential for 50x50)
# Logical deduction: Detects +15 vs +10 eval bonus
# ################################################################

SYSTEM_PROMPT = 'Analyze the Minesweeper board. CRITICAL: You can ONLY target cells marked "." (unknown). NEVER pick a numbered cell (0-8) or flagged cell (F) - those are already revealed/flagged. For each numbered cell, count adjacent flags(F) and unknowns(.). If number equals flag count, unknowns are safe to reveal. If number minus flags equals unknown count, unknowns are mines to flag. Only act on certain deductions. Output JSON: {"think":"<brief constraint reasoning>","type":"reveal"or"flag","row":N,"col":N}'

def build_compact_prompt(game_or_state):
    """Build compact prompt. Accepts MinesweeperGame or dict game state."""
    if isinstance(game_or_state, dict):
        board = game_or_state["board"]
        rows = game_or_state["rows"]
        cols = game_or_state["cols"]
        mines = game_or_state["mines"]
        flagged = game_or_state.get("flags_placed", 0)
        revealed = game_or_state.get("cells_revealed", 0)
    else:
        board = game_or_state.get_visible_board()
        rows = game_or_state.rows
        cols = game_or_state.cols
        mines = game_or_state.num_mines
        flagged = len(game_or_state._flagged)
        revealed = len(game_or_state._revealed)

    board_lines = []
    for r in range(rows):
        board_lines.append(f"{r:>2}|{''.join(board[r])}")
    board_str = "\n".join(board_lines)

    prompt = f"""Minesweeper {rows}x{cols}, {mines} mines, {flagged} flagged, {revealed} revealed.
.=unknown F=flag 0-8=adjacent mines

{board_str}

JSON action:"""
    return prompt


def parse_llm_action(response):
    """Extract JSON action from LLM response. Takes LAST valid match."""
    best = None
    for match in re.finditer(r'\{[^{}]*\}', response):
        try:
            action = json.loads(match.group())
            if ("type" in action and "row" in action and "col" in action
                    and action["type"] in ["reveal", "flag"]):
                best = action
        except json.JSONDecodeError:
            continue
    return best


def get_neighbors(r, c, rows, cols):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    return neighbors


def generate_reasoning(board, rows, cols, action):
    """Generate brief chain-of-thought reasoning for a solver action.
    Finds the numbered cell that directly constrains the target."""
    row, col = action["row"], action["col"]
    atype = action["type"]
    # Find a numbered neighbor that directly constrains this cell
    for r in range(rows):
        for c in range(cols):
            if board[r][c] not in '12345678':
                continue
            num = int(board[r][c])
            nbrs = get_neighbors(r, c, rows, cols)
            if (row, col) not in nbrs:
                continue
            flags = sum(1 for nr, nc in nbrs if board[nr][nc] == 'F')
            unknowns = [(nr, nc) for nr, nc in nbrs if board[nr][nc] == '.']
            rem = num - flags
            if atype == "reveal" and (row, col) in unknowns and rem == 0:
                return f"({r},{c})={num}, {flags} flags, 0 mines left -> ({row},{col}) safe"
            if atype == "flag" and (row, col) in unknowns and rem == len(unknowns):
                return f"({r},{c})={num}, {flags}F, {len(unknowns)}U={rem} mines -> ({row},{col}) mine"
    # Phase 2 / coupled deduction - generic trace
    if atype == "flag":
        return f"Constraints -> ({row},{col}) must be mine"
    return f"Constraints -> ({row},{col}) is safe"


def is_logically_deducible(board, rows, cols, action_type, tr, tc):
    """Check if a move can be logically deduced from board constraints."""
    cf = set()  # certain flags
    cr = set()  # certain reveals

    # Phase 1: Single-cell constraint propagation
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if board[r][c] not in '12345678':
                    continue
                num = int(board[r][c])
                nbrs = get_neighbors(r, c, rows, cols)
                fn = sum(1 for nr, nc in nbrs if board[nr][nc] == 'F' or (nr, nc) in cf)
                un = [(nr, nc) for nr, nc in nbrs
                      if board[nr][nc] == '.' and (nr, nc) not in cf and (nr, nc) not in cr]
                rem = num - fn
                if rem < 0:
                    continue
                if rem == len(un) and un:
                    for n in un:
                        if n not in cf:
                            cf.add(n)
                            changed = True
                if rem == 0 and un:
                    for n in un:
                        if n not in cr:
                            cr.add(n)
                            changed = True

    # Phase 2: Coupled constraints (pair-wise subset analysis)
    numbered = [(r, c) for r in range(rows) for c in range(cols) if board[r][c] in '12345678']
    changed = True
    iters = 0
    while changed and iters < 30:
        changed = False
        iters += 1
        for i, (r1, c1) in enumerate(numbered):
            n1 = int(board[r1][c1])
            nb1 = get_neighbors(r1, c1, rows, cols)
            f1 = sum(1 for nr, nc in nb1 if board[nr][nc] == 'F' or (nr, nc) in cf)
            u1 = set(n for n in nb1 if board[n[0]][n[1]] == '.' and n not in cf and n not in cr)
            rm1 = n1 - f1
            if not u1:
                continue
            for j in range(i + 1, len(numbered)):
                r2, c2 = numbered[j]
                if abs(r1 - r2) > 2 or abs(c1 - c2) > 2:
                    continue
                n2 = int(board[r2][c2])
                nb2 = get_neighbors(r2, c2, rows, cols)
                f2 = sum(1 for nr, nc in nb2 if board[nr][nc] == 'F' or (nr, nc) in cf)
                u2 = set(n for n in nb2 if board[n[0]][n[1]] == '.' and n not in cf and n not in cr)
                rm2 = n2 - f2
                if not u2:
                    continue
                for sa, sb, ra, rb in [(u1, u2, rm1, rm2), (u2, u1, rm2, rm1)]:
                    if sa.issubset(sb):
                        diff = sb - sa
                        dm = rb - ra
                        if diff and dm == len(diff):
                            for cell in diff:
                                if cell not in cf:
                                    cf.add(cell)
                                    changed = True
                        elif diff and dm == 0:
                            for cell in diff:
                                if cell not in cr:
                                    cr.add(cell)
                                    changed = True

    target = (tr, tc)
    return (action_type == "flag" and target in cf) or (action_type == "reveal" and target in cr)


class MinesweeperSolver:
    """Expert solver using constraint propagation + coupled constraints."""

    def analyze_board(self, board, rows, cols, num_mines, num_flagged):
        cf = set()
        cr = set()
        frontier = [(r, c) for r in range(rows) for c in range(cols)
                     if board[r][c] in '12345678'
                     and any(board[nr][nc] == '.' for nr, nc in get_neighbors(r, c, rows, cols))]

        # Phase 1: Single-cell constraints
        changed = True
        while changed:
            changed = False
            for r, c in frontier:
                num = int(board[r][c])
                nbrs = get_neighbors(r, c, rows, cols)
                fn = [n for n in nbrs if board[n[0]][n[1]] == 'F' or n in cf]
                un = [n for n in nbrs if board[n[0]][n[1]] == '.' and n not in cf and n not in cr]
                rem = num - len(fn)
                if rem < 0:
                    continue
                if rem == len(un) and un:
                    for n in un:
                        if n not in cf:
                            cf.add(n)
                            changed = True
                if rem == 0 and un:
                    for n in un:
                        if n not in cr:
                            cr.add(n)
                            changed = True

        # Phase 2: Coupled constraints with spatial grid indexing (fast on 50x50)
        gi = defaultdict(list)
        for r, c in frontier:
            gi[(r // 3, c // 3)].append((r, c))
        changed = True
        it = 0
        while changed and it < 50:
            changed = False
            it += 1
            for (gr, gc), fc in gi.items():
                nearby = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nearby.extend(gi.get((gr + dr, gc + dc), []))
                for r1, c1 in fc:
                    n1 = int(board[r1][c1])
                    nb1 = get_neighbors(r1, c1, rows, cols)
                    f1 = sum(1 for n in nb1 if board[n[0]][n[1]] == 'F' or n in cf)
                    u1 = set(n for n in nb1 if board[n[0]][n[1]] == '.' and n not in cf and n not in cr)
                    rm1 = n1 - f1
                    if not u1:
                        continue
                    for r2, c2 in nearby:
                        if (r1, c1) >= (r2, c2) or abs(r1 - r2) > 2 or abs(c1 - c2) > 2:
                            continue
                        n2 = int(board[r2][c2])
                        nb2 = get_neighbors(r2, c2, rows, cols)
                        f2 = sum(1 for n in nb2 if board[n[0]][n[1]] == 'F' or n in cf)
                        u2 = set(n for n in nb2 if board[n[0]][n[1]] == '.' and n not in cf and n not in cr)
                        rm2 = n2 - f2
                        if not u2:
                            continue
                        for sa, sb, ra, rb in [(u1, u2, rm1, rm2), (u2, u1, rm2, rm1)]:
                            if sa.issubset(sb):
                                diff = sb - sa
                                dm = rb - ra
                                if diff and dm == len(diff):
                                    for cell in diff:
                                        if cell not in cf:
                                            cf.add(cell)
                                            changed = True
                                elif diff and dm == 0:
                                    for cell in diff:
                                        if cell not in cr:
                                            cr.add(cell)
                                            changed = True
        cf -= cr
        return {"certain_flags": cf, "certain_reveals": cr}

    def estimate_probabilities(self, board, rows, cols, num_mines, cf, cr):
        cur_flags = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == 'F')
        rem_mines = max(0, num_mines - cur_flags - len(cf))
        uncertain = set()
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == '.' and (r, c) not in cf and (r, c) not in cr:
                    uncertain.add((r, c))
        if not uncertain:
            return {}
        gp = rem_mines / len(uncertain) if uncertain else 0
        cp = defaultdict(list)
        for r in range(rows):
            for c in range(cols):
                if board[r][c] not in '12345678':
                    continue
                num = int(board[r][c])
                nbrs = get_neighbors(r, c, rows, cols)
                fn = sum(1 for n in nbrs if board[n[0]][n[1]] == 'F' or n in cf)
                un = [n for n in nbrs if n in uncertain]
                if un:
                    lp = max(0, min(1, (num - fn) / len(un)))
                    for n in un:
                        cp[n].append(lp)
        probs = {}
        for cell in uncertain:
            probs[cell] = sum(cp[cell]) / len(cp[cell]) if cell in cp else gp
        return probs

    def get_best_action(self, game):
        board = game.get_visible_board()
        rows, cols = game.rows, game.cols
        a = self.analyze_board(board, rows, cols, game.num_mines, len(game._flagged))
        cf, cr = a["certain_flags"], a["certain_reveals"]
        if cf:
            r, c = min(cf)
            return {"type": "flag", "row": r, "col": c}, True
        if cr:
            r, c = min(cr)
            return {"type": "reveal", "row": r, "col": c}, True
        probs = self.estimate_probabilities(board, rows, cols, game.num_mines, cf, cr)
        if probs:
            safest = min(probs.keys(), key=lambda k: (probs[k], k))
            return {"type": "reveal", "row": safest[0], "col": safest[1]}, False
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == '.':
                    return {"type": "reveal", "row": r, "col": c}, False
        return None, False

solver = MinesweeperSolver()

# Quick solver test
test_game = MinesweeperGame(rows=8, cols=8, num_mines=10, seed=42)
test_game.do_action({"type": "reveal", "row": 4, "col": 4})
action, is_logical = solver.get_best_action(test_game)
print(f"Solver recommends: {action} (logical: {is_logical})")
print(f"\nCompact prompt ({len(build_compact_prompt(test_game))} chars):")
print(build_compact_prompt(test_game))


# ################################################################
# CELL 5: TEST BASE MODEL BEFORE TRAINING (optional)
# ################################################################
from transformers import TextStreamer

FastLanguageModel.for_inference(model)

game = MinesweeperGame(rows=8, cols=8, num_mines=10, seed=42)
game.do_action({"type": "reveal", "row": 4, "col": 4})
prompt = build_compact_prompt(game)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt},
]
try:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("=== Base Model Response ===")
output = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature=1.0, do_sample=True, max_new_tokens=128,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
FastLanguageModel.for_training(model)


# ################################################################
# CELL 6: GENERATE SFT TRAINING DATASET
# ################################################################
# Expert solver generates optimal (state, action) pairs.
# Variable board sizes 5x5->50x50 for generalization.
# ################################################################
def generate_sft_dataset(num_samples=10000, rng_seed=42):
    """Generate expert-solved training data (logical-only, no guesses).
    KEY CHANGES from previous version:
    1. LOGICAL-ONLY: Skip examples where solver had to guess (~30-50% noise removed)
    2. RANDOMIZED selection: Random choice from cf/cr sets (removes top-left bias)
    3. More attempts to compensate for logical-only filtering
    """
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    dataset_items = []
    slvr = MinesweeperSolver()
    skipped_non_logical = 0

    # Board size distribution: (rows, cols, mine_pct, weight)
    board_configs = [
        # Square boards
        (5, 5, 0.12, 0.03), (6, 6, 0.14, 0.05), (8, 8, 0.15, 0.08),
        (10, 10, 0.15, 0.08), (12, 12, 0.15, 0.05), (16, 16, 0.15, 0.05),
        (20, 20, 0.15, 0.04), (25, 25, 0.15, 0.02), (30, 30, 0.15, 0.02),
        (40, 40, 0.12, 0.02), (50, 50, 0.10, 0.02),
        (50, 50, 0.15, 0.02), (50, 50, 0.20, 0.01),
        # Rectangular boards (wide)
        (5, 8, 0.14, 0.03), (6, 10, 0.14, 0.04), (8, 12, 0.15, 0.04),
        (8, 16, 0.15, 0.03), (10, 16, 0.15, 0.04), (12, 20, 0.15, 0.03),
        (15, 25, 0.15, 0.03), (20, 30, 0.15, 0.03), (25, 40, 0.15, 0.02),
        (30, 50, 0.15, 0.02),
        # Rectangular boards (tall)
        (8, 5, 0.14, 0.02), (10, 6, 0.14, 0.03), (12, 8, 0.15, 0.03),
        (16, 10, 0.15, 0.03), (20, 12, 0.15, 0.02), (25, 15, 0.15, 0.02),
        (30, 20, 0.15, 0.02), (40, 25, 0.12, 0.01), (50, 30, 0.15, 0.01),
    ]

    total_w = sum(w for _, _, _, w in board_configs)
    targets = [(r, c, mp, max(1, int(num_samples * w / total_w)))
               for r, c, mp, w in board_configs]

    for rows, cols, mine_pct, target in targets:
        mines = max(1, int(rows * cols * mine_pct))
        gen = 0
        attempts = 0
        # More attempts since we filter out non-logical examples
        while gen < target and attempts < target * 20:
            attempts += 1
            seed = np.random.randint(1000000)
            try:
                game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
            except ValueError:
                continue

            # Random first move
            fr, fc = np.random.randint(0, rows), np.random.randint(0, cols)
            game.do_action({"type": "reveal", "row": int(fr), "col": int(fc)})
            if game.state() != "ongoing":
                continue

            # Play solver moves to reach a mid/late game state
            move_history = [{"type": "reveal", "row": int(fr), "col": int(fc)}]
            max_depth = max(min(rows * cols // 2, 40), 4)
            # Bias toward mid/late game where logical deductions exist
            r_val = np.random.random()
            if r_val < 0.1:
                num_extra = np.random.randint(0, min(3, max_depth))
            elif r_val < 0.5:
                num_extra = np.random.randint(2, max(max_depth * 2 // 3, 3))
            else:
                num_extra = np.random.randint(max(max_depth // 3, 2), max_depth)
            for _ in range(num_extra):
                if game.state() != "ongoing":
                    break
                act, _ = slvr.get_best_action(game)
                if act is None:
                    break
                game.do_action(act)
                move_history.append(act)

            if game.state() != "ongoing":
                continue

            # Get solver analysis - use analyze_board directly for random selection
            board = game.get_visible_board()
            analysis = slvr.analyze_board(board, rows, cols, mines, len(game._flagged))
            cf, cr = analysis["certain_flags"], analysis["certain_reveals"]

            # LOGICAL-ONLY: Skip if no certain moves exist
            if not cf and not cr:
                skipped_non_logical += 1
                continue

            # RANDOM SELECTION: Pick random from cf/cr (removes top-left bias)
            # When both exist, 50/50 between flag and reveal for balanced training
            if cf and cr:
                if random.random() < 0.5:
                    r_act, c_act = random.choice(list(cf))
                    act = {"type": "flag", "row": r_act, "col": c_act}
                else:
                    r_act, c_act = random.choice(list(cr))
                    act = {"type": "reveal", "row": r_act, "col": c_act}
            elif cf:
                r_act, c_act = random.choice(list(cf))
                act = {"type": "flag", "row": r_act, "col": c_act}
            else:
                r_act, c_act = random.choice(list(cr))
                act = {"type": "reveal", "row": r_act, "col": c_act}

            # ADD CHAIN-OF-THOUGHT: include reasoning in JSON output
            # Research shows 60% accuracy boost from "Think Inside the JSON"
            reasoning = generate_reasoning(board, rows, cols, act)
            act_with_think = {"think": reasoning, **act}
            response_text = json.dumps(act_with_think)

            prompt_text = build_compact_prompt(game)
            dataset_items.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": response_text},
                ],
                "seed": seed,
                "move_history": json.dumps(move_history),
                "game_rows": rows,
                "game_cols": cols,
                "game_mines": mines,
            })
            gen += 1

        print(f"  {rows}x{cols} ({mines} mines): {gen}/{target} examples")

    random.shuffle(dataset_items)
    print(f"\nSkipped {skipped_non_logical} non-logical examples (noise removed)")
    return Dataset.from_list(dataset_items)

print("Generating SFT dataset (logical-only, balanced flag/reveal)...")
sft_dataset = generate_sft_dataset(num_samples=3000)
print(f"\nGenerated {len(sft_dataset)} SFT examples (all logically deducible)")

# Show distribution
sizes = defaultdict(int)
for item in sft_dataset:
    sizes[f"{item['game_rows']}x{item['game_cols']}"] += 1
print("\nBoard size distribution:")
for s, c in sorted(sizes.items(), key=lambda x: int(x[0].split('x')[0])):
    print(f"  {s}: {c}")


# ################################################################
# CELL 7: SFT TRAINING
# ################################################################
# Teaches the model:
# 1. Output format: pure JSON action (no reasoning text)
# 2. Constraint-based minesweeper logic (logical-only examples, no guesses)
# 3. When to flag vs reveal based on neighbor constraints
# ################################################################
from trl import SFTConfig, SFTTrainer

# Pre-format dataset: apply chat template to create a "text" column
# This avoids Unsloth's formatting_func quirks with batched tokenization
def _format_to_text(example):
    try:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False,
            add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False,
            add_generation_prompt=False
        )
    return {"text": text}

sft_dataset = sft_dataset.map(_format_to_text)
print(f"Sample formatted text (first 300 chars):\n{sft_dataset[0]['text'][:300]}")

sft_config = SFTConfig(
    output_dir="minesweeper_sft_output",
    per_device_train_batch_size=8,       # 256GB GPU can handle large batches
    gradient_accumulation_steps=2,
    num_train_epochs=1,                     # 1 epoch to fit in 1-hour time budget (~19 min)
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    max_seq_length=max_seq_length,
    optim="adamw_8bit",
    report_to="none",
    dataset_text_field="text",
)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=sft_dataset,
    args=sft_config,
)

print("Starting SFT training...")
sft_trainer.train()
print("SFT training complete!")


# ################################################################
# CELL 8: SAVE SFT CHECKPOINT + QUICK EVALUATION
# ################################################################
model.save_pretrained("minesweeper_sft_checkpoint_oss")
tokenizer.save_pretrained("minesweeper_sft_checkpoint_oss")
print("SFT checkpoint saved")

def quick_eval(model, tokenizer, num_games=10, label=""):
    """Score-based eval matching competition rules. Game continues after non-fatal errors."""
    FastLanguageModel.for_inference(model)
    wins = 0
    valid_json = 0
    total_moves = 0
    total_score = 0.0
    for seed_i in range(num_games):
        game = MinesweeperGame(rows=8, cols=8, num_mines=10, seed=seed_i + 10000)
        game.do_action({"type": "reveal", "row": 4, "col": 4})
        moves = 0
        game_score = 0.0
        consecutive_bad = 0
        while game.state() == "ongoing" and moves < 100 and consecutive_bad < 5:
            prompt = build_compact_prompt(game)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(msgs, tokenize=False,
                    add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt").to(model.device)
            out = model.generate(**inp, temperature=0.3, max_new_tokens=128, do_sample=True)
            resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            action = parse_llm_action(resp)
            moves += 1
            if action is None:
                game_score -= 10.0
                consecutive_bad += 1
                continue
            valid_json += 1
            row, col = int(action["row"]), int(action["col"])
            atype = action["type"]
            # Pre-check action without breaking game state
            if not (0 <= row < 8 and 0 <= col < 8):
                game_score -= 15.0; consecutive_bad += 1; continue
            if atype == "reveal":
                if (row, col) in game._revealed:
                    game_score -= 12.0; consecutive_bad += 1; continue
                if (row, col) in game._flagged:
                    game_score -= 8.0; consecutive_bad += 1; continue
                if game._board[row][col] == -1:
                    game_score -= 25.0; break  # Mine = game over
                consecutive_bad = 0
                board = game.get_visible_board()
                is_log = is_logically_deducible(board, 8, 8, "reveal", row, col)
                game_score += 15.0 if is_log else 10.0
                game.do_action(action)
                if game.state() == "success":
                    game_score += 100.0
            elif atype == "flag":
                if (row, col) in game._revealed:
                    game_score -= 8.0; consecutive_bad += 1; continue
                if (row, col) in game._flagged:
                    game_score -= 8.0; consecutive_bad += 1; continue
                consecutive_bad = 0
                if len(game._flagged) + 1 > 10:
                    game_score -= 10.0
                if game._board[row][col] == -1:
                    game_score += 15.0
                else:
                    game_score -= 10.0
                game.do_action(action)
        total_moves += moves
        total_score += game_score
        if game.state() == "success":
            wins += 1
    FastLanguageModel.for_training(model)
    avg_score = total_score / num_games
    print(f"[{label}] {wins}/{num_games} wins, {valid_json} valid JSON, "
          f"avg {total_moves/num_games:.1f} moves/game, avg score {avg_score:.1f}")

print("\nEvaluating SFT model...")
quick_eval(model, tokenizer, num_games=3, label="Post-SFT")


# ################################################################
# CELL 9: GRPO REWARD FUNCTIONS (all 12 eval criteria)
# ################################################################
# 1. Format reward: Valid JSON output, bonus for conciseness
# 2. Gameplay reward: All 12 scoring criteria + logical deduction
# 3. Conciseness reward: Penalizes verbose output (128 token limit)
# ################################################################
def valid_json_reward(completions, **kwargs):
    """Strict JSON format reward - harsh invalid penalty."""
    scores = []
    for completion in completions:
        try:
            response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        except Exception:
            scores.append(-25.0)
            continue
        action = parse_llm_action(response)
        if action is None:
            scores.append(-25.0)
        else:
            stripped = response.strip()
            if stripped.startswith('{') and len(stripped) < 150:
                scores.append(8.0)    # JSON with think field (~100-130 chars)
            elif stripped.startswith('{'):
                scores.append(5.0)
            else:
                scores.append(1.0)
    return scores

def gameplay_reward(completions, **kwargs):
    """Improved gameplay reward v3 - lessons from Qwen Phase 2 GRPO.
    Key fix: frontier bonus was CANCELING random penalty (-5+5=0).
    Now: frontier bonus ONLY for logical reveals. Random = always punished.
    Wider gap between logical (+30) and random (-15) reveals."""
    scores = []
    seeds = kwargs.get("seed", [])
    mh_list = kwargs.get("move_history", [])
    gr_list = kwargs.get("game_rows", [])
    gc_list = kwargs.get("game_cols", [])
    gm_list = kwargs.get("game_mines", [])

    # Ensure lists (some trl versions pass scalars for batch=1)
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    if not isinstance(mh_list, (list, tuple)):
        mh_list = [mh_list]
    if not isinstance(gr_list, (list, tuple)):
        gr_list = [gr_list]
    if not isinstance(gc_list, (list, tuple)):
        gc_list = [gc_list]
    if not isinstance(gm_list, (list, tuple)):
        gm_list = [gm_list]

    for idx, completion in enumerate(completions):
        try:
            response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        except Exception:
            scores.append(-10.0)
            continue
        action = parse_llm_action(response)
        if action is None:
            scores.append(-10.0)
            continue
        if not seeds or not mh_list:
            scores.append(0.0)
            continue

        # Handle both repeated and non-repeated kwargs across trl versions
        pi = idx % max(1, len(seeds))
        seed = seeds[pi]
        mh_raw = mh_list[pi % max(1, len(mh_list))]
        rows = gr_list[pi % max(1, len(gr_list))] if gr_list else 6
        cols = gc_list[pi % max(1, len(gc_list))] if gc_list else 6
        mines = gm_list[pi % max(1, len(gm_list))] if gm_list else 5
        mh = json.loads(mh_raw) if isinstance(mh_raw, str) else mh_raw

        game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
        for prev in mh:
            game.do_action(prev)
        if game.state() != "ongoing":
            scores.append(0.0)
            continue

        board = game.get_visible_board()
        try:
            row, col = int(action["row"]), int(action["col"])
        except (ValueError, TypeError):
            scores.append(-10.0)
            continue

        atype = action["type"]

        # Criterion 7: Out of bounds -> -20
        if not (0 <= row < rows and 0 <= col < cols):
            scores.append(-20.0)
            continue

        score = 0.0
        if atype == "reveal":
            # Criterion 6: Already revealed -> -15
            if (row, col) in game._revealed:
                scores.append(-15.0)
                continue
            # Criterion 11: Reveal flagged cell -> -10
            if (row, col) in game._flagged:
                scores.append(-10.0)
                continue
            # Criterion 3: Reveal mine -> -100 (4x actual, MUST avoid mines)
            if game._board[row][col] == -1:
                score = -100.0
            else:
                # Criterion 4: Reveal safe
                is_log = is_logically_deducible(board, rows, cols, "reveal", row, col)
                if is_log:
                    # Logical reveal -> +30 base
                    score = 30.0
                    # Frontier bonus ONLY for logical reveals (+5)
                    nbrs = get_neighbors(row, col, rows, cols)
                    near_revealed = any((nr, nc) in game._revealed for nr, nc in nbrs)
                    if near_revealed:
                        score += 5.0
                else:
                    # Random reveal -> -15 ALWAYS (no frontier bonus!)
                    # This is the KEY fix: Qwen's -5+5=0 gave no signal
                    score = -15.0
                # Criterion 10: Check win bonus +200
                tg = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
                for prev in mh:
                    tg.do_action(prev)
                tg.do_action(action)
                if tg.state() == "success":
                    score += 200.0
        elif atype == "flag":
            # Criterion 12: Flag revealed cell -> -10
            if (row, col) in game._revealed:
                scores.append(-10.0)
                continue
            # Criterion 5: Flag already flagged -> -10
            if (row, col) in game._flagged:
                scores.append(-10.0)
                continue
            # Criterion 8: Too many flags -> -15
            if len(game._flagged) + 1 > mines:
                score -= 15.0
            # Criterion 1: Flag mine -> +30, Criterion 2: Flag non-mine -> -10
            if game._board[row][col] == -1:
                score += 30.0
            else:
                score -= 10.0

        scores.append(score)
    return scores

def conciseness_reward(completions, **kwargs):
    """Reward concise pure JSON output. High rewards prevent format degradation."""
    scores = []
    for completion in completions:
        try:
            response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        except Exception:
            scores.append(-3.0)
            continue
        stripped = response.strip()
        action = parse_llm_action(response)
        if action is None:
            scores.append(-3.0)
            continue
        tok_est = len(stripped) / 4
        if stripped.startswith('{') and tok_est < 40:
            scores.append(5.0)     # Perfect: pure JSON with think field (~30 tokens)
        elif stripped.startswith('{') and tok_est < 60:
            scores.append(2.5)     # Good: starts with JSON
        elif tok_est < 80:
            scores.append(1.0)     # Acceptable
        else:
            scores.append(-3.0)    # Too verbose
    return scores

print("Reward functions defined (3 functions, 12 criteria)")


# ################################################################
# CELL 10: GENERATE GRPO TRAINING DATASET
# ################################################################
# Similar to SFT but prompt-only (no expert answer).
# Model explores, reward functions guide learning.
# ################################################################
def generate_grpo_dataset(num_samples=5000, rng_seed=123):
    """Generate diverse game states for GRPO training.
    Includes rectangular boards and varied game depths."""
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    items = []
    slvr = MinesweeperSolver()

    configs = [
        # Square boards
        (5, 5, 0.12, 0.03), (6, 6, 0.14, 0.05), (8, 8, 0.15, 0.08),
        (10, 10, 0.15, 0.08), (12, 12, 0.15, 0.05), (16, 16, 0.15, 0.05),
        (20, 20, 0.15, 0.04), (25, 25, 0.15, 0.02), (30, 30, 0.15, 0.02),
        (40, 40, 0.12, 0.02), (50, 50, 0.10, 0.02),
        (50, 50, 0.15, 0.02), (50, 50, 0.20, 0.01),
        # Rectangular boards
        (5, 8, 0.14, 0.03), (6, 10, 0.14, 0.04), (8, 12, 0.15, 0.04),
        (8, 16, 0.15, 0.03), (10, 16, 0.15, 0.04), (12, 20, 0.15, 0.03),
        (15, 25, 0.15, 0.03), (20, 30, 0.15, 0.02), (25, 40, 0.15, 0.02),
        (30, 50, 0.15, 0.02),
        (10, 6, 0.14, 0.03), (16, 10, 0.15, 0.03), (20, 12, 0.15, 0.02),
        (50, 30, 0.15, 0.01), (40, 25, 0.12, 0.01),
    ]
    total_w = sum(w for _, _, _, w in configs)

    for rows, cols, mp, weight in configs:
        mines = max(1, int(rows * cols * mp))
        target = max(1, int(num_samples * weight / total_w))
        gen = 0
        attempts = 0
        while gen < target and attempts < target * 10:
            attempts += 1
            seed = np.random.randint(1000000)
            try:
                game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
            except ValueError:
                continue
            fr, fc = np.random.randint(0, rows), np.random.randint(0, cols)
            fa = {"type": "reveal", "row": int(fr), "col": int(fc)}
            game.do_action(fa)
            if game.state() != "ongoing":
                continue
            mh = [fa]
            max_depth = max(min(rows * cols // 2, 40), 4)
            # Bias toward mid/late game where logical deductions exist
            # (matches SFT distribution for better reward variance in GRPO)
            r_val = np.random.random()
            if r_val < 0.1:
                num_extra = np.random.randint(0, min(3, max_depth))
            elif r_val < 0.5:
                num_extra = np.random.randint(2, max(max_depth * 2 // 3, 3))
            else:
                num_extra = np.random.randint(max(max_depth // 3, 2), max_depth)
            for _ in range(num_extra):
                if game.state() != "ongoing":
                    break
                act, _ = slvr.get_best_action(game)
                if act is None:
                    break
                game.do_action(act)
                mh.append(act)
            if game.state() != "ongoing":
                continue

            items.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_compact_prompt(game)},
                ],
                "seed": seed,
                "move_history": json.dumps(mh),
                "game_rows": rows,
                "game_cols": cols,
                "game_mines": mines,
            })
            gen += 1
        print(f"  {rows}x{cols} ({mines} mines): {gen}/{target}")

    random.shuffle(items)
    return Dataset.from_list(items)

print("Generating GRPO dataset...")
grpo_dataset = generate_grpo_dataset(num_samples=2000)
print(f"Generated {len(grpo_dataset)} GRPO examples")


# ################################################################
# CELL 11: GRPO TRAINING
# ################################################################
# FIX for training_loss=0: Explicitly ensure model is in training mode,
# use non-zero beta, and verify trainable parameters before starting.
# ################################################################
from trl import GRPOConfig, GRPOTrainer
import inspect

# CRITICAL: Ensure model is in training mode with gradients
# (Unsloth's for_inference from eval cell may persist)
FastLanguageModel.for_training(model)
model.train()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
assert trainable_params > 0, "ERROR: No trainable parameters! LoRA may not be attached."

# Build GRPO config - max_prompt_length may not exist in all trl versions
_grpo_kwargs = dict(
    # Generation - lower temp since Qwen showed too-random exploration was wasteful
    temperature=0.7,
    num_generations=4,            # Reduced from 8 to save time (halves generation cost)

    # Optimizer
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.15,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    max_grad_norm=0.5,

    # Batching
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,

    # Lengths
    max_completion_length=128,

    # Training duration - 1 hour total budget, ~20 min for GRPO
    max_steps=100,
    save_steps=50,

    # GRPO specific - use DAPO-style (beta=0.0) with loss_type
    # Qwen's beta=0.3 was too conservative, model barely moved
    beta=0.04,

    # Logging
    logging_steps=1,
    report_to="none",
    output_dir="minesweeper_grpo_output",
)
# Only add max_prompt_length if this trl version supports it
if "max_prompt_length" in inspect.signature(GRPOConfig).parameters:
    _grpo_kwargs["max_prompt_length"] = 3500
grpo_config = GRPOConfig(**_grpo_kwargs)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[valid_json_reward, gameplay_reward, conciseness_reward],
    args=grpo_config,
    train_dataset=grpo_dataset,
)

print("Starting GRPO training...")
print("NOTE: Rewards may not improve for first ~100 steps - this is NORMAL!")
grpo_trainer.train()
print("GRPO training complete!")


# ################################################################
# CELL 12: FINAL EVALUATION
# ################################################################
def play_full_game(model, tokenizer, rows=8, cols=8, num_mines=10, seed=None, max_moves=200):
    """Play a full game with competition-style scoring.
    Game continues after non-fatal errors (only mine hit ends the game)."""
    game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
    game.do_action({"type": "reveal", "row": rows // 2, "col": cols // 2})
    moves = 0
    score = 0.0
    consecutive_bad = 0
    while game.state() == "ongoing" and moves < max_moves and consecutive_bad < 5:
        prompt = build_compact_prompt(game)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(**inp, temperature=0.3, max_new_tokens=128, do_sample=True)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        action = parse_llm_action(resp)
        moves += 1
        if action is None:
            score -= 10.0; consecutive_bad += 1; continue
        try:
            row, col = int(action["row"]), int(action["col"])
        except (ValueError, TypeError):
            score -= 10.0; consecutive_bad += 1; continue
        atype = action["type"]
        # Pre-check: don't let bad moves kill game state
        if not (0 <= row < rows and 0 <= col < cols):
            score -= 15.0; consecutive_bad += 1; continue
        if atype == "reveal":
            if (row, col) in game._revealed:
                score -= 12.0; consecutive_bad += 1; continue
            if (row, col) in game._flagged:
                score -= 8.0; consecutive_bad += 1; continue
            if game._board[row][col] == -1:
                score -= 25.0; break  # Mine = game over
            consecutive_bad = 0
            board = game.get_visible_board()
            is_log = is_logically_deducible(board, rows, cols, "reveal", row, col)
            score += 15.0 if is_log else 10.0
            game.do_action(action)
            if game.state() == "success":
                score += 100.0
        elif atype == "flag":
            if (row, col) in game._revealed:
                score -= 8.0; consecutive_bad += 1; continue
            if (row, col) in game._flagged:
                score -= 8.0; consecutive_bad += 1; continue
            consecutive_bad = 0
            if len(game._flagged) + 1 > num_mines:
                score -= 10.0
            if game._board[row][col] == -1:
                score += 15.0
            else:
                score -= 10.0
            game.do_action(action)
    return game, moves, score

FastLanguageModel.for_inference(model)

eval_configs = [
    # Quick eval - 3 games each to save time
    (8, 8, 10, 3, "8x8"),
    (10, 10, 15, 3, "10x10"),
    (6, 10, 8, 3, "6x10"),
]

print("=" * 60)
print("FINAL EVALUATION (competition-style scoring)")
print("=" * 60)
for rows, cols, mines, num_games, label in eval_configs:
    wins = 0
    total_moves = 0
    total_score = 0.0
    for i in range(num_games):
        game, moves, sc = play_full_game(model, tokenizer, rows, cols, mines,
                                         seed=20000 + i, max_moves=2 * rows * cols)
        if game.state() == "success":
            wins += 1
        total_moves += moves
        total_score += sc
    avg_sc = total_score / num_games
    print(f"{label}: {wins}/{num_games} wins ({wins/num_games*100:.0f}%), "
          f"avg {total_moves/num_games:.1f} moves, avg score {avg_sc:.1f}")
print("=" * 60)


# ################################################################
# CELL 13: SAVE FINAL MODEL
# ################################################################
# Save LoRA adapters
model.save_pretrained("my_minesweeper_model_oss")
tokenizer.save_pretrained("my_minesweeper_model_oss")
print("LoRA adapters saved to: my_minesweeper_model_oss/")

# Save merged model (this is what the inference agent loads)
# FIX: Unsloth bug - cache dir exists but lacks permissions, causing
# UnboundLocalError on 'copied_tokenizer_model_from_cache'.
# Workaround: point HF_HOME to a writable directory.
_old_hf_home = os.environ.get("HF_HOME", "")
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)
try:
    model.save_pretrained_merged(
        "your_fine_tuned_model_oss",
        tokenizer,
        save_method="merged_16bit"
    )
    print("Merged model saved to: your_fine_tuned_model_oss/")
except Exception as e:
    print(f"save_pretrained_merged failed: {e}")
    print("Falling back to manual merge + save...")
    model = model.merge_and_unload()
    model.save_pretrained("your_fine_tuned_model_oss")
    tokenizer.save_pretrained("your_fine_tuned_model_oss")
    print("Merged model saved (manual fallback) to: your_fine_tuned_model_oss/")
finally:
    os.environ["HF_HOME"] = _old_hf_home

# Also copy to /workspace path that the agent expects
import shutil
src = os.path.abspath("your_fine_tuned_model_oss")
dst = "/workspace/your_fine_tuned_model_oss"
if src != dst and not os.path.exists(dst):
    try:
        os.symlink(src, dst)
        print(f"Symlinked {src} -> {dst}")
    except Exception as e:
        print(f"Note: Could not symlink to {dst}: {e}")
        print(f"Model is at: {src}")


# ################################################################
# CELL 14: UPDATE INFERENCE AGENT FILES
# ################################################################
# CRITICAL: The inference agent's prompt format MUST match training.
# This cell writes the updated agent files for evaluation.
# ################################################################

# --- Write agents/minesweeper_agent.py (clean - no post-processing) ---
AGENT_CODE = r'''#!/usr/bin/python3
"""Minesweeper Agent - Competition Version (no post-processing)"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from .minesweeper_model import MinesweeperAgent


class MinesweeperPlayer:
    """Agent responsible for playing Minesweeper.
    Only uses: prompt engineering + JSON parsing. No post-processing."""

    def __init__(self, **kwargs):
        self.agent = MinesweeperAgent(**kwargs)

    def build_prompt(self, game_state: Dict[str, Any]) -> tuple:
        board = game_state["board"]
        rows = game_state["rows"]
        cols = game_state["cols"]
        mines = game_state["mines"]
        flagged = game_state.get("flags_placed", 0)
        revealed = game_state.get("cells_revealed", 0)

        board_lines = []
        for r in range(rows):
            board_lines.append(f"{r:>2}|{''.join(board[r])}")
        board_str = "\n".join(board_lines)

        prompt = f"""Minesweeper {rows}x{cols}, {mines} mines, {flagged} flagged, {revealed} revealed.
.=unknown F=flag 0-8=adjacent mines

{board_str}

JSON action:"""

        sys_prompt = 'Analyze the Minesweeper board. CRITICAL: You can ONLY target cells marked "." (unknown). NEVER pick a numbered cell (0-8) or flagged cell (F) - those are already revealed/flagged. For each numbered cell, count adjacent flags(F) and unknowns(.). If number equals flag count, unknowns are safe to reveal. If number minus flags equals unknown count, unknowns are mines to flag. Only act on certain deductions. Output JSON: {"think":"<brief constraint reasoning>","type":"reveal"or"flag","row":N,"col":N}'
        return prompt, sys_prompt

    def play_action(self, game_state, **gen_kwargs):
        prompt, sys_prompt = self.build_prompt(game_state)
        response, tl, gt = self.agent.generate_response(prompt, sys_prompt, **gen_kwargs)
        action = self.parse_action(response)
        return action, tl, gt

    def parse_action(self, response: str) -> Optional[Dict]:
        try:
            potential_jsons = []
            i = 0
            while i < len(response):
                start = response.find("{", i)
                if start == -1:
                    break
                brace_count = 0
                end = start
                while end < len(response):
                    if response[end] == '{':
                        brace_count += 1
                    elif response[end] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response[start:end+1]
                            try:
                                obj = json.loads(json_str)
                                potential_jsons.append(obj)
                            except:
                                pass
                            break
                    end += 1
                i = end + 1 if end < len(response) else len(response)

            for obj in potential_jsons:
                if (isinstance(obj, dict) and
                    "type" in obj and "row" in obj and "col" in obj and
                    obj["type"] in ["reveal", "flag"]):
                    obj["row"] = int(obj["row"])
                    obj["col"] = int(obj["col"])
                    return obj
        except Exception as e:
            print(f"Failed to parse action: {e}")
            return None
        return None

    @staticmethod
    def save_action(action: Dict, file_path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(action, f, indent=2)


if __name__ == "__main__":
    import argparse
    import yaml

    argparser = argparse.ArgumentParser(description="Play Minesweeper using fine-tuned LLM.")
    argparser.add_argument("--game_state_file", type=str, required=True)
    argparser.add_argument("--output_file", type=str, default="outputs/action.json")
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    with open(args.game_state_file, "r") as f:
        game_state = json.load(f)

    player = MinesweeperPlayer()
    gen_kwargs = {"tgps_show": args.verbose}
    config_file = Path("minesweeper_config.yaml")
    if config_file.exists():
        with open(config_file, "r") as f:
            gen_kwargs.update(yaml.safe_load(f))

    action, tl, gt = player.play_action(game_state, **gen_kwargs)
    if args.verbose:
        print(f"Generated Action: {json.dumps(action, indent=2)}")
    if action:
        player.save_action(action, args.output_file)
        print(f"Action saved to {args.output_file}")
    else:
        print("ERROR: Failed to generate valid action!")
        player.save_action({"error": "parse_failed"}, args.output_file)
'''

os.makedirs("agents", exist_ok=True)
with open("agents/minesweeper_agent.py", "w") as f:
    f.write(AGENT_CODE)
print("Updated agents/minesweeper_agent.py")


# --- Write agents/minesweeper_model.py ---
MODEL_CODE = r'''"""Minesweeper Model - Competition Version"""
import time
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class MinesweeperAgent(object):
    def __init__(self, **kwargs):
        model_name = "/workspace/your_fine_tuned_model_oss"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate_response(self, message, system_prompt=None, **kwargs):
        if system_prompt is None:
            system_prompt = 'Analyze the Minesweeper board. For each numbered cell, count adjacent flags(F) and unknowns(.). If number equals flag count, unknowns are safe to reveal. If number minus flags equals unknown count, unknowns are mines to flag. Only act on certain deductions. Output JSON with reasoning: {"think":"<brief constraint reasoning>","type":"reveal"or"flag","row":N,"col":N}'

        if isinstance(message, str):
            message = [message]

        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        texts = []
        for messages in all_messages:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)
        if tgps_show_var:
            start_time = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 128),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=kwargs.get("temperature", 0.1),
            do_sample=kwargs.get("do_sample", True),
        )

        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        batch_outs = [output.strip() for output in batch_outs]
        print(batch_outs)

        if tgps_show_var:
            token_len = sum(len(generated_ids[i]) - model_inputs.input_ids.shape[1]
                          for i in range(len(generated_ids)))
            return (batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time)

        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None
'''

with open("agents/minesweeper_model.py", "w") as f:
    f.write(MODEL_CODE)
print("Updated agents/minesweeper_model.py")


# --- Write minesweeper_config.yaml ---
CONFIG_YAML = """## Minesweeper Agent Configuration ##
max_new_tokens: 128
temperature: 0.1
do_sample: true
"""

with open("minesweeper_config.yaml", "w") as f:
    f.write(CONFIG_YAML)
print("Updated minesweeper_config.yaml")

print("\n" + "=" * 60)
print("ALL DONE! Inference agent files updated.")
print("Model saved to: your_fine_tuned_model_oss/")
print("=" * 60)
print("""
TROUBLESHOOTING:
- OOM: Reduce per_device_train_batch_size or num_generations
- GRPO rewards flat: Normal for first 150 steps. If flat at 300, check reward variance.
- Invalid JSON: Increase SFT epochs or dataset size
- Bad on large boards: Add more large-board examples to training data
""")


# ################################################################
# CELL 15: DETAILED EVALUATION WITH SCORING BREAKDOWN
# ################################################################
# Per-move scoring breakdown showing exactly how the model earns/loses
# points across all 12 competition criteria. Essential for diagnosing
# weaknesses before Phase 2 training.
# ################################################################

def detailed_game_eval(model, tokenizer, rows, cols, num_mines, seed, max_moves=None, verbose=True):
    """Play a full game with detailed per-move scoring breakdown.
    Tracks all 12 competition scoring criteria individually."""
    if max_moves is None:
        max_moves = 2 * rows * cols

    categories = {
        "safe_logical": 0.0,
        "safe_random": 0.0,
        "mine_hit": 0.0,
        "correct_flag": 0.0,
        "wrong_flag": 0.0,
        "invalid_json": 0.0,
        "oob": 0.0,
        "already_revealed": 0.0,
        "already_flagged": 0.0,
        "reveal_flagged": 0.0,
        "flag_revealed": 0.0,
        "excess_flags": 0.0,
        "win": 0.0,
    }

    game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
    game.do_action({"type": "reveal", "row": rows // 2, "col": cols // 2})

    total_score = 0.0
    moves = 0
    consecutive_bad = 0
    result = "ongoing"

    while game.state() == "ongoing" and moves < max_moves and consecutive_bad < 5:
        prompt = build_compact_prompt(game)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(**inp, temperature=0.1, max_new_tokens=128, do_sample=True)
        resp = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        action = parse_llm_action(resp)
        moves += 1
        delta = 0.0
        category = ""

        if action is None:
            delta = -10.0
            category = "invalid_json"
            categories["invalid_json"] += delta
            consecutive_bad += 1
            if verbose:
                print(f"  Move {moves}: INVALID JSON ({resp[:60]}) -> {delta:+.0f}")
            total_score += delta
            continue

        try:
            row, col = int(action["row"]), int(action["col"])
        except (ValueError, TypeError):
            delta = -10.0
            category = "invalid_json"
            categories["invalid_json"] += delta
            consecutive_bad += 1
            if verbose:
                print(f"  Move {moves}: BAD ROW/COL ({action}) -> {delta:+.0f}")
            total_score += delta
            continue

        atype = action["type"]

        # Criterion 7: Out of bounds
        if not (0 <= row < rows and 0 <= col < cols):
            delta = -15.0
            category = "oob"
            categories["oob"] += delta
            consecutive_bad += 1
            if verbose:
                print(f"  Move {moves}: OOB ({atype} {row},{col}) -> {delta:+.0f}")
            total_score += delta
            continue

        if atype == "reveal":
            # Criterion 6: Already revealed
            if (row, col) in game._revealed:
                delta = -12.0
                category = "already_revealed"
                categories["already_revealed"] += delta
                consecutive_bad += 1
                if verbose:
                    print(f"  Move {moves}: ALREADY REVEALED ({row},{col}) -> {delta:+.0f}")
                total_score += delta
                continue
            # Criterion 11: Reveal flagged cell
            if (row, col) in game._flagged:
                delta = -8.0
                category = "reveal_flagged"
                categories["reveal_flagged"] += delta
                consecutive_bad += 1
                if verbose:
                    print(f"  Move {moves}: REVEAL FLAGGED ({row},{col}) -> {delta:+.0f}")
                total_score += delta
                continue
            # Criterion 3: Reveal mine
            if game._board[row][col] == -1:
                delta = -25.0
                category = "mine_hit"
                categories["mine_hit"] += delta
                consecutive_bad = 0
                if verbose:
                    print(f"  Move {moves}: MINE HIT ({row},{col}) -> {delta:+.0f} *** GAME OVER ***")
                total_score += delta
                result = "mine_hit"
                break
            # Criterion 4: Reveal safe
            consecutive_bad = 0
            board = game.get_visible_board()
            is_log = is_logically_deducible(board, rows, cols, "reveal", row, col)
            if is_log:
                delta = 15.0
                category = "safe_logical"
                categories["safe_logical"] += delta
            else:
                delta = 10.0
                category = "safe_random"
                categories["safe_random"] += delta
            game.do_action(action)
            # Criterion 10: Win bonus
            if game.state() == "success":
                win_bonus = 100.0
                categories["win"] += win_bonus
                delta += win_bonus
                result = "success"
                if verbose:
                    print(f"  Move {moves}: {category.upper()} ({row},{col}) -> +{delta:.0f} *** WIN! ***")
                total_score += delta
                break
            if verbose:
                print(f"  Move {moves}: {category.upper()} ({row},{col}) -> {delta:+.0f}")

        elif atype == "flag":
            # Criterion 12: Flag revealed cell
            if (row, col) in game._revealed:
                delta = -8.0
                category = "flag_revealed"
                categories["flag_revealed"] += delta
                consecutive_bad += 1
                if verbose:
                    print(f"  Move {moves}: FLAG REVEALED ({row},{col}) -> {delta:+.0f}")
                total_score += delta
                continue
            # Criterion 5: Flag already flagged
            if (row, col) in game._flagged:
                delta = -8.0
                category = "already_flagged"
                categories["already_flagged"] += delta
                consecutive_bad += 1
                if verbose:
                    print(f"  Move {moves}: ALREADY FLAGGED ({row},{col}) -> {delta:+.0f}")
                total_score += delta
                continue
            consecutive_bad = 0
            # Criterion 8: Excess flags
            if len(game._flagged) + 1 > num_mines:
                excess_pen = -10.0
                categories["excess_flags"] += excess_pen
                delta += excess_pen
                if verbose:
                    print(f"  Move {moves}: EXCESS FLAG penalty -> {excess_pen:+.0f}")
            # Criterion 1/2: Flag mine/non-mine
            if game._board[row][col] == -1:
                flag_delta = 15.0
                category = "correct_flag"
                categories["correct_flag"] += flag_delta
            else:
                flag_delta = -10.0
                category = "wrong_flag"
                categories["wrong_flag"] += flag_delta
            delta += flag_delta
            game.do_action(action)
            if verbose:
                print(f"  Move {moves}: {category.upper()} ({row},{col}) -> {delta:+.0f}")

        total_score += delta

    if result == "ongoing":
        if game.state() == "success":
            result = "success"
        elif consecutive_bad >= 5:
            result = "stopped_bad_moves"
        elif moves >= max_moves:
            result = "max_moves"
        else:
            result = game.state()

    if verbose:
        print(f"\n  --- Summary (seed={seed}) ---")
        print(f"  Result: {result} | Moves: {moves} | Total Score: {total_score:+.1f}")
        print(f"  {'Category':<20} {'Score':>8}")
        print(f"  {'-'*28}")
        for cat, val in categories.items():
            if val != 0.0:
                print(f"  {cat:<20} {val:>+8.1f}")

    return {
        "total_score": total_score,
        "categories": dict(categories),
        "result": result,
        "moves": moves,
        "seed": seed,
    }


FastLanguageModel.for_inference(model)
print("=" * 70)
print("DETAILED EVALUATION - Per-move scoring breakdown")
print("=" * 70)
all_results = []
for bl, r, c, mi, ng in [("8x8", 8, 8, 10, 2), ("10x10", 10, 10, 15, 2)]:
    print(f"\n{'='*50} {bl} {'='*50}")
    for i in range(ng):
        print(f"\nGame {i+1} (seed={42+i}):")
        res = detailed_game_eval(model, tokenizer, r, c, mi, 42 + i)
        all_results.append((bl, res))
print("\n" + "=" * 70)
print("OVERALL SUMMARY")
for bl in ["8x8", "10x10"]:
    rs = [r for b, r in all_results if b == bl]
    print(f"  {bl}: {sum(1 for r in rs if r['result']=='success')}/{len(rs)} wins, "
          f"avg {sum(r['total_score'] for r in rs)/len(rs):+.1f}, "
          f"avg {sum(r['moves'] for r in rs)/len(rs):.1f} moves")

