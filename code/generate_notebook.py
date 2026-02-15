#!/usr/bin/env python3
"""Generate the final competition notebook as .ipynb"""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]})

# ============================================================
# CELL 0: Title
# ============================================================
md("""# Minesweeper RL Competition - SFT + GRPO Pipeline

## Strategy: Expert Solver SFT → GRPO Refinement

**Our Edge:**
- Expert solver generates optimal training data (56-80% win rates)
- Compact board representation: 50x50 = ~800 tokens (vs ~8000 JSON = impossible)
- Complete reward function: all 12 eval criteria + logical deduction (+15 vs +10)
- Variable board sizes (5x5 to 50x50) for generalization
- SFT teaches format + logic first, GRPO refines strategy
- DAPO loss for stable, efficient training""")

# ============================================================
# CELL 1: Setup
# ============================================================
code("""import os
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"

import json
import random
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
from datasets import Dataset
print("Imports ready")""")

# ============================================================
# CELL 2: Load Model
# ============================================================
md("""# Load Model with Unsloth

**Model choice: Qwen2.5-14B-Instruct** (14B dense params > gpt-oss-20b's 3.6B active MoE params for reasoning)

If Qwen2.5-14B fails, uncomment alternatives below.""")

code("""from unsloth import FastLanguageModel
import torch

max_seq_length = 4096  # MUST be 4096+ for 50x50 boards (compact prompt ~1500 tokens)
lora_rank = 32          # Higher rank = better reasoning capacity

# PRIMARY: Qwen2.5-14B-Instruct
model_name = "Qwen/Qwen2.5-14B-Instruct"

# ALTERNATIVES (uncomment if primary fails):
# model_name = "unsloth/gpt-oss-20b-BF16"
# model_name = "Unsloth/Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen3-4B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded: {model_name} on {model.device}")""")

# ============================================================
# CELL 3: LoRA
# ============================================================
md("# Add LoRA Adapters")

code("""model = FastLanguageModel.get_peft_model(
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
print(f"LoRA adapters added (rank={lora_rank}, alpha={lora_rank * 2})")""")

# ============================================================
# CELL 4: MinesweeperGame
# ============================================================
md("""# Minesweeper Game Implementation

**DO NOT MODIFY this class** - it must match the evaluation environment exactly.""")

code("""from dataclasses import dataclass, field
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
        lines.append("  " + chr(9472) * (self.cols * 3 + 1))
        for r, row in enumerate(visible):
            line = f"{r:2d}" + chr(9474) + " " + "  ".join(row)
            lines.append(line)
        return "\\n".join(lines)

# Quick test
game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=42)
game.do_action({"type": "reveal", "row": 3, "col": 3})
print(game.pretty_print())
print(f"State: {game.state()}")""")

# ============================================================
# CELL 5: Expert Solver + Prompt + Parser
# ============================================================
md("""# Expert Solver + Compact Prompt + Parser

**Solver**: Constraint propagation + coupled subset analysis (tested: 56-80% win rates)
**Compact prompt**: 90% token savings vs JSON format (essential for 50x50 boards)
**Logical deduction**: Detects whether moves are logically deducible (+15 vs +10 eval bonus)""")

code("""# ============================
# SYSTEM PROMPT (used everywhere)
# ============================
SYSTEM_PROMPT = "You output JSON actions for Minesweeper. No text, only JSON."

# ============================
# COMPACT PROMPT BUILDER
# ============================
def build_compact_prompt(game_or_state):
    \"\"\"Build compact prompt. Accepts MinesweeperGame or dict game state.\"\"\"
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
    board_str = "\\n".join(board_lines)

    prompt = f\"\"\"Minesweeper {rows}x{cols}, {mines} mines, {flagged} flagged, {revealed} revealed.
.=unknown F=flag 0-8=adjacent mines

{board_str}

JSON action:\"\"\"
    return prompt

# ============================
# JSON ACTION PARSER
# ============================
def parse_llm_action(response):
    \"\"\"Extract JSON action from LLM response. Takes LAST valid match.\"\"\"
    best = None
    for match in re.finditer(r'\\{[^{}]*\\}', response):
        try:
            action = json.loads(match.group())
            if ("type" in action and "row" in action and "col" in action
                    and action["type"] in ["reveal", "flag"]):
                best = action
        except json.JSONDecodeError:
            continue
    return best

# ============================
# HELPER: Get neighbors
# ============================
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

# ============================
# LOGICAL DEDUCTION CHECKER
# ============================
def is_logically_deducible(board, rows, cols, action_type, tr, tc):
    \"\"\"Check if a move can be logically deduced from board constraints.\"\"\"
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

    # Phase 2: Coupled constraints
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

# ============================
# EXPERT SOLVER
# ============================
class MinesweeperSolver:
    \"\"\"Expert solver using constraint propagation + coupled constraints.\"\"\"

    def analyze_board(self, board, rows, cols, num_mines, num_flagged):
        cf = set()
        cr = set()
        frontier = [(r, c) for r in range(rows) for c in range(cols)
                     if board[r][c] in '12345678'
                     and any(board[nr][nc] == '.' for nr, nc in get_neighbors(r, c, rows, cols))]

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

        # Coupled constraints with spatial indexing
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
                        if (r1, c1) >= (r2, c2) or abs(r1-r2) > 2 or abs(c1-c2) > 2:
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
print(f"Compact prompt ({len(build_compact_prompt(test_game))} chars):")
print(build_compact_prompt(test_game))""")

# ============================================================
# CELL 6: Test Base Model
# ============================================================
md("# Test Base Model Before Training")

code("""from transformers import TextStreamer

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
    temperature=1.0, max_new_tokens=128,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)""")

# ============================================================
# CELL 7: SFT Dataset Generation
# ============================================================
md("""# Phase 1: Generate SFT Training Data

Using the expert solver to generate optimal (state, action) pairs.
Variable board sizes from 5x5 to 50x50 for generalization.""")

code("""def generate_sft_dataset(num_samples=5000, rng_seed=42):
    \"\"\"Generate expert-solved training data using constraint solver.\"\"\"
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    dataset_items = []
    slvr = MinesweeperSolver()

    # Board size distribution - diverse sizes for generalization
    board_configs = [
        (5, 5, 0.12, 0.05), (6, 6, 0.14, 0.10), (8, 8, 0.15, 0.15),
        (10, 10, 0.15, 0.15), (12, 12, 0.15, 0.10), (16, 16, 0.15, 0.10),
        (20, 20, 0.15, 0.10), (25, 25, 0.15, 0.05), (30, 30, 0.15, 0.05),
        (40, 40, 0.12, 0.03), (50, 50, 0.10, 0.05),
        (50, 50, 0.15, 0.04), (50, 50, 0.20, 0.03),
    ]

    total_w = sum(w for _, _, _, w in board_configs)
    targets = [(r, c, mp, max(1, int(num_samples * w / total_w)))
               for r, c, mp, w in board_configs]

    for rows, cols, mine_pct, target in targets:
        mines = max(1, int(rows * cols * mine_pct))
        gen = 0
        attempts = 0
        while gen < target and attempts < target * 10:
            attempts += 1
            seed = np.random.randint(1000000)
            try:
                game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
            except ValueError:
                continue

            # Random first move
            fr, fc = np.random.randint(0, rows), np.random.randint(0, cols)
            result = game.do_action({"type": "reveal", "row": int(fr), "col": int(fc)})
            if game.state() != "ongoing":
                continue

            # Play 0-N moves with solver to create diverse mid-game states
            move_history = [{"type": "reveal", "row": int(fr), "col": int(fc)}]
            num_extra = np.random.randint(0, min(20, rows * cols // 4))
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

            # Get expert next action
            act, is_logical = slvr.get_best_action(game)
            if act is None:
                continue

            prompt_text = build_compact_prompt(game)
            response_text = json.dumps(act)

            dataset_items.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": response_text},
                ],
                # Metadata for GRPO phase later
                "seed": seed,
                "move_history": json.dumps(move_history),
                "game_rows": rows,
                "game_cols": cols,
                "game_mines": mines,
            })
            gen += 1

        print(f"  {rows}x{cols} ({mines} mines): {gen}/{target} examples")

    random.shuffle(dataset_items)
    return Dataset.from_list(dataset_items)

print("Generating SFT dataset (this takes a few minutes)...")
sft_dataset = generate_sft_dataset(num_samples=5000)
print(f"\\nGenerated {len(sft_dataset)} SFT examples")

# Stats
sizes = defaultdict(int)
for item in sft_dataset:
    sizes[f"{item['game_rows']}x{item['game_cols']}"] += 1
print("\\nBoard size distribution:")
for s, c in sorted(sizes.items(), key=lambda x: int(x[0].split('x')[0])):
    print(f"  {s}: {c}")""")

# ============================================================
# CELL 8: SFT Training
# ============================================================
md("""# Phase 1: SFT Training

Teaches the model:
1. Output format (pure JSON, no reasoning text)
2. Basic minesweeper logic from expert solver
3. When to flag vs reveal""")

code("""from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="minesweeper_sft_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    max_seq_length=max_seq_length,
    optim="adamw_8bit",
    report_to="none",
    dataset_text_field=None,  # Use messages format
)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=sft_dataset,
    args=sft_config,
)

print("Starting SFT training...")
sft_trainer.train()
print("SFT training complete!")""")

# ============================================================
# CELL 9: Save SFT checkpoint + Quick eval
# ============================================================
md("# Save SFT Checkpoint + Quick Evaluation")

code("""# Save SFT checkpoint
model.save_pretrained("minesweeper_sft_checkpoint")
tokenizer.save_pretrained("minesweeper_sft_checkpoint")
print("SFT checkpoint saved")

# Quick evaluation
def quick_eval(model, tokenizer, num_games=10, label=""):
    wins = 0
    valid = 0
    total_moves = 0
    for seed in range(num_games):
        game = MinesweeperGame(rows=8, cols=8, num_mines=10, seed=seed + 10000)
        game.do_action({"type": "reveal", "row": 4, "col": 4})
        moves = 0
        while game.state() == "ongoing" and moves < 100:
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
            if action is None:
                break
            valid += 1
            result = game.do_action(action)
            moves += 1
            if result in ("mine", "already_revealed", "flagged_cell", "invalid_flag", "out_of_bounds"):
                break
        total_moves += moves
        if game.state() == "success":
            wins += 1
    print(f"[{label}] {wins}/{num_games} wins, {valid} valid JSON, avg {total_moves/num_games:.1f} moves/game")

print("\\nEvaluating SFT model...")
quick_eval(model, tokenizer, num_games=10, label="Post-SFT")""")

# ============================================================
# CELL 10: GRPO Reward Functions
# ============================================================
md("""# Phase 2: GRPO Reward Functions

Three reward functions matching all 12 evaluation criteria:
1. **Format reward**: Valid JSON output, bonus for conciseness
2. **Gameplay reward**: All 12 scoring criteria with logical deduction detection
3. **Conciseness reward**: Penalizes verbose output (128 token hard limit)""")

code("""def valid_json_reward(completions, **kwargs):
    \"\"\"Reward valid JSON format. Bonus for concise output.\"\"\"
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        action = parse_llm_action(response)
        if action is None:
            scores.append(-5.0)
        else:
            stripped = response.strip()
            if stripped.startswith('{') and len(stripped) < 80:
                scores.append(2.0)
            else:
                scores.append(1.0)
    return scores

def gameplay_reward(completions, **kwargs):
    \"\"\"Complete gameplay reward - all 12 eval criteria.\"\"\"
    scores = []
    seeds = kwargs.get("seed", [])
    mh_list = kwargs.get("move_history", [])
    gr_list = kwargs.get("game_rows", [])
    gc_list = kwargs.get("game_cols", [])
    gm_list = kwargs.get("game_mines", [])

    for idx, completion in enumerate(completions):
        response = completion[0]["content"]
        action = parse_llm_action(response)
        if action is None:
            scores.append(-10.0)
            continue
        if idx >= len(seeds) or idx >= len(mh_list):
            scores.append(0.0)
            continue

        seed = seeds[idx]
        mh_raw = mh_list[idx]
        rows = gr_list[idx] if idx < len(gr_list) else 6
        cols = gc_list[idx] if idx < len(gc_list) else 6
        mines = gm_list[idx] if idx < len(gm_list) else 5
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

        # Out of bounds
        if not (0 <= row < rows and 0 <= col < cols):
            scores.append(-15.0)
            continue

        score = 0.0
        if atype == "reveal":
            if (row, col) in game._revealed:
                scores.append(-12.0)
                continue
            if (row, col) in game._flagged:
                scores.append(-8.0)
                continue
            if game._board[row][col] == -1:
                score = -25.0
            else:
                is_log = is_logically_deducible(board, rows, cols, "reveal", row, col)
                score = 15.0 if is_log else 10.0
                # Check win
                tg = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
                for prev in mh:
                    tg.do_action(prev)
                tg.do_action(action)
                if tg.state() == "success":
                    score += 100.0
        elif atype == "flag":
            if (row, col) in game._revealed:
                scores.append(-8.0)
                continue
            if (row, col) in game._flagged:
                scores.append(-8.0)
                continue
            if len(game._flagged) + 1 > mines:
                score -= 10.0
            if game._board[row][col] == -1:
                score += 15.0
            else:
                score -= 10.0

        scores.append(score)
    return scores

def conciseness_reward(completions, **kwargs):
    \"\"\"Reward concise output. 128 token hard limit means verbosity = death.\"\"\"
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        stripped = response.strip()
        action = parse_llm_action(response)
        if action is None:
            scores.append(-2.0)
            continue
        tok_est = len(stripped) / 4
        if stripped.startswith('{') and tok_est < 20:
            scores.append(3.0)
        elif stripped.startswith('{') and tok_est < 40:
            scores.append(1.5)
        elif tok_est < 60:
            scores.append(0.5)
        else:
            scores.append(-2.0)
    return scores

print("Reward functions defined (3 functions, 12 criteria)")""")

# ============================================================
# CELL 11: GRPO Dataset
# ============================================================
md("""# Phase 2: Generate GRPO Training Data

Similar to SFT data but prompt-only format (no expert answer).
The model explores and reward functions guide learning.""")

code("""def generate_grpo_dataset(num_samples=3000, rng_seed=123):
    \"\"\"Generate diverse game states for GRPO training.\"\"\"
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    items = []
    slvr = MinesweeperSolver()

    configs = [
        (5, 5, 0.12, 0.05), (6, 6, 0.14, 0.10), (8, 8, 0.15, 0.15),
        (10, 10, 0.15, 0.15), (12, 12, 0.15, 0.10), (16, 16, 0.15, 0.10),
        (20, 20, 0.15, 0.10), (25, 25, 0.15, 0.05), (30, 30, 0.15, 0.05),
        (40, 40, 0.12, 0.03), (50, 50, 0.10, 0.04),
        (50, 50, 0.15, 0.04), (50, 50, 0.20, 0.04),
    ]
    total_w = sum(w for _, _, _, w in configs)

    for rows, cols, mp, weight in configs:
        mines = max(1, int(rows * cols * mp))
        target = max(1, int(num_samples * weight / total_w))
        gen = 0
        while gen < target:
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
            for _ in range(np.random.randint(0, min(15, rows * cols // 4))):
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
grpo_dataset = generate_grpo_dataset(num_samples=3000)
print(f"Generated {len(grpo_dataset)} GRPO examples")""")

# ============================================================
# CELL 12: GRPO Training
# ============================================================
md("""# Phase 2: GRPO Training

Key settings from research:
- `learning_rate=5e-6` (lower than SFT - critical for RL stability)
- `num_generations=8` (diverse exploration per prompt)
- `beta=0.0` (no KL penalty, saves VRAM)
- `temperature=1.0` (diverse generations during training)
- **Rewards won't improve for ~150 steps - this is normal, do NOT stop early!**""")

code("""from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    # Generation
    temperature=1.0,
    num_generations=8,

    # Optimizer
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",

    # Batching
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # Lengths
    max_prompt_length=3000,
    max_completion_length=128,

    # Training
    max_steps=500,        # Increase to 1000 if time allows
    save_steps=200,

    # GRPO specific
    beta=0.0,

    # Logging
    logging_steps=1,
    report_to="none",
    output_dir="minesweeper_grpo_output",
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[valid_json_reward, gameplay_reward, conciseness_reward],
    args=grpo_config,
    train_dataset=grpo_dataset,
)

print("Starting GRPO training...")
print("NOTE: Rewards may not improve for first 150+ steps - this is NORMAL!")
grpo_trainer.train()
print("GRPO training complete!")""")

# ============================================================
# CELL 13: Final Evaluation
# ============================================================
md("# Final Evaluation")

code("""def play_full_game(model, tokenizer, rows=8, cols=8, num_mines=10, seed=None, max_moves=50):
    game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
    # First move
    game.do_action({"type": "reveal", "row": rows // 2, "col": cols // 2})
    moves = 0
    while game.state() == "ongoing" and moves < max_moves:
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
        if action is None:
            break
        result = game.do_action(action)
        moves += 1
        if result in ("mine", "already_revealed", "flagged_cell", "invalid_flag", "out_of_bounds"):
            break
    return game, moves

# Evaluate on multiple board sizes
eval_configs = [
    (6, 6, 5, 20, "6x6"),
    (8, 8, 10, 20, "8x8"),
    (10, 10, 15, 10, "10x10"),
    (16, 16, 40, 5, "16x16"),
]

print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)
for rows, cols, mines, num_games, label in eval_configs:
    wins = 0
    total_moves = 0
    for i in range(num_games):
        game, moves = play_full_game(model, tokenizer, rows, cols, mines,
                                     seed=20000 + i, max_moves=2 * rows * cols)
        if game.state() == "success":
            wins += 1
        total_moves += moves
    print(f"{label}: {wins}/{num_games} wins ({wins/num_games*100:.0f}%), avg {total_moves/num_games:.1f} moves")
print("=" * 60)""")

# ============================================================
# CELL 14: Save Model
# ============================================================
md("""# Save Final Model

Save merged model for the evaluation agent.""")

code("""# Save LoRA adapters
model.save_pretrained("my_minesweeper_model")
tokenizer.save_pretrained("my_minesweeper_model")
print("LoRA adapters saved to: my_minesweeper_model/")

# Save merged model (used by eval agent)
model.save_pretrained_merged(
    "your_finetuned_model",
    tokenizer,
    save_method="merged_16bit"
)
print("Merged model saved to: your_finetuned_model/")""")

# ============================================================
# CELL 15: Update Inference Agent
# ============================================================
md("""# Update Inference Agent Files

**CRITICAL**: The inference agent's prompt MUST match our training prompt format.
This cell updates the agent files for evaluation.""")

code("""# Update minesweeper_agent.py with our compact prompt
agent_code = '''#!/usr/bin/python3
"""Minesweeper Agent - Competition Version"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from .minesweeper_model import MinesweeperAgent

class MinesweeperPlayer:
    """Agent responsible for playing Minesweeper"""

    def __init__(self, **kwargs):
        self.agent = MinesweeperAgent(**kwargs)

    def build_prompt(self, game_state: Dict[str, Any]) -> tuple[str, str]:
        board = game_state["board"]
        rows = game_state["rows"]
        cols = game_state["cols"]
        mines = game_state["mines"]
        flagged = game_state.get("flags_placed", 0)
        revealed = game_state.get("cells_revealed", 0)

        board_lines = []
        for r in range(rows):
            board_lines.append(f"{r:>2}|{''.join(board[r])}")
        board_str = "\\\\n".join(board_lines)

        prompt = f"""Minesweeper {rows}x{cols}, {mines} mines, {flagged} flagged, {revealed} revealed.
.=unknown F=flag 0-8=adjacent mines

{board_str}

JSON action:"""

        sys_prompt = "You output JSON actions for Minesweeper. No text, only JSON."
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

with open("agents/minesweeper_agent.py", "w") as f:
    f.write(agent_code)
print("Updated agents/minesweeper_agent.py")

# Update model path
model_code = '''"""Minesweeper Model - Competition Version"""
import time
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer

class MinesweeperAgent(object):
    def __init__(self, **kwargs):
        model_name = "/workspace/your_finetuned_model"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate_response(self, message, system_prompt=None, **kwargs):
        if system_prompt is None:
            system_prompt = "You output JSON actions for Minesweeper. No text, only JSON."

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
            temperature=kwargs.get("temperature", 0.3),
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
    f.write(model_code)
print("Updated agents/minesweeper_model.py")

# Update config
config = \"\"\"## Minesweeper Agent Configuration ##
max_new_tokens: 128
temperature: 0.3
top_p: 0.9
repetition_penalty: 1.1
do_sample: true
\"\"\"

with open("minesweeper_config.yaml", "w") as f:
    f.write(config)
print("Updated minesweeper_config.yaml")

print("\\nAll inference agent files updated! Ready for evaluation.")""")

# ============================================================
# CELL 16: Final notes
# ============================================================
md("""# Competition Checklist

- [x] Model fine-tuned with SFT + GRPO
- [x] Compact board representation (handles 50x50)
- [x] Complete reward function (all 12 criteria)
- [x] Variable board sizes in training data
- [x] Inference agent updated with matching prompt
- [x] Model saved to `your_finetuned_model/`

## Troubleshooting
- **OOM**: Reduce `per_device_train_batch_size` or `num_generations`
- **GRPO rewards flat**: Normal for first 150 steps. If still flat at 300, check reward function variance.
- **Invalid JSON outputs**: Increase SFT training epochs or data size
- **Bad performance on large boards**: Add more large-board examples to training data
- **Model too verbose**: Increase `conciseness_reward` weight

## Inference Test
```bash
python -m agents.minesweeper_agent --game_state_file inputs/game_state.json --output_file outputs/action.json --verbose
```""")

# ============================================================
# BUILD NOTEBOOK
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("/Users/tusharchandra/Downloads/AMD_Hack_Initial_State_Backup/minesweeper_final.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook generated: minesweeper_final.ipynb")
print(f"Total cells: {len(cells)}")
print(f"  Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
