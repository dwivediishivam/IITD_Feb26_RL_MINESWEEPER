#!/usr/bin/env python3
"""
Standalone Prompt Strategy Comparison Test
6 strategies x 4 models = 24 configurations
Tests which prompt + model combo performs best (no post-processing).
"""

# ################################################################
# CELL 0: IMPORTS AND SETUP
# ################################################################
import os
import gc
import json
import re
import glob
import random
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from unsloth import FastLanguageModel

# ################################################################
# CELL 1: MINESWEEPER GAME CLASS (DO NOT MODIFY)
# ################################################################

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

    def _reveal_cell(self, row, col):
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

    def _flag_cell(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed:
            return False
        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action):
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

    def get_visible_board(self):
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

    def state(self):
        return self._state

print("MinesweeperGame class loaded.")

# ################################################################
# CELL 2: HELPER FUNCTIONS
# ################################################################

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


def is_logically_deducible(board, rows, cols, action_type, tr, tc):
    """Check if a move can be logically deduced from board constraints."""
    cf = set()
    cr = set()

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

print("Helper functions loaded.")

# ################################################################
# CELL 3: SCORING RULES AND 6 PROMPT STRATEGIES
# ################################################################

SCORING_RULES = (
    "\nSCORING RULES (maximize your total score!):\n"
    "GOOD MOVES:\n"
    "- Reveal a safe unknown(.) cell: +10 pts (or +15 if logically deducible from constraints!)\n"
    "- Flag a cell that IS a mine: +15 pts\n"
    "- Win the game (all safe cells revealed): +100 pts BONUS\n"
    "BAD MOVES (AVOID THESE AT ALL COSTS):\n"
    "- Reveal a mine: -25 pts AND GAME OVER\n"
    "- Flag a cell that is NOT a mine: -10 pts\n"
    "- Reveal an already-revealed cell (showing 0-8): -12 pts\n"
    "- Reveal a flagged cell (F): -8 pts\n"
    "- Flag an already-flagged cell (F): -8 pts\n"
    "- Flag a revealed cell (0-8): -8 pts\n"
    "- Out of bounds row/col: -15 pts\n"
    "- Placing more flags than total mines: -10 pts\n"
    "- Invalid/missing JSON: -10 pts\n"
    "STRATEGY: ONLY pick cells marked '.' (dot). Use constraint logic for +15 bonus. Never guess blindly."
)


def strategy_v1_simple(game, add_scoring=False):
    """V1: Original simple prompt"""
    prompt = build_compact_prompt(game)
    sys = "You output JSON actions for Minesweeper. No text, only JSON."
    if add_scoring:
        sys += SCORING_RULES
    return sys, prompt


def strategy_v2_constraint(game, add_scoring=False):
    """V2: Constraint logic prompt"""
    prompt = build_compact_prompt(game)
    sys = 'Analyze the Minesweeper board. For each numbered cell, count adjacent flags(F) and unknowns(.). If number equals flag count, unknowns are safe to reveal. If number minus flags equals unknown count, unknowns are mines to flag. Only act on certain deductions. Output ONLY JSON: {"type":"reveal","row":N,"col":N} or {"type":"flag","row":N,"col":N}'
    if add_scoring:
        sys += SCORING_RULES
    return sys, prompt


def strategy_v3_aggressive(game, add_scoring=False):
    """V3: Extremely aggressive DO NOT repeated warnings"""
    prompt = build_compact_prompt(game)
    sys = (
        'You play Minesweeper. Output ONLY valid JSON: {"type":"reveal"or"flag","row":N,"col":N}\n'
        'ABSOLUTE RULES - VIOLATION = INSTANT PENALTY:\n'
        '1. ONLY target cells showing "." on the board. These are UNKNOWN cells.\n'
        '2. Cells showing 0,1,2,3,4,5,6,7,8 are ALREADY REVEALED. DO NOT pick them. DO NOT DO NOT DO NOT.\n'
        '3. Cells showing F are ALREADY FLAGGED. DO NOT reveal them. DO NOT flag them again. DO NOT DO NOT.\n'
        '4. NEVER reveal an already-revealed cell. CHECK the board at your target (row,col). Is it "."? If NOT, STOP and pick another.\n'
        '5. NEVER reveal a flagged cell (F). CHECK AGAIN. Is your target "."? If NOT, pick another cell.\n'
        '6. NEVER flag an already-flagged cell. CHECK AGAIN.\n'
        '7. NEVER flag a revealed cell (showing 0-8). CHECK AGAIN.\n'
        '8. BEFORE outputting: look at board[row][col]. If it is NOT ".", you MUST change your answer.\n'
        '9. DOUBLE CHECK: Is your target cell "."? YES -> output. NO -> pick a different cell.\n'
        '10. TRIPLE CHECK: Are you absolutely sure the cell shows "."? Confirmed? Then output.\n'
        'LOGIC: Count F and "." around numbered cells. number=flags -> unknowns safe. number-flags=unknowns -> unknowns are mines.'
    )
    if add_scoring:
        sys += SCORING_RULES
    return sys, prompt


def strategy_v4_rules_list(game, add_scoring=False):
    """V4: Step-by-step verification rules"""
    prompt = build_compact_prompt(game)
    sys = (
        'Minesweeper action rules:\n'
        'STEP 1: Scan the board. Cells with "." are UNKNOWN (valid targets). Cells with 0-8 are REVEALED (forbidden). F = FLAGGED (forbidden).\n'
        'STEP 2: For each numbered cell (1-8), count adjacent F (flags) and "." (unknowns).\n'
        'STEP 3: If number = flag_count, all adjacent "." are safe -> reveal one.\n'
        'STEP 4: If number - flag_count = unknown_count, all adjacent "." are mines -> flag one.\n'
        'STEP 5: Pick your target (row, col). VERIFY: What does board[row][col] show?\n'
        '  - Shows "." -> GOOD, proceed.\n'
        '  - Shows 0-8 -> FORBIDDEN! Already revealed. Pick a different cell.\n'
        '  - Shows F -> FORBIDDEN! Already flagged. Pick a different cell.\n'
        'Output ONLY: {"type":"reveal","row":N,"col":N} or {"type":"flag","row":N,"col":N}'
    )
    if add_scoring:
        sys += SCORING_RULES
    return sys, prompt


def strategy_v5_annotated_board(game, add_scoring=False):
    """V5: List valid target cells directly in user prompt"""
    board = game.get_visible_board()
    rows, cols = game.rows, game.cols
    flagged = len(game._flagged)
    revealed = len(game._revealed)
    mines = game.num_mines

    board_lines = []
    for r in range(rows):
        board_lines.append(f"{r:>2}|{''.join(board[r])}")
    board_str = "\n".join(board_lines)

    valid_targets = []
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == '.':
                valid_targets.append(f"({r},{c})")
    valid_str = " ".join(valid_targets[:30])

    prompt = (
        f"Minesweeper {rows}x{cols}, {mines} mines, {flagged} flagged, {revealed} revealed.\n"
        f".=unknown F=flag 0-8=adjacent mines\n\n"
        f"{board_str}\n\n"
        f"VALID TARGETS (only these cells show '.'): {valid_str}\n"
        f"You MUST pick row,col from this list. Any cell NOT showing '.' = HEAVY PENALTY.\n\n"
        f"JSON action:"
    )
    sys = 'Analyze the Minesweeper board. Pick ONLY from the VALID TARGETS list shown in the prompt. Output JSON: {"type":"reveal"or"flag","row":N,"col":N}'
    if add_scoring:
        sys += SCORING_RULES
    return sys, prompt


def strategy_v6_cot_verify(game, add_scoring=False):
    """V6: Model self-verifies target cell in think field"""
    prompt = build_compact_prompt(game)
    sys = (
        'Play Minesweeper. Output JSON with self-verification:\n'
        '{"think":"<reasoning>. Cell at (row,col) shows: <symbol>. Is it dot? YES.","type":"reveal"or"flag","row":N,"col":N}\n'
        'MANDATORY: In your "think" field, CHECK what your target cell shows on the board.\n'
        'If board[row][col] is NOT "." (dot), you picked wrong. Numbers 0-8 = already revealed. F = already flagged.\n'
        'Only "." cells are valid targets. VERIFY before output.\n'
        'Logic: count F and "." neighbors of each number. number=F_count -> "." safe. number-F_count=unknown_count -> "." mines.'
    )
    if add_scoring:
        sys += SCORING_RULES
    return sys, prompt


strategies = [
    ("V1-simple", strategy_v1_simple),
    ("V2-constraint", strategy_v2_constraint),
    ("V3-aggressive", strategy_v3_aggressive),
    ("V4-rules", strategy_v4_rules_list),
    ("V5-annotated", strategy_v5_annotated_board),
    ("V6-cot-verify", strategy_v6_cot_verify),
]

print(f"Defined {len(strategies)} prompt strategies.")
print("Scoring rules will be added to BASE model prompts only.")

# ################################################################
# CELL 4: GAME RUNNER + TEST LOOP (24 configs: 6 strategies x 4 models)
# ################################################################

def play_game_with_strategy(model, tokenizer, strategy_fn, add_scoring=False,
                            rows=8, cols=8, num_mines=10, seed=None, max_moves=200):
    """Play one full game with a given prompt strategy. No post-processing."""
    game = MinesweeperGame(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
    game.do_action({"type": "reveal", "row": rows // 2, "col": cols // 2})
    moves = 0
    score = 0.0
    consecutive_bad = 0
    invalid_counts = {"already_revealed": 0, "reveal_flagged": 0, "already_flagged": 0,
                      "flag_revealed": 0, "oob": 0, "mine_hit": 0, "wrong_flag": 0, "invalid_json": 0}

    while game.state() == "ongoing" and moves < max_moves and consecutive_bad < 5:
        sys_prompt, user_prompt = strategy_fn(game, add_scoring=add_scoring)
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}]
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
            score -= 10.0; consecutive_bad += 1; invalid_counts["invalid_json"] += 1; continue
        try:
            row, col = int(action["row"]), int(action["col"])
        except (ValueError, TypeError):
            score -= 10.0; consecutive_bad += 1; invalid_counts["invalid_json"] += 1; continue
        atype = action["type"]

        if not (0 <= row < rows and 0 <= col < cols):
            score -= 15.0; consecutive_bad += 1; invalid_counts["oob"] += 1; continue
        if atype == "reveal":
            if (row, col) in game._revealed:
                score -= 12.0; consecutive_bad += 1; invalid_counts["already_revealed"] += 1; continue
            if (row, col) in game._flagged:
                score -= 8.0; consecutive_bad += 1; invalid_counts["reveal_flagged"] += 1; continue
            if game._board[row][col] == -1:
                score -= 25.0; invalid_counts["mine_hit"] += 1; break
            consecutive_bad = 0
            board = game.get_visible_board()
            is_log = is_logically_deducible(board, rows, cols, "reveal", row, col)
            score += 15.0 if is_log else 10.0
            game.do_action(action)
            if game.state() == "success":
                score += 100.0
        elif atype == "flag":
            if (row, col) in game._revealed:
                score -= 8.0; consecutive_bad += 1; invalid_counts["flag_revealed"] += 1; continue
            if (row, col) in game._flagged:
                score -= 8.0; consecutive_bad += 1; invalid_counts["already_flagged"] += 1; continue
            consecutive_bad = 0
            if len(game._flagged) + 1 > num_mines:
                score -= 10.0
            if game._board[row][col] == -1:
                score += 15.0
            else:
                score -= 10.0; invalid_counts["wrong_flag"] += 1
            game.do_action(action)

    return {
        "result": game.state(),
        "moves": moves,
        "score": score,
        "invalid": invalid_counts,
    }


# ---- Board configs: 3 normal + 1 bigger ----
test_configs = [
    (8, 8, 10, 3, "8x8"),
    (10, 10, 15, 3, "10x10"),
    (6, 10, 8, 3, "6x10"),
    (16, 16, 40, 2, "16x16"),     # bigger board
]
total_games_per_strat = sum(n for _, _, _, n, _ in test_configs)
print(f"Test configs: {[lbl for *_, lbl in test_configs]}")
print(f"Games per strategy: {total_games_per_strat}")

# ---- Model configs: BASE models only (no fine-tuning) ----
# Auto-detect base model paths from cache

def _find_model_path(search_dirs):
    """Find the snapshot path for a model in the HF cache."""
    for md in search_dirs:
        if os.path.exists(md):
            snaps = sorted(glob.glob(os.path.join(md, "snapshots", "*")))
            if snaps:
                return snaps[-1]
    return None

_qwen_base = _find_model_path([
    "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct",
])
if _qwen_base is None:
    _qwen_base = "Qwen/Qwen2.5-14B-Instruct"  # fallback

_oss_base = _find_model_path([
    "/root/.cache/huggingface/models--unsloth--gpt-oss-20b-BF16",
    "/root/.cache/huggingface/models--gpt-oss-20b",
])
if _oss_base is None:
    _oss_base = "unsloth/gpt-oss-20b-BF16"  # fallback

# (label, path, add_scoring_to_prompt)
# Both are base models -> scoring rules added to all prompts
model_configs = [
    ("BASE-QWEN-14B", _qwen_base, True),
    ("BASE-OSS-20B", _oss_base, True),
]

print(f"\nModels to test (BASE only, no fine-tuning):")
for label, path, _ in model_configs:
    print(f"  {label}: {path}")
print(f"Total configs: {len(model_configs)} models x {len(strategies)} strategies = {len(model_configs) * len(strategies)}")
print(f"Total games: {len(model_configs) * len(strategies) * total_games_per_strat}")

# ---- RUN ALL TESTS ----
grand_results = {}

for model_label, model_path, is_base in model_configs:
    print("\n" + "#" * 80)
    print(f"#  LOADING: {model_label}")
    print(f"#  Path: {model_path}")
    print("#" * 80)

    # Check if model exists
    if not os.path.exists(model_path) and not model_path.startswith("/root"):
        print(f"  WARNING: {model_path} not found, SKIPPING this model.")
        for strat_name, _ in strategies:
            grand_results[f"{model_label}|{strat_name}"] = {
                "model": model_label, "strategy": strat_name,
                "wins": 0, "total": 0, "avg_score": 0, "invalid": {},
                "skipped": True,
            }
        continue

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    try:
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=False,
            max_seq_length=4096,
            torch_dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(_model)
        print(f"  Loaded on {_model.device}")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        for strat_name, _ in strategies:
            grand_results[f"{model_label}|{strat_name}"] = {
                "model": model_label, "strategy": strat_name,
                "wins": 0, "total": 0, "avg_score": 0, "invalid": {},
                "skipped": True,
            }
        continue

    for strat_name, strat_fn in strategies:
        config_key = f"{model_label}|{strat_name}"
        print(f"\n{'='*60}")
        print(f"  {config_key}")
        print(f"{'='*60}")

        strat_results = []
        total_invalid = {"already_revealed": 0, "reveal_flagged": 0, "already_flagged": 0,
                         "flag_revealed": 0, "oob": 0, "mine_hit": 0, "wrong_flag": 0, "invalid_json": 0}

        for rows, cols, mines, num_seeds, label in test_configs:
            wins = 0
            total_score = 0.0
            total_moves = 0
            for seed_i in range(num_seeds):
                res = play_game_with_strategy(
                    _model, _tokenizer, strat_fn,
                    add_scoring=is_base,
                    rows=rows, cols=cols, num_mines=mines, seed=42 + seed_i
                )
                strat_results.append(res)
                total_score += res["score"]
                total_moves += res["moves"]
                if res["result"] == "success":
                    wins += 1
                for k, v in res["invalid"].items():
                    total_invalid[k] += v
            avg_sc = total_score / num_seeds
            print(f"    {label}: {wins}/{num_seeds} wins, avg {total_moves/num_seeds:.1f} moves, avg score {avg_sc:+.1f}")

        total_games = len(strat_results)
        total_wins = sum(1 for r in strat_results if r["result"] == "success")
        total_avg = sum(r["score"] for r in strat_results) / total_games if total_games else 0
        print(f"\n    TOTAL: {total_wins}/{total_games} wins, avg score {total_avg:+.1f}")
        inv_str = ", ".join(f"{k}={v}" for k, v in total_invalid.items() if v > 0)
        print(f"    Invalid: {inv_str if inv_str else 'NONE'}")

        grand_results[config_key] = {
            "model": model_label, "strategy": strat_name,
            "wins": total_wins, "total": total_games,
            "avg_score": total_avg, "invalid": total_invalid,
            "skipped": False,
        }

    # Cleanup model
    del _model
    del _tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Unloaded {model_label}.")

# ################################################################
# CELL 5: RESULTS TABLE AND RANKING
# ################################################################

# Filter out skipped
active = {k: v for k, v in grand_results.items() if not v.get("skipped")}

print("\n" + "=" * 95)
print("GRAND RANKING: ALL CONFIGURATIONS (sorted by average score)")
print("=" * 95)
print(f"{'#':>3} {'Model':>16} | {'Strategy':>15} | {'Avg Score':>10} | {'Wins':>8} | {'Invalids':>8} | Top Penalties")
print("-" * 95)
ranked = sorted(active.items(), key=lambda x: x[1]["avg_score"], reverse=True)
for rank, (key, data) in enumerate(ranked, 1):
    inv_total = sum(data["invalid"].values())
    top_penalties = sorted(data["invalid"].items(), key=lambda x: x[1], reverse=True)
    top_str = ", ".join(f"{k}={v}" for k, v in top_penalties[:3] if v > 0)
    print(f"{rank:>3} {data['model']:>16} | {data['strategy']:>15} | {data['avg_score']:>+10.1f} | "
          f"{data['wins']:>3}/{data['total']:<3} | {inv_total:>8} | {top_str}")
print("=" * 95)

# Best per model
print("\nBEST STRATEGY PER MODEL:")
for ml in ["BASE-QWEN-14B", "BASE-OSS-20B"]:
    model_results = [(k, v) for k, v in active.items() if v["model"] == ml]
    if model_results:
        best_key, best_data = max(model_results, key=lambda x: x[1]["avg_score"])
        inv_total = sum(best_data["invalid"].values())
        print(f"  {ml:>16}: {best_data['strategy']:>15} | avg {best_data['avg_score']:+.1f} | "
              f"{best_data['wins']}/{best_data['total']} wins | {inv_total} invalids")

# Overall best
if ranked:
    best_key, best_data = ranked[0]
    print(f"\n{'='*60}")
    print(f">>> OVERALL BEST: {best_key}")
    print(f">>>   avg score: {best_data['avg_score']:+.1f}")
    print(f">>>   wins: {best_data['wins']}/{best_data['total']}")
    print(f">>>   invalids: {sum(best_data['invalid'].values())}")
    print(f"{'='*60}")
    print("\nGive these results to Claude to write the winning combo to agents/!")

# Skipped models
skipped = {k: v for k, v in grand_results.items() if v.get("skipped")}
if skipped:
    print(f"\nSKIPPED (model not found): {set(v['model'] for v in skipped.values())}")
