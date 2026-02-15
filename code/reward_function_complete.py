#!/usr/bin/env python3
"""
Complete Reward Function for Minesweeper GRPO Training
Implements all 12 scoring criteria from the hackathon evaluation.

Also includes logical deduction detection for the +15 vs +10 bonus.
"""
import json
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict


# ============================================================
# MinesweeperGame (EXACT copy from notebook - DO NOT MODIFY)
# ============================================================
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
        lines.append("  " + "─" * (self.cols * 3 + 1))
        for r, row in enumerate(visible):
            line = f"{r:2d}│ " + "  ".join(row)
            lines.append(line)
        return "\n".join(lines)


# ============================================================
# Logical Deduction Checker
# ============================================================
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


def is_logically_deducible(board, rows, cols, action_type, target_row, target_col):
    """
    Check if a move (reveal/flag at target_row, target_col) can be logically
    deduced from the current board state using single-cell constraint propagation
    and coupled constraints.

    Returns True if the move is logically deducible.
    """
    # Phase 1: Single-cell constraints
    certain_flags = set()
    certain_reveals = set()

    changed = True
    iterations = 0
    while changed and iterations < 50:
        changed = False
        iterations += 1
        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]
                if cell not in '12345678':
                    continue
                num = int(cell)
                neighbors = get_neighbors(r, c, rows, cols)

                flagged_n = sum(1 for nr, nc in neighbors
                               if board[nr][nc] == 'F' or (nr, nc) in certain_flags)
                unrevealed_n = [(nr, nc) for nr, nc in neighbors
                                if board[nr][nc] == '.' and (nr, nc) not in certain_flags and (nr, nc) not in certain_reveals]

                remaining = num - flagged_n
                if remaining < 0:
                    continue

                if remaining == len(unrevealed_n) and unrevealed_n:
                    for nr, nc in unrevealed_n:
                        if (nr, nc) not in certain_flags:
                            certain_flags.add((nr, nc))
                            changed = True

                if remaining == 0 and unrevealed_n:
                    for nr, nc in unrevealed_n:
                        if (nr, nc) not in certain_reveals:
                            certain_reveals.add((nr, nc))
                            changed = True

    # Phase 2: Coupled constraints
    numbered_cells = [(r, c) for r in range(rows) for c in range(cols)
                      if board[r][c] in '12345678']

    changed = True
    iterations = 0
    while changed and iterations < 30:
        changed = False
        iterations += 1
        for i, (r1, c1) in enumerate(numbered_cells):
            num1 = int(board[r1][c1])
            n1 = get_neighbors(r1, c1, rows, cols)
            flagged1 = sum(1 for nr, nc in n1 if board[nr][nc] == 'F' or (nr, nc) in certain_flags)
            unrev1 = set((nr, nc) for nr, nc in n1
                        if board[nr][nc] == '.' and (nr, nc) not in certain_flags and (nr, nc) not in certain_reveals)
            rem1 = num1 - flagged1
            if not unrev1:
                continue

            for j in range(i + 1, len(numbered_cells)):
                r2, c2 = numbered_cells[j]
                if abs(r1 - r2) > 2 or abs(c1 - c2) > 2:
                    continue
                num2 = int(board[r2][c2])
                n2 = get_neighbors(r2, c2, rows, cols)
                flagged2 = sum(1 for nr, nc in n2 if board[nr][nc] == 'F' or (nr, nc) in certain_flags)
                unrev2 = set((nr, nc) for nr, nc in n2
                            if board[nr][nc] == '.' and (nr, nc) not in certain_flags and (nr, nc) not in certain_reveals)
                rem2 = num2 - flagged2
                if not unrev2:
                    continue

                if unrev1.issubset(unrev2):
                    diff = unrev2 - unrev1
                    dm = rem2 - rem1
                    if diff and dm == len(diff):
                        for cell in diff:
                            if cell not in certain_flags:
                                certain_flags.add(cell)
                                changed = True
                    elif diff and dm == 0:
                        for cell in diff:
                            if cell not in certain_reveals:
                                certain_reveals.add(cell)
                                changed = True

                if unrev2.issubset(unrev1):
                    diff = unrev1 - unrev2
                    dm = rem1 - rem2
                    if diff and dm == len(diff):
                        for cell in diff:
                            if cell not in certain_flags:
                                certain_flags.add(cell)
                                changed = True
                    elif diff and dm == 0:
                        for cell in diff:
                            if cell not in certain_reveals:
                                certain_reveals.add(cell)
                                changed = True

    target = (target_row, target_col)
    if action_type == "flag" and target in certain_flags:
        return True
    if action_type == "reveal" and target in certain_reveals:
        return True
    return False


# ============================================================
# JSON Parser (same as notebook)
# ============================================================
def parse_llm_action(response):
    """Extract JSON action from LLM response. Takes LAST valid match."""
    import re
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


# ============================================================
# COMPLETE REWARD FUNCTIONS
# ============================================================

def valid_json_reward(completions, **kwargs):
    """Reward valid JSON action format - weighted to teach format first"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        action = parse_llm_action(response)

        if action is None:
            scores.append(-5.0)
        else:
            # Bonus for concise output (starts with { and is short)
            stripped = response.strip()
            if stripped.startswith('{') and len(stripped) < 80:
                scores.append(2.0)  # Extra reward for clean output
            else:
                scores.append(1.0)
    return scores


def gameplay_reward(completions, **kwargs):
    """
    COMPLETE gameplay reward matching ALL 12 evaluation criteria.

    Scoring Criteria (from hackathon eval):
    1.  Flag cell that IS a mine        → +15
    2.  Flag cell that is NOT a mine    → -10
    3.  Reveal cell that IS a mine      → -25
    4.  Reveal safe cell (random +10, logical +15)
    5.  Flag already flagged cell       → -8
    6.  Reveal already revealed cell    → -12
    7.  Out of bounds                   → -15
    8.  Total flags > total mines       → -10
    9.  Invalid JSON                    → -10
    10. Win the game                    → +100
    11. Reveal a flagged cell           → -8
    12. Flag a revealed cell            → -8
    """
    scores = []
    seeds = kwargs.get("seed", [])
    move_histories = kwargs.get("move_history", [])
    game_rows = kwargs.get("game_rows", [])
    game_cols = kwargs.get("game_cols", [])
    game_mines = kwargs.get("game_mines", [])

    for idx, completion in enumerate(completions):
        response = completion[0]["content"]
        action = parse_llm_action(response)

        # Criterion 9: Invalid JSON → -10
        if action is None:
            scores.append(-10.0)
            continue

        if idx >= len(seeds) or idx >= len(move_histories):
            scores.append(0.0)
            continue

        seed = seeds[idx]
        move_history_raw = move_histories[idx]
        rows = game_rows[idx] if idx < len(game_rows) else 6
        cols = game_cols[idx] if idx < len(game_cols) else 6
        mines = game_mines[idx] if idx < len(game_mines) else 5

        if isinstance(move_history_raw, str):
            move_history = json.loads(move_history_raw)
        else:
            move_history = move_history_raw

        # Reconstruct exact game state
        game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
        for prev_action in move_history:
            game.do_action(prev_action)

        if game.state() != "ongoing":
            scores.append(0.0)
            continue

        board = game.get_visible_board()
        row = action.get("row")
        col = action.get("col")
        action_type = action.get("type")

        try:
            row, col = int(row), int(col)
        except (ValueError, TypeError):
            scores.append(-10.0)
            continue

        score = 0.0

        # Criterion 7: Out of bounds → -15
        if not (0 <= row < rows and 0 <= col < cols):
            scores.append(-15.0)
            continue

        if action_type == "reveal":
            # Criterion 6: Reveal already revealed cell → -12
            if (row, col) in game._revealed:
                scores.append(-12.0)
                continue

            # Criterion 11: Reveal a flagged cell → -8
            if (row, col) in game._flagged:
                scores.append(-8.0)
                continue

            # Check if cell is mine
            if game._board[row][col] == -1:
                # Criterion 3: Reveal mine → -25
                score = -25.0
            else:
                # Criterion 4: Reveal safe cell
                is_logical = is_logically_deducible(board, rows, cols, "reveal", row, col)
                score = 15.0 if is_logical else 10.0

                # Check if this move would win the game
                # (revealing this cell + auto-reveals = all safe cells revealed)
                test_game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
                for prev_action in move_history:
                    test_game.do_action(prev_action)
                test_game.do_action(action)
                if test_game.state() == "success":
                    # Criterion 10: Win bonus
                    score += 100.0

        elif action_type == "flag":
            # Criterion 12: Flag a revealed cell → -8
            if (row, col) in game._revealed:
                scores.append(-8.0)
                continue

            # Criterion 5: Flag already flagged cell → -8
            if (row, col) in game._flagged:
                scores.append(-8.0)
                continue

            # Criterion 8: Total flags > total mines → -10
            current_flags = len(game._flagged) + 1  # +1 for this new flag
            if current_flags > mines:
                score -= 10.0

            # Check if cell is actually a mine
            if game._board[row][col] == -1:
                # Criterion 1: Flag correct mine → +15
                score += 15.0
            else:
                # Criterion 2: Flag non-mine → -10
                score += -10.0

        scores.append(score)

    return scores


def conciseness_reward(completions, **kwargs):
    """
    Reward concise output. Critical because 128 token limit.
    Train model to output ONLY the JSON action.
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        stripped = response.strip()

        # Ideal: starts with { and is very short
        if not stripped:
            scores.append(-3.0)
            continue

        action = parse_llm_action(response)
        if action is None:
            scores.append(-2.0)
            continue

        # Count tokens approximately (chars / 4)
        token_est = len(stripped) / 4

        if stripped.startswith('{') and token_est < 20:
            scores.append(3.0)  # Perfect: just JSON
        elif stripped.startswith('{') and token_est < 40:
            scores.append(1.5)  # Good: JSON with minor extra
        elif token_est < 60:
            scores.append(0.5)  # Acceptable
        elif token_est < 100:
            scores.append(-0.5)  # Getting verbose
        else:
            scores.append(-2.0)  # Too verbose - risk of 128 token limit

    return scores


# ============================================================
# TESTS
# ============================================================
def test_reward_functions():
    """Comprehensive tests for all reward function criteria"""
    print("=" * 70)
    print("REWARD FUNCTION TESTS")
    print("=" * 70)

    # Setup: create a game with known state
    seed = 42
    game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=seed)

    # Make first move to get some revealed cells
    game.do_action({"type": "reveal", "row": 3, "col": 3})

    board = game.get_visible_board()
    print("Test board:")
    print(game.pretty_print())
    print(f"\nMine locations: ", end="")
    mines = [(r, c) for r in range(6) for c in range(6) if game._board[r][c] == -1]
    print(mines)
    print(f"Revealed: {game._revealed}")
    print(f"Flagged: {game._flagged}")

    move_history = [{"type": "reveal", "row": 3, "col": 3}]

    # Find a safe unrevealed cell and a mine cell
    safe_cell = None
    mine_cell = None
    revealed_cell = None
    for r in range(6):
        for c in range(6):
            if game._board[r][c] != -1 and (r, c) not in game._revealed and safe_cell is None:
                safe_cell = (r, c)
            if game._board[r][c] == -1 and mine_cell is None:
                mine_cell = (r, c)
            if (r, c) in game._revealed and revealed_cell is None:
                revealed_cell = (r, c)

    print(f"\nTest cells: safe={safe_cell}, mine={mine_cell}, revealed={revealed_cell}")

    # Test cases
    test_cases = []

    # 1. Valid reveal of safe cell
    if safe_cell:
        test_cases.append({
            "name": "Reveal safe cell",
            "response": json.dumps({"type": "reveal", "row": safe_cell[0], "col": safe_cell[1]}),
            "expected_range": (10, 115),  # +10 to +15, or +110-115 if wins
        })

    # 2. Reveal mine
    if mine_cell:
        test_cases.append({
            "name": "Reveal mine",
            "response": json.dumps({"type": "reveal", "row": mine_cell[0], "col": mine_cell[1]}),
            "expected_range": (-25, -25),
        })

    # 3. Flag correct mine
    if mine_cell:
        test_cases.append({
            "name": "Flag correct mine",
            "response": json.dumps({"type": "flag", "row": mine_cell[0], "col": mine_cell[1]}),
            "expected_range": (15, 15),
        })

    # 4. Flag non-mine
    if safe_cell:
        test_cases.append({
            "name": "Flag non-mine",
            "response": json.dumps({"type": "flag", "row": safe_cell[0], "col": safe_cell[1]}),
            "expected_range": (-10, -10),
        })

    # 5. Reveal already revealed cell
    if revealed_cell:
        test_cases.append({
            "name": "Reveal already revealed",
            "response": json.dumps({"type": "reveal", "row": revealed_cell[0], "col": revealed_cell[1]}),
            "expected_range": (-12, -12),
        })

    # 6. Flag revealed cell
    if revealed_cell:
        test_cases.append({
            "name": "Flag revealed cell",
            "response": json.dumps({"type": "flag", "row": revealed_cell[0], "col": revealed_cell[1]}),
            "expected_range": (-8, -8),
        })

    # 7. Out of bounds
    test_cases.append({
        "name": "Out of bounds",
        "response": json.dumps({"type": "reveal", "row": 10, "col": 10}),
        "expected_range": (-15, -15),
    })

    # 8. Invalid JSON
    test_cases.append({
        "name": "Invalid JSON",
        "response": "I think we should reveal cell at row 2",
        "expected_range": (-10, -10),
    })

    # 9. Concise valid JSON
    test_cases.append({
        "name": "Concise JSON output",
        "response": json.dumps({"type": "reveal", "row": 0, "col": 0}),
        "expected_range": None,  # Just testing conciseness
    })

    # 10. Verbose output
    test_cases.append({
        "name": "Verbose output",
        "response": "Let me analyze the board carefully. Looking at the numbers, I can see that cell (0,0) is likely safe because... " + json.dumps({"type": "reveal", "row": 0, "col": 0}),
        "expected_range": None,  # Just testing conciseness
    })

    # Run tests
    print(f"\n{'='*70}")
    all_passed = True
    for tc in test_cases:
        completions = [[{"content": tc["response"]}]]

        gp_scores = gameplay_reward(
            completions,
            seed=[seed],
            move_history=[json.dumps(move_history)],
            game_rows=[6],
            game_cols=[6],
            game_mines=[5],
        )

        json_scores = valid_json_reward(completions)
        concise_scores = conciseness_reward(completions)

        gp = gp_scores[0]
        js = json_scores[0]
        cs = concise_scores[0]

        status = "PASS"
        if tc["expected_range"] is not None:
            lo, hi = tc["expected_range"]
            if not (lo <= gp <= hi):
                status = "FAIL"
                all_passed = False

        print(f"\n  {tc['name']}: {status}")
        print(f"    Gameplay: {gp:+.1f}  JSON: {js:+.1f}  Concise: {cs:+.1f}")
        if tc["expected_range"]:
            print(f"    Expected gameplay: [{tc['expected_range'][0]}, {tc['expected_range'][1]}]")
        print(f"    Response: {tc['response'][:80]}...")

    print(f"\n{'='*70}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


def test_logical_deduction():
    """Test that logical deduction detection works correctly"""
    print("\n" + "=" * 70)
    print("LOGICAL DEDUCTION TESTS")
    print("=" * 70)

    # Create a known board state where logical deduction is possible
    # Board: after first reveal, we have numbered cells with deterministic neighbors
    game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=42)
    game.do_action({"type": "reveal", "row": 3, "col": 3})

    board = game.get_visible_board()
    print("Board state:")
    print(game.pretty_print())

    # Find cells that are logically deducible
    certain_flags = set()
    certain_reveals = set()

    for r in range(6):
        for c in range(6):
            if board[r][c] == '.':
                if is_logically_deducible(board, 6, 6, "flag", r, c):
                    certain_flags.add((r, c))
                    is_mine = game._board[r][c] == -1
                    print(f"  Logically deducible FLAG at ({r},{c}) - actually mine: {is_mine}")

                if is_logically_deducible(board, 6, 6, "reveal", r, c):
                    certain_reveals.add((r, c))
                    is_mine = game._board[r][c] == -1
                    print(f"  Logically deducible REVEAL at ({r},{c}) - actually mine: {is_mine}")

    print(f"\n  Total logically deducible flags: {len(certain_flags)}")
    print(f"  Total logically deducible reveals: {len(certain_reveals)}")

    # Verify correctness: logically deduced flags should be actual mines
    for r, c in certain_flags:
        if game._board[r][c] != -1:
            print(f"  ERROR: Flagged ({r},{c}) but it's not a mine!")
    for r, c in certain_reveals:
        if game._board[r][c] == -1:
            print(f"  ERROR: Revealed ({r},{c}) but it IS a mine!")


def test_flag_overflow():
    """Test that flagging more than total mines gets penalized"""
    print("\n" + "=" * 70)
    print("FLAG OVERFLOW TEST")
    print("=" * 70)

    seed = 42
    game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=seed)
    game.do_action({"type": "reveal", "row": 3, "col": 3})

    # Flag 5 cells (= total mines)
    mines = [(r, c) for r in range(6) for c in range(6) if game._board[r][c] == -1]
    move_history = [{"type": "reveal", "row": 3, "col": 3}]

    for i, (mr, mc) in enumerate(mines):
        game.do_action({"type": "flag", "row": mr, "col": mc})
        move_history.append({"type": "flag", "row": mr, "col": mc})

    # Now try to flag one more (overflow)
    unrevealed = [(r, c) for r in range(6) for c in range(6)
                  if (r, c) not in game._revealed and (r, c) not in game._flagged]
    if unrevealed:
        extra_r, extra_c = unrevealed[0]
        response = json.dumps({"type": "flag", "row": extra_r, "col": extra_c})
        completions = [[{"content": response}]]

        scores = gameplay_reward(
            completions,
            seed=[seed],
            move_history=[json.dumps(move_history)],
            game_rows=[6],
            game_cols=[6],
            game_mines=[5],
        )

        print(f"  Flagging with {len(mines)} already flagged (max {5}):")
        print(f"  Score: {scores[0]} (expected: -20 = -10 wrong flag + -10 overflow)")


if __name__ == "__main__":
    test_reward_functions()
    test_logical_deduction()
    test_flag_overflow()
