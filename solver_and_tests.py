#!/usr/bin/env python3
"""
Minesweeper Expert Solver + Test Suite
Generates expert training data for SFT phase.

Implements:
1. Single-cell constraint propagation
2. Coupled constraint analysis (pair-wise)
3. Global mine count constraint
4. Probability-based guessing when no certain move exists
"""
import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
import itertools

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
# Expert Minesweeper Solver
# ============================================================
class MinesweeperSolver:
    """
    Expert solver using constraint propagation + coupled constraints.
    Generates optimal training data for LLM fine-tuning.
    """

    def get_neighbors(self, r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
        """Get all valid neighbor coordinates"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append((nr, nc))
        return neighbors

    def analyze_board(self, board: List[List[str]], rows: int, cols: int,
                      num_mines: int, num_flagged: int) -> Dict:
        """
        Full constraint analysis of the board.
        Returns sets of certain_flags, certain_reveals, and probabilities.
        Optimized with spatial indexing for large boards.
        """
        certain_flags = set()
        certain_reveals = set()

        # Pre-compute: find frontier numbered cells (those adjacent to unrevealed)
        # This avoids scanning the entire board repeatedly
        frontier_cells = []
        for r in range(rows):
            for c in range(cols):
                cell = board[r][c]
                if cell not in '12345678':
                    continue
                neighbors = self.get_neighbors(r, c, rows, cols)
                has_unrevealed = any(board[nr][nc] == '.' for nr, nc in neighbors)
                if has_unrevealed:
                    frontier_cells.append((r, c))

        # Phase 1: Single-cell constraint propagation
        changed = True
        iterations = 0
        while changed and iterations < 100:
            changed = False
            iterations += 1
            for r, c in frontier_cells:
                num = int(board[r][c])
                neighbors = self.get_neighbors(r, c, rows, cols)

                flagged_n = []
                unrevealed_n = []
                for nr, nc in neighbors:
                    if board[nr][nc] == 'F' or (nr, nc) in certain_flags:
                        flagged_n.append((nr, nc))
                    elif board[nr][nc] == '.' and (nr, nc) not in certain_flags and (nr, nc) not in certain_reveals:
                        unrevealed_n.append((nr, nc))

                remaining_mines = num - len(flagged_n)
                if remaining_mines < 0:
                    continue

                if remaining_mines == len(unrevealed_n) and len(unrevealed_n) > 0:
                    for nr, nc in unrevealed_n:
                        if (nr, nc) not in certain_flags:
                            certain_flags.add((nr, nc))
                            changed = True

                if remaining_mines == 0 and len(unrevealed_n) > 0:
                    for nr, nc in unrevealed_n:
                        if (nr, nc) not in certain_reveals:
                            certain_reveals.add((nr, nc))
                            changed = True

        # Phase 2: Coupled constraint analysis (pair-wise)
        # Use spatial grid to only compare nearby cells
        from collections import defaultdict as dd
        grid_index = dd(list)
        for r, c in frontier_cells:
            grid_index[(r // 3, c // 3)].append((r, c))

        changed = True
        iterations = 0
        while changed and iterations < 50:
            changed = False
            iterations += 1
            for (gr, gc), cells in grid_index.items():
                # Check cells in this grid block and adjacent blocks
                nearby_cells = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nearby_cells.extend(grid_index.get((gr + dr, gc + dc), []))

                for i, (r1, c1) in enumerate(cells):
                    num1 = int(board[r1][c1])
                    n1_neighbors = self.get_neighbors(r1, c1, rows, cols)
                    flagged1 = sum(1 for nr, nc in n1_neighbors
                                  if board[nr][nc] == 'F' or (nr, nc) in certain_flags)
                    unrevealed1 = set()
                    for nr, nc in n1_neighbors:
                        if board[nr][nc] == '.' and (nr, nc) not in certain_flags and (nr, nc) not in certain_reveals:
                            unrevealed1.add((nr, nc))
                    remaining1 = num1 - flagged1
                    if not unrevealed1:
                        continue

                    for r2, c2 in nearby_cells:
                        if (r1, c1) >= (r2, c2):
                            continue
                        if abs(r1 - r2) > 2 or abs(c1 - c2) > 2:
                            continue

                        num2 = int(board[r2][c2])
                        n2_neighbors = self.get_neighbors(r2, c2, rows, cols)
                        flagged2 = sum(1 for nr, nc in n2_neighbors
                                      if board[nr][nc] == 'F' or (nr, nc) in certain_flags)
                        unrevealed2 = set()
                        for nr, nc in n2_neighbors:
                            if board[nr][nc] == '.' and (nr, nc) not in certain_flags and (nr, nc) not in certain_reveals:
                                unrevealed2.add((nr, nc))
                        remaining2 = num2 - flagged2
                        if not unrevealed2:
                            continue

                        # Check both subset directions
                        if unrevealed1.issubset(unrevealed2):
                            diff = unrevealed2 - unrevealed1
                            diff_mines = remaining2 - remaining1
                            if diff and diff_mines == len(diff):
                                for cell in diff:
                                    if cell not in certain_flags:
                                        certain_flags.add(cell)
                                        changed = True
                            elif diff and diff_mines == 0:
                                for cell in diff:
                                    if cell not in certain_reveals:
                                        certain_reveals.add(cell)
                                        changed = True

                        if unrevealed2.issubset(unrevealed1):
                            diff = unrevealed1 - unrevealed2
                            diff_mines = remaining1 - remaining2
                            if diff and diff_mines == len(diff):
                                for cell in diff:
                                    if cell not in certain_flags:
                                        certain_flags.add(cell)
                                        changed = True
                            elif diff and diff_mines == 0:
                                for cell in diff:
                                    if cell not in certain_reveals:
                                        certain_reveals.add(cell)
                                        changed = True

        certain_flags -= certain_reveals

        return {
            "certain_flags": certain_flags,
            "certain_reveals": certain_reveals,
        }

    def estimate_probabilities(self, board: List[List[str]], rows: int, cols: int,
                               num_mines: int, certain_flags: set, certain_reveals: set) -> Dict[Tuple[int, int], float]:
        """
        Estimate mine probability for unrevealed cells.
        Uses local constraint information + global mine count.
        """
        # Count current flags
        current_flags = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == 'F')
        total_flagged = current_flags + len(certain_flags)
        remaining_mines = num_mines - total_flagged

        # Get all unrevealed, non-flagged, non-certain cells
        uncertain_cells = set()
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == '.' and (r, c) not in certain_flags and (r, c) not in certain_reveals:
                    uncertain_cells.add((r, c))

        if not uncertain_cells:
            return {}

        # Global probability
        if len(uncertain_cells) > 0:
            global_prob = max(0, remaining_mines) / len(uncertain_cells)
        else:
            global_prob = 0

        # Local probability based on neighboring constraints
        probs = {}
        constraint_counts = defaultdict(list)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] not in '12345678':
                    continue
                num = int(board[r][c])
                neighbors = self.get_neighbors(r, c, rows, cols)
                flagged_n = sum(1 for nr, nc in neighbors
                               if board[nr][nc] == 'F' or (nr, nc) in certain_flags)
                unrevealed_n = [(nr, nc) for nr, nc in neighbors
                                if (nr, nc) in uncertain_cells]

                if unrevealed_n:
                    remaining = num - flagged_n
                    local_prob = max(0, min(1, remaining / len(unrevealed_n)))
                    for nr, nc in unrevealed_n:
                        constraint_counts[(nr, nc)].append(local_prob)

        for cell in uncertain_cells:
            if cell in constraint_counts:
                # Average of local constraint probabilities
                probs[cell] = sum(constraint_counts[cell]) / len(constraint_counts[cell])
            else:
                # Interior cell (not adjacent to any number) - use global probability
                probs[cell] = global_prob

        return probs

    def get_best_action(self, game: MinesweeperGame) -> Optional[Dict]:
        """
        Get the best action for the current game state.
        Returns (action_dict, is_logical) where is_logical indicates
        whether the move was deduced logically vs guessed.
        """
        board = game.get_visible_board()
        rows, cols = game.rows, game.cols
        num_mines = game.num_mines
        num_flagged = len(game._flagged)

        analysis = self.analyze_board(board, rows, cols, num_mines, num_flagged)
        certain_flags = analysis["certain_flags"]
        certain_reveals = analysis["certain_reveals"]

        # Priority 1: Flag certain mines (highest value: +15 each)
        if certain_flags:
            r, c = min(certain_flags)  # Deterministic ordering
            return {"type": "flag", "row": r, "col": c}, True

        # Priority 2: Reveal certain safe cells (+10-15 each)
        if certain_reveals:
            r, c = min(certain_reveals)
            return {"type": "reveal", "row": r, "col": c}, True

        # Priority 3: Probability-based guess
        probs = self.estimate_probabilities(board, rows, cols, num_mines,
                                            certain_flags, certain_reveals)

        if probs:
            # Pick cell with LOWEST mine probability
            safest_cell = min(probs.keys(), key=lambda k: (probs[k], k))
            return {"type": "reveal", "row": safest_cell[0], "col": safest_cell[1]}, False

        # Fallback: pick any unrevealed cell
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == '.':
                    return {"type": "reveal", "row": r, "col": c}, False

        return None, False

    def play_full_game(self, game: MinesweeperGame, first_move: Optional[Dict] = None) -> List[Dict]:
        """
        Play a complete game using the solver.
        Returns list of (action, is_logical, result) tuples.
        """
        trajectory = []
        max_moves = 2 * game.rows * game.cols

        # Make first move if provided
        if first_move:
            result = game.do_action(first_move)
            trajectory.append({
                "action": first_move,
                "is_logical": False,  # First move is given
                "result": result
            })
            if game.state() != "ongoing":
                return trajectory

        move_count = 0
        while game.state() == "ongoing" and move_count < max_moves:
            action, is_logical = self.get_best_action(game)
            if action is None:
                break

            result = game.do_action(action)
            trajectory.append({
                "action": action,
                "is_logical": is_logical,
                "result": result
            })
            move_count += 1

        return trajectory


# ============================================================
# Board Representation Analysis
# ============================================================
def compact_board_repr(board: List[List[str]], rows: int, cols: int) -> str:
    """Ultra-compact board representation: one char per cell, row per line"""
    lines = []
    for r in range(rows):
        lines.append(f"{r:>2}|{''.join(board[r])}")
    return "\n".join(lines)


def json_board_repr(board: List[List[str]]) -> str:
    """Standard JSON representation"""
    return json.dumps(board, indent=2)


def medium_compact_repr(board: List[List[str]], rows: int, cols: int) -> str:
    """Medium compact: space-separated cells"""
    lines = []
    # Column headers
    header = "   " + " ".join(f"{c}" for c in range(cols))
    lines.append(header)
    for r in range(rows):
        lines.append(f"{r:>2}|" + " ".join(board[r]))
    return "\n".join(lines)


def count_chars(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text"""
    return len(text)


# ============================================================
# Prompt Templates
# ============================================================
def build_compact_prompt(game: MinesweeperGame) -> str:
    """Compact prompt optimized for token efficiency"""
    board = game.get_visible_board()
    board_str = compact_board_repr(board, game.rows, game.cols)

    prompt = f"""Minesweeper {game.rows}x{game.cols}, {game.num_mines} mines, {len(game._flagged)} flagged, {len(game._revealed)} revealed.
.=unknown F=flag 0-8=adjacent mines

{board_str}

JSON action:"""
    return prompt


def build_full_prompt(game: MinesweeperGame) -> str:
    """Full JSON prompt (from original notebook)"""
    state = {
        "board": game.get_visible_board(),
        "rows": game.rows,
        "cols": game.cols,
        "mines": game.num_mines,
        "flags_placed": len(game._flagged),
        "cells_revealed": len(game._revealed),
    }

    prompt = f"""You are playing Minesweeper. Analyze the game state and output your next move.

You must output ONLY a valid JSON object. No explanation, no analysis, no text.

Start your response immediately with {{ and end with }}.

Do NOT output cell which is already revealed or flagged in the current state.

Game state:
{json.dumps(state, indent=2)}

Legend:
- "." = unrevealed cell
- "F" = flagged cell (suspected mine)
- "0"-"8" = number of adjacent mines
- "*" = revealed mine (game over)

Output your next action as JSON:
{{"type": "reveal", "row": <row_index>, "col": <col_index>}}
or
{{"type": "flag", "row": <row_index>, "col": <col_index>}}

Your action:"""
    return prompt


# ============================================================
# Tests
# ============================================================
def test_solver_win_rates():
    """Test solver on various board sizes and report win rates"""
    print("=" * 70)
    print("SOLVER WIN RATE TESTING")
    print("=" * 70)

    configs = [
        (5, 5, 3, "Easy 5x5", 200),
        (6, 6, 5, "Standard 6x6", 200),
        (8, 8, 10, "Medium 8x8", 100),
        (10, 10, 15, "Hard 10x10", 100),
        (16, 16, 40, "Expert 16x16", 50),
        (20, 20, 60, "Large 20x20", 30),
        (30, 30, 130, "XL 30x30 ~14%", 20),
        (50, 50, 250, "XXL 50x50 10%", 10),
        (50, 50, 500, "XXL 50x50 20%", 10),
    ]

    solver = MinesweeperSolver()

    for rows, cols, mines, label, num_games in configs:
        wins = 0
        total_moves = 0
        logical_moves = 0
        guess_moves = 0
        mine_hits = 0

        for seed in range(num_games):
            try:
                game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)

                # First move: pick a safe corner or random cell
                first_r, first_c = 0, 0
                first_action = {"type": "reveal", "row": first_r, "col": first_c}
                trajectory = solver.play_full_game(game, first_action)

                for step in trajectory:
                    total_moves += 1
                    if step["is_logical"]:
                        logical_moves += 1
                    else:
                        guess_moves += 1
                    if step["result"] == "mine":
                        mine_hits += 1

                if game.state() == "success":
                    wins += 1
            except Exception as e:
                pass  # Skip errored games

        avg_moves = total_moves / num_games if num_games > 0 else 0
        logical_pct = logical_moves / total_moves * 100 if total_moves > 0 else 0

        print(f"\n{label} ({rows}x{cols}, {mines} mines):")
        print(f"  Win rate: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
        print(f"  Avg moves/game: {avg_moves:.1f}")
        print(f"  Logical moves: {logical_pct:.1f}% ({logical_moves}/{total_moves})")
        print(f"  Mine hits: {mine_hits}")


def test_token_counts():
    """Analyze token counts for different board sizes and representations"""
    print("\n" + "=" * 70)
    print("TOKEN COUNT ANALYSIS")
    print("=" * 70)

    configs = [
        (6, 6, 5),
        (8, 8, 10),
        (10, 10, 15),
        (16, 16, 40),
        (20, 20, 60),
        (30, 30, 130),
        (50, 50, 250),
        (50, 50, 500),
    ]

    for rows, cols, mines in configs:
        game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=42)
        # Make a few moves to get a realistic board
        solver = MinesweeperSolver()
        first_action = {"type": "reveal", "row": 0, "col": 0}
        game.do_action(first_action)

        # Get prompts
        compact_prompt = build_compact_prompt(game)
        full_prompt = build_full_prompt(game)

        # Character counts (rough estimate: 1 token ≈ 3.5-4 chars)
        compact_chars = len(compact_prompt)
        full_chars = len(full_prompt)

        compact_tokens_est = compact_chars / 3.5
        full_tokens_est = full_chars / 3.5

        print(f"\n{rows}x{cols} ({mines} mines):")
        print(f"  Compact: {compact_chars:>6} chars ≈ {compact_tokens_est:>6.0f} tokens")
        print(f"  Full JSON: {full_chars:>6} chars ≈ {full_tokens_est:>6.0f} tokens")
        print(f"  Savings: {(1 - compact_chars/full_chars)*100:.1f}%")


def test_solver_trajectory_quality():
    """Analyze quality of solver trajectories for training data"""
    print("\n" + "=" * 70)
    print("TRAJECTORY QUALITY ANALYSIS")
    print("=" * 70)

    solver = MinesweeperSolver()
    rows, cols, mines = 8, 8, 10
    num_games = 50

    action_type_counts = defaultdict(int)
    logical_by_type = defaultdict(lambda: [0, 0])  # [logical, total]
    result_counts = defaultdict(int)

    for seed in range(num_games):
        game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
        first_action = {"type": "reveal", "row": 0, "col": 0}
        trajectory = solver.play_full_game(game, first_action)

        for step in trajectory:
            action = step["action"]
            action_type_counts[action["type"]] += 1
            logical_by_type[action["type"]][1] += 1
            if step["is_logical"]:
                logical_by_type[action["type"]][0] += 1
            result_counts[step["result"]] += 1

    print(f"\nBoard: {rows}x{cols}, {mines} mines, {num_games} games:")
    print(f"\nAction type distribution:")
    for atype, count in sorted(action_type_counts.items()):
        logical, total = logical_by_type[atype]
        print(f"  {atype}: {count} ({logical}/{total} logical = {logical/total*100:.1f}%)")

    print(f"\nResult distribution:")
    for result, count in sorted(result_counts.items()):
        print(f"  {result}: {count}")


def test_scoring_simulation():
    """Simulate scoring based on eval criteria"""
    print("\n" + "=" * 70)
    print("SCORING SIMULATION")
    print("=" * 70)

    solver = MinesweeperSolver()
    configs = [
        (6, 6, 5, "6x6"),
        (10, 10, 15, "10x10"),
        (20, 20, 60, "20x20"),
        (50, 50, 250, "50x50 10%"),
        (50, 50, 500, "50x50 20%"),
    ]

    for rows, cols, mines, label in configs:
        total_score = 0
        num_games = 50

        for seed in range(num_games):
            score = 0
            game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
            first_action = {"type": "reveal", "row": 0, "col": 0}
            trajectory = solver.play_full_game(game, first_action)

            for step in trajectory:
                action = step["action"]
                result = step["result"]
                is_logical = step["is_logical"]

                if result == "ok":
                    if action["type"] == "flag":
                        # Check if it's actually a mine
                        r, c = action["row"], action["col"]
                        if game._board[r][c] == -1:
                            score += 15  # Flag correct mine
                        else:
                            score -= 10  # Flag non-mine
                    elif action["type"] == "reveal":
                        score += 15 if is_logical else 10  # Safe reveal
                elif result == "mine":
                    score -= 25  # Hit mine
                elif result == "win":
                    # Last move scored + win bonus
                    if action["type"] == "reveal":
                        score += 15 if is_logical else 10
                    elif action["type"] == "flag":
                        score += 15
                    score += 100  # Win bonus

            total_score += score

        avg_score = total_score / num_games
        print(f"\n{label} ({rows}x{cols}, {mines} mines):")
        print(f"  Average score: {avg_score:.1f} points")


def test_edge_cases():
    """Test solver on edge cases"""
    print("\n" + "=" * 70)
    print("EDGE CASE TESTING")
    print("=" * 70)

    solver = MinesweeperSolver()

    # Test 1: Very small board
    print("\nTest 1: 4x4 board with 2 mines")
    game = MinesweeperGame(rows=4, cols=4, num_mines=2, seed=42)
    first_action = {"type": "reveal", "row": 0, "col": 0}
    trajectory = solver.play_full_game(game, first_action)
    print(f"  Result: {game.state()}, Moves: {len(trajectory)}")
    print(f"  Final board:")
    print(f"  {game.pretty_print()}")

    # Test 2: Dense mines (20%)
    print("\nTest 2: 10x10 board with 20 mines (20%)")
    game = MinesweeperGame(rows=10, cols=10, num_mines=20, seed=42)
    first_action = {"type": "reveal", "row": 5, "col": 5}
    trajectory = solver.play_full_game(game, first_action)
    print(f"  Result: {game.state()}, Moves: {len(trajectory)}")

    # Test 3: Sparse mines (10%)
    print("\nTest 3: 10x10 board with 10 mines (10%)")
    game = MinesweeperGame(rows=10, cols=10, num_mines=10, seed=42)
    first_action = {"type": "reveal", "row": 5, "col": 5}
    trajectory = solver.play_full_game(game, first_action)
    print(f"  Result: {game.state()}, Moves: {len(trajectory)}")

    # Test 4: Max size
    print("\nTest 4: 50x50 board with 500 mines (20%)")
    start = time.time()
    game = MinesweeperGame(rows=50, cols=50, num_mines=500, seed=42)
    first_action = {"type": "reveal", "row": 25, "col": 25}
    trajectory = solver.play_full_game(game, first_action)
    elapsed = time.time() - start
    print(f"  Result: {game.state()}, Moves: {len(trajectory)}, Time: {elapsed:.2f}s")


def test_compact_prompt_examples():
    """Show example compact prompts"""
    print("\n" + "=" * 70)
    print("COMPACT PROMPT EXAMPLES")
    print("=" * 70)

    # Small board example
    game = MinesweeperGame(rows=6, cols=6, num_mines=5, seed=42)
    game.do_action({"type": "reveal", "row": 3, "col": 3})
    print("\n--- 6x6 Compact Prompt ---")
    print(build_compact_prompt(game))
    print("\nExpected output: {\"type\": \"reveal\", \"row\": X, \"col\": Y}")
    print(f"Prompt length: {len(build_compact_prompt(game))} chars")

    # Large board
    game = MinesweeperGame(rows=20, cols=20, num_mines=40, seed=42)
    game.do_action({"type": "reveal", "row": 10, "col": 10})
    print("\n--- 20x20 Compact Prompt ---")
    print(build_compact_prompt(game))
    print(f"Prompt length: {len(build_compact_prompt(game))} chars")


if __name__ == "__main__":
    print("Running Minesweeper Solver Test Suite...")
    print("This validates our expert solver for training data generation.\n")

    test_solver_win_rates()
    test_token_counts()
    test_solver_trajectory_quality()
    test_scoring_simulation()
    test_edge_cases()
    test_compact_prompt_examples()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
