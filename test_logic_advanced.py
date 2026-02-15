#!/usr/bin/env python3
"""Test logical deduction on board states where deductions ARE possible"""
import json
import random
from reward_function_complete import (
    MinesweeperGame, is_logically_deducible, get_neighbors
)
from solver_and_tests import MinesweeperSolver

def test_deduction_on_mid_game():
    """Create mid-game states where logical deduction is possible"""
    solver = MinesweeperSolver()

    print("Testing logical deduction on mid-game states...\n")

    for seed in range(20):
        game = MinesweeperGame(rows=8, cols=8, num_mines=10, seed=seed)

        # Play a few moves to create a revealing board
        first_action = {"type": "reveal", "row": 0, "col": 0}
        game.do_action(first_action)

        if game.state() != "ongoing":
            continue

        # Play more moves using solver
        for _ in range(5):
            if game.state() != "ongoing":
                break
            action, is_logical = solver.get_best_action(game)
            if action:
                game.do_action(action)

        if game.state() != "ongoing":
            continue

        board = game.get_visible_board()

        # Check for logically deducible moves
        deducible_flags = []
        deducible_reveals = []

        for r in range(8):
            for c in range(8):
                if board[r][c] == '.':
                    if is_logically_deducible(board, 8, 8, "flag", r, c):
                        is_mine = game._board[r][c] == -1
                        deducible_flags.append(((r, c), is_mine))
                    if is_logically_deducible(board, 8, 8, "reveal", r, c):
                        is_mine = game._board[r][c] == -1
                        deducible_reveals.append(((r, c), is_mine))

        if deducible_flags or deducible_reveals:
            print(f"Seed {seed}: {len(deducible_flags)} deducible flags, {len(deducible_reveals)} deducible reveals")

            # Verify correctness
            flag_errors = sum(1 for _, is_mine in deducible_flags if not is_mine)
            reveal_errors = sum(1 for _, is_mine in deducible_reveals if is_mine)

            if flag_errors > 0:
                print(f"  ERROR: {flag_errors} flag deductions are WRONG!")
                print(game.pretty_print())
            if reveal_errors > 0:
                print(f"  ERROR: {reveal_errors} reveal deductions are WRONG!")
                print(game.pretty_print())

            if flag_errors == 0 and reveal_errors == 0:
                print(f"  All deductions verified CORRECT")


def test_deduction_vs_solver():
    """Compare logical deduction checker with solver's analysis"""
    solver = MinesweeperSolver()

    print("\nComparing deduction checker vs solver on 50 games...\n")

    total_deduction_matches = 0
    total_solver_finds = 0
    total_deduction_finds = 0

    for seed in range(50):
        game = MinesweeperGame(rows=8, cols=8, num_mines=10, seed=seed)
        game.do_action({"type": "reveal", "row": 4, "col": 4})

        if game.state() != "ongoing":
            continue

        board = game.get_visible_board()

        # Solver analysis
        analysis = solver.analyze_board(board, 8, 8, 10, len(game._flagged))
        solver_flags = analysis["certain_flags"]
        solver_reveals = analysis["certain_reveals"]

        # Deduction checker
        deduction_flags = set()
        deduction_reveals = set()
        for r in range(8):
            for c in range(8):
                if board[r][c] == '.':
                    if is_logically_deducible(board, 8, 8, "flag", r, c):
                        deduction_flags.add((r, c))
                    if is_logically_deducible(board, 8, 8, "reveal", r, c):
                        deduction_reveals.add((r, c))

        # Compare
        total_solver_finds += len(solver_flags) + len(solver_reveals)
        total_deduction_finds += len(deduction_flags) + len(deduction_reveals)
        total_deduction_matches += len(solver_flags & deduction_flags) + len(solver_reveals & deduction_reveals)

    print(f"Solver found: {total_solver_finds} certain moves")
    print(f"Deduction checker found: {total_deduction_finds} certain moves")
    print(f"Overlap: {total_deduction_matches}")
    if total_solver_finds > 0:
        print(f"Deduction/Solver agreement: {total_deduction_matches/total_solver_finds*100:.1f}%")


def test_training_data_snapshot():
    """Show what a training example looks like with compact prompt"""
    solver = MinesweeperSolver()

    print("\n" + "="*70)
    print("SAMPLE TRAINING DATA (SFT format)")
    print("="*70)

    game = MinesweeperGame(rows=8, cols=8, num_mines=10, seed=5)
    game.do_action({"type": "reveal", "row": 4, "col": 4})

    if game.state() != "ongoing":
        print("Game already over from first move")
        return

    board = game.get_visible_board()
    print("\nBoard:")
    print(game.pretty_print())

    # Get solver's recommended action
    action, is_logical = solver.get_best_action(game)

    # Build compact prompt
    board_lines = []
    for r in range(8):
        board_lines.append(f"{r:>2}|{''.join(board[r])}")
    board_str = "\n".join(board_lines)

    prompt = f"""Minesweeper 8x8, 10 mines, 0 flagged, {len(game._revealed)} revealed.
.=unknown F=flag 0-8=adjacent mines

{board_str}

JSON action:"""

    response = json.dumps(action)

    print(f"\n--- PROMPT ({len(prompt)} chars) ---")
    print(prompt)
    print(f"\n--- EXPECTED RESPONSE ---")
    print(response)
    print(f"\n--- META ---")
    print(f"Is logical: {is_logical}")
    print(f"Action: {action}")

    # Verify with actual game
    result = game.do_action(action)
    print(f"Result: {result}")
    print(f"Game state after: {game.state()}")


if __name__ == "__main__":
    test_deduction_on_mid_game()
    test_deduction_vs_solver()
    test_training_data_snapshot()
