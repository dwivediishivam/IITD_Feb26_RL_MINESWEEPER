#!/usr/bin/env python3
"""
Verify training data format is correct for SFTTrainer and GRPOTrainer.
Test that prompt sizes fit within context windows.
"""
import json
import random
from collections import defaultdict

# Simulate what the SFT dataset looks like
def simulate_sft_example():
    """Show what a single SFT training example looks like"""
    # Simulate a compact prompt for 50x50 board
    rows, cols, mines = 50, 50, 250
    # Generate fake board (mostly dots with some numbers)
    board = [['.' for _ in range(cols)] for _ in range(rows)]
    # Add some revealed cells
    for r in range(20, 35):
        for c in range(20, 35):
            board[r][c] = str(random.randint(0, 3))
    board[25][25] = 'F'

    board_lines = []
    for r in range(rows):
        board_lines.append(f"{r:>2}|{''.join(board[r])}")
    board_str = "\n".join(board_lines)

    prompt = f"""Minesweeper {rows}x{cols}, {mines} mines, 1 flagged, 225 revealed.
.=unknown F=flag 0-8=adjacent mines

{board_str}

JSON action:"""

    response = '{"type": "flag", "row": 19, "col": 22}'

    return {
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "total_chars": len(prompt) + len(response),
        "prompt_tokens_est": len(prompt) / 3.5,
        "response_tokens_est": len(response) / 3.5,
        "prompt": prompt,
        "response": response,
    }


def simulate_various_sizes():
    """Test all board sizes"""
    configs = [
        (5, 5, 3), (6, 6, 5), (8, 8, 10), (10, 10, 15),
        (16, 16, 40), (20, 20, 60), (30, 30, 130),
        (40, 40, 200), (50, 50, 250), (50, 50, 500),
    ]

    print("Board Size Analysis:")
    print(f"{'Size':<12} {'Prompt Chars':<15} {'Est Tokens':<12} {'Fits 2048?':<12} {'Fits 4096?':<12}")
    print("-" * 63)

    for rows, cols, mines in configs:
        board = [['.' for _ in range(cols)] for _ in range(rows)]
        board_lines = [f"{r:>2}|{''.join(board[r])}" for r in range(rows)]
        board_str = "\n".join(board_lines)
        prompt = f"Minesweeper {rows}x{cols}, {mines} mines, 0 flagged, 0 revealed.\n.=unknown F=flag 0-8=adjacent mines\n\n{board_str}\n\nJSON action:"

        chars = len(prompt)
        tokens_est = chars / 3.5  # Conservative estimate
        fits_2048 = "YES" if tokens_est + 128 < 2048 else "NO"
        fits_4096 = "YES" if tokens_est + 128 < 4096 else "NO"

        print(f"{rows}x{cols:<8} {chars:<15} {tokens_est:<12.0f} {fits_2048:<12} {fits_4096:<12}")


def verify_sft_format():
    """Verify SFT data format matches what SFTTrainer expects"""
    print("\n\nSFT Data Format Verification:")
    print("=" * 50)

    # SFTTrainer expects either:
    # 1. A "text" field (raw text)
    # 2. A "prompt" + "completion" pair (chat format)
    # 3. Messages format [{"role": "user", ...}, {"role": "assistant", ...}]

    example = {
        "prompt": [
            {"role": "system", "content": "You output JSON actions for Minesweeper. No text, only JSON."},
            {"role": "user", "content": 'Minesweeper 8x8, 10 mines, 0 flagged, 5 revealed.\n.=unknown F=flag 0-8=adjacent mines\n\n 0|........\n 1|..1.....\n 2|..12....\n 3|...1....\n 4|........\n 5|........\n 6|........\n 7|........\n\nJSON action:'},
        ],
        "completion": [
            {"role": "assistant", "content": '{"type": "reveal", "row": 0, "col": 0}'},
        ],
    }

    print(f"\nPrompt format: {json.dumps(example['prompt'], indent=2)[:300]}...")
    print(f"\nCompletion format: {json.dumps(example['completion'], indent=2)}")


def verify_grpo_format():
    """Verify GRPO data format matches what GRPOTrainer expects"""
    print("\n\nGRPO Data Format Verification:")
    print("=" * 50)

    # GRPOTrainer expects:
    # - "prompt" field: list of message dicts [{"role": "user", "content": "..."}]
    # - Additional fields passed as kwargs to reward functions

    example = {
        "prompt": [
            {"role": "system", "content": "You output JSON actions for Minesweeper. No text, only JSON."},
            {"role": "user", "content": "Minesweeper 8x8, 10 mines...\n\nJSON action:"},
        ],
        "seed": 42,
        "move_history": '[{"type": "reveal", "row": 4, "col": 4}]',
        "game_rows": 8,
        "game_cols": 8,
        "game_mines": 10,
    }

    print(f"\nPrompt: {json.dumps(example['prompt'], indent=2)[:200]}...")
    print(f"\nMetadata (passed to reward functions):")
    print(f"  seed: {example['seed']}")
    print(f"  move_history: {example['move_history']}")
    print(f"  game_rows: {example['game_rows']}")
    print(f"  game_cols: {example['game_cols']}")
    print(f"  game_mines: {example['game_mines']}")


def check_output_token_budget():
    """Verify the model's output fits in 128 tokens"""
    print("\n\nOutput Token Budget:")
    print("=" * 50)

    outputs = [
        '{"type": "reveal", "row": 0, "col": 0}',
        '{"type": "flag", "row": 49, "col": 49}',
        '{"type": "reveal", "row": 25, "col": 33}',
    ]

    for output in outputs:
        chars = len(output)
        tokens_est = chars / 3.5
        print(f"  '{output}' → {chars} chars ≈ {tokens_est:.0f} tokens (budget: 128)")

    print(f"\n  Maximum output is ~12-15 tokens. Well within 128 limit.")
    print(f"  The danger is verbose reasoning BEFORE the JSON. SFT prevents this.")


if __name__ == "__main__":
    simulate_various_sizes()
    verify_sft_format()
    verify_grpo_format()
    check_output_token_budget()

    print("\n\n50x50 PROMPT EXAMPLE:")
    print("=" * 50)
    ex = simulate_sft_example()
    print(f"Prompt chars: {ex['prompt_chars']}")
    print(f"Response chars: {ex['response_chars']}")
    print(f"Estimated prompt tokens: {ex['prompt_tokens_est']:.0f}")
    print(f"Estimated response tokens: {ex['response_tokens_est']:.0f}")
    print(f"\nFirst 500 chars of prompt:")
    print(ex["prompt"][:500])
    print("...")
    print(f"\nLast 200 chars of prompt:")
    print(ex["prompt"][-200:])
    print(f"\nResponse: {ex['response']}")
