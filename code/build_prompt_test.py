#!/usr/bin/env python3
"""Build prompt_test.ipynb from prompt_test.py"""
import json
import re

with open("/Users/tusharchandra/Downloads/AMD_Hack_Initial_State_Backup/prompt_test.py") as f:
    content = f.read()

pattern = r'# ################################################################\n# (.+?)\n# ################################################################'
parts = re.split(pattern, content)

markdown_cells = {
    "CELL 0: IMPORTS AND SETUP": "# Prompt Strategy Comparison Test\n\n**24 configs**: 6 prompt strategies x 4 models. No post-processing.",

    "CELL 1: MINESWEEPER GAME CLASS (DO NOT MODIFY)": "# Minesweeper Game Class\n\n**DO NOT MODIFY** - must match evaluation environment.",

    "CELL 2: HELPER FUNCTIONS": "# Helper Functions\n\nCompact prompt builder, JSON parser, logical deduction checker.",

    "CELL 3: SCORING RULES AND 6 PROMPT STRATEGIES": """# 6 Prompt Strategies + Scoring Rules

| # | Strategy | Key Idea |
|---|----------|----------|
| V1 | Simple | "You output JSON actions" |
| V2 | Constraint | Count flags/unknowns logic |
| V3 | Aggressive | "DO NOT DO NOT DO NOT" x10, TRIPLE CHECK |
| V4 | Rules list | STEP 1-5 verification process |
| V5 | Annotated | Valid target cells listed: `(2,3) (2,4)...` |
| V6 | CoT verify | Self-check: "Cell shows: . Is it dot? YES" |

For BASE model only: full 12-criteria scoring schedule added to every prompt.""",

    "CELL 4: GAME RUNNER + TEST LOOP (24 configs: 6 strategies x 4 models)": """# Run All 12 Tests (2 base models x 6 strategies)

**2 Base Models (no fine-tuning):**
- `BASE-QWEN-14B`: Qwen2.5-14B-Instruct (14B dense, instruction-tuned)
- `BASE-OSS-20B`: gpt-oss-20b (20B MoE, 3.6B active, NOT instruction-tuned)

Both get the full **12-criteria scoring schedule** injected into every prompt.

**4 Board configs**: 8x8(x3), 10x10(x3), 6x10(x3), 16x16(x2) = 11 games per config

**Total: 12 configs x 11 games = 132 games**""",

    "CELL 5: RESULTS TABLE AND RANKING": """# Results: Grand Ranking

Sorted by average score. Shows best strategy per model and overall winner.""",
}

cells = []

def add_md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source.strip()],
    })

add_md("# Prompt Strategy Comparison: 6 Strategies x 2 Base Models\n\n"
       "Test which base model + prompt combo works best **without any fine-tuning**.\n\n"
       "**No post-processing** - only prompt changes (within competition rules).\n\n"
       "Both models get the full 12-criteria scoring schedule in the prompt.\n\n"
       "Models: Qwen2.5-14B-Instruct (dense) vs gpt-oss-20b (MoE)")

i = 1
while i < len(parts):
    if i + 1 >= len(parts):
        break
    cell_desc = parts[i].strip()
    cell_code = parts[i + 1].strip()
    if cell_desc in markdown_cells:
        add_md(markdown_cells[cell_desc])
    else:
        add_md(f"# {cell_desc}")
    if cell_code:
        code_lines = cell_code.split('\n')
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        code_text = '\n'.join(code_lines)
        if code_text.strip():
            add_code(code_text)
    i += 2

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
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = "/Users/tusharchandra/Downloads/AMD_Hack_Initial_State_Backup/prompt_test.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook: {output_path}")
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")
for i, cell in enumerate(cells):
    ctype = cell['cell_type']
    src = cell['source'][0] if cell['source'] else ""
    first_line = src.split('\n')[0][:80]
    print(f"  Cell {i}: [{ctype:8s}] {first_line}")
