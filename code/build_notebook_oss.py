#!/usr/bin/env python3
"""
Build .ipynb notebook from minesweeper_pipeline_oss.py (gpt-oss-20b version)
"""
import json
import re

with open("/Users/tusharchandra/Downloads/AMD_Hack_Initial_State_Backup/minesweeper_pipeline_oss.py") as f:
    content = f.read()

pattern = r'# ################################################################\n# (.+?)\n# ################################################################'
parts = re.split(pattern, content)

markdown_cells = {
    "CELL 0: SETUP - Imports & Model Discovery": "# Setup & Imports",

    "CELL 1: LOAD GPT-OSS-20B MODEL WITH UNSLOTH": """# Load gpt-oss-20b Model

**gpt-oss-20b**: MoE model (20B total, 3.6B active params)
- NOT instruction-tuned → needs more SFT training (2 epochs)
- BF16 precision (no 4-bit quantization for MoE)
- LoRA rank 64 for more capacity""",

    "CELL 2: ADD LoRA ADAPTERS": """# Add LoRA Adapters (r=64)

Higher rank LoRA for non-instruct model:""",

    "CELL 3: MINESWEEPER GAME CLASS (DO NOT MODIFY)": """# Minesweeper Game Implementation

**DO NOT MODIFY** - must match evaluation environment.""",

    "CELL 4: EXPERT SOLVER + COMPACT PROMPT + PARSER + DEDUCTION": """# Expert Solver + Compact Prompt + JSON Parser

System prompt includes constraint logic instructions for better reasoning.""",

    "CELL 5: TEST BASE MODEL BEFORE TRAINING (optional)": "# Test Base Model (optional)",

    "CELL 6: GENERATE SFT TRAINING DATASET": """# Generate SFT Training Data

10,000 expert-solved examples across all board sizes.""",

    "CELL 7: SFT TRAINING": """# SFT Training (2 epochs for non-instruct model)

More epochs needed since gpt-oss-20b is not instruction-tuned.""",

    "CELL 8: SAVE SFT CHECKPOINT + QUICK EVALUATION": "# Save SFT Checkpoint + Evaluation",

    "CELL 9: GRPO REWARD FUNCTIONS (all 12 eval criteria)": """# GRPO Reward Functions (v3 - Lessons from Qwen)

**Key fix: frontier bonus no longer cancels random penalty**
- Mine hit: **-100** (4x penalty)
- Random reveal: **-15** (no frontier bonus!)
- Logical reveal: **+30** (+5 frontier bonus)
- Correct flag: **+30**
- Invalid JSON: **-25** (harsh)
- Win: **+200**""",

    "CELL 10: GENERATE GRPO TRAINING DATASET": "# Generate GRPO Training Dataset",

    "CELL 11: GRPO TRAINING": """# GRPO Training (v3)

- `beta=0.04`, `temperature=0.7`
- `learning_rate=5e-6`, `max_grad_norm=0.5`
- 300 steps (SFT with CoT is the main driver)""",

    "CELL 12: FINAL EVALUATION": "# Final Evaluation",

    "CELL 13: SAVE FINAL MODEL": """# Save Model

Saved to `your_fine_tuned_model_oss/` (separate from Qwen model).""",

    "CELL 14: UPDATE INFERENCE AGENT FILES": """# Update Agent Files

Points to gpt-oss-20b model with constraint-logic prompt.""",

    "CELL 15: DETAILED EVALUATION WITH SCORING BREAKDOWN": """# Detailed Evaluation - Per-Move Breakdown

Track all 12 scoring criteria per move.""",
}

title_md = """# Minesweeper LLM - gpt-oss-20b Pipeline

## Model: gpt-oss-20b (MoE, 3.6B active params)

- **Not instruction-tuned** → 2 SFT epochs
- **BF16** precision, no 4-bit quantization
- **LoRA r=64** for more capacity
- All Phase 2 improvements baked in from the start"""

tips_md = """# Notes

- Model saved to `your_fine_tuned_model_oss/`
- Agent files point to `_oss` model
- Can run simultaneously with Qwen pipeline on separate GPU allocation"""

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

add_md(title_md)

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

add_md(tips_md)

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

output_path = "/Users/tusharchandra/Downloads/AMD_Hack_Initial_State_Backup/minesweeper_oss.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook generated: {output_path}")
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")
for i, cell in enumerate(cells):
    ctype = cell['cell_type']
    src = cell['source'][0] if cell['source'] else ""
    first_line = src.split('\n')[0][:80]
    print(f"  Cell {i}: [{ctype:8s}] {first_line}")
