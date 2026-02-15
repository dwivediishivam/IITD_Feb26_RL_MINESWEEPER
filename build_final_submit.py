#!/usr/bin/env python3
"""Build final_submit.ipynb"""
import json, re

with open("/Users/tusharchandra/Downloads/AMD_Hack_Initial_State_Backup/final_submit.py") as f:
    content = f.read()

pattern = r'# ################################################################\n# (.+?)\n# ################################################################'
parts = re.split(pattern, content)

markdown_cells = {
    "CELL 0: WRITE AGENT FILES (run this FIRST - takes 5 seconds)": """# Step 1: Write Agent Files

Run this cell first. Writes `agents/minesweeper_agent.py` and `agents/minesweeper_model.py`.

**Prompt features:**
- Constraint logic with worked example
- Full 12-criteria scoring schedule
- VALID TARGETS list (enumerates all '.' cells)
- No fine-tuning needed, no post-processing""",

    "CELL 1: QUICK TEST (run this to verify - takes ~5 min)": """# Step 2: Quick Test

Loads base Qwen2.5-14B-Instruct and plays 11 games (8x8, 10x10, 6x10, 16x16).
If scores look good, submit the agents/ folder.""",
}

cells = []
def add_md(s): cells.append({"cell_type":"markdown","metadata":{},"source":[s]})
def add_code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s.strip()]})

add_md("# Final Submission - Base Qwen + Expert Prompt\n\n20-minute sprint. No fine-tuning. Best possible prompt.")

i = 1
while i < len(parts):
    if i+1 >= len(parts): break
    desc = parts[i].strip()
    code = parts[i+1].strip()
    add_md(markdown_cells.get(desc, f"# {desc}"))
    if code:
        lines = code.split('\n')
        while lines and not lines[0].strip(): lines.pop(0)
        while lines and not lines[-1].strip(): lines.pop()
        t = '\n'.join(lines)
        if t.strip(): add_code(t)
    i += 2

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12.11"}},"nbformat":4,"nbformat_minor":4}
out = "/Users/tusharchandra/Downloads/AMD_Hack_Initial_State_Backup/final_submit.ipynb"
with open(out,"w") as f: json.dump(nb, f, indent=1)
print(f"Notebook: {out}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} md + {sum(1 for c in cells if c['cell_type']=='code')} code)")
