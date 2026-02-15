#!/usr/bin/env python3
"""
FINAL SUBMISSION - 20 minute speed-run
Lightning SFT (1500 samples, 1 epoch, ~10 min) + expert prompt agent
"""

# ################################################################
# CELL 0: SETUP + LOAD MODEL
# ################################################################
import os, glob, json, re, random, gc
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from unsloth import FastLanguageModel

max_seq_length = 4096

# Auto-detect model
model_name = None
for md in [
    "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct",
    "/root/.cache/huggingface/models--google--gemma-3-12b-it",
    "/root/.cache/huggingface/models--Unsloth--Llama-3.1-8B-Instruct",
    "/root/.cache/huggingface/models--unsloth--Llama-3.1-8B-Instruct",
]:
    if os.path.exists(md):
        snaps = sorted(glob.glob(os.path.join(md, "snapshots", "*")))
        if snaps:
            model_name = snaps[-1]
            print(f"Found: {os.path.basename(md)} -> {model_name}")
            break
if model_name is None:
    all_m = sorted(glob.glob("/root/.cache/huggingface/models--*/snapshots/*"))
    model_name = all_m[0] if all_m else "Qwen/Qwen2.5-14B-Instruct"
    print(f"Fallback: {model_name}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded: {model_name}")

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print(f"LoRA added (r=32, alpha=64)")


# ################################################################
# CELL 1: GAME CLASS + SOLVER + HELPERS (compact)
# ################################################################

@dataclass
class MinesweeperGame:
    rows: int; cols: int; num_mines: int; seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _revealed: Set[Tuple[int,int]] = field(init=False, repr=False, default_factory=set)
    _flagged: Set[Tuple[int,int]] = field(init=False, repr=False, default_factory=set)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._board = [[0]*self.cols for _ in range(self.rows)]
        pos = [(r,c) for r in range(self.rows) for c in range(self.cols)]
        for r,c in self._rng.sample(pos, self.num_mines):
            self._board[r][c] = -1
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c]==-1: continue
                ct=0
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr,nc=r+dr,c+dc
                        if 0<=nr<self.rows and 0<=nc<self.cols and self._board[nr][nc]==-1: ct+=1
                self._board[r][c]=ct

    def _reveal_cell(self, row, col):
        if (row,col) in self._revealed or (row,col) in self._flagged: return False
        stack=[(row,col)]
        while stack:
            r,c=stack.pop()
            if (r,c) in self._revealed: continue
            self._revealed.add((r,c))
            if self._board[r][c]==-1: self._state="failed"; return True
            if self._board[r][c]==0:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr,nc=r+dr,c+dc
                        if 0<=nr<self.rows and 0<=nc<self.cols and (nr,nc) not in self._revealed and (nr,nc) not in self._flagged:
                            stack.append((nr,nc))
        return True

    def do_action(self, action):
        if self._state!="ongoing": return "game_over"
        at=action.get("type"); row=int(action.get("row",0)); col=int(action.get("col",0))
        if at=="reveal":
            if (row,col) in self._revealed: return "already_revealed"
            if (row,col) in self._flagged: return "flagged_cell"
            self._reveal_cell(row,col)
        elif at=="flag":
            if (row,col) in self._revealed: return "invalid_flag"
            if (row,col) in self._flagged: self._flagged.remove((row,col))
            else: self._flagged.add((row,col))
        total=self.rows*self.cols-self.num_mines
        if len(self._revealed)==total: self._state="success"
        if self._state=="failed": return "mine"
        if self._state=="success": return "win"
        return "ok"

    def get_visible_board(self):
        v=[]
        for r in range(self.rows):
            row=[]
            for c in range(self.cols):
                if (r,c) in self._flagged: row.append('F')
                elif (r,c) in self._revealed:
                    val=self._board[r][c]; row.append('*' if val==-1 else str(val))
                else: row.append('.')
            v.append(row)
        return v

    def state(self): return self._state


def get_neighbors(r,c,rows,cols):
    n=[]
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            nr,nc=r+dr,c+dc
            if 0<=nr<rows and 0<=nc<cols: n.append((nr,nc))
    return n


SYSTEM_PROMPT = (
    'You are an expert Minesweeper AI. Analyze the board and output ONE JSON action.\n'
    'BOARD: "."=unknown (valid target), "F"=flagged (never target), "0"-"8"=revealed (never target).\n'
    'LOGIC: For numbered cell N, count adjacent F and ".". If N==F_count -> "." are safe. If N-F==unknowns -> "." are mines.\n'
    'RULES: ONLY target "." cells. NEVER pick 0-8 or F. Prefer logical deductions over guessing.\n'
    'Output ONLY: {"type":"reveal","row":N,"col":N} or {"type":"flag","row":N,"col":N}'
)


def build_compact_prompt(game_or_state):
    if isinstance(game_or_state, dict):
        board=game_or_state["board"]; rows=game_or_state["rows"]; cols=game_or_state["cols"]
        mines=game_or_state["mines"]; flagged=game_or_state.get("flags_placed",0); revealed=game_or_state.get("cells_revealed",0)
    else:
        board=game_or_state.get_visible_board(); rows=game_or_state.rows; cols=game_or_state.cols
        mines=game_or_state.num_mines; flagged=len(game_or_state._flagged); revealed=len(game_or_state._revealed)
    lines=[f"{r:>2}|{''.join(board[r])}" for r in range(rows)]
    return f"Minesweeper {rows}x{cols}, {mines} mines, {flagged} flagged, {revealed} revealed.\n.=unknown F=flag 0-8=adjacent mines\n\n" + "\n".join(lines) + "\n\nJSON action:"


def parse_llm_action(response):
    best=None
    for m in re.finditer(r'\{[^{}]*\}', response):
        try:
            a=json.loads(m.group())
            if "type" in a and "row" in a and "col" in a and a["type"] in ["reveal","flag"]: best=a
        except: continue
    return best


def is_logically_deducible(board, rows, cols, action_type, tr, tc):
    cf,cr=set(),set()
    changed=True
    while changed:
        changed=False
        for r in range(rows):
            for c in range(cols):
                if board[r][c] not in '12345678': continue
                num=int(board[r][c]); nbrs=get_neighbors(r,c,rows,cols)
                fn=sum(1 for nr,nc in nbrs if board[nr][nc]=='F' or (nr,nc) in cf)
                un=[(nr,nc) for nr,nc in nbrs if board[nr][nc]=='.' and (nr,nc) not in cf and (nr,nc) not in cr]
                rem=num-fn
                if rem<0: continue
                if rem==len(un) and un:
                    for n in un:
                        if n not in cf: cf.add(n); changed=True
                if rem==0 and un:
                    for n in un:
                        if n not in cr: cr.add(n); changed=True
    # Coupled constraints
    numbered=[(r,c) for r in range(rows) for c in range(cols) if board[r][c] in '12345678']
    changed=True; iters=0
    while changed and iters<20:
        changed=False; iters+=1
        for i,(r1,c1) in enumerate(numbered):
            n1=int(board[r1][c1]); nb1=get_neighbors(r1,c1,rows,cols)
            f1=sum(1 for nr,nc in nb1 if board[nr][nc]=='F' or (nr,nc) in cf)
            u1=set(n for n in nb1 if board[n[0]][n[1]]=='.' and n not in cf and n not in cr)
            rm1=n1-f1
            if not u1: continue
            for j in range(i+1,len(numbered)):
                r2,c2=numbered[j]
                if abs(r1-r2)>2 or abs(c1-c2)>2: continue
                n2=int(board[r2][c2]); nb2=get_neighbors(r2,c2,rows,cols)
                f2=sum(1 for nr,nc in nb2 if board[nr][nc]=='F' or (nr,nc) in cf)
                u2=set(n for n in nb2 if board[n[0]][n[1]]=='.' and n not in cf and n not in cr)
                rm2=n2-f2
                if not u2: continue
                for sa,sb,ra,rb in [(u1,u2,rm1,rm2),(u2,u1,rm2,rm1)]:
                    if sa.issubset(sb):
                        diff=sb-sa; dm=rb-ra
                        if diff and dm==len(diff):
                            for cell in diff:
                                if cell not in cf: cf.add(cell); changed=True
                        elif diff and dm==0:
                            for cell in diff:
                                if cell not in cr: cr.add(cell); changed=True
    return (action_type=="flag" and (tr,tc) in cf) or (action_type=="reveal" and (tr,tc) in cr)


class MinesweeperSolver:
    def analyze_board(self, board, rows, cols, num_mines, num_flagged):
        cf,cr=set(),set()
        frontier=[(r,c) for r in range(rows) for c in range(cols)
                   if board[r][c] in '12345678'
                   and any(board[nr][nc]=='.' for nr,nc in get_neighbors(r,c,rows,cols))]
        changed=True
        while changed:
            changed=False
            for r,c in frontier:
                num=int(board[r][c]); nbrs=get_neighbors(r,c,rows,cols)
                fn=[n for n in nbrs if board[n[0]][n[1]]=='F' or n in cf]
                un=[n for n in nbrs if board[n[0]][n[1]]=='.' and n not in cf and n not in cr]
                rem=num-len(fn)
                if rem<0: continue
                if rem==len(un) and un:
                    for n in un:
                        if n not in cf: cf.add(n); changed=True
                if rem==0 and un:
                    for n in un:
                        if n not in cr: cr.add(n); changed=True
        # Coupled
        gi=defaultdict(list)
        for r,c in frontier: gi[(r//3,c//3)].append((r,c))
        changed=True; it=0
        while changed and it<30:
            changed=False; it+=1
            for (gr,gc_),fc in gi.items():
                nearby=[]
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        nearby.extend(gi.get((gr+dr,gc_+dc),[]))
                for r1,c1 in fc:
                    n1=int(board[r1][c1]); nb1=get_neighbors(r1,c1,rows,cols)
                    f1=sum(1 for n in nb1 if board[n[0]][n[1]]=='F' or n in cf)
                    u1=set(n for n in nb1 if board[n[0]][n[1]]=='.' and n not in cf and n not in cr)
                    rm1=n1-f1
                    if not u1: continue
                    for r2,c2 in nearby:
                        if (r1,c1)>=(r2,c2) or abs(r1-r2)>2 or abs(c1-c2)>2: continue
                        n2=int(board[r2][c2]); nb2=get_neighbors(r2,c2,rows,cols)
                        f2=sum(1 for n in nb2 if board[n[0]][n[1]]=='F' or n in cf)
                        u2=set(n for n in nb2 if board[n[0]][n[1]]=='.' and n not in cf and n not in cr)
                        rm2=n2-f2
                        if not u2: continue
                        for sa,sb,ra,rb in [(u1,u2,rm1,rm2),(u2,u1,rm2,rm1)]:
                            if sa.issubset(sb):
                                diff=sb-sa; dm=rb-ra
                                if diff and dm==len(diff):
                                    for cell in diff:
                                        if cell not in cf: cf.add(cell); changed=True
                                elif diff and dm==0:
                                    for cell in diff:
                                        if cell not in cr: cr.add(cell); changed=True
        cf-=cr
        return {"certain_flags":cf, "certain_reveals":cr}

    def get_best_action(self, board, rows, cols, num_mines, num_flagged):
        res=self.analyze_board(board,rows,cols,num_mines,num_flagged)
        if res["certain_reveals"]:
            r,c=next(iter(res["certain_reveals"]))
            return {"type":"reveal","row":r,"col":c}, True
        if res["certain_flags"]:
            r,c=next(iter(res["certain_flags"]))
            return {"type":"flag","row":r,"col":c}, True
        # Random frontier
        frontier=[]
        for r in range(rows):
            for c in range(cols):
                if board[r][c]!='.': continue
                for nr,nc in get_neighbors(r,c,rows,cols):
                    if board[nr][nc] not in '.F':
                        frontier.append((r,c)); break
        if frontier:
            r,c=random.choice(frontier)
            return {"type":"reveal","row":r,"col":c}, False
        for r in range(rows):
            for c in range(cols):
                if board[r][c]=='.':
                    return {"type":"reveal","row":r,"col":c}, False
        return {"type":"reveal","row":0,"col":0}, False

solver = MinesweeperSolver()
print("Game class + solver + helpers loaded.")


# ################################################################
# CELL 2: GENERATE SFT DATA (1500 samples, ~1 min)
# ################################################################
from datasets import Dataset

def generate_sft_dataset(num_samples=1500, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    board_configs = [
        (6,6,5), (8,8,10), (10,10,15), (12,12,20), (16,16,40),
        (6,10,8), (8,12,14), (10,16,24), (20,20,50),
    ]
    items = []
    attempts = 0
    while len(items) < num_samples and attempts < num_samples * 20:
        attempts += 1
        rows, cols, mines = board_configs[rng.randint(len(board_configs))]
        seed = int(rng.randint(1000000))
        game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
        game.do_action({"type":"reveal","row":rows//2,"col":cols//2})
        if game.state() != "ongoing":
            continue
        # Play random depth
        max_depth = max(min(rows*cols//2, 40), 4)
        depth = int(rng.randint(0, max_depth))
        for _ in range(depth):
            if game.state() != "ongoing":
                break
            board = game.get_visible_board()
            act, logical = solver.get_best_action(board, rows, cols, mines, len(game._flagged))
            if not logical:
                break  # Only use logical moves for training
            game.do_action(act)

        if game.state() != "ongoing":
            continue

        board = game.get_visible_board()
        act, logical = solver.get_best_action(board, rows, cols, mines, len(game._flagged))
        if not logical:
            continue  # Skip random guesses

        prompt = build_compact_prompt(game)
        response = json.dumps(act)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        items.append({"messages": messages})

    random.shuffle(items)
    return Dataset.from_list(items)

print("Generating SFT data (1500 logical-only examples)...")
sft_dataset = generate_sft_dataset(num_samples=1500)
print(f"Generated {len(sft_dataset)} SFT examples")


# ################################################################
# CELL 3: SFT TRAINING (~10 min at 0.33 it/s)
# ################################################################
from trl import SFTConfig, SFTTrainer

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
print(f"Sample: {sft_dataset[0]['text'][:200]}...")

sft_config = SFTConfig(
    output_dir="sft_speedrun",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,           # 1 epoch = ~94 steps at batch 16
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=9999,              # Don't save checkpoints (speed)
    max_seq_length=max_seq_length,
    optim="adamw_8bit",
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=sft_dataset, args=sft_config)
print(f"SFT: {len(sft_dataset)} samples, 1 epoch, batch 16 -> {len(sft_dataset)//16} steps")
print("Starting SFT training...")
trainer.train()
print("SFT DONE!")


# ################################################################
# CELL 4: SAVE MODEL + WRITE AGENT FILES (~2 min)
# ################################################################

# Save merged model
model.save_pretrained("my_sft_model")
tokenizer.save_pretrained("my_sft_model")
print("LoRA saved to my_sft_model/")

try:
    model.save_pretrained_merged("your_fine_tuned_model", tokenizer, save_method="merged_16bit")
    print("Merged model saved to your_fine_tuned_model/")
except Exception as e:
    print(f"Merge failed ({e}), saving LoRA only...")
    model.save_pretrained("your_fine_tuned_model")
    tokenizer.save_pretrained("your_fine_tuned_model")
    print("Saved to your_fine_tuned_model/")

# Symlink to /workspace if possible
try:
    src = os.path.abspath("your_fine_tuned_model")
    dst = "/workspace/your_fine_tuned_model"
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"Symlinked {src} -> {dst}")
except Exception as e:
    print(f"Symlink: {e}")


# --- Write agent files ---
os.makedirs("agents", exist_ok=True)

AGENT_CODE = r'''#!/usr/bin/python3
"""Minesweeper Agent - SFT Fine-tuned + Expert Prompt"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from .minesweeper_model import MinesweeperAgent


class MinesweeperPlayer:
    def __init__(self, **kwargs):
        self.agent = MinesweeperAgent(**kwargs)

    def build_prompt(self, game_state: Dict[str, Any]) -> tuple:
        board = game_state["board"]
        rows = game_state["rows"]
        cols = game_state["cols"]
        mines = game_state["mines"]
        flagged = game_state.get("flags_placed", 0)
        revealed = game_state.get("cells_revealed", 0)

        lines = []
        for r in range(rows):
            lines.append(f"{r:>2}|{''.join(board[r])}")
        board_str = "\n".join(lines)

        prompt = (
            f"Minesweeper {rows}x{cols}, {mines} mines, {flagged} flagged, {revealed} revealed.\n"
            f".=unknown F=flag 0-8=adjacent mines\n\n"
            f"{board_str}\n\n"
            f"JSON action:"
        )

        sys_prompt = (
            'You are an expert Minesweeper AI. Analyze the board and output ONE JSON action.\n'
            'BOARD: "."=unknown (valid target), "F"=flagged (never target), "0"-"8"=revealed (never target).\n'
            'LOGIC: For numbered cell N, count adjacent F and ".". If N==F_count -> "." are safe. If N-F==unknowns -> "." are mines.\n'
            'RULES: ONLY target "." cells. NEVER pick 0-8 or F. Prefer logical deductions over guessing.\n'
            'Output ONLY: {"type":"reveal","row":N,"col":N} or {"type":"flag","row":N,"col":N}'
        )
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
                if start == -1: break
                brace_count = 0; end = start
                while end < len(response):
                    if response[end] == '{': brace_count += 1
                    elif response[end] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                obj = json.loads(response[start:end+1])
                                potential_jsons.append(obj)
                            except: pass
                            break
                    end += 1
                i = end + 1 if end < len(response) else len(response)
            for obj in potential_jsons:
                if (isinstance(obj, dict) and "type" in obj and "row" in obj and "col" in obj
                    and obj["type"] in ["reveal", "flag"]):
                    obj["row"] = int(obj["row"])
                    obj["col"] = int(obj["col"])
                    return obj
        except: pass
        return None

    @staticmethod
    def save_action(action, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(action, f, indent=2)

if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_state_file", required=True)
    ap.add_argument("--output_file", default="outputs/action.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    with open(args.game_state_file) as f: gs = json.load(f)
    player = MinesweeperPlayer()
    gk = {"tgps_show": args.verbose}
    cf = Path("minesweeper_config.yaml")
    if cf.exists():
        with open(cf) as f: gk.update(yaml.safe_load(f))
    action, tl, gt = player.play_action(gs, **gk)
    if action: player.save_action(action, args.output_file); print(f"Saved: {args.output_file}")
    else: player.save_action({"error":"parse_failed"}, args.output_file); print("ERROR: No valid action")
'''

MODEL_CODE = r'''"""Minesweeper Model Loader"""
import time, os, glob
from transformers import AutoModelForCausalLM, AutoTokenizer

class MinesweeperAgent(object):
    def __init__(self, **kwargs):
        model_name = "/workspace/your_fine_tuned_model"
        if not os.path.exists(model_name):
            model_name = "your_fine_tuned_model"
        if not os.path.exists(model_name):
            # Fallback to base
            for md in ["/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct"]:
                if os.path.exists(md):
                    snaps = sorted(glob.glob(os.path.join(md, "snapshots", "*")))
                    if snaps: model_name = snaps[-1]; break
        print(f"Loading: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.model.eval()

    def generate_response(self, prompt, sys_prompt="", **kwargs):
        msgs = []
        if sys_prompt: msgs.append({"role":"system","content":sys_prompt})
        msgs.append({"role":"user","content":prompt})
        try: text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        outputs = self.model.generate(**inputs, max_new_tokens=128, temperature=0.3, do_sample=True)
        gt = time.time()-t0
        resp = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return resp, outputs.shape[1], gt
'''

with open("agents/minesweeper_agent.py","w") as f: f.write(AGENT_CODE)
with open("agents/minesweeper_model.py","w") as f: f.write(MODEL_CODE)
with open("minesweeper_config.yaml","w") as f: f.write("temperature: 0.3\nmax_new_tokens: 128\n")
print("\nAgent files written to agents/")


# ################################################################
# CELL 5: QUICK EVAL (~5 min)
# ################################################################

FastLanguageModel.for_inference(model)

def play_game(rows, cols, mines, seed, max_moves=200):
    game = MinesweeperGame(rows=rows, cols=cols, num_mines=mines, seed=seed)
    game.do_action({"type":"reveal","row":rows//2,"col":cols//2})
    moves=0; score=0.0; bad=0
    inv={"already_revealed":0,"reveal_flagged":0,"already_flagged":0,"flag_revealed":0,"oob":0,"mine_hit":0,"wrong_flag":0,"invalid_json":0}
    while game.state()=="ongoing" and moves<max_moves and bad<5:
        prompt=build_compact_prompt(game)
        msgs=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
        try: text=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: text=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tokenizer(text,return_tensors="pt").to(model.device)
        out=model.generate(**inp,temperature=0.3,max_new_tokens=128,do_sample=True)
        resp=tokenizer.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True)
        action=parse_llm_action(resp)
        moves+=1
        if action is None: score-=10; bad+=1; inv["invalid_json"]+=1; continue
        try: row,col=int(action["row"]),int(action["col"])
        except: score-=10; bad+=1; inv["invalid_json"]+=1; continue
        at=action["type"]
        if not(0<=row<rows and 0<=col<cols): score-=15; bad+=1; inv["oob"]+=1; continue
        if at=="reveal":
            if (row,col) in game._revealed: score-=12; bad+=1; inv["already_revealed"]+=1; continue
            if (row,col) in game._flagged: score-=8; bad+=1; inv["reveal_flagged"]+=1; continue
            if game._board[row][col]==-1: score-=25; inv["mine_hit"]+=1; break
            bad=0; board=game.get_visible_board()
            il=is_logically_deducible(board,rows,cols,"reveal",row,col)
            score+=15 if il else 10; game.do_action(action)
            if game.state()=="success": score+=100
        elif at=="flag":
            if (row,col) in game._revealed: score-=8; bad+=1; inv["flag_revealed"]+=1; continue
            if (row,col) in game._flagged: score-=8; bad+=1; inv["already_flagged"]+=1; continue
            bad=0
            if len(game._flagged)+1>mines: score-=10
            if game._board[row][col]==-1: score+=15
            else: score-=10; inv["wrong_flag"]+=1
            game.do_action(action)
    return game.state(), moves, score, inv


print("\n" + "="*60)
print("EVALUATION: SFT Fine-tuned Model")
print("="*60)
configs=[(8,8,10,3,"8x8"),(10,10,15,3,"10x10"),(6,10,8,3,"6x10"),(16,16,40,2,"16x16")]
gw=0; gg=0; gs=0
gi={"already_revealed":0,"reveal_flagged":0,"already_flagged":0,"flag_revealed":0,"oob":0,"mine_hit":0,"wrong_flag":0,"invalid_json":0}
for rows,cols,mines,ns,label in configs:
    w=0; ts=0; tm=0
    for si in range(ns):
        res,mv,sc,inv=play_game(rows,cols,mines,seed=42+si)
        ts+=sc; tm+=mv
        if res=="success": w+=1
        for k,v in inv.items(): gi[k]+=v
    gw+=w; gg+=ns; gs+=ts
    print(f"  {label}: {w}/{ns} wins, avg {tm/ns:.1f} moves, avg score {ts/ns:+.1f}")

print(f"\nTOTAL: {gw}/{gg} wins, avg score {gs/gg:+.1f}")
inv_str=", ".join(f"{k}={v}" for k,v in gi.items() if v>0)
print(f"Invalid: {inv_str if inv_str else 'NONE'}")
print("="*60)
print("\nDONE! Agent files in agents/. Model at your_fine_tuned_model/")
print("SUBMIT NOW!")
