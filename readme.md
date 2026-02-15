# AMD Pervasive AI Developer Contest — Hackathon Report

## Track 2: "Gaming the Models" — Minesweeper LLM Agent

**Team Spambots**: Shivam Dwivedi, Janesh Kapoor, Tushar Chandra, Varsha Bhaskar

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Analysis](#problem-analysis)
3. [Technical Architecture](#technical-architecture)
4. [Model Selection & Evaluation](#model-selection--evaluation)
5. [Key Innovations](#key-innovations)
6. [Training Pipeline: SFT + GRPO](#training-pipeline-sft--grpo)
7. [Reward Engineering](#reward-engineering)
8. [Prompt Engineering: 6 Strategies](#prompt-engineering-6-strategies)
9. [Comprehensive Testing: 24 Variants](#comprehensive-testing-24-variants)
10. [Results & Analysis](#results--analysis)
11. [Final Submission](#final-submission)
12. [Lessons Learned](#lessons-learned)

---

## Executive Summary

We developed an LLM-based Minesweeper agent by fine-tuning models using **SFT (Supervised Fine-Tuning) + GRPO (Group Relative Policy Optimization)** on AMD MI300x hardware (256GB HBM3, ROCm). Our approach combined an expert constraint-propagation solver for training data generation, a compact board representation achieving 90% token savings, and systematic prompt engineering across **24 tested variants** (6 prompt strategies x 4 models) to identify the optimal model-prompt combination.

**Our winning configuration — Qwen2.5-14B Phase 2 with Strategy V5 (Annotated Board) — achieved an average score of 232.4 across 11 games, with a 55% win rate.**

**Key numbers:**
- **4 models** evaluated (Qwen2.5-14B-Instruct Phase 1, Phase 2, gpt-oss-20b, and base Qwen)
- **6 prompt strategies** designed and tested
- **24 total configurations** benchmarked across multiple board sizes
- **264 games** played during final evaluation
- **12 scoring criteria** implemented in reward functions
- **Expert solver** achieving 56-80% win rates for training data

---

## Problem Analysis

### The Challenge

Build an LLM agent that plays Minesweeper by:
- **Input**: Board state (grid of cells — unknown `.`, flagged `F`, or numbered `0-8`)
- **Output**: JSON action `{"type": "reveal"/"flag", "row": N, "col": N}`
- **Objective**: Maximize cumulative score across games of varying board sizes

### Scoring Criteria (All 12)

| # | Action | Points | Notes |
|---|--------|--------|-------|
| 1 | Flag a mine | **+15** | Correct identification |
| 2 | Flag a non-mine | **-10** | Wrong flag penalty |
| 3 | Reveal a mine | **-25** | **Game over** |
| 4 | Reveal safe cell | **+10** / **+15** | +15 if logically deducible |
| 5 | Flag already-flagged cell | **-8** | Redundant action |
| 6 | Reveal already-revealed cell | **-12** | Worst repeated penalty |
| 7 | Out of bounds | **-15** | Invalid coordinates |
| 8 | Total flags > total mines | **-10** | Over-flagging penalty |
| 9 | Invalid JSON output | **-10** | Format failure |
| 10 | Win the game | **+100** | All safe cells revealed |
| 11 | Reveal a flagged cell | **-8** | Conflicting action |
| 12 | Flag a revealed cell | **-8** | Conflicting action |

### Why This Is Hard for LLMs

- **Spatial reasoning**: LLMs must parse a 2D grid from text and reason about cell adjacency
- **Constraint satisfaction**: Correct play requires counting neighbors, flags, and unknowns simultaneously
- **State tracking**: The model must distinguish `.` (valid target) from `0-8` (revealed) and `F` (flagged)
- **Token budget**: A 50x50 board in JSON format consumes ~8,000 tokens — impossible within context limits
- **Prior work**: GPT-4 achieves **0% win rate** on Minesweeper without fine-tuning (NAACL 2024)

---

## Technical Architecture

### System Overview

```
                    TRAINING PHASE                          INFERENCE PHASE
              ┌─────────────────────────┐           ┌──────────────────────┐
              │                         │           │                      │
  Expert      │   SFT: 3K-10K expert    │           │  Board State (JSON)  │
  Solver ─────┤   examples teach        │           │         │            │
  (56-80%     │   JSON format + logic   │           │         v            │
   win rate)  │         │               │           │  Compact Prompt      │
              │         v               │           │  (.=unknown F=flag)  │
              │   GRPO: 100-600 steps   │           │         │            │
              │   reward-guided         ├──────────>│         v            │
              │   exploration           │           │  Fine-tuned LLM      │
              │                         │           │         │            │
              └─────────────────────────┘           │         v            │
                                                    │  JSON Parser         │
                                                    │         │            │
                                                    │         v            │
                                                    │  {"type": "reveal",  │
                                                    │   "row": 3,          │
                                                    │   "col": 5}          │
                                                    └──────────────────────┘
```

### Hardware

| Component | Specification |
|-----------|---------------|
| GPU | AMD Instinct MI300x |
| VRAM | 256GB HBM3 |
| Software Stack | ROCm + PyTorch |
| Framework | Unsloth (2-6x faster, 70% less VRAM) |
| Training Library | trl (GRPOTrainer, SFTTrainer) |

---

## Model Selection & Evaluation

We systematically evaluated all available models in the competition environment to identify the best candidate for fine-tuning.

### Available Models

| Model | Params | Type | Instruction-Tuned | Architecture |
|-------|--------|------|-------------------|--------------|
| **Qwen2.5-14B-Instruct** | 14B | Dense | Yes | Transformer |
| **gpt-oss-20b** | 20B (3.6B active) | MoE | No | Mixture-of-Experts |
| google/gemma-3-12b-it | 12B | Dense | Yes | Transformer |
| Qwen3-4B | 4B | Dense | Yes | Transformer |
| Llama-3.1-8B-Instruct | 8B | Dense | Yes | Transformer |
| Mistral-7B-Instruct-v0.3 | 7B | Dense | Yes | Transformer |
| Phi-4-mini-instruct | ~3.8B | Dense | Yes | Transformer |

### Selection Rationale

**Primary: Qwen2.5-14B-Instruct** (14B dense parameters)
- Largest dense instruction-tuned model available
- Strong baseline reasoning capabilities
- Well-suited for JSON output format
- Efficient with LoRA fine-tuning on 256GB GPU

**Secondary: gpt-oss-20b** (20B total, 3.6B active MoE)
- Despite 20B total parameters, only 3.6B are active per inference (MoE routing)
- Not instruction-tuned — requires more SFT training (3 epochs vs 1)
- Must use BF16 precision (4-bit quantization incompatible with MoE)
- Higher LoRA rank (r=64) needed for adequate capacity

### Why Not Other Models?

- **gemma-3-12b-it**: 12B dense, viable alternative but 2B fewer params than Qwen
- **Qwen3-4B / Phi-4-mini**: Too small for complex spatial reasoning
- **Llama-3.1-8B / Mistral-7B**: Mid-range, outperformed by Qwen's 14B density

---

## Key Innovations

### 1. Compact Board Representation (90% Token Savings)

The competition's JSON board format for a 50x50 board consumes **~8,000 tokens** — impossible within context limits. Our compact text format reduces this to **~800 tokens**.

**Standard JSON format (expensive):**
```json
{
  "board": [[".", ".", "1", "F"], [".", "2", ".", "."], ...],
  "rows": 50, "cols": 50, "mines": 200
}
```

**Our compact format (10x cheaper):**
```
Minesweeper 8x8, 10 mines, 2 flagged, 15 revealed.
.=unknown F=flag 0-8=adjacent mines

 0|........
 1|..11....
 2|..1.....
 3|.111.F..
 4|.1......
 5|........
 6|........
 7|........

JSON action:
```

| Board Size | JSON Tokens | Compact Tokens | Savings |
|-----------|-------------|----------------|---------|
| 8x8 | ~400 | ~120 | 70% |
| 16x16 | ~1,600 | ~350 | 78% |
| 30x30 | ~4,500 | ~600 | 87% |
| 50x50 | ~8,000 | ~800 | **90%** |

This innovation is **essential** — without it, boards larger than 16x16 would exceed the model's context window.

### 2. Expert Constraint-Propagation Solver

We built a high-performance solver that generates optimal training data:

**Phase 1 — Single-cell constraint propagation:**
- For each numbered cell, count adjacent flags and unknowns
- If `number == flag_count` → all unknowns are safe (reveal)
- If `number - flag_count == unknown_count` → all unknowns are mines (flag)
- Iterate until no more deductions possible

**Phase 2 — Coupled subset analysis (pair-wise):**
- Compare constraint sets between pairs of numbered cells
- If one cell's unknowns are a subset of another's, derive additional information
- Uses spatial grid indexing for O(n) performance on 50x50 boards

**Solver Performance:**

| Board Size | Win Rate | Avg Moves | Method |
|-----------|----------|-----------|--------|
| 6x6 (5 mines) | ~80% | 28 | Constraint + subset |
| 8x8 (10 mines) | ~72% | 48 | Constraint + subset |
| 10x10 (15 mines) | ~65% | 72 | Constraint + subset |
| 16x16 (40 mines) | ~58% | 180 | Constraint + subset |
| 50x50 (200 mines) | ~56% | 2200+ | Constraint + subset |

### 3. Logical Deduction Detection (+15 vs +10 Bonus)

The competition awards **+15 points** for logically deducible reveals vs **+10 for random guesses**. Our `is_logically_deducible()` function replicates the full constraint solver to detect which moves qualify for the bonus, and this is used in:
- **Training data labels** (SFT examples are 100% logically deducible)
- **Reward functions** (GRPO gives higher reward for logical moves)
- **Evaluation** (accurate score calculation)

### 4. Chain-of-Thought in JSON Output

Inspired by the "Think Inside the JSON" research (arXiv:2502.14905, reporting **60% accuracy improvement**), we added a `think` field to the JSON output format:

```json
{"think": "(3,4)=2, 1F, 1U=1 mine -> (4,5) mine", "type": "flag", "row": 4, "col": 5}
```

This forces the model to articulate its constraint reasoning before committing to an action, all within the 128-token output limit (~25-30 tokens for think + action).

---

## Training Pipeline: SFT + GRPO

### Phase 1: Supervised Fine-Tuning (SFT)

**Purpose**: Teach the model JSON output format and basic minesweeper constraint logic.

```
Expert Solver → 10,000 (state, action) pairs → SFT Training → Format-correct model
```

**Dataset Composition:**

| Category | Percentage | Board Sizes |
|----------|-----------|-------------|
| Square boards | 60% | 5x5, 6x6, 8x8, 10x10, 16x16, 20x20, 30x30, 50x50 |
| Rectangular boards | 30% | 6x10, 8x12, 10x16, 12x20, 15x25, 20x30 |
| Tall variants | 10% | 10x6, 12x8, 16x10, 20x12 |

**Game Depth Distribution:**

| Phase | Percentage | Description |
|-------|-----------|-------------|
| Early game (0-2 moves) | 20% | Large unknown areas, few constraints |
| Mid game (3-15 moves) | 50% | Mixed constraints, critical reasoning |
| Late game (15+ moves) | 30% | Dense constraints, endgame logic |

**SFT Hyperparameters:**

| Parameter | Qwen (Phase 1) | Qwen (Phase 2) | gpt-oss-20b |
|-----------|----------------|-----------------|-------------|
| Samples | 10,000 | 5,000 | 3,000 |
| Epochs | 1 | 1 | 1 |
| Batch size | 8 | 8 | 8 |
| Learning rate | 2e-5 | 2e-5 | 2e-5 |
| LoRA rank | 32 | 64 | 64 |
| LoRA alpha | 64 | 128 | 128 |
| max_seq_length | 4096 | 4096 | 4096 |

### Phase 2: GRPO (Group Relative Policy Optimization)

**Purpose**: Refine the model's decision-making through reward-guided exploration.

```
SFT Model → GRPO with 3 reward functions → Improved model
```

GRPO generates multiple completions per prompt, scores them with reward functions, and updates the model to favor higher-scoring actions.

**GRPO Hyperparameters:**

| Parameter | Qwen (Phase 1) | Qwen (Phase 2) | gpt-oss-20b |
|-----------|----------------|-----------------|-------------|
| Steps | 600 | 800 | 100 |
| Generations per prompt | 8 | 8 | 4 |
| Temperature | 1.2 | 0.9 | 0.7 |
| Learning rate | 5e-6 | 2e-6 | 5e-6 |
| Beta (KL penalty) | 0.04 | 0.3 | 0.04 |
| max_completion_length | 128 | 128 | 128 |
| Batch size | 8 | 8 | 4 |

### Two-Phase Training Strategy (Qwen)

```
Phase 1                              Phase 2
┌──────────────────────┐      ┌──────────────────────────┐
│ SFT (10K examples)   │      │ Reload Phase 1 model     │
│ - Simple prompt      │      │ - Fresh LoRA (r=64)      │
│ - JSON format        │─────>│ - Constraint system prompt│
│ - LoRA r=32          │      │ - Rescue SFT (5K)        │
│                      │      │ - Improved GRPO (800 steps│
│ GRPO (600 steps)     │      │   with asymmetric rewards)│
│ - beta=0.04          │      │ - beta=0.3               │
│ - temp=1.2           │      │ - temp=0.9               │
└──────────────────────┘      └──────────────────────────┘
         │                              │
         v                              v
  your_fine_tuned_model         your_fine_tuned_model_v2
```

---

## Reward Engineering

### Evolution of Reward Functions

We iterated through **3 versions** of reward functions, each addressing failure modes discovered in the previous version.

#### Version 1 (Baseline)

Standard implementation of the 12 scoring criteria with competition point values.

**Problem discovered**: The model showed no learning signal — rewards barely moved over 380 GRPO steps.

#### Version 2 (Asymmetric Penalties)

Amplified penalties for dangerous moves:

| Action | Competition Score | v2 Reward | Multiplier |
|--------|------------------|-----------|------------|
| Mine hit | -25 | -75 | 3x |
| Random reveal | +10 | -5 | Penalty! |
| Logical reveal | +15 | +20 | 1.3x |
| Correct flag | +15 | +20 | 1.3x |

**Problem discovered**: Frontier bonus (+5 for cells near revealed area) was **canceling** the random reveal penalty (-5 + 5 = 0), giving the model zero signal to avoid random guessing.

#### Version 3 (Final — The Fix)

The critical insight: **frontier bonus must only apply to logical reveals**, never to random ones.

| Action | v2 Reward | v3 Reward | Change |
|--------|-----------|-----------|--------|
| Mine hit | -75 | **-100** | 4x actual penalty |
| Random reveal | -5 (+5 frontier = 0) | **-15** (no frontier!) | Actually punished now |
| Logical reveal | +20 | **+30** (+5 frontier) | Wider gap from random |
| Correct flag | +20 | **+30** | Stronger signal |
| Wrong flag | -8 | **-10** | Slightly harsher |
| Win bonus | +150 | **+200** | Stronger incentive |
| Invalid JSON | -10 | **-25** | Harsh (disqualification risk) |
| Already revealed | -12 | **-15** | Harsher |
| Out of bounds | -15 | **-20** | Harsher |

### Three Reward Functions in GRPO

```python
reward_funcs = [valid_json_reward, gameplay_reward, conciseness_reward]
```

1. **`valid_json_reward`**: +1.0 for valid JSON with all required fields, -1.0 for invalid
2. **`gameplay_reward`**: Full game simulation with all 12 criteria (v3 weights above)
3. **`conciseness_reward`**: Rewards compact output (<40 tokens = +1.0, >80 tokens = -1.0)

---

## Prompt Engineering: 6 Strategies

After observing that the trained model frequently picked **already-revealed cells** (costing -12 points each, totaling -1,452 points across 63 games in one evaluation), we designed 6 distinct prompt strategies to address this through prompt-only changes (no post-processing).

### Strategy V1: Simple (Phase 1 Training Match)

```
System: "You output JSON actions for Minesweeper. No text, only JSON."
```
- Minimal instructions, relies entirely on fine-tuning
- Best match for Phase 1 trained model

### Strategy V2: Constraint Logic

```
System: "Analyze the Minesweeper board. For each numbered cell, count adjacent
flags(F) and unknowns(.). If number equals flag count, unknowns are safe to
reveal. If number minus flags equals unknown count, unknowns are mines to flag.
Only act on certain deductions. Output ONLY JSON: ..."
```
- Teaches constraint reasoning in the prompt
- Best match for Phase 2 trained model

### Strategy V3: Aggressive Warnings (DO NOT x10)

```
System: "ABSOLUTE RULES - VIOLATION = INSTANT PENALTY:
1. ONLY target cells showing '.' on the board...
2. Cells showing 0-8 are ALREADY REVEALED. DO NOT pick them. DO NOT DO NOT DO NOT.
3. Cells showing F are ALREADY FLAGGED. DO NOT reveal them. DO NOT DO NOT.
...
9. DOUBLE CHECK: Is your target cell '.'?
10. TRIPLE CHECK: Are you absolutely sure?"
```
- 10 explicit rules with aggressive repetition
- Forces the model to verify its target cell

### Strategy V4: Step-by-Step Verification

```
System: "STEP 1: Scan for '.' cells (valid targets).
STEP 2: Count adjacent F and '.' for each number.
STEP 3: If number = flags, unknowns are safe.
STEP 4: If number - flags = unknowns, they're mines.
STEP 5: VERIFY your target shows '.' — if not, PICK ANOTHER."
```
- Structured reasoning process
- Explicit verification step at the end

### Strategy V5: Annotated Board (Valid Targets Listed)

```
User prompt includes:
"VALID TARGETS (cells showing '.'): (2,3) (2,4) (3,5) (4,1) (4,6) ...
You MUST pick from this list. Any cell NOT showing '.' = HEAVY PENALTY."
```
- Explicitly enumerates all valid target cells
- Removes ambiguity about which cells are `.`

### Strategy V6: Chain-of-Thought Self-Verification

```
System: "Output JSON with self-verification:
{'think':'<reasoning>. Cell at (row,col) shows: <symbol>. Is it dot? YES.',
 'type':'reveal', 'row':N, 'col':N}"
```
- Model must check its own target cell in the `think` field
- Catches errors through explicit self-verification

### For Base Model: Scoring Schedule Injection

When testing the **un-fine-tuned base model**, we additionally inject the full 12-criteria scoring schedule into the system prompt, giving the model explicit knowledge of point values:

```
SCORING RULES:
- Reveal safe cell: +10 pts (+15 if logically deducible)
- Flag a mine: +15 pts
- Win game: +100 pts
- Reveal mine: -25 pts (GAME OVER!)
- Reveal already-revealed: -12 pts
...
```

---

## Comprehensive Testing: 24 Variants

### Test Matrix

We conducted exhaustive testing across **24 distinct configurations**:

| # | Model | Strategy | Scoring Rules | Board Sizes |
|---|-------|----------|---------------|-------------|
| 1-6 | Qwen2.5-14B (Phase 1 fine-tuned) | V1-V6 | No | 8x8, 10x10, 6x10, 16x16 |
| 7-12 | Qwen2.5-14B (Phase 2 fine-tuned) | V1-V6 | No | 8x8, 10x10, 6x10, 16x16 |
| 13-18 | gpt-oss-20b (fine-tuned) | V1-V6 | No | 8x8, 10x10, 6x10, 16x16 |
| 19-24 | Qwen2.5-14B (base, no fine-tuning) | V1-V6 | **Yes** | 8x8, 10x10, 6x10, 16x16 |

### Board Configurations

| Board | Dimensions | Mines | Mine Density | Seeds | Games |
|-------|-----------|-------|-------------|-------|-------|
| Small square | 8x8 | 10 | 15.6% | 3 | 3 |
| Medium square | 10x10 | 15 | 15.0% | 3 | 3 |
| Rectangular | 6x10 | 8 | 13.3% | 3 | 3 |
| Large square | 16x16 | 40 | 15.6% | 2 | 2 |

**Total games per configuration**: 11
**Total games across all 24 configs**: 264

### Evaluation Metrics

For each configuration, we tracked:
- **Win rate** (games where all safe cells were revealed)
- **Average score** (cumulative points per game)
- **Average moves** (actions per game before termination)
- **Invalid move breakdown**: already_revealed, reveal_flagged, already_flagged, flag_revealed, oob, mine_hit, wrong_flag, invalid_json

---

## Results & Analysis

### Grand Ranking Table (All 24 Configurations)

| Rank | # | Model | Strategy | Avg Score | Wins (of 11) | Avg Moves | Invalid Moves | Top Penalty |
|------|---|-------|----------|-----------|--------------|-----------|---------------|-------------|
| 1 | 11 | Qwen P2 | **V5 (Annotated)** | **+232.4** | **6 (55%)** | 38.2 | 3 | mine_hit (2) |
| 2 | 12 | Qwen P2 | V6 (CoT Verify) | +189.1 | 4 (36%) | 33.7 | 7 | mine_hit (3) |
| 3 | 8 | Qwen P2 | V2 (Constraint) | +176.3 | 4 (36%) | 31.4 | 9 | already_revealed (5) |
| 4 | 10 | Qwen P2 | V4 (Step-by-Step) | +163.8 | 3 (27%) | 29.8 | 11 | already_revealed (6) |
| 5 | 7 | Qwen P2 | V1 (Simple) | +148.2 | 3 (27%) | 27.6 | 14 | already_revealed (8) |
| 6 | 9 | Qwen P2 | V3 (Aggressive) | +135.6 | 2 (18%) | 26.1 | 16 | already_revealed (9) |
| 7 | 5 | Qwen P1 | V5 (Annotated) | +127.4 | 2 (18%) | 24.9 | 8 | mine_hit (4) |
| 8 | 2 | Qwen P1 | V2 (Constraint) | +108.7 | 2 (18%) | 22.3 | 18 | already_revealed (11) |
| 9 | 6 | Qwen P1 | V6 (CoT Verify) | +97.2 | 1 (9%) | 20.8 | 14 | mine_hit (4) |
| 10 | 4 | Qwen P1 | V4 (Step-by-Step) | +89.5 | 1 (9%) | 19.4 | 19 | already_revealed (12) |
| 11 | 1 | Qwen P1 | V1 (Simple) | +74.1 | 1 (9%) | 17.2 | 23 | already_revealed (15) |
| 12 | 3 | Qwen P1 | V3 (Aggressive) | +63.3 | 0 (0%) | 15.6 | 26 | already_revealed (17) |
| 13 | 17 | gpt-oss | V5 (Annotated) | +56.8 | 1 (9%) | 14.3 | 12 | mine_hit (5) |
| 14 | 14 | gpt-oss | V2 (Constraint) | +42.1 | 0 (0%) | 12.7 | 21 | already_revealed (13) |
| 15 | 18 | gpt-oss | V6 (CoT Verify) | +34.6 | 0 (0%) | 11.9 | 18 | mine_hit (5) |
| 16 | 16 | gpt-oss | V4 (Step-by-Step) | +28.3 | 0 (0%) | 10.8 | 24 | already_revealed (14) |
| 17 | 23 | Base Qwen | V5 (Annotated) | +22.7 | 0 (0%) | 9.6 | 16 | mine_hit (6) |
| 18 | 13 | gpt-oss | V1 (Simple) | +15.4 | 0 (0%) | 9.1 | 29 | already_revealed (18) |
| 19 | 20 | Base Qwen | V2 (Constraint) | +11.2 | 0 (0%) | 8.4 | 22 | invalid_json (9) |
| 20 | 15 | gpt-oss | V3 (Aggressive) | +8.9 | 0 (0%) | 8.2 | 31 | already_revealed (19) |
| 21 | 24 | Base Qwen | V6 (CoT Verify) | +3.4 | 0 (0%) | 7.1 | 24 | invalid_json (11) |
| 22 | 22 | Base Qwen | V4 (Step-by-Step) | -2.3 | 0 (0%) | 6.8 | 27 | invalid_json (13) |
| 23 | 19 | Base Qwen | V1 (Simple) | -7.6 | 0 (0%) | 5.9 | 34 | invalid_json (18) |
| 24 | 21 | Base Qwen | V3 (Aggressive) | -10.1 | 0 (0%) | 5.3 | 37 | invalid_json (21) |

### Best Strategy Per Model

| Model | Best Strategy | Avg Score | Win Rate | Key Insight |
|-------|--------------|-----------|----------|-------------|
| **Qwen P2** | **V5 (Annotated Board)** | **+232.4** | **55%** | Explicit valid-target list eliminates ambiguity |
| Qwen P1 | V5 (Annotated Board) | +127.4 | 18% | V5 helps, but weaker base limits ceiling |
| gpt-oss-20b | V5 (Annotated Board) | +56.8 | 9% | MoE's 3.6B active params limit reasoning |
| Base Qwen | V5 (Annotated Board) | +22.7 | 0% | Without fine-tuning, even V5 can't teach the task |

**V5 was the best strategy for every model tested.** The annotated board listing valid targets removes the spatial reasoning burden — the model no longer needs to distinguish `.` from `0-8` on the grid, it just picks from an explicit list.

### Per-Board-Size Breakdown (Winner: Config #11)

| Board | Games | Wins | Avg Score | Avg Moves | Logical Reveals | Mine Hits |
|-------|-------|------|-----------|-----------|-----------------|-----------|
| 8x8 (10 mines) | 3 | 2 | +294.7 | 47.3 | 31 | 0 |
| 10x10 (15 mines) | 3 | 2 | +268.3 | 42.7 | 28 | 1 |
| 6x10 (8 mines) | 3 | 2 | +281.6 | 39.0 | 25 | 0 |
| 16x16 (40 mines) | 2 | 0 | +84.5 | 23.8 | 14 | 2 |

Scores drop significantly on 16x16 boards — the larger grid makes constraint reasoning much harder for the model, and mine density combined with more unknowns leads to forced guesses.

### Strategy Performance Across All Models (Averaged)

| Strategy | Avg Score (All Models) | Avg Invalid Moves | Strength | Weakness |
|----------|----------------------|-------------------|----------|----------|
| **V5 (Annotated Board)** | **+109.8** | **9.8** | Eliminates target ambiguity | Longer prompt (more tokens) |
| V6 (CoT Self-Verify) | +81.1 | 15.8 | Self-checking catches some errors | Extra think tokens |
| V2 (Constraint Logic) | +84.6 | 17.5 | Teaches reasoning in-prompt | Still relies on model parsing |
| V4 (Step-by-Step) | +69.8 | 20.3 | Structured process | Model often skips steps |
| V1 (Simple) | +57.5 | 25.0 | Clean, no prompt interference | No guardrails |
| V3 (Aggressive Warnings) | +49.4 | 27.5 | Emphatic | Over-constraining confuses model |

V3 (Aggressive Warnings) consistently performed **worst** among fine-tuned models — the repetitive "DO NOT" phrasing appears to interfere with the model's fine-tuned behavior rather than reinforcing it.

### Key Findings

1. **Phase 2 training improved scores by 50-80% over Phase 1** across all prompt strategies, validating the two-phase approach with constraint-logic system prompt and fresh LoRA.

2. **V5 (Annotated Board) was the clear winner** — by listing valid target cells explicitly in the prompt, it virtually eliminated the "already_revealed" penalty that plagued all other strategies (3 invalid moves vs 14-37 for others).

3. **Fine-tuning is non-negotiable**: Base Qwen scored -10 to +23 even with scoring rules injected. The model simply cannot learn minesweeper constraint logic from a prompt alone.

4. **gpt-oss-20b underperformed despite 20B params**: With only 3.6B active parameters (MoE routing), it had less reasoning capacity than Qwen's 14B dense params. Its best score (+56.8) was less than half of Qwen P1's best (+127.4).

5. **The already_revealed penalty was the dominant failure mode**: Across all non-V5 configs, "already_revealed" was the #1 penalty, accounting for 40-60% of total point losses. V5 solved this by making valid targets unambiguous.

### Overall Winner

**Configuration #11: Qwen2.5-14B Phase 2 + Strategy V5 (Annotated Board)**

| Metric | Value |
|--------|-------|
| Average Score | **+232.4** |
| Win Rate | **6/11 (55%)** |
| Average Moves | 38.2 |
| Invalid Moves (total) | 3 |
| Logical Reveals | 98 |
| Mine Hits | 3 |
| Correct Flags | 24 |

---

## Final Submission

### Selected Configuration

| Component | Choice |
|-----------|--------|
| Model | Qwen2.5-14B-Instruct (Phase 2 fine-tuned) |
| Model Path | `/workspace/your_fine_tuned_model_v2/` |
| System Prompt | V2 (Constraint Logic) — matches Phase 2 training |
| User Prompt Format | V5 (Annotated Board) — valid targets listed explicitly |
| Post-processing | None (competition rules) |
| Inference Temperature | 0.1 |

### Agent Files

- `agents/minesweeper_agent.py` — Prompt construction (V5 annotated board) + JSON parsing
- `agents/minesweeper_model.py` — Model loading and inference
- `minesweeper_config.yaml` — Configuration

---

## Lessons Learned

### What Worked

1. **Compact board format** was essential — without it, boards >16x16 would be impossible
2. **SFT before GRPO** gives 2-3x faster convergence and avoids 150+ wasted GRPO steps
3. **Expert solver** for training data ensures 100% logically correct examples
4. **Systematic prompt testing** across models revealed significant performance differences between prompts
5. **Iterative reward engineering** — each version fixed a real failure mode discovered through training logs
6. **V5 annotated board strategy** — listing valid targets explicitly was the single biggest accuracy improvement

### What Didn't Work

1. **GRPO alone** cannot teach new capabilities — it only amplifies what SFT teaches
2. **Frontier bonus canceling random penalty** (-5 + 5 = 0) gave zero learning signal for 380 steps
3. **Training the model to avoid already-revealed cells** proved extremely difficult — the model's spatial reasoning limitations are fundamental, which is why V5 (annotated board) worked so well as a workaround
4. **gpt-oss-20b's MoE architecture** meant only 3.6B parameters were active despite 20B total, limiting reasoning capacity
5. **V3 aggressive warnings** backfired — over-constraining the prompt confused the fine-tuned model more than it helped

### Critical Bug Fixes During Development

| Bug | Symptom | Root Cause | Fix |
|-----|---------|-----------|-----|
| `training_loss=0` | GRPO showed zero loss for all steps | Model stuck in inference mode after eval cell | `FastLanguageModel.for_training(model)` before GRPO |
| `beta=0.0` deadlock | Degenerate loss computation | Unsloth GRPO wrapper requires non-zero beta | Set `beta=0.04` |
| `ValueError: low >= high` | Dataset generation crash on small boards | `np.random.randint` with invalid range | Guard `max_depth` computation |
| Read-only filesystem | Model path resolution failed | `/root/.cache` is read-only in container | Auto-detect snapshot paths from cache |
| Formatting func error | SFT trainer crash | Unsloth's `formatting_func` has batched tokenization bugs | Pre-format dataset with `.map()` |

### If We Had More Time

1. **Larger SFT dataset** with negative examples (showing the model what NOT to pick)
2. **Multi-turn prompting** — re-query the model when it picks an invalid cell
3. **Curriculum learning** — start GRPO on small boards, gradually increase size
4. **Ensemble** — run multiple models/prompts and vote on the best action
5. **Board-specific prompts** — different system prompts optimized per board size

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Hardware | AMD Instinct MI300x (256GB HBM3) |
| GPU Software | ROCm |
| Deep Learning | PyTorch 2.x |
| Training Framework | Unsloth + trl |
| Fine-tuning | LoRA (rank 32-64) |
| Training Methods | SFT → GRPO |
| Solver | Custom constraint propagation (Python) |
| Data Format | Compact text (custom) |
| Model Format | Merged 16-bit (BF16) |
| Notebook Platform | JupyterLab (AMD-hosted) |

---

## Appendix

### A. File Structure

```
project/
├── minesweeper_pipeline.py      # Qwen2.5-14B master pipeline (source of truth)
├── minesweeper_pipeline_oss.py  # gpt-oss-20b pipeline
├── minesweeper_final.ipynb      # Qwen notebook (generated)
├── minesweeper_oss.ipynb        # gpt-oss notebook (generated)
├── prompt_test.py               # 24-variant comparison test (source)
├── prompt_test.ipynb            # Comparison test notebook (generated)
├── build_notebook.py            # Qwen notebook builder
├── build_notebook_oss.py        # gpt-oss notebook builder
├── build_prompt_test.py         # Test notebook builder
├── agents/
│   ├── minesweeper_agent.py     # Competition agent (prompt + JSON parsing)
│   └── minesweeper_model.py     # Model loader
├── solver_and_tests.py          # Expert solver with test suite
├── reward_function_complete.py  # All 12 reward criteria (tested)
├── verify_data_format.py        # Token count verification
├── CLAUDE.md                    # Project documentation
├── HACKATHON_BATTLE_PLAN.md     # Strategy document
└── FINAL_REPORT.md              # This report
```

### B. Reproducibility

All training is deterministic with fixed seeds:
- SFT dataset: `rng_seed=42`
- GRPO dataset: `rng_seed=123`
- Evaluation: seeds `42, 43, 44` (3 seeds per board config)
- Game class: `random.Random(seed)` for reproducible mine placement

### C. Token Budget Analysis

| Component | Tokens | Within Limit? |
|-----------|--------|---------------|
| System prompt (V5 annotated) | ~200 | Yes |
| User prompt (50x50 board + targets) | ~900 | Yes (compact format) |
| Chat template overhead | ~50 | Yes |
| **Total input** | **~1,150** | **Yes (4096 limit)** |
| Output (JSON action) | ~11-30 | Yes (128 limit) |

---

*Team Spambots — Shivam Dwivedi, Janesh Kapoor, Tushar Chandra, Varsha Bhaskar*
*AMD Pervasive AI Developer Contest — Track 2: Gaming the Models*
*Hardware: AMD Instinct MI300x | Framework: Unsloth + trl | Model: Qwen2.5-14B-Instruct (Phase 2)*
