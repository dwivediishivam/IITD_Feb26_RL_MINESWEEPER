# Minesweeper AI: Deep Research Report
## Comprehensive Analysis for Expert Solver & LLM Training Data Generation

---

## 1. Constraint Satisfaction for Minesweeper

### 1.1 Problem Formulation

Minesweeper maps directly to a Constraint Satisfaction Problem (CSP). Each unrevealed cell
is a boolean variable x_i in {0, 1} (0 = safe, 1 = mine). Each revealed number generates a
constraint: the sum of adjacent unrevealed cells equals (number - already_flagged_neighbors).

For example, if cell (3,4) shows "2" and has one flagged neighbor and three unrevealed
neighbors A, B, C:

    A + B + C = 1

Minesweeper consistency (determining if a partial board is consistent) is NP-complete,
proven by Richard Kaye (2000) via reduction from Circuit-SAT.

### 1.2 Single-Cell Constraint Propagation (Single Point Strategy)

The simplest and fastest approach. For each revealed number N with neighbors:

```
SINGLE_POINT_PROPAGATION(board):
    changed = True
    while changed:
        changed = False
        for each revealed cell (r,c) with number N:
            flagged = count_flagged_neighbors(r, c)
            unrevealed = get_unrevealed_neighbors(r, c)
            remaining_mines = N - flagged

            # Rule 1: All remaining neighbors are mines
            if remaining_mines == len(unrevealed):
                for cell in unrevealed:
                    flag_as_mine(cell)
                    changed = True

            # Rule 2: All mines accounted for - remaining are safe
            if remaining_mines == 0:
                for cell in unrevealed:
                    mark_as_safe(cell)
                    changed = True
    return board
```

Win rates (Single Point only):
- Beginner (9x9/10): ~70%
- Intermediate (16x16/40): ~55%
- Expert (30x16/99): <1%

### 1.3 Coupled/Paired Constraint Analysis (Subset Method)

Compares constraints pairwise. If one constraint's variables are a subset of another's, the
difference can be deduced.

```
COUPLED_CONSTRAINT_ANALYSIS(constraints):
    for each pair (C1, C2) in constraints:
        # C1: vars1 summing to n1
        # C2: vars2 summing to n2

        if vars1 is subset of vars2:
            # New constraint: (vars2 - vars1) sums to (n2 - n1)
            new_vars = vars2 - vars1
            new_sum = n2 - n1
            add_constraint(new_vars, new_sum)

            if new_sum == 0:
                mark_all_safe(new_vars)
            if new_sum == len(new_vars):
                flag_all_mines(new_vars)
```

This is the mathematical basis for pattern recognition (1-1, 1-2, 1-2-1 patterns).

### 1.4 Gaussian Elimination on Constraint Matrix

Treats the constraint system as a linear algebra problem over binary variables.

**Matrix Construction:**
1. Each unrevealed border cell gets a column index
2. Each revealed number touching unrevealed cells gets a row
3. Entry M[i][j] = 1 if constraint i involves cell j, else 0
4. Augmented column = remaining mine count for that constraint

```
GAUSSIAN_ELIMINATION_SOLVER(board):
    # Step 1: Build augmented matrix
    cells = get_border_unrevealed_cells(board)
    constraints = get_active_constraints(board)
    M = build_augmented_matrix(constraints, cells)  # m x (n+1)

    # Step 2: Reduce to row echelon form
    M = reduced_row_echelon_form(M)

    # Step 3: Apply binary constraint heuristic
    for each row r in M (bottom to top):
        coefficients = M[r][0..n-1]
        rhs = M[r][n]
        pos_sum = sum(c for c in coefficients if c > 0)  # max possible
        neg_sum = sum(c for c in coefficients if c < 0)  # min possible

        if rhs == pos_sum:
            # All positive-coefficient vars are mines, negative are safe
            for j where coefficients[j] > 0: flag_mine(cells[j])
            for j where coefficients[j] < 0: mark_safe(cells[j])
        elif rhs == neg_sum:
            # All negative-coefficient vars are mines, positive are safe
            for j where coefficients[j] < 0: flag_mine(cells[j])
            for j where coefficients[j] > 0: mark_safe(cells[j])

    return results
```

Complexity: O(n^3) for elimination, but interpretation is NP-hard in general because
we need solutions over {0,1} (system of Diophantine equations).

### 1.5 Backtracking Search with Constraint Propagation (Tank Algorithm)

The most powerful deterministic approach. Enumerates ALL valid mine configurations for
border cells, then identifies cells that are mines or safe in every configuration.

```
TANK_SOLVER(board):
    # Step 1: Identify border cells (unrevealed, adjacent to a number)
    border_cells = get_border_cells(board)

    # Step 2: CRITICAL OPTIMIZATION - Segregate into independent components
    components = find_connected_components(border_cells, constraints)

    all_results = {}
    for component in components:
        # Step 3: Enumerate valid configurations via backtracking
        configs = []
        backtrack(component, index=0, assignment={}, configs, constraints)

        # Step 4: Analyze configurations
        for cell in component:
            mine_count = sum(1 for c in configs if c[cell] == MINE)
            if mine_count == 0:
                all_results[cell] = SAFE        # Safe in ALL configs
            elif mine_count == len(configs):
                all_results[cell] = MINE         # Mine in ALL configs
            else:
                all_results[cell] = mine_count / len(configs)  # Probability

    return all_results

BACKTRACK(cells, index, assignment, valid_configs, constraints):
    if index == len(cells):
        if all_constraints_satisfied(assignment, constraints):
            valid_configs.append(copy(assignment))
        return

    cell = cells[index]

    # Try cell = SAFE
    assignment[cell] = 0
    if is_consistent(assignment, constraints):  # Prune!
        backtrack(cells, index+1, assignment, valid_configs, constraints)

    # Try cell = MINE
    assignment[cell] = 1
    if is_consistent(assignment, constraints):  # Prune!
        backtrack(cells, index+1, assignment, valid_configs, constraints)

    del assignment[cell]
```

**Segregation Optimization**: The key insight is that border cells can often be split into
independent connected components. If component A has 10 cells and component B has 7 cells,
instead of 2^17 = 131,072 combinations, you evaluate 2^10 + 2^7 = 1,152 -- a ~114x speedup.

---

## 2. Probability Estimation & The Tank Algorithm

### 2.1 Exact Probability Computation

Mine probability for cell X = (number of valid configurations where X is a mine) /
(total number of valid configurations).

This is #P-complete in general, but practical with component decomposition and pruning.

### 2.2 The Tank Algorithm (Detailed)

Named from the blog post by "luckytoilet" (2012), the Tank algorithm is a complete
enumeration approach:

**Phase 1 - Simple Solver**: Apply single-point constraint propagation until stuck.

**Phase 2 - Border Identification**: Collect all unrevealed cells adjacent to at least
one revealed number. These are "border" cells.

**Phase 3 - Component Decomposition**: Build a graph where border cells are nodes, and
two cells share an edge if they appear in the same constraint. Find connected components.
Each component can be solved independently.

**Phase 4 - Exhaustive Enumeration**: For each component, use backtracking with constraint
checking to enumerate every valid mine/safe assignment. Prune branches early when a partial
assignment violates any constraint.

**Phase 5 - Certainty Extraction**: If a cell is a mine in 100% of valid configs, it IS a
mine. If 0%, it IS safe. Otherwise, compute exact probability.

**Phase 6 - Global Mine Count Integration**: The total number of remaining mines constrains
which combinations of component solutions are jointly valid. Weight configurations accordingly.

**Phase 7 - Interior Cell Probability**: For cells not on the border (no adjacent numbers),
their probability is: (remaining_mines - border_mines) / (total_unrevealed - border_cells).
This uniform probability often makes interior cells the safest guess.

### 2.3 Component-Area Optimization (mrgris/minesweepr)

Further optimization groups cells within a component into "areas" -- sets of cells subject
to exactly the same constraints. Since all cells in an area have identical probability, solving
areas instead of individual cells massively reduces enumeration space.

### 2.4 Probability-Weighted Guessing

When forced to guess, the optimal strategy is NOT just "pick the lowest mine probability."
Advanced heuristics consider:
- **Progress potential**: How much information will clicking this cell reveal?
- **Secondary safety**: If the guess is correct, what's the probability of surviving the next move?
- **Information entropy**: How much will the result narrow down mine configurations?

---

## 3. Win Rates for Minesweeper Solvers

### 3.1 Comprehensive Win Rate Table

| Strategy / Solver                     | Beginner (9x9/10) | Intermediate (16x16/40) | Expert (30x16/99) |
|---------------------------------------|-------------------:|------------------------:|-------------------:|
| Single Point (SP) only                |            ~70%    |              ~55%       |           <1%      |
| Double Set Single Point               |            ~74%    |              ~60%       |           ~5%      |
| Equation Strategy                     |            ~76%    |              ~65%       |           ~15%     |
| CSP Backtracking (basic)              |            ~85%    |              ~72%       |           ~30%     |
| Coupled Subsets CSP (Becerra 2015)    |             91.25% |               75.94%    |            32.90%  |
| PSEQ Strategy (Tu et al. 2017)        |             81.63% |               78.12%    |            39.62%  |
| Quasi-Optimal (Tu et al. 2017)        |             81.79% |               78.22%    |            40.06%  |
| JSMinesweeper (Hill) - Classic Expert |               --   |                 --      |            41%     |
| JSMinesweeper (Hill) - Modern Expert  |               --   |                 --      |            54.3%   |
| RL/DQN (6x6/4 mines)                 |             93.3%  |                 --      |              --    |

### 3.2 Key Observations

- **Classic vs Modern rules matter enormously**: Classic expert (safe start in corner) gives
  ~41%, modern expert (guaranteed opening at (3,3)) gives ~54.3%.
- **Average guesses needed**: To win classic expert, a solver must make ~3.3 correct guesses
  on average.
- **First-click death rate on expert**: 99/480 = ~20.6% chance of hitting a mine on the
  very first click (classic rules). This caps theoretical max at ~80%.
- **Deterministic solvability**: For 8x8/10, approximately 81.6% of boards are solvable
  without any guessing. For expert, this drops dramatically.
- **Games with 10%+ mine density** have >99% chance of being solvable (with optimal play
  including probabilistic guessing).

### 3.3 Theoretical Maximum Win Rate

The theoretical maximum is bounded by:
1. **First click survival** (classic rules): ~79.4% on expert
2. **Board solvability**: Most boards require at least some guessing
3. **Optimal guessing**: Perfect probability computation maximizes survival

Best known automated solver: **~54% on modern expert** (Hill's JSMinesweeper with open start).
With perfect play and optimal guessing, theoretical maximum for expert is estimated at ~55-60%.

---

## 4. LLMs Playing Minesweeper

### 4.1 Key Academic Paper

**"Assessing Logical Puzzle Solving in Large Language Models: Insights from a Minesweeper
Case Study"** (Li, Wang, Zhang -- NAACL 2024)

**Setup:**
- Tested GPT-4 and GPT-3.5 variants
- Board sizes: 5x5 with 4 mines (gameplay), 9x9 with 10 mines (comprehension)
- Two text representations tested:
  - **Table format**: Grid with LaTeX-style cell markers
  - **Coordinate format**: "(1,1): ?, (1,2): 1, ..." -- significantly more effective

**GPT-4 Results:**
- 82.9% valid actions (coordinate format)
- 45% correctly flagged mines
- **0% boards fully solved**
- 26.4% of reasoning chains were accurate and coherent

**Key Failure Modes:**
1. **Board comprehension gaps**: Struggled with "neighboring cells" concept and counting
2. **Fragmented reasoning**: Could not chain multi-step logical deductions
3. **Non-linear planning**: Actions appeared ad-hoc rather than logical progressions

**Conclusion:** LLMs possess foundational abilities but "struggle to integrate these into a
coherent, multi-step logical reasoning process." They function more as "a comprehensive
dictionary, useful for reference but lacking in comprehension."

### 4.2 Other LLM Minesweeper Projects

- **Mukesh Barnwal (2024)**: Reported 92-95% success rate training LLMs on minesweeper
  (LinkedIn/GitHub). Methodology details limited, likely on small boards.

- **MCP-based approach** (BioErrorLog): Connected LLMs to Minesweeper via Model Context
  Protocol (MCP), allowing tool-use rather than pure reasoning.

- **General observation**: No published work has achieved expert-level play with LLMs alone.
  The gap between LLM reasoning and algorithmic solving remains very large.

### 4.3 Implications for Training Data Generation

The research strongly suggests that:
1. **Pure prompting is insufficient** -- LLMs need fine-tuning on structured reasoning traces
2. **Coordinate representation outperforms grid/table format** for LLM comprehension
3. **Chain-of-thought training data** should include explicit constraint enumeration
4. **Small boards (5x5 to 9x9)** are the right starting difficulty for LLM training
5. **Expert solver traces** showing each deduction step would be ideal training data

---

## 5. Key Patterns in Minesweeper

### 5.1 Foundational Patterns

**B1 - Trivial Mine**: If a number N has exactly N unrevealed neighbors, all are mines.
**B2 - Trivial Safe**: If a number N already has N flagged neighbors, all other unrevealed
neighbors are safe.

### 5.2 The 1-1 Pattern (Subset Reduction)

```
Wall/Edge
  ?  ?  ?
  1  1  .
```
The left 1 constrains its 2 unknown neighbors. The right 1 shares those 2 plus has a 3rd.
Since the mine must be in the shared subset, the 3rd cell (extending beyond the left 1's
range) is SAFE.

**Generalized**: When constraint A's variables are a subset of constraint B's variables,
and A's count equals B's count, then (B_vars - A_vars) are all safe.

### 5.3 The 1-2 Pattern

```
Wall/Edge
  ?  ?  ?
  1  2  .
```
The 1 says there's one mine in its 2 cells. The 2 says there are 2 mines in its 3 cells.
Subtracting: the 3rd cell (only in the 2's range) must be a MINE.

**Generalized**: If B_count - A_count == len(B_vars - A_vars), then all of (B_vars - A_vars)
are mines.

### 5.4 The 1-2-1 Pattern

```
Wall/Edge
  ?  ?  ?  ?
  1  2  1  .
```
Apply 1-2 from left: rightmost of the three shared cells is a mine.
Apply 1-1 from right: the extension cell is safe.
Result: mines are at the two ends, middle is safe. The 1-2-1 pattern always resolves
completely.

### 5.5 The 1-2-2-1 Pattern

```
Wall/Edge
  ?  ?  ?  ?  ?
  1  2  2  1  .
```
Resolves completely via cascading subset reductions from both ends. Mines at positions
2 and 4, safe at 1, 3, 5.

### 5.6 Advanced Patterns

**1-2C (Corner variant)**: When a 1 and 2 share cells in a corner configuration, the
non-overlapping cell of the 2 must be a mine.

**Reduction patterns (R-suffix)**: After flagging mines, numbers reduce. A "3" next to
a flagged mine becomes effectively a "2", potentially revealing new simple patterns.

**Triangle patterns**: Three numbers forming a triangle with overlapping constraints.

**Hole patterns**: Numbers separated by revealed cells sharing distant unknowns.

**Dependency chains**: Long sequences where solving one end cascades deductions through
the entire chain.

### 5.7 Endgame Patterns

**Global mine counting**: When remaining_mines is known, and border analysis accounts for
most of them, interior cells can be deduced.

**Exhaustive combination elimination**: With few cells remaining, enumerate all valid
configurations to find certainties.

---

## 6. Board Representation for AI

### 6.1 For Neural Networks (CNNs/DQNs)

**One-Hot Encoding (Best performing):**
- Dimension: ROWS x COLS x 12 channels
- Channels 0-8: one-hot encoding of numbers 0-8 (e.g., "3" -> channel 3 = 1)
- Channel 9: cell is unrevealed (1 if unknown)
- Channel 10: cell is flagged (1 if flagged mine)
- Channel 11: cell is out of bounds (for padding)

This avoids imposing false ordinal relationships between states.

**Condensed Encoding (Slightly worse but compact):**
- Dimension: ROWS x COLS x 1
- Values scaled to [-1, 1]: numbers/8 for revealed, -1 for unknown, special values for flags
- More memory-efficient but loses information about state boundaries

**Research finding**: Full one-hot encoding yields best results, with CNN architectures
using 3x3 kernels naturally capturing the 8-neighbor relationship that defines minesweeper
constraints.

### 6.2 For Language Models (Text-Based)

**Coordinate representation (Best for LLMs):**
```
(0,0): 1, (0,1): ?, (0,2): 2, (0,3): F
(1,0): ?, (1,1): ?, (1,2): ?, (1,3): 1
```
- Explicitly maps coordinates to states
- Avoids spatial parsing errors
- Significantly outperformed table format in GPT-4 experiments

**Table/Grid representation:**
```
  0 1 2 3
0 1 ? 2 F
1 ? ? ? 1
```
- More human-readable but LLMs struggle with spatial indexing
- Requires robust row/column parsing

**Proposed optimal for LLM training data:**
```
Board (5x5, 4 mines remaining):
(0,0)=1 (0,1)=? (0,2)=2 (0,3)=? (0,4)=1
(1,0)=? (1,1)=? (1,2)=? (1,3)=? (1,4)=?

Analysis:
- Constraint from (0,0)=1: (0,1) + (1,0) + (1,1) = 1
- Constraint from (0,2)=2: (0,1) + (0,3) + (1,1) + (1,2) + (1,3) = 2
- Subset: {(0,1),(1,1)} in (0,0)'s constraint is subset of (0,2)'s vars
- Deduction: (0,3) + (1,2) + (1,3) = 2 - 1 = 1

Action: SAFE (0,1) -- or -- FLAG (x,y)
```

### 6.3 Recommended Representation for Training Data

For generating expert solver traces to train an LLM:

1. **Input format**: Coordinate-based board state with clear cell status markers
2. **Reasoning trace**: Explicit constraint formulation, subset identification, and
   deduction chain -- mimicking chain-of-thought
3. **Output format**: Single action (reveal cell or flag mine) with confidence
4. **Metadata**: Remaining mine count, board dimensions, move number

The trace should teach the LLM to:
- Enumerate constraints from visible numbers
- Identify subset relationships between constraints
- Apply the 1-1, 1-2, 1-2-1 reduction rules
- When stuck, reason about probability (even if approximate)

---

## Summary: Recommended Expert Solver Architecture

For generating training data, build a solver with this hierarchy:

1. **Single-point propagation** (fast, handles ~60% of moves)
2. **Coupled constraint / subset analysis** (handles ~25% more)
3. **Gaussian elimination** (catches remaining deterministic cases)
4. **Tank algorithm with component decomposition** (complete enumeration for certainty)
5. **Probability-weighted guessing** (when no deterministic move exists)
6. **Global mine count integration** (endgame optimization)

Expected win rates with this full stack:
- Beginner: ~90%+
- Intermediate: ~78%+
- Expert (modern): ~50%+

Each move should emit a structured reasoning trace suitable for LLM fine-tuning.

---

## Sources

- [Lucky's Notes - Tank Algorithm](https://luckytoilet.wordpress.com/2012/12/23/2125/)
- [Solving Minesweeper with Matrices](https://massaioli.wordpress.com/2013/01/12/solving-minesweeper-with-matricies/)
- [Minesweepr Probability Engine](https://mrgris.com/projects/minesweepr/)
- [Minesweeper as a CSP - Studholme](https://www.cs.toronto.edu/~cvs/minesweeper/minesweeper.pdf)
- [Algorithmic Approaches to Playing Minesweeper - Becerra 2015](https://dash.harvard.edu/bitstreams/7312037d-80a6-6bd4-e053-0100007fdf3b/download)
- [Exploring Efficient Strategies for Minesweeper - Tu et al. 2017](https://cdn.aaai.org/ocs/ws/ws0294/15091-68459-1-PB.pdf)
- [JSMinesweeper - David Hill](https://github.com/DavidNHill/JSMinesweeper)
- [LLM Minesweeper Study - Li et al. NAACL 2024](https://arxiv.org/abs/2311.07387)
- [Minesweeper Patterns - minesweeper.online](https://minesweeper.online/help/patterns)
- [Minesweeper Patterns - minesweepergame.com](https://minesweepergame.com/strategy/patterns.php)
- [CNN Minesweeper Agent - MDPI 2025](https://www.mdpi.com/2076-3417/15/5/2490)
- [Gaussian Elimination Solver - GitHub](https://github.com/davidz-repo/Minesweeper-AI-Solver)
- [Minesweeper Probability Calculation](https://www.lrvideckis.com/blog/2020/07/17/minesweeper_probability.html)
- [Mine-Sweeper-Solver - nabenabe0928](https://github.com/nabenabe0928/mine-sweeper-solver)
- [Fast CSP for Minesweeper - 2021](https://arxiv.org/abs/2105.04120)
- [Minesweeper is NP-complete - Kaye](https://medium.com/smith-hcv/minesweeper-is-np-complete-47e37895cc52)
- [Minesweeper-DDQN](https://github.com/AlexMGitHub/Minesweeper-DDQN)
- [Minesweeper RL - sdlee94](https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/)
