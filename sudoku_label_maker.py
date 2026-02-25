import pandas as pd
import json
import io
import sys
import os

# da bi importovali Puzzle modul koji je u folderu
sys.path.insert(0, os.path.abspath("sudoku-solver-main"))
from sudoku_solver_tim.puzzle import Puzzle

DATASET_PATH = "sudoku.csv"
OUTPUT_CSV = "sudoku_steps.csv"
NUM_PUZZLES = 1000
MAX_RATING = 5

def get_technique_and_actions(puzzle):
    """
    Vraca ime tehnike i akcije, zajedno sa snapshot-om trenutnog stanja pre poteza.
    Ovo je snapshot "pre step-a" kako bi model mogao kasnije da uci.
    """
    # Snapshot pre step-a
    values_before = [[c.value for c in puzzle.rows[r].cells] for r in range(9)]
    markup_before = [[set(c.markup) for c in puzzle.rows[r].cells] for r in range(9)]

    # Uradi jedan step
    puzzle.solve_step()

    # Ime tehnike
    technique = getattr(puzzle, "last_strategy", None)
    if technique is None:
        return None, [], None, None  # npr. brute force ili nema napretka

    if technique == "brute_force":
        return None, [], None, None

    # Snapshot posle poteza (samo za detekciju sta se desilo, da znamo sta se izmenilo zapravo)
    values_after = [[c.value for c in puzzle.rows[r].cells] for r in range(9)]
    markup_after = [[set(c.markup) for c in puzzle.rows[r].cells] for r in range(9)]

    actions = []
    has_set_value = False
    # provera po cell-ovima nakon resenog koraka (nakon solve_step())
    for r in range(9):
        for c in range(9):
            before_val = values_before[r][c]
            after_val = values_after[r][c]

            before_markup = markup_before[r][c]
            after_markup = markup_after[r][c]

            # 1) Ako je postavljen broj - belezimo set_value
            if before_val == 0 and after_val != 0:
                actions.append({
                    "row": r,
                    "col": c,
                    "action_type": "set_value",
                    "value": after_val,
                    "candidate": None
                })
            # 2) Ako vrednost nije postavljena, ali su kandidati uklonjeni - belezimo remove_candidate
            elif not has_set_value and before_val == 0 and before_markup != after_markup:
                removed = before_markup - after_markup
                for cand in removed:
                    actions.append({
                        "row": r,
                        "col": c,
                        "action_type": "remove_candidate",
                        "value": None,
                        "candidate": cand
                    })
    
    # ako je bilo postavljanja broja, izbacimo sve remove_candidate (jer logicno je da su izbaceni neki kandidati)
    if any(a["action_type"] == "set_value" for a in actions):
        actions = [a for a in actions if a["action_type"] == "set_value"]

    return technique, actions, values_before, markup_before

def process_puzzle(grid_str, puzzle_idx):
    # Pretvori '.' u 0
    grid = [[int(c) if c != '.' else 0 for c in grid_str[i*9:(i+1)*9]] for i in range(9)]
    p = Puzzle(grid)
    records = []
    step_idx = 0

    # while loop za dok puzzle nije resen logickim potezima
    while any(cell.value == 0 for cell in p.cells):
        technique, actions, values_before, candidates_before_snapshot = get_technique_and_actions(p)

        if not actions:
            continue  # predji na sledeci korak

        # za value i candidate radimo konverziju u string da ne bi cuvali npr. "2.0" ili "6.0" vec "2" i "6" (nzm zbog cega dolazi do toga :))
        # candidates_before nas ne interesuje ako je akcija "set_value"

        values_before_json = json.dumps(values_before)
        candidates_before_json = json.dumps([[list(c) for c in row] for row in candidates_before_snapshot])

        for act in actions:
            records.append({
                "puzzle_id": puzzle_idx,
                "step_idx": step_idx,
                "technique": technique,
                "row": act["row"],
                "col": act["col"],
                "action_type": act["action_type"],
                "value": str(act["value"]) if act["value"] is not None else "",
                "candidate": str(act["candidate"]) if act["candidate"] is not None else "",
                "values_before": values_before_json,
                "candidates_before": candidates_before_json
            })

        step_idx += 1

    return records

df = pd.read_csv(DATASET_PATH)
selected_df = df[df['difficulty'] <= MAX_RATING].head(NUM_PUZZLES).reset_index(drop=True)

all_records = []
for idx, row in selected_df.iterrows():
    puzzle_records = process_puzzle(row['puzzle'], idx)
    all_records.extend(puzzle_records)

output_df = pd.DataFrame(all_records)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(output_df)} moves to {OUTPUT_CSV}")