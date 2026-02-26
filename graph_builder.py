import torch
import json
import pandas as pd
from torch_geometric.data import Dataset, Data

ACTION_MAP = {
    "set_value": 0,
    "remove_candidate": 1
}

class SudokuDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.df = pd.read_csv(csv_path)

        # mapiranje tehnika
        techniques = sorted(self.df["technique"].unique())
        self.tech_map = {t: i for i, t in enumerate(techniques)}

        self.edge_index = build_sudoku_edge_index()

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]

        values = json.loads(row["values_before"])
        candidates = json.loads(row["candidates_before"])

        node_features = []

        for r in range(9):
            for c in range(9):

                val = values[r][c]

                # ---------- VALUE ONE HOT ----------
                value_one_hot = [0]*10
                value_one_hot[val] = 1

                # ---------- CANDIDATE MASK ----------
                cand_mask = [0]*9
                for cand in candidates[r][c]:
                    cand_mask[cand-1] = 1

                # ---------- POSITION FEATURES ----------
                row_one_hot = [0]*9
                col_one_hot = [0]*9
                box_one_hot = [0]*9

                row_one_hot[r] = 1
                col_one_hot[c] = 1
                box_one_hot[(r//3)*3 + c//3] = 1

                node_features.append(
                    value_one_hot +
                    cand_mask +
                    row_one_hot +
                    col_one_hot +
                    box_one_hot
                )

        x = torch.tensor(node_features, dtype=torch.float)

        # ---------- LABELS ----------
        y_action_type = torch.tensor(ACTION_MAP[row["action_type"]])

        cell_index = int(row["row"])*9 + int(row["col"])
        y_cell = torch.tensor(cell_index)

        if row["action_type"] == "set_value":
            y_number = torch.tensor(int(row["value"]) - 1)
        else:
            y_number = torch.tensor(int(row["candidate"]) - 1)

        y_technique = torch.tensor(self.tech_map[row["technique"]])

        return Data(
            x=x,
            edge_index=self.edge_index,
            y_action_type=y_action_type,
            y_cell=y_cell,
            y_number=y_number,
            y_technique=y_technique,
            puzzle_id=row["puzzle_id"],
            step_idx=row["step_idx"]
        )
    # mozda izbaciti step_idx nekad kasnije i u okviru dataset-a dodati action_id da svaki potez ima nezavisan id ukoliko to bude bilo neophodno
    # zasto je step_idx los: jer solver koji generise dataset u okviru jednog stepa uradi vise akcija istom strategijom (tehnikom) i onda imamo vise "akcija" (unosa vrednosti ili uklanjanja kandidata) sa istim step_idx

def build_sudoku_edge_index():
    """
    Kreira edge_index tensor za Sudoku graf
    Cvorovi: 81 celija
    Veze: cvorovi su povezani ako su u istom redu koloni ili 3x3 bloku
    """
    edges = set()

    # helper funkcija za dodavanje edge-ova
    def add_edges(group):
        for i in group:
            for j in group:
                if i != j:
                    edges.add((i, j))

    # redovi
    for r in range(9):
        add_edges([r*9 + c for c in range(9)])
    # kolone
    for c in range(9):
        add_edges([r*9 + c for r in range(9)])
    # blokovi 3x3
    for br in range(3):
        for bc in range(3):
            block = []
            for r in range(br*3, br*3 + 3):
                for c in range(bc*3, bc*3 + 3):
                    block.append(r*9 + c)
            add_edges(block)

    # konvertuj u tensor [2, num_edges]
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index

def build_graph(values, candidates):
    """
    Pretvara trenutno stanje Sudoku puzzle u PyG graf.
    values: 9x9 matrica brojeva (0 ako prazno)
    candidates: 9x9 matrica setova kandidata
    """
    x = []
    for r in range(9):
        for c in range(9):
            cell_val = values[r][c]
            candidate_vec = [0]*9
            for cand in candidates[r][c]:
                candidate_vec[cand-1] = 1
            # feature vector = one-hot value + candidates
            val_onehot = [0]*9
            if cell_val != 0:
                val_onehot[cell_val-1] = 1
            feat = val_onehot + candidate_vec
            x.append(feat)
    
    x = torch.tensor(x, dtype=torch.float)

    # edges: poveži svaka dva čvora u istom redu, koloni i kvadratu 3x3
    edge_index = []
    for i in range(81):
        r1, c1 = divmod(i, 9)
        for j in range(81):
            if i == j:
                continue
            r2, c2 = divmod(j, 9)
            if r1 == r2 or c1 == c2 or (r1//3 == r2//3 and c1//3 == c2//3):
                edge_index.append([i,j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data