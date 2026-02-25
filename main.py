import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from graph_builder import ACTION_MAP, SudokuDataset
from sudoku_gnn import SudokuGNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# -------- hyperparametri --------
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 5
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1

TOP_K = 3

# main pipeline: ucitavanje podataka, podela na train/val/test, kreiranje modela, trening, evaluacija
if __name__ == "__main__":
    # ----------------- dataset i podela -----------------
    dataset = SudokuDataset("sudoku_steps.csv")
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, shuffle=True, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=VALID_SPLIT/(1-TEST_SPLIT), shuffle=True, random_state=42)

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=BATCH_SIZE)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=BATCH_SIZE)

    # ----------------- model i optimizer -----------------
    num_action_types = len(ACTION_MAP)
    num_cells = 81
    num_numbers = 9
    num_techniques = len(dataset.tech_map)
    in_channels = len(dataset[0].x[0])
    hidden_channels = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SudokuGNN(in_channels, hidden_channels, num_action_types, num_cells, num_numbers, num_techniques).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # koristicemo za vizuelizaciju kasnije
    train_losses, val_losses = [], []
    acc_actions, acc_cells, acc_numbers, acc_techs = [], [], [], []

    # ----------------- training loop -----------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # forward
            y_action_pred, y_cell_pred, y_number_pred, y_tech_pred = model(batch)
            y_action = batch.y_action_type
            y_cell = batch.y_cell
            y_number = batch.y_number
            y_tech = batch.y_technique

            number_mask = (y_action == ACTION_MAP["set_value"])

            loss_action = F.cross_entropy(y_action_pred, y_action)
            loss_cell = F.cross_entropy(y_cell_pred, y_cell)
            loss_number = F.cross_entropy(y_number_pred[number_mask], y_number[number_mask])
            loss_tech = F.cross_entropy(y_tech_pred, y_tech)

            loss = loss_action + loss_cell + loss_number + loss_tech
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----------------- validacija
        model.eval()
        val_loss = 0
        correct_action = 0
        correct_cell = 0
        correct_number = 0
        correct_tech = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_action_pred, y_cell_pred, y_number_pred, y_tech_pred = model(batch)
                y_action = batch.y_action_type
                y_cell = batch.y_cell
                y_number = batch.y_number
                y_tech = batch.y_technique
                number_mask = (y_action == ACTION_MAP["set_value"])

                loss_action = F.cross_entropy(y_action_pred, y_action)
                loss_cell = F.cross_entropy(y_cell_pred, y_cell)
                loss_number = F.cross_entropy(y_number_pred[number_mask], y_number[number_mask])
                loss_tech = F.cross_entropy(y_tech_pred, y_tech)
                loss = loss_action + loss_cell + loss_number + loss_tech
                val_loss += loss.item()

                # prezicnost (Accuracy)
                correct_action += (y_action_pred.argmax(dim=1) == y_action).sum().item()
                correct_cell += (y_cell_pred.argmax(dim=1) == y_cell).sum().item()
                correct_number += (y_number_pred[number_mask].argmax(dim=1) == y_number[number_mask]).sum().item()
                correct_tech += (y_tech_pred.argmax(dim=1) == y_tech).sum().item()
                total_samples += batch.num_graphs

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        acc_action = correct_action / total_samples
        acc_cell = correct_cell / total_samples
        acc_number = correct_number / max(1, correct_number+0)  # da izbegnemo deljenje sa nulom
        acc_tech = correct_tech / total_samples

        acc_actions.append(acc_action)
        acc_cells.append(acc_cell)
        acc_numbers.append(acc_number)
        acc_techs.append(acc_tech)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Acc Action: {acc_action:.3f}, Cell: {acc_cell:.3f}, Number: {acc_number:.3f}, Tech: {acc_tech:.3f}")
    
    # ----------------- cuvanje modela (ako zatreba kasnije) -----------------
    torch.save(model.state_dict(), "sudoku_gnn.pth")
    print("Model sacuvan u: sudoku_gnn.pth")

    # ----------------- test i evaluacija -----------------
    model.eval()
    correct_action = correct_cell = correct_number = correct_tech = 0
    step_level_correct = 0
    topk_correct_action = topk_correct_cell = topk_correct_number = topk_correct_tech = 0
    test_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            y_action_pred, y_cell_pred, y_number_pred, y_tech_pred = model(batch)
            y_action = batch.y_action_type
            y_cell = batch.y_cell
            y_number = batch.y_number
            y_tech = batch.y_technique
            number_mask = (y_action == ACTION_MAP["set_value"])

            loss_action = F.cross_entropy(y_action_pred, y_action)
            loss_cell = F.cross_entropy(y_cell_pred, y_cell)
            loss_number = F.cross_entropy(y_number_pred[number_mask], y_number[number_mask])
            loss_tech = F.cross_entropy(y_tech_pred, y_tech)
            loss = loss_action + loss_cell + loss_number + loss_tech
            test_loss += loss.item()

            # standardna preciznost po akciji, celiji, broju i tehnici
            correct_action += (y_action_pred.argmax(dim=1) == y_action).sum().item()
            correct_cell += (y_cell_pred.argmax(dim=1) == y_cell).sum().item()
            correct_number += (y_number_pred[number_mask].argmax(dim=1) == y_number[number_mask]).sum().item()
            correct_tech += (y_tech_pred.argmax(dim=1) == y_tech).sum().item()

            # Step-level accuracy: ceo tuple mora da bude tacan
            y_action_pred_idx = y_action_pred.argmax(dim=1)
            y_cell_pred_idx = y_cell_pred.argmax(dim=1)
            y_number_pred_idx = y_number_pred.argmax(dim=1)
            y_tech_pred_idx = y_tech_pred.argmax(dim=1)

            for i in range(batch.num_graphs):
                # broj grafova u batch-u
                mask = number_mask[i].item() if i < len(number_mask) else True  # za number
                is_step_correct = (
                    y_action_pred_idx[i] == y_action[i] and
                    y_cell_pred_idx[i] == y_cell[i] and
                    (y_number_pred_idx[i] == y_number[i] if mask else True) and
                    y_tech_pred_idx[i] == y_tech[i]
                )
                step_level_correct += int(is_step_correct)

            # Top-k accuracy
            k_action = min(TOP_K, y_action_pred.size(1))
            topk_action = y_action_pred.topk(k_action, dim=1).indices

            k_cell = min(TOP_K, y_cell_pred.size(1))
            topk_cell = y_cell_pred.topk(k_cell, dim=1).indices

            k_number = min(TOP_K, y_number_pred.size(1))
            topk_number = y_number_pred.topk(k_number, dim=1).indices

            k_tech = min(TOP_K, y_tech_pred.size(1))
            topk_tech = y_tech_pred.topk(k_tech, dim=1).indices

            topk_correct_action += (topk_action == y_action.unsqueeze(1)).any(dim=1).sum().item()
            topk_correct_cell += (topk_cell == y_cell.unsqueeze(1)).any(dim=1).sum().item()
            topk_correct_number += (topk_number[number_mask] == y_number[number_mask].unsqueeze(1)).any(dim=1).sum().item()
            topk_correct_tech += (topk_tech == y_tech.unsqueeze(1)).any(dim=1).sum().item()

            total_samples += batch.num_graphs

    avg_test_loss = test_loss / len(test_loader)
    acc_action = correct_action / total_samples
    acc_cell = correct_cell / total_samples
    acc_number = correct_number / max(1, correct_number+0)
    acc_tech = correct_tech / total_samples
    step_level_acc = step_level_correct / total_samples
    topk_acc_action = topk_correct_action / total_samples
    topk_acc_cell = topk_correct_cell / total_samples
    topk_acc_number = topk_correct_number / max(1, topk_correct_number+0)
    topk_acc_tech = topk_correct_tech / total_samples

    print(f"TEST | Loss: {avg_test_loss:.4f} | Acc Action: {acc_action:.3f}, Cell: {acc_cell:.3f}, "
          f"Number: {acc_number:.3f}, Tech: {acc_tech:.3f}")
    
    print("\n--- FINAL TEST METRICS ---")
    print(f"Step-Level Accuracy (whole tuple correct): {step_level_acc:.3f}")
    print(f"Top-{TOP_K} Accuracy per component:")
    print(f"  Action: {topk_acc_action:.3f}")
    print(f"  Cell:   {topk_acc_cell:.3f}")
    print(f"  Number: {topk_acc_number:.3f}")
    print(f"  Tech:   {topk_acc_tech:.3f}")

    # ----------------- vizualizacija -----------------
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)

    # ---------- loss plot ----------
    plt.figure(figsize=(10,5))
    plt.plot(range(1,EPOCHS+1), train_losses, label="Train Loss")
    plt.plot(range(1,EPOCHS+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()
    
    # ---------- accuracy po epoch-i ----------
    plt.figure(figsize=(10,5))
    plt.plot(range(1,EPOCHS+1), acc_actions, label="Acc Action")
    plt.plot(range(1,EPOCHS+1), acc_cells, label="Acc Cell")
    plt.plot(range(1,EPOCHS+1), acc_numbers, label="Acc Number")
    plt.plot(range(1,EPOCHS+1), acc_techs, label="Acc Tech")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_epoch_plot.png"))
    plt.close()

    # ---------- top-K i step-Level Accuracy ----------
    components = ["Action", "Cell", "Number", "Tech"]
    topk_values = [topk_acc_action, topk_acc_cell, topk_acc_number, topk_acc_tech]

    plt.figure(figsize=(8,5))
    bars = plt.bar(components, topk_values, color="skyblue")
    plt.ylim(0,1)
    plt.ylabel(f"Top-{TOP_K} Accuracy")
    plt.title(f"Step-Level Accuracy: {step_level_acc:.3f}, Top-{TOP_K} per Component")
    plt.axhline(step_level_acc, color='red', linestyle='--', label='Step-Level Accuracy')
    plt.legend()

    # dodaj numericke vrednosti iznad bar-ova
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

    plt.savefig(os.path.join(save_dir, "topk_step_plot.png"))
    plt.close()