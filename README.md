# Sudoku solver baziran na Grafovskim neuronskim mrezama
## Definicija problema
Cilj projekta je razviti Sudoku solving model baziran na Graph Neural Network (GNN) koji uči da rešava Sudoku puzzle koristeći ljudski-prihvatljive heuristike (npr. izbegavanje backtracking-a). Model predvidja sledeći potez na osnovu trenutnog stanja table, imitirajući sekvencu poteza rule-based solver-a. Svaki potez je tuple `(tehnika, ćelija, vrednost)`.

Cilj je omogućiti GNN-u da uči i generalizuje human-like poteze i samostalno rešava zagonetku bez backtracking-a.

## Motivacija problema
Sudoku je primer Constraint Satisfaction Problema (CSP) koji zahteva relational reasoning i sekvencijalno planiranje.

Razvijeni GNN model omogućava učenje human-like strategija i heuristika, što ne samo da daje uvid u proces rešavanja puzzle-a, već može poslužiti i kao osnova za AI asistente, igre i rešavanje sličnih problemskih domena.

## Skup podataka
Koristićemo Sudoku dataset sa Kaggle-a:  
[https://www.kaggle.com/datasets/bryanpark/sudoku](https://www.kaggle.com/datasets/bryanpark/sudoku)

Svaki zapis sadrži nepopunjenu sudoku-zagonetku i odgovarajuće rešenje; koristićemo samo puzzle koje se mogu rešiti ograničenim skupom tehnika, a ostale izbaciti radi konzistentnosti.

Zbog toga što svaka sudoku zagonetka podrazumeva veliki broj poteza (step-ova za GNN), verovatno ćemo koristiti samo mali podskup originalnog dataset-a kako bismo broj trening primera držali u razumnim granicama. Ukoliko ovakvo smanjenje bude uzrokovalo lošijim performansama modela, korišćeni skup podataka će biti povećan.

Za GNN, ulaz predstavlja trenutno stanje puzzle‑a u svakom step-u (graf čvorova za svaku ćeliju sa kandidatima), dok je label tuple `(tehnika, ćelija, vrednost)` koji označava potez koji solver odigrava u tom step-u.

## Metodologija

### Rule-based solver
- Svaka sudoku zagonetka se rešava korišćenjem ograničenog skupa tehnika:  
  `Naked/Hidden Single, Naked/Hidden Pair, Pointing Pair, X-Wing, XY-Wing, Naked/Hidden Triple, Box-Line Reduction`  
- Generiše sekvencu poteza `(tehnika, ćelija, vrednost)` koja služi kao label za GNN

### Graph Neural Network (GNN)
- Zagonetka predstavlja graf: ćelije kao čvorovi, veze između ćelija u istom redu, koloni i bloku  
- **Ulaz:** trenutno stanje puzzle‑a sa kandidatima za svaku ćeliju  
- **Output:** predviđanje sledećeg poteza kao tuple `(tehnika, ćelija, vrednost)`; svaka komponenta tuple-a se tretira kao kategorijska promenljiva, tj. model radi **multi-task klasifikaciju**

## Evaluacija

### Step-level evaluacija
- **Step-Level Accuracy:** koliko često GNN predviđa tačan potez `(tehnika, ćelija, vrednost)` u odnosu na sekvencu koju je generisao rule-based solver  
- **Top-k Accuracy:** koliko često je tačan potez među top-k predikcija modela (npr. k=3)

### Puzzle-level evaluacija
- **Completion rate:** procenat Sudoku puzzlova koje GNN uspešno reši samostalno, korak po korak  
- **Average number of correct moves per puzzle:** daje uvid koliko dobro model imitira solver po potezima

Kombinovanjem ovih metrika dobijamo bolju procenu kako GNN imitira solver po stepovima, ali i koliko zapravo uspešno rešava ceo sudoku.
