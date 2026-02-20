from __future__ import annotations

"""
Choice of framework:
 PyTorch was selected (not sklearn MLPClassifier) because the assignment requires
  "early stopping with VAL". sklearn's early_stopping=True uses an internal
  split taken from TRAIN- not the provided VAL split- so it does not match
  the spec strictly. The provided VAL split explicitly was monitored.
"""
import os
import sys
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# PATH SETUP #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from load_data import load_data
from plots import plot_confusion_matrix


# CONFIG #
LABEL = "killer_id"

# Continuous features dc = 8 in fixed order
CONT_FEATURES = [
    "hour_float",
    "latitude",
    "longitude",
    "victim_age",
    "temp_c",
    "humidity",
    "dist_precinct_km",
    "pop_density",
]

# Raw categorical integer-coded cols
CAT_COLS = ["weapon_code", "scene_type", "weather", "vic_gender"]
# dcat = 17
CAT_SIZES = [6, 4, 5, 2]

# Output
OUT_DIR = PROJECT_ROOT / "results" / "Q6_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42

# numerical safety
EPS = 1e-12


PERM_REPEATS = 5  # optional repeats to reduce randomness
TOPK = 5

# Training limits
MAX_EPOCHS = 300
PATIENCE = 25  # early stopping patience on VAL accuracy
BATCH_SIZE = 128


# REPRODUCIBILITY #
def set_all_seeds(seed: int) -> None:
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism knobs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# FEATURE ENGINEERING (Q4/Q5)
def one_hot_encode_categoricals(df) -> np.ndarray:
    #codes are clipped into valid range [0, C-1] to avoid crashes
    N = df.shape[0]
    total_dim = int(np.sum(CAT_SIZES))
    X_cat = np.zeros((N, total_dim), dtype=float)

    offset = 0
    for col, C in zip(CAT_COLS, CAT_SIZES):
        codes = df[col].astype(int).values
        codes = np.clip(codes, 0, C - 1)

        rows = np.arange(N)
        X_cat[rows, offset + codes] = 1.0
        offset += C

    return X_cat


def build_full_features(df) -> np.ndarray:
    #full feature vector x = [continuous; one-hot categorical].
    #input dimension d = 8 + 17 = 25.

    Xc = df[CONT_FEATURES].values.astype(float)
    Xcat = one_hot_encode_categoricals(df)
    return np.hstack([Xc, Xcat])


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# TRAIN only
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return mu, std


def standardize_apply(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mu) / std


def build_feature_names() -> List[str]:
    #names in the same order as build_full_features()
    names = list(CONT_FEATURES)

    # Weapon one-hot
    for j in range(CAT_SIZES[0]):
        names.append(f"weapon_code={j}")

    # Scene one-hot
    for j in range(CAT_SIZES[1]):
        names.append(f"scene_type={j}")

    # Weather one-hot
    for j in range(CAT_SIZES[2]):
        names.append(f"weather={j}")

    # Gender one-hot
    for j in range(CAT_SIZES[3]):
        names.append(f"vic_gender={j}")

    # Sanity
    assert len(names) == (len(CONT_FEATURES) + int(np.sum(CAT_SIZES)))
    return names


# DATA LOADING / SPLITS #
def get_split(df, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    d = df[df["split"] == split_name].copy()
    d = d.dropna(subset=CONT_FEATURES + CAT_COLS + [LABEL])

    X = build_full_features(d)
    y = d[LABEL].astype(int).values
    return X, y


def make_label_mapping(y_train: np.ndarray) -> Tuple[List[int], Dict[int, int], Dict[int, int]]:
    #labels map to contiguous indices 0..S-1 for PyTorch CE loss.

    class_labels = sorted(np.unique(y_train).tolist())
    to_index = {lab: i for i, lab in enumerate(class_labels)}
    to_label = {i: lab for lab, i in to_index.items()}
    return class_labels, to_index, to_label



# MODEL #
#MLP classifier
#ReLU hidden layers, output logits-no softmax inside forward
# and CrossEntropyLoss which applies log-softmax internally
class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], num_classes: int, dropout: float = 0.0):
        super().__init__()

        layers: List[nn.Module] = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits



# TRAINING UTILITIES #
@dataclass
class TrainConfig:
    hidden_sizes: Tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float  # L2 in AdamW

#shuffles each epoch
def make_batches(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    N = X.shape[0]
    idx = rng.permutation(N)
    for start in range(0, N, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(X).float().to(device)
    logits = model(xb)
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    return pred


def train_one_model(
    X_tr: np.ndarray,
    y_tr_idx: np.ndarray,
    X_va: np.ndarray,
    y_va_idx: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    seed: int,
) -> Tuple[nn.Module, float]:

    #train one MLP candidate with early stopping on VAL accuracy.
    #returns best_model , best_val_acc

    # RNG for determinism
    rng = np.random.default_rng(seed)

    input_dim = X_tr.shape[1]
    num_classes = int(np.max(y_tr_idx)) + 1

    model = MLP(
        input_dim=input_dim,
        hidden_sizes=cfg.hidden_sizes,
        num_classes=num_classes,
        dropout=cfg.dropout,
    ).to(device)

    # Cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # AdamW is stable and standard
    # weight_decay implements L2 regularisation
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0

        for Xb, yb in make_batches(X_tr, y_tr_idx, BATCH_SIZE, rng):
            xb = torch.from_numpy(Xb).float().to(device)
            yb_t = torch.from_numpy(yb).long().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb_t)
            loss.backward()

            # Simple gradient stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            bs = xb.shape[0]
            epoch_loss += float(loss.item()) * bs
            seen += bs

        # VAL accuracy (early stopping monitor) #
        y_va_pred = predict_labels(model, X_va, device)
        val_acc = float(accuracy_score(y_va_idx, y_va_pred))

        # maximise VAL accuracy
        if val_acc > best_val_acc + 1e-12:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


# PERMUTATION IMPORTANCE (VAL) #
def permutation_importance(
    model: nn.Module,
    X_val: np.ndarray,
    y_val_idx: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    repeats: int,
    seed: int,
) -> Tuple[float, np.ndarray]:

    rng = np.random.default_rng(seed)

    # baseline
    y_pred_base = predict_labels(model, X_val, device)
    A_base = float(accuracy_score(y_val_idx, y_pred_base))

    d = X_val.shape[1]
    delta = np.zeros((d,), dtype=float)

    # for each feature index j
    for j in range(d):
        drops = []

        for _ in range(repeats):
            Xp = X_val.copy()

            # permute column j across incidents
            perm = rng.permutation(Xp.shape[0])
            Xp[:, j] = Xp[perm, j]

            y_pred = predict_labels(model, Xp, device)
            Aj = float(accuracy_score(y_val_idx, y_pred))

            drops.append(A_base - Aj)

        delta[j] = float(np.mean(drops))

    #ensure same length as feature_names
    assert len(feature_names) == d
    return A_base, delta

#bar plot of top-k ΔA_j values (largest drops = most important)
def plot_topk_importance(feature_names: List[str], delta: np.ndarray, topk: int, save_path: Path) -> None:
    idx = np.argsort(-delta)  # descending
    top = idx[:topk]

    names_top = [feature_names[i] for i in top]
    vals_top = delta[top]

    plt.figure(figsize=(9, 4.5))
    plt.bar(range(topk), vals_top)
    plt.xticks(range(topk), names_top, rotation=35, ha="right")
    plt.ylabel(r"Importance $\Delta A_j = A_{base} - A_j$")
    plt.title(f"Top-{topk} permutation feature importances (VAL)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def read_val_acc_from_file(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            low = line.lower()
            if ("val" in low) and ("acc" in low):
                m = re.search(r"([0-9]*\.[0-9]+)", line)
                if m:
                    return float(m.group(1))
        return None
    except Exception:
        return None



# MAIN #
def main() -> None:
    set_all_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()

    # split
    X_tr_raw, y_tr = get_split(df, "TRAIN")
    X_va_raw, y_va = get_split(df, "VAL")

    #label mapping
    class_labels, to_index, to_label = make_label_mapping(y_tr)

    y_tr_idx = np.array([to_index[int(y)] for y in y_tr], dtype=int)
    y_va_idx = np.array([to_index[int(y)] for y in y_va], dtype=int)

    # Standardise on TRAIN only (Q4/Q5)
    mu, std = standardize_fit(X_tr_raw)
    X_tr = standardize_apply(X_tr_raw, mu, std)
    X_va = standardize_apply(X_va_raw, mu, std)

    #  Hyperparameter tuning on VAL #
    # tuning grid
    candidates: List[TrainConfig] = [
        TrainConfig(hidden_sizes=(64,), dropout=0.0, lr=1e-3, weight_decay=0.0),
        TrainConfig(hidden_sizes=(64,), dropout=0.2, lr=1e-3, weight_decay=1e-4),
        TrainConfig(hidden_sizes=(64, 32), dropout=0.0, lr=1e-3, weight_decay=0.0),
        TrainConfig(hidden_sizes=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4),
        TrainConfig(hidden_sizes=(128, 64), dropout=0.2, lr=5e-4, weight_decay=1e-4),
    ]

    best = {"val_acc": -np.inf, "cfg": None, "model": None}

    # for reproducibility in candidates
    #vary seed deterministically
    for i, cfg in enumerate(candidates):
        cand_seed = SEED + 1000 + i

        model, val_acc = train_one_model(
            X_tr=X_tr,
            y_tr_idx=y_tr_idx,
            X_va=X_va,
            y_va_idx=y_va_idx,
            cfg=cfg,
            device=device,
            seed=cand_seed,
        )

        if val_acc > best["val_acc"]:
            best["val_acc"] = val_acc
            best["cfg"] = cfg
            best["model"] = model

    assert best["model"] is not None and best["cfg"] is not None
    best_model: nn.Module = best["model"]
    best_cfg: TrainConfig = best["cfg"]
    best_val_acc = float(best["val_acc"])

    # Final VAL evaluation
    y_va_pred_idx = predict_labels(best_model, X_va, device)
    val_acc = float(accuracy_score(y_va_idx, y_va_pred_idx))

    y_va_true_lbl = np.array([to_label[int(i)] for i in y_va_idx], dtype=int)
    y_va_pred_lbl = np.array([to_label[int(i)] for i in y_va_pred_idx], dtype=int)

    cm = confusion_matrix(y_va_true_lbl, y_va_pred_lbl, labels=class_labels)

    #Save outputs
    (OUT_DIR / "best_hyperparams.txt").write_text(
        "\n".join([
            "Best MLP hyperparameters (selected by VAL accuracy):",
            f"hidden_sizes = {best_cfg.hidden_sizes}",
            f"dropout      = {best_cfg.dropout}",
            f"lr           = {best_cfg.lr}",
            f"weight_decay = {best_cfg.weight_decay}",
            f"VAL accuracy = {val_acc:.6f}",
        ]) + "\n",
        encoding="utf-8"
    )

    (OUT_DIR / "acc_VAL.txt").write_text(f"VAL accuracy = {val_acc:.6f}\n", encoding="utf-8")
    np.savetxt(OUT_DIR / "cm_VAL.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(
        cm=cm,
        class_labels=class_labels,
        save_path=OUT_DIR / "cm_VAL.png",
        title="Q6 MLP - Confusion Matrix (VAL)"
    )

    # Comparison with Q3 and Q5
    q3_acc = read_val_acc_from_file(PROJECT_ROOT / "results" / "Q3_bayes" / "acc_VAL.txt")
    q5_acc = read_val_acc_from_file(PROJECT_ROOT / "results" / "Q5_svm" / "Q5_acc_VAL.txt")

    comp_lines = []
    comp_lines.append("Comparison summary (VAL):")
    comp_lines.append(f"Q6 MLP VAL acc = {val_acc:.6f}")
    comp_lines.append(f"Q3 Gaussian Bayes VAL acc = {q3_acc:.6f}" if q3_acc is not None else "Q3 Gaussian Bayes VAL acc = (not found)")
    comp_lines.append(f"Q5 SVM VAL acc = {q5_acc:.6f}" if q5_acc is not None else "Q5 SVM VAL acc = (not found)")
    (OUT_DIR / "comparison_summary.txt").write_text("\n".join(comp_lines) + "\n", encoding="utf-8")

    # Permutation feature importance on VAL
    feature_names = build_feature_names()

    A_base, delta_A = permutation_importance(
        model=best_model,
        X_val=X_va,
        y_val_idx=y_va_idx,
        feature_names=feature_names,
        device=device,
        repeats=PERM_REPEATS,
        seed=SEED + 9999,
    )


    order = np.argsort(-delta_A)
    rows = []
    rows.append("feature_index,feature_name,A_base,delta_A")
    for j in range(len(feature_names)):
        rows.append(f"{j},{feature_names[j]},{A_base:.6f},{delta_A[j]:.6f}")
    (OUT_DIR / "permutation_importance_all.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Top-5
    top_idx = order[:TOPK]
    top_rows = []
    top_rows.append("rank,feature_index,feature_name,delta_A")
    for r, j in enumerate(top_idx, start=1):
        top_rows.append(f"{r},{j},{feature_names[j]},{delta_A[j]:.6f}")
    (OUT_DIR / "permutation_importance_top5.csv").write_text("\n".join(top_rows) + "\n", encoding="utf-8")

    # Bar plot (top-5)
    plot_topk_importance(
        feature_names=feature_names,
        delta=delta_A,
        topk=TOPK,
        save_path=OUT_DIR / "permutation_importance_top5.png"
    )

    print("Q6 done.")
    print(f"Best config: hidden={best_cfg.hidden_sizes}, dropout={best_cfg.dropout}, lr={best_cfg.lr}, wd={best_cfg.weight_decay}")
    print(f"VAL accuracy = {val_acc:.6f}")
    print(f"Permutation importance saved: permutation_importance_all.csv, top5.csv, top5.png")
    print(f"Saved results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
