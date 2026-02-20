from __future__ import annotations

import sys
from pathlib import Path
import argparse
import warnings
import re  #for parsing best_hyperparams.txt

import numpy as np
import pandas as pd

# PyTorch (MLP with true posteriors via softmax)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score

# PATH SETUP #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from load_data import load_data


# CONFIG #
LABEL = "killer_id"
INCIDENT_ID_COL = "incident_id"

# dc=8
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

# Categorical integer-coded columns
CAT_COLS = ["weapon_code", "scene_type", "weather", "vic_gender"]
CAT_SIZES = [6, 4, 5, 2]  # dcat=17
D_TOTAL = len(CONT_FEATURES) + int(np.sum(CAT_SIZES))  # 25

# num safety
EPS = 1e-12

# Reproducibility
SEED = 42

Q6_BEST_PATH = PROJECT_ROOT / "results" / "Q6_mlp" / "best_hyperparams.txt"


def _read_best_hyperparams_q6(path: Path) -> dict:
    defaults = {
        "hidden1": 128,
        "hidden2": 64,
        "dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }
    if not path.exists():
        return defaults

    txt = path.read_text(encoding="utf-8", errors="ignore")

    #hidden_sizes = (64,) or (64, 32)
    hidden1, hidden2 = defaults["hidden1"], defaults["hidden2"]
    m_h = re.search(r"hidden_sizes\s*=\s*([^\n\r]+)", txt, flags=re.IGNORECASE)
    if m_h:
        nums = [int(x) for x in re.findall(r"\d+", m_h.group(1))]
        if len(nums) >= 1:
            hidden1 = int(nums[0])
            hidden2 = int(nums[1]) if len(nums) >= 2 else 0  # 0 => no 2nd layer

    # dropout
    dropout = defaults["dropout"]
    m_d = re.search(r"dropout\s*=\s*([0-9]*\.?[0-9]+)", txt, flags=re.IGNORECASE)
    if m_d:
        try:
            dropout = float(m_d.group(1))
        except Exception:
            dropout = defaults["dropout"]

    # lr
    lr = defaults["lr"]
    m_lr = re.search(r"\blr\s*=\s*([0-9.eE+-]+)", txt)
    if m_lr:
        try:
            lr = float(m_lr.group(1))
        except Exception:
            lr = defaults["lr"]

    # weight_decay
    wd = defaults["weight_decay"]
    m_wd = re.search(r"weight_decay\s*=\s*([0-9.eE+-]+)", txt, flags=re.IGNORECASE)
    if m_wd:
        try:
            wd = float(m_wd.group(1))
        except Exception:
            wd = defaults["weight_decay"]

    return {
        "hidden1": hidden1,
        "hidden2": hidden2,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": wd,
    }


Q6_DEFAULTS = _read_best_hyperparams_q6(Q6_BEST_PATH)


# FEATURE PIPELINE (match Q4/Q5/Q6)
# full features + standardize ALL dims fit on TRAIN) #
# TRAIN : fit preprocessing (means/stds)
# VAL: evaluate accuracy
# TEST : final prediction
def one_hot_encode_categoricals(df: pd.DataFrame) -> np.ndarray:
    N = df.shape[0]
    total_dim = int(np.sum(CAT_SIZES))
    X_cat = np.zeros((N, total_dim), dtype=float)

    offset = 0
    for col, C in zip(CAT_COLS, CAT_SIZES):
        # fill NaNs with 0 then cast to int
        codes = df[col].fillna(0).astype(int).values
        codes = np.clip(codes, 0, C - 1)  # defensive clamp
        rows = np.arange(N)
        X_cat[rows, offset + codes] = 1.0
        offset += C

    return X_cat


def build_full_features_raw(df: pd.DataFrame, mu_cont: np.ndarray) -> np.ndarray:
    # x = [continuous(8 dims), one-hot(17 dims)] => d=25
    Xc = df[CONT_FEATURES].values.astype(float)

    # Continuous NaNs impute with TRAIN mean (per-feature)
    Xc = np.where(np.isnan(Xc), mu_cont.reshape(1, -1), Xc)

    Xcat = one_hot_encode_categoricals(df)
    X = np.hstack([Xc, Xcat])
    return X


def standardize_fit_all(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Fit μ, std on TRAIN only for ALL 25 dims
    mu = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return mu, std


def standardize_apply_all(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mu.reshape(1, -1)) / std.reshape(1, -1)


def ensure_incident_id(df: pd.DataFrame) -> pd.Series:
    if INCIDENT_ID_COL in df.columns:
        return df[INCIDENT_ID_COL]
    # fallback: use row index (still deterministic)
    return pd.Series(df.index.astype(int), index=df.index, name=INCIDENT_ID_COL)


def infer_S_and_check_labels(y_tr: np.ndarray) -> int:
    # Formal statement says killer_id ∈ {1,...,S}
    labs = np.unique(y_tr.astype(int))
    S = int(labs.max())
    expected = np.arange(1, S + 1, dtype=int)

    if len(labs) != S or not np.array_equal(np.sort(labs), expected):
        raise ValueError(
            "Label set in TRAIN is not exactly {1,...,S}. "
            f"Found labels={np.sort(labs).tolist()} but expected {expected.tolist()}."
        )
    return S


# MLP MODEL #
class MLP(nn.Module):
    def __init__(self, d_in: int, S: int, hidden1: int, hidden2: int | None, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden1)
        self.drop1 = nn.Dropout(p=dropout)

        if hidden2 is None:
            self.fc2 = None
            self.out = nn.Linear(hidden1, S)
        else:
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.drop2 = nn.Dropout(p=dropout)
            self.out = nn.Linear(hidden2, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        if self.fc2 is not None:
            x = F.relu(self.fc2(x))
            x = self.drop2(x)
        logits = self.out(x)
        return logits


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_with_internal_early_stopping(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    S: int,
    *,
    hidden1: int,
    hidden2: int | None,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: str,
) -> MLP:

    #TRAIN ONLY.
    set_all_seeds(seed)

    N = X_tr.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    # Internal split from TRAIN (10% as train-internal-val)
    n_val_int = max(1, int(0.10 * N))
    idx_val_int = idx[:n_val_int]
    idx_train_int = idx[n_val_int:]

    X_train_int = torch.tensor(X_tr[idx_train_int], dtype=torch.float32)
    y_train_int = torch.tensor(y_tr[idx_train_int] - 1, dtype=torch.long)  # 0..S-1
    X_val_int = torch.tensor(X_tr[idx_val_int], dtype=torch.float32)
    y_val_int = torch.tensor(y_tr[idx_val_int] - 1, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_int, y_train_int),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    model = MLP(d_in=X_tr.shape[1], S=S, hidden1=hidden1, hidden2=hidden2, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    bad = 0

    for epoch in range(int(max_epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Training diverged: loss is NaN/Inf.")
            loss.backward()
            # mild safety clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        # internal validation loss
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_int.to(device))
            loss_val = float(loss_fn(logits_val, y_val_int.to(device)).item())

        if loss_val + 1e-10 < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("Early stopping failed: no best state captured.")
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_proba_mlp(model: MLP, X: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:

    #softmax : posterior probs then renormalization per row to enforce sum=1 up to 1e-8

    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=int(batch_size), shuffle=False, drop_last=False)

    probs_list = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)

            # compute softmax in float64 for better numerical accuracy
            probs = F.softmax(logits.double(), dim=1)  # float64
            probs_list.append(probs.detach().cpu().numpy())

    P = np.vstack(probs_list).astype(np.float64)

    # Hard safety: remove tiny negatives due to numerical noise
    P[P < 0.0] = 0.0

    # Strict renormalization per row (enforces sum=1)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < EPS, 1.0, row_sums)
    P = P / row_sums

    return P


def forensic_checks(P: np.ndarray, y_hat: np.ndarray, S: int, tol: float = 1e-8) -> None:
    if P.ndim != 2 or P.shape[1] != S:
        raise ValueError(f"P has wrong shape: {P.shape}, expected (N,{S}).")

    row_sums = P.sum(axis=1)
    min_sum = float(np.min(row_sums))
    max_sum = float(np.max(row_sums))

    min_p = float(np.nanmin(P))
    max_p = float(np.nanmax(P))

    n_nan = int(np.isnan(P).sum())
    n_neg = int((P < -tol).sum())

    print(f"[CHECK] row_sum: min={min_sum:.12f}, max={max_sum:.12f}")
    print(f"[CHECK] prob range: min={min_p:.12f}, max={max_p:.12f}")
    print(f"[CHECK] NaN probs: {n_nan}")
    print(f"[CHECK] negative probs (< {tol}): {n_neg}")

    if n_nan > 0:
        raise RuntimeError("Found NaN probabilities. Aborting.")
    if n_neg > 0:
        raise RuntimeError("Found negative probabilities. Aborting.")
    if not (np.allclose(row_sums, 1.0, atol=tol, rtol=0.0)):
        bad = np.max(np.abs(row_sums - 1.0))
        raise RuntimeError(f"Row sums are not 1 within tol={tol}. Worst |sum-1|={bad:.3e}. Aborting.")

    # predicted_killer = argmax_k π̂_i(k)  (map 0..S-1 -> 1..S)
    argmax_labels = (np.argmax(P, axis=1) + 1).astype(int)
    if not np.array_equal(argmax_labels, y_hat.astype(int)):
        mism = int(np.sum(argmax_labels != y_hat))
        raise RuntimeError(f"predicted_killer does not match argmax in {mism} rows. Aborting.")

    # predicted_killer ∈ {1,...,S}
    if np.any((y_hat < 1) | (y_hat > S)):
        raise RuntimeError("predicted_killer out of range {1..S}. Aborting.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    #  defaults come from Q6 best_hyperparams.txt
    parser.add_argument("--hidden1", type=int, default=int(Q6_DEFAULTS["hidden1"]))
    parser.add_argument("--hidden2", type=int, default=int(Q6_DEFAULTS["hidden2"]))
    parser.add_argument("--dropout", type=float, default=float(Q6_DEFAULTS["dropout"]))
    parser.add_argument("--lr", type=float, default=float(Q6_DEFAULTS["lr"]))
    parser.add_argument("--weight_decay", type=float, default=float(Q6_DEFAULTS["weight_decay"]))

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # Load data
    df = load_data()

    # Split
    df_tr = df[df["split"] == "TRAIN"].copy()
    df_va = df[df["split"] == "VAL"].copy()
    df_te = df[df["split"] == "TEST"].copy()

    if LABEL not in df_tr.columns:
        raise ValueError("TRAIN split does not contain killer_id. Cannot train final model.")

    # Labels + S check
    y_tr = df_tr[LABEL].astype(int).values
    S = infer_S_and_check_labels(y_tr)

    # Preprocessing fit on TRAIN only
    # Continuous μ for NaN imputation
    Xc_tr = df_tr[CONT_FEATURES].values.astype(float)
    mu_cont = np.nanmean(Xc_tr, axis=0)
    mu_cont = np.where(np.isnan(mu_cont), 0.0, mu_cont)

    X_tr_raw = build_full_features_raw(df_tr, mu_cont)
    X_va_raw = build_full_features_raw(df_va, mu_cont)
    X_te_raw = build_full_features_raw(df_te, mu_cont)

    if X_tr_raw.shape[1] != D_TOTAL:
        raise RuntimeError(f"Feature dimension mismatch: got {X_tr_raw.shape[1]}, expected {D_TOTAL}.")

    mu_all, std_all = standardize_fit_all(X_tr_raw)  # TRAIN only
    X_tr = standardize_apply_all(X_tr_raw, mu_all, std_all)
    X_va = standardize_apply_all(X_va_raw, mu_all, std_all)
    X_te = standardize_apply_all(X_te_raw, mu_all, std_all)

    # Device fallback
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # Train model on TRAIN only (no VAL labels)
    model = train_with_internal_early_stopping(
        X_tr=X_tr,
        y_tr=y_tr,
        S=S,
        hidden1=int(args.hidden1),
        hidden2=int(args.hidden2) if int(args.hidden2) > 0 else None,
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        seed=int(args.seed),
        device=device,
    )

    # Print VAL accuracy
    out_dir = PROJECT_ROOT / "results" / "final_submission"
    out_dir.mkdir(parents=True, exist_ok=True)

    if LABEL in df_va.columns and df_va[LABEL].notna().any():
        y_va = df_va[LABEL].astype(int).values
        P_va = predict_proba_mlp(model, X_va, device=device, batch_size=int(args.batch_size))
        yhat_va = (np.argmax(P_va, axis=1) + 1).astype(int)
        acc_va = float(accuracy_score(y_va, yhat_va))
        print(f"[VAL] accuracy = {acc_va:.6f}")
        (out_dir / "acc_VAL.txt").write_text(f"VAL accuracy = {acc_va:.6f}\n", encoding="utf-8")
    else:
        print("[VAL] labels not available in dataframe (hidden). Skipping direct accuracy computation.")
        P_va = None

    # Build probabilities for ALL rows
    P_tr = predict_proba_mlp(model, X_tr, device=device, batch_size=int(args.batch_size))
    yhat_tr = (np.argmax(P_tr, axis=1) + 1).astype(int)
    forensic_checks(P_tr, yhat_tr, S, tol=1e-8)

    P_te = predict_proba_mlp(model, X_te, device=device, batch_size=int(args.batch_size))
    yhat_te = (np.argmax(P_te, axis=1) + 1).astype(int)
    forensic_checks(P_te, yhat_te, S, tol=1e-8)

    # check VAL probabilities if computed
    if P_va is not None:
        yhat_va = (np.argmax(P_va, axis=1) + 1).astype(int)
        forensic_checks(P_va, yhat_va, S, tol=1e-8)

    # Write submission.csv
    test_ids = ensure_incident_id(df_te).values
    sub = pd.DataFrame(
        {
            "incident_id": test_ids,
            "predicted_killer": yhat_te.astype(int),
        }
    )
    for k in range(1, S + 1):
        sub[f"p_killer_{k}"] = P_te[:, k - 1]

    expected_cols = ["incident_id", "predicted_killer"] + [f"p_killer_{k}" for k in range(1, S + 1)]
    if list(sub.columns) != expected_cols:
        raise RuntimeError("Submission columns do not match required template exactly. Aborting.")
    if sub.isna().any().any():
        raise RuntimeError("submission.csv contains NaNs. Aborting.")

    row_sums = P_te.sum(axis=1)
    print(f"[TEST] row_sum: min={float(row_sums.min()):.12f}, max={float(row_sums.max()):.12f}")
    print(f"[TEST] min prob={float(P_te.min()):.12f}, max prob={float(P_te.max()):.12f}")
    print(f"[TEST] NaN probs={int(np.isnan(P_te).sum())}, negative probs={int((P_te < -1e-8).sum())}")

    out_path = out_dir / "submission.csv"
    sub.to_csv(out_path, index=False)

    root_path = PROJECT_ROOT / "submission.csv"
    sub.to_csv(root_path, index=False)

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Also wrote: {root_path}")
    print("[OK] No leakage: preprocessing fit on TRAIN; model trained on TRAIN only; VAL used only for evaluation.")
    print("[OK] Posteriors: softmax probabilities; rows sum to 1; predicted_killer matches argmax.")


if __name__ == "__main__":
    main()
