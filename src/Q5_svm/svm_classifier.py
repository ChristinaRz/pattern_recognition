from __future__ import annotations

import sys
from pathlib import Path
import warnings
import re

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PATH SETUP #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from load_data import load_data
from plots import plot_confusion_matrix


# CONFIG #
LABEL = "killer_id"

CONT_FEATURES = [
    "hour_float", "latitude", "longitude", "victim_age",
    "temp_c", "humidity", "dist_precinct_km", "pop_density",
]

CAT_COLS = ["weapon_code", "scene_type", "weather", "vic_gender"]
CAT_SIZES = [6, 4, 5, 2]  # 17 one-hot dims

OUT_DIR = PROJECT_ROOT / "results" / "Q5_svm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
SEED = 42
np.random.seed(SEED)

# Hyperparameter grids tuning on VAL
C_GRID = [0.1, 1.0, 10.0, 100.0]
GAMMA_GRID_RBF = ["scale", 1e-3, 1e-2, 1e-1, 1.0]
GAMMA_GRID_POLY = ["scale", 1e-3, 1e-2, 1e-1]
COEF0_GRID = [0.0, 1.0, 10.0]

# Plotting control in case of SVM returns too many SVs
MAX_SV_TO_PLOT = 2000


# FEATURE PIPELINE #
def one_hot_encode_categoricals(df):
    # assume codes are valid in-range (0..C-1) from above
    # NOTE (added): to match Q4 exactly, we CLIP codes instead of dropping rows.
    N = df.shape[0]
    total_dim = int(np.sum(CAT_SIZES))
    X_cat = np.zeros((N, total_dim), dtype=float)

    offset = 0
    for col, C in zip(CAT_COLS, CAT_SIZES):
        codes = df[col].astype(int).values

        # Defensive clamp (exactly like Q4)
        codes = np.clip(codes, 0, C - 1)

        rows = np.arange(N)
        X_cat[rows, offset + codes] = 1.0
        offset += C

    return X_cat


def build_full_features(df):
    # x = [continuous; one-hot categoricals]
    Xc = df[CONT_FEATURES].values.astype(float)
    Xcat = one_hot_encode_categoricals(df)
    return np.hstack([Xc, Xcat])


def standardize_fit(X):
    # standardization on TRAIN only
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return mu, std


def standardize_apply(X, mu, std):
    return (X - mu) / std


# DATA HELPERS#
def get_split(df, split_name: str):
    # to match Q4 i do not drop invalid categorical codes
    # categorical codes clipped during one-hot encoding
    d = df[df["split"] == split_name].copy()

    # all required columns
    d = d.dropna(subset=CONT_FEATURES + CAT_COLS + [LABEL])

    X = build_full_features(d)
    y = d[LABEL].astype(int).values
    return X, y


# MODEL HELPERS #
def make_ovr_svm(kernel: str, C: float, gamma=None, coef0=None) -> OneVsRestClassifier:
    # One-vs-rest wrapper around SVC
    if kernel == "rbf":
        base = SVC(kernel="rbf", C=float(C), gamma=gamma, cache_size=2000)
    elif kernel == "poly":
        base = SVC(kernel="poly", degree=2, C=float(C), gamma=gamma, coef0=float(coef0), cache_size=2000)
        # cache_size helps speed on bigger datasets
    else:
        raise ValueError("kernel must be 'rbf' or 'poly'")
    return OneVsRestClassifier(base)


def tune_hyperparams(X_tr_std, y_tr, X_va_std, y_va):
    # fit on TRAIN only select by VAL accuracy
    best = {"val_acc": -np.inf, "kernel": None, "params": None}

    # RBF
    for C in C_GRID:
        for gamma in GAMMA_GRID_RBF:
            model = make_ovr_svm("rbf", C=C, gamma=gamma)
            model.fit(X_tr_std, y_tr)
            acc = float(accuracy_score(y_va, model.predict(X_va_std)))
            if acc > best["val_acc"]:
                best = {"val_acc": acc, "kernel": "rbf", "params": {"C": float(C), "gamma": gamma}}

    # Poly degree=2
    for C in C_GRID:
        for gamma in GAMMA_GRID_POLY:
            for coef0 in COEF0_GRID:
                model = make_ovr_svm("poly", C=C, gamma=gamma, coef0=coef0)
                model.fit(X_tr_std, y_tr)
                acc = float(accuracy_score(y_va, model.predict(X_va_std)))
                if acc > best["val_acc"]:
                    best = {
                        "val_acc": acc,
                        "kernel": "poly",
                        "params": {"degree": 2, "C": float(C), "gamma": gamma, "coef0": float(coef0)},
                    }

    return best


def collect_support_vectors_std(ovr: OneVsRestClassifier) -> np.ndarray:
    sv_list = []
    for est in getattr(ovr, "estimators_", []):
        if hasattr(est, "support_vectors_"):
            sv_list.append(est.support_vectors_)
    if not sv_list:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(sv_list)


def count_total_support_vectors(ovr: OneVsRestClassifier) -> int:
    total = 0
    for est in getattr(ovr, "estimators_", []):
        if hasattr(est, "n_support_"):
            total += int(np.sum(est.n_support_))
        elif hasattr(est, "support_"):
            total += int(len(est.support_))
        elif hasattr(est, "support_vectors_"):
            total += int(est.support_vectors_.shape[0])
    return total


def plot_pca_regions_with_sv(pca: PCA, Z_train, y_train, ovr_model, class_labels, save_path: Path):
    # PCA decision regions#
    # grid in PCA plane inverse_transform to standardize full space
    # predict on grid overlay TRAIN points and support vectors

    Z_train = np.asarray(Z_train)
    y_train = np.asarray(y_train)

    xmin, xmax = Z_train[:, 0].min(), Z_train[:, 0].max()
    ymin, ymax = Z_train[:, 1].min(), Z_train[:, 1].max()
    dx = 0.10 * (xmax - xmin + EPS)
    dy = 0.10 * (ymax - ymin + EPS)
    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy

    H, W = 250, 250
    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)

    ZZ = np.array([[x, y] for y in ys for x in xs], dtype=float)
    X_grid_std = pca.inverse_transform(ZZ)
    y_grid = ovr_model.predict(X_grid_std).astype(int)

    # plotting consistency
    # safe label to index mapping
    label_to_idx = {lab: i for i, lab in enumerate(class_labels)}
    grid_idx_flat = np.array([label_to_idx[int(l)] for l in y_grid], dtype=int)
    grid_idx = grid_idx_flat.reshape(H, W)

    SV_std = collect_support_vectors_std(ovr_model)
    if SV_std.shape[0] > 0:
        SV_pca = pca.transform(SV_std)

        # subsample for readability
        if SV_pca.shape[0] > MAX_SV_TO_PLOT:
            idx = np.random.choice(SV_pca.shape[0], size=MAX_SV_TO_PLOT, replace=False)
            SV_pca = SV_pca[idx]
    else:
        SV_pca = np.zeros((0, 2), dtype=float)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        grid_idx,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
        alpha=0.35
    )
    plt.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, s=10, alpha=0.75)

    if SV_pca.shape[0] > 0:
        plt.scatter(
            SV_pca[:, 0], SV_pca[:, 1],
            s=28,
            facecolors="none",
            edgecolors="k",
            linewidths=0.8,
            alpha=0.9,
            label=f"Support vectors (shown: {SV_pca.shape[0]})"
        )
        plt.legend(loc="best")

    plt.title("Q5 SVM (OVR) - Decision regions in PCA space (TRAIN + support vectors)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# METRIC FILE PARSER#
def read_val_acc_from_file(path: Path) -> float | None:
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


def main():
    # Load
    df = load_data()

    # Splits
    X_tr_raw, y_tr = get_split(df, "TRAIN")
    X_va_raw, y_va = get_split(df, "VAL")

    # class labels derived from TRAIN
    CLASS_LABELS = sorted(np.unique(y_tr).tolist())

    # Standardize TRAIN only
    mu, std = standardize_fit(X_tr_raw)
    X_tr = standardize_apply(X_tr_raw, mu, std)
    X_va = standardize_apply(X_va_raw, mu, std)

    # PCA(2) fit on TRAIN only
    pca = PCA(n_components=2, random_state=0)
    Z_tr = pca.fit_transform(X_tr)

    # Tune on VAL
    best = tune_hyperparams(X_tr, y_tr, X_va, y_va)

    # Refit best model on TRAIN only
    best_model = make_ovr_svm(
        kernel=best["kernel"],
        C=best["params"]["C"],
        gamma=best["params"].get("gamma", None),
        coef0=best["params"].get("coef0", None),
    )
    best_model.fit(X_tr, y_tr)

    # VAL evaluation
    y_pred_va = best_model.predict(X_va).astype(int)
    acc_va = float(accuracy_score(y_va, y_pred_va))
    cm_va = confusion_matrix(y_va, y_pred_va, labels=CLASS_LABELS)

    # Support vectors count
    sv_total = count_total_support_vectors(best_model)
    (OUT_DIR / "sv_count.txt").write_text(
        f"Total support vectors across OVR estimators (sum): {sv_total}\n",
        encoding="utf-8"
    )

    # accuracy + best params
    (OUT_DIR / "acc_VAL.txt").write_text(f"VAL accuracy = {acc_va:.6f}\n", encoding="utf-8")
    (OUT_DIR / "best_params.txt").write_text(
        f"Best kernel = {best['kernel']}\nBest params = {best['params']}\n",
        encoding="utf-8"
    )

    # confusion matrix
    np.savetxt(OUT_DIR / "cm_VAL.csv", cm_va, delimiter=",", fmt="%d")
    plot_confusion_matrix(
        cm_va,
        class_labels=CLASS_LABELS,
        save_path=OUT_DIR / "cm_VAL.png",
        title="Q5 SVM (OVR) - Confusion Matrix (VAL)",
    )

    # PCA decision regions + SV overlay
    plot_pca_regions_with_sv(
        pca=pca,
        Z_train=Z_tr,
        y_train=y_tr,
        ovr_model=best_model,
        class_labels=CLASS_LABELS,
        save_path=OUT_DIR / "pca_decision_regions_train_with_sv.png",
    )

    # Comparison summary
    q3_acc = read_val_acc_from_file(PROJECT_ROOT / "results" / "Q3_bayes" / "acc_VAL.txt")
    q4_acc = read_val_acc_from_file(PROJECT_ROOT / "results" / "Q4_linear_sse" / "acc.txt")

    lines = []
    lines.append("Comparison summary (VAL):")
    lines.append(f"Q5 SVM VAL acc = {acc_va:.6f}")
    lines.append(f"Q3 Gaussian Bayes VAL acc = {q3_acc:.6f}" if q3_acc is not None else "Q3 Gaussian Bayes VAL acc = (not found)")
    lines.append(f"Q4 Linear SSE VAL acc = {q4_acc:.6f}" if q4_acc is not None else "Q4 Linear SSE VAL acc = (not found)")
    lines.append("")
    lines.append("Logic check")
    lines.append(" Q4 is linear in full feature space so it cannot model nonlinear boundaries.")
    lines.append(" Q5 adds nonlinearity (RBF / poly-2) so it can improve when separation is nonlinear after encoding.")
    lines.append(" Q3 is generative on continuous features only so it strong if Gaussian assumption matches the data structure.")
    (OUT_DIR / "comparison_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Console summary including SV count
    print("Q5 done.")
    print(f"Best kernel/params: {best['kernel']} | {best['params']}")
    print(f"VAL accuracy = {acc_va:.6f}")
    print(f"Total support vectors (sum across OVR estimators) = {sv_total}")
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
