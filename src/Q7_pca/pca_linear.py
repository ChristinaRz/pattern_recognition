from __future__ import annotations
import sys
from pathlib import Path
import re
import argparse  # elbow m via CLI (to avoid hardcoded raw "4")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from load_data import load_data

# CONFIG #
LABEL = "killer_id"

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
CAT_SIZES = [6, 4, 5, 2]  # dcat = 17

# Output
OUT_DIR = PROJECT_ROOT / "results" / "Q7_pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# num safety
EPS = 1e-12

# Plot settings
SCATTER_SIZE = 10
SCATTER_ALPHA = 0.75

EXPLAINED_VAR_TARGET = 0.95
SAVE_CUMULATIVE_PLOT = True


# FEATURE PIPELINE (match Q4/Q5/Q6) #
#input codes are clamped into [0, C-1] to avoid crash
#output has dimension 17
def one_hot_encode_categoricals(df) -> np.ndarray:
    N = df.shape[0]
    total_dim = int(np.sum(CAT_SIZES))
    X_cat = np.zeros((N, total_dim), dtype=float)

    offset = 0
    for col, C in zip(CAT_COLS, CAT_SIZES):
        codes = df[col].astype(int).values
        codes = np.clip(codes, 0, C - 1)  # defensive clamp

        rows = np.arange(N)
        X_cat[rows, offset + codes] = 1.0
        offset += C

    return X_cat


# x = [x_continuous (8 dims), x_onehot (17 dims)] => d=25
def build_full_features(df) -> np.ndarray:
    Xc = df[CONT_FEATURES].values.astype(float)
    Xcat = one_hot_encode_categoricals(df)
    return np.hstack([Xc, Xcat])


def standardize_fit_continuous(Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
     # z = (x - μ) / std
    #TRAIN only.
    mu = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return mu, std


def standardize_apply_continuous(Xc: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (Xc - mu) / std


def build_q7_features(df, mu_c: np.ndarray, std_c: np.ndarray) -> np.ndarray:
    Xc = df[CONT_FEATURES].values.astype(float)
    Xc_std = standardize_apply_continuous(Xc, mu_c, std_c)
    Xcat = one_hot_encode_categoricals(df)
    return np.hstack([Xc_std, Xcat])


# DATA HELPERS #
def get_split_df(df, split_name: str):
    d = df[df["split"] == split_name].copy()
    d = d.dropna(subset=CONT_FEATURES + CAT_COLS)  # PCA needs features only
    return d


def read_q5_best_params(q5_dir: Path) -> dict | None:
    #Returns dict with keys: kernel, C, γ, coef0 (optional), degree (optional)
    path = q5_dir / "best_params.txt"
    if not path.exists():
        return None

    txt = path.read_text(encoding="utf-8").splitlines()
    kernel = None
    params_line = None
    for line in txt:
        if "Best kernel" in line:
            kernel = line.split("=")[-1].strip()
        if "Best params" in line:
            params_line = line.split("=", 1)[-1].strip()

    if kernel is None or params_line is None:
        return None


    out = {"kernel": kernel}

    # extract C
    mC = re.search(r"'C'\s*:\s*([0-9]*\.?[0-9]+)", params_line)
    if mC:
        out["C"] = float(mC.group(1))

    # extract γ
    mg = re.search(r"'gamma'\s*:\s*('scale'|[0-9]*\.?[0-9]+)", params_line)
    if mg:
        g = mg.group(1)
        out["gamma"] = g.strip("'") if "scale" in g else float(g)

    # extract coef0 if present
    mc0 = re.search(r"'coef0'\s*:\s*([0-9]*\.?[0-9]+)", params_line)
    if mc0:
        out["coef0"] = float(mc0.group(1))

    # degree if present
    md = re.search(r"'degree'\s*:\s*([0-9]+)", params_line)
    if md:
        out["degree"] = int(md.group(1))

    return out


def train_q5_svm_on_train(X_tr: np.ndarray, y_tr: np.ndarray, q5_params: dict) -> OneVsRestClassifier:
    kernel = q5_params["kernel"]
    C = float(q5_params["C"])

    if kernel == "rbf":
        gamma = q5_params.get("gamma", "scale")
        base = SVC(kernel="rbf", C=C, gamma=gamma, cache_size=2000)
    elif kernel == "poly":
        gamma = q5_params.get("gamma", "scale")
        degree = int(q5_params.get("degree", 2))
        coef0 = float(q5_params.get("coef0", 0.0))
        base = SVC(kernel="poly", degree=degree, C=C, gamma=gamma, coef0=coef0, cache_size=2000)
    else:
        raise ValueError("Unsupported kernel in Q5 params. Expected 'rbf' or 'poly'.")

    clf = OneVsRestClassifier(base)
    clf.fit(X_tr, y_tr)
    return clf


# PLOTTING #
def plot_scree(eigvals: np.ndarray, save_path: Path, title: str = "PCA eigenvalues (scree plot)") -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o")
    plt.xlabel("Component index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_cumulative_explained(cum: np.ndarray, save_path: Path, target: float) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o")
    plt.axhline(target, linestyle="--")
    plt.xlabel("Number of components m")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("Cumulative explained variance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# Scatter plot in PC1-PC2 colored by predicted killer (from Q5 SVM)
def plot_val_scatter(Z_val: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(Z_val[:, 0], Z_val[:, 1], c=y_pred, s=SCATTER_SIZE, alpha=SCATTER_ALPHA)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("VAL projected to PC1-PC2 (colored by Q5 SVM predictions)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



# MATH CHECKS #
""" This function verifies that the eigenvalues obtained from PCA
 match those of the empirical covariance matrix of the centered TRAIN data.
  Since scikit-learn defines covariance using the unbiased estimator
  (division by 𝑁−1),Ι compute the covariance with the same denominator
  to ensure consistency. The maximum absolute difference between the sorted
  covariance eigenvalues and
 pca.explained_variance_ is returned as a numerical correctness check."""
def covariance_eigvals_check(X_tr_centered: np.ndarray, pca: PCA) -> float:

    N = X_tr_centered.shape[0]
    # Match sklearn: covariance denominator (N-1)
    Sigma = (X_tr_centered.T @ X_tr_centered) / max(N - 1, 1)
    w = np.linalg.eigvalsh(Sigma)  # ascending
    w = w[::-1]  # descending
    w_pca = pca.explained_variance_
    m = min(len(w), len(w_pca))
    diff = float(np.max(np.abs(w[:m] - w_pca[:m])))
    return diff


def _parse_args() -> argparse.Namespace:
    # elbow m as a user decision after inspecting the scree plot
    parser = argparse.ArgumentParser(description="Q7 PCA (TRAIN-only) + VAL scatter colored by Q5 SVM predictions.")
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Elbow-based number of principal components to keep (chosen after inspecting the scree plot). "
             "If omitted, the script will still compute m95 (target cumulative variance) as a reference."
    )
    return parser.parse_args()


# MAIN #
def main() -> None:
    args = _parse_args()

    df = load_data()

    #  separate TRAIN / VAL
    df_tr = get_split_df(df, "TRAIN")
    df_va = get_split_df(df, "VAL")

    #  TRAIN only
    Xc_tr = df_tr[CONT_FEATURES].values.astype(float)
    mu_c, std_c = standardize_fit_continuous(Xc_tr)

    X_tr = build_q7_features(df_tr, mu_c, std_c)
    X_va = build_q7_features(df_va, mu_c, std_c)

    # PCA fit  on TRAIN
    pca = PCA(n_components=X_tr.shape[1], random_state=0)
    Z_tr = pca.fit_transform(X_tr)  # Fit+transform TRAIN
    Z_va = pca.transform(X_va)      # Transform VAL

    # Eigenvalues (variance along PCs)
    eigvals = pca.explained_variance_.copy()
    evr = pca.explained_variance_ratio_.copy()
    cum = np.cumsum(evr)

    # choose m based on explained variance threshold
    m95 = int(np.searchsorted(cum, EXPLAINED_VAR_TARGET) + 1)
    m95 = max(1, min(m95, X_tr.shape[1]))

    # elbow
    m_elbow = args.m
    if m_elbow is not None:
        m_elbow = int(m_elbow)
        m_elbow = max(1, min(m_elbow, X_tr.shape[1]))

    # Save eigenvalues + m
    lines = []
    lines.append("Q7 PCA summary")
    lines.append(f"Feature dimension d = {X_tr.shape[1]}")
    lines.append("PCA fitted on TRAIN only.")
    lines.append("")
    lines.append("Explained variance ratio (first 10 components):")
    for i in range(min(10, len(evr))):
        lines.append(f"  PC{i+1}: evr={evr[i]:.6f}, cum={cum[i]:.6f}")
    lines.append("")
    lines.append(f"Reference m95 (smallest m with cumulative explained variance >= {EXPLAINED_VAR_TARGET:.2f}): m95 = {m95}")
    lines.append(f"Cumulative explained variance at m95: {cum[m95-1]:.6f}")
    lines.append("")
    if m_elbow is None:
        lines.append("Elbow-based m was NOT provided (--m missing).")
        lines.append("The scree plot must be inspected and an elbow-based m should be chosen for the report (e.g., --m 4).")
    else:
        lines.append(f"Chosen elbow m (--m): m = {m_elbow}")
        lines.append(f"Cumulative explained variance at elbow m: {cum[m_elbow-1]:.6f}")

    (OUT_DIR / "pca_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # plots
    plot_scree(eigvals, OUT_DIR / "pca_eigenvalues_scree.png")
    if SAVE_CUMULATIVE_PLOT:
        plot_cumulative_explained(cum, OUT_DIR / "pca_explained_variance_cumulative.png", EXPLAINED_VAR_TARGET)

    # Color VAL by Q5 SVM predictions
    #  SVM from Q5 re-train on TRAIN only using best params
    q5_dir = PROJECT_ROOT / "results" / "Q5_svm"
    q5_params = read_q5_best_params(q5_dir)
    if q5_params is None:
        raise FileNotFoundError(
            f"Q5 best params not found in {q5_dir}/best_params.txt. "
            "Run Q5 first (or ensure best_params.txt exists)."
        )

#Q5:the SVM was trained on fully standardized features (continuous + one-hot)
#for consistent predictions in Q7 replicate the same preprocessing pipeline
#ensure y_tr has no NaNs by filtering LABEL
    df_tr_svm = df_tr.dropna(subset=CONT_FEATURES + CAT_COLS + [LABEL]).copy()
    df_va_svm = df_va.dropna(subset=CONT_FEATURES + CAT_COLS).copy()

    X_tr_q5_raw = build_full_features(df_tr_svm)
    X_va_q5_raw = build_full_features(df_va_svm)

    mu_all = X_tr_q5_raw.mean(axis=0)
    std_all = X_tr_q5_raw.std(axis=0)
    std_all = np.where(std_all < EPS, 1.0, std_all)
    X_tr_q5 = (X_tr_q5_raw - mu_all) / std_all
    X_va_q5 = (X_va_q5_raw - mu_all) / std_all

    y_tr = df_tr_svm[LABEL].astype(int).values
    svm = train_q5_svm_on_train(X_tr_q5, y_tr, q5_params)

    # Predict killers on VAL (no labels)
    y_pred_va = svm.predict(X_va_q5).astype(int)

    plot_val_scatter(Z_va[:, :2], y_pred_va, OUT_DIR / "val_scatter_pc1_pc2_colored_by_q5.png")

    #  covariance eigenvalues
    X_tr_centered = X_tr - X_tr.mean(axis=0, keepdims=True)
    max_diff = covariance_eigvals_check(X_tr_centered, pca)
    (OUT_DIR / "math_check.txt").write_text(
        f"Max abs difference between covariance eigenvalues (N-1 denom) and PCA explained_variance_: {max_diff:.6e}\n",
        encoding="utf-8"
    )

    print("Q7 done.")
    print(f"Reference m95 = {m95} (target cumulative explained variance >= {EXPLAINED_VAR_TARGET:.2f}, achieved {cum[m95-1]:.4f})")
    if m_elbow is None:
        print("Elbow m not provided. Rerun with --m <elbow_m> after inspecting the scree plot (e.g., --m 4).")
    else:
        print(f"Chosen elbow m (--m) = {m_elbow} (cumulative {cum[m_elbow-1]:.4f})")
    print(f"Saved plots in: {OUT_DIR}")
    print(f"Math check max eigenvalue diff: {max_diff:.3e}")


if __name__ == "__main__":
    main()
