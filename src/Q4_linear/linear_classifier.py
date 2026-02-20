import numpy as np
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# PATH SETUP #
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # pattern_recognition/
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from load_data import load_data
from plots import plot_confusion_matrix, plot_pca_decision_regions

# CONFIG
LABEL = "killer_id"

# Continuous features (dc=8) in fixed order
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

# Raw categorical integer-coded columns
CAT_COLS = ["weapon_code", "scene_type", "weather", "vic_gender"]

# Cardinalities from the statement: C1=6, C2=4, C3=5, C4=2
# dcat=17
CAT_SIZES = [6, 4, 5, 2]

# Output folder
OUT_DIR = PROJECT_ROOT / "results" / "Q4_linear_sse"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparams
LR = 0.05          # learning rate
EPOCHS = 1200      #  convergence
L2_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

# Numerical stability
EPS = 1e-12



# ENCODING + STANDARDISATION #
def one_hot_encode_categoricals(df, cat_cols, cat_sizes):
    N = df.shape[0]
    total_dim = int(np.sum(cat_sizes))
    X_cat = np.zeros((N, total_dim), dtype=float)

    offset = 0
    for col, C in zip(cat_cols, cat_sizes):
        codes = df[col].astype(int).values

        # Defensive clamp
        codes = np.clip(codes, 0, C - 1)

        rows = np.arange(N)
        X_cat[rows, offset + codes] = 1.0
        offset += C

    return X_cat

#Fit standardization parameters on TRAIN only
def standardize_fit(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < EPS, 1.0, std)  # avoid division by ~0
    return mu, std


def standardize_apply(X, mu, std):
    return (X - mu) / std

# x_i = [x_i^(c); x_i^(cat)]
def build_full_features(df, cont_features, cat_cols, cat_sizes):
    Xc = df[cont_features].values.astype(float)
    Xcat = one_hot_encode_categoricals(df, cat_cols, cat_sizes)
    X = np.hstack([Xc, Xcat])
    return X

#matrix T (N,S) aligned with class_labels order
def one_hot_targets(y, class_labels):
    y = y.astype(int)
    S = len(class_labels)
    N = y.shape[0]
    T = np.zeros((N, S), dtype=float)

    # map class id : index 0..S-1
    idx = {k: j for j, k in enumerate(class_labels)}
    for i in range(N):
        T[i, idx[y[i]]] = 1.0
    return T


# LINEAR NETWORK TRAINING (SSE + L2)
def train_linear_sse(X, T, l2=0.0, lr=0.05, epochs=1000, seed=0):
    rng = np.random.default_rng(seed)
    N, d = X.shape
    S = T.shape[1]

    # Small random init
    W = 0.01 * rng.standard_normal((S, d))
    b = np.zeros((S,), dtype=float)

    # Gradient descent
    for _ in range(epochs):
        # scores: (N,S)
        Y = X @ W.T + b

        # residuals: (N,S)
        E = Y - T

        # dJ/dW = (1/N) E^T X + l2 W
        gradW = (E.T @ X) / N + l2 * W

        # dJ/db = (1/N) sum_i E_i
        gradb = E.mean(axis=0)

        # update
        W -= lr * gradW
        b -= lr * gradb

    return W, b

#argmax_k f_k(x)
def predict_linear(X, W, b, class_labels):
    scores = X @ W.T + b
    idx = np.argmax(scores, axis=1)
    return np.array([class_labels[j] for j in idx], dtype=int)



# EVALUATION + PCA DECISION REGIONS #
def evaluate_split(df, split_name, X_mu, X_std, class_labels, W, b):
    d = df[df["split"] == split_name].copy()
    d = d.dropna(subset=CONT_FEATURES + CAT_COLS + [LABEL])

    X = build_full_features(d, CONT_FEATURES, CAT_COLS, CAT_SIZES)
    X = standardize_apply(X, X_mu, X_std)

    y_true = d[LABEL].astype(int).values
    y_pred = predict_linear(X, W, b, class_labels)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    return acc, cm

#c): In the 2D PCA projection (PC1, PC2) overlay decision regions
# PCA is fit on TRAIN only.
""" Decision regions are computed by evaluating the linear classifier
  on a grid in PCA space mapping grid points back to the original
  standardized feature space using inverse projection (approximate) """
def pca_projection_and_regions(X_train_std, y_train, W, b, class_labels, out_dir):
    pca = PCA(n_components=2, random_state=0)
    Z_train = pca.fit_transform(X_train_std)  # (N,2)

    # margin
    xmin, xmax = Z_train[:, 0].min(), Z_train[:, 0].max()
    ymin, ymax = Z_train[:, 1].min(), Z_train[:, 1].max()
    dx = 0.10 * (xmax - xmin + EPS)
    dy = 0.10 * (ymax - ymin + EPS)
    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy

    H, Wg = 250, 250
    xs = np.linspace(xmin, xmax, Wg)
    ys = np.linspace(ymin, ymax, H)

    # grid in PCA space
    ZZ = np.array([[x, y] for y in ys for x in xs], dtype=float)  # (H*Wg, 2)


    # X_approx = ZZ @ P^T + mean
    X_grid = pca.inverse_transform(ZZ)  # (H*Wg, d) in standardized space
    # Predict on grid
    y_grid = predict_linear(X_grid, W, b, class_labels)
    # Reshape to image (H, Wg)
    grid_pred = y_grid.reshape(H, Wg)

    # Plot decision regions + TRAIN points
    plot_pca_decision_regions(
        Z_train=Z_train,
        y_train=y_train,
        grid_pred=grid_pred,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        class_labels=class_labels,
        save_path=out_dir / "pca_decision_regions_train.png",
        title="Q4 Linear (SSE) - Decision regions in PCA space (TRAIN points)"
    )




def main():

    df = load_data()

    df = df.dropna(subset=CONT_FEATURES + CAT_COLS + [LABEL])

    df_tr = df[df["split"] == "TRAIN"].copy()
    df_va = df[df["split"] == "VAL"].copy()

    # Classes (S=8)
    class_labels = sorted(df_tr[LABEL].astype(int).unique().tolist())

    X_tr = build_full_features(df_tr, CONT_FEATURES, CAT_COLS, CAT_SIZES)
    X_va = build_full_features(df_va, CONT_FEATURES, CAT_COLS, CAT_SIZES)

    y_tr = df_tr[LABEL].astype(int).values
    y_va = df_va[LABEL].astype(int).values

    # One-hot targets for SSE
    T_tr = one_hot_targets(y_tr, class_labels)


    X_mu, X_std = standardize_fit(X_tr)
    X_tr_std = standardize_apply(X_tr, X_mu, X_std)
    X_va_std = standardize_apply(X_va, X_mu, X_std)

    best = {"l2": None, "acc_val": -np.inf, "W": None, "b": None}

    for l2 in L2_GRID:
        W_lin, b_lin = train_linear_sse(
            X_tr_std, T_tr, l2=l2, lr=LR, epochs=EPOCHS, seed=0
        )

        y_pred_va = predict_linear(X_va_std, W_lin, b_lin, class_labels)
        acc_va = accuracy_score(y_va, y_pred_va)

        if acc_va > best["acc_val"]:
            best.update({"l2": l2, "acc_val": acc_va, "W": W_lin, "b": b_lin})

    W_best, b_best, l2_best = best["W"], best["b"], best["l2"]


    # Final evaluation: TRAIN (sanity) + VAL (report)
    y_pred_tr = predict_linear(X_tr_std, W_best, b_best, class_labels)
    acc_tr = accuracy_score(y_tr, y_pred_tr)

    y_pred_va = predict_linear(X_va_std, W_best, b_best, class_labels)
    acc_va = accuracy_score(y_va, y_pred_va)

    cm_va = confusion_matrix(y_va, y_pred_va, labels=class_labels)

    # Save metrics
    with open(OUT_DIR / "acc.txt", "w", encoding="utf-8") as f:
        f.write(f"Best L2 = {l2_best}\n")
        f.write(f"TRAIN accuracy = {acc_tr:.6f}\n")
        f.write(f"VAL accuracy   = {acc_va:.6f}\n")

    np.savetxt(OUT_DIR / "cm_VAL.csv", cm_va, delimiter=",", fmt="%d")

    plot_confusion_matrix(
        cm_va,
        class_labels=class_labels,
        save_path=OUT_DIR / "cm_VAL.png",
        title="Q4 Linear (SSE) - Confusion Matrix (VAL)"
    )

    # c) PCA decision regions with TRAIN points
    pca_projection_and_regions(
        X_train_std=X_tr_std,
        y_train=y_tr,
        W=W_best,
        b=b_best,
        class_labels=class_labels,
        out_dir=OUT_DIR
    )

    print("Q4 done.")
    print(f"Best L2={l2_best} | TRAIN acc={acc_tr:.4f} | VAL acc={acc_va:.4f}")
    print("Saved results in results/Q4_linear_sse/")


if __name__ == "__main__":
    main()
