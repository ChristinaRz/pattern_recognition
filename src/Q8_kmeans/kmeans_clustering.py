from __future__ import annotations
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# PATH SETUP #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
CAT_SIZES = [6, 4, 5, 2]  # dcat = 17

# Output
OUT_DIR = PROJECT_ROOT / "results" / "Q8_kmeans"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
SEED = 42
np.random.seed(SEED)


# FEATURE PIPELINE (match Q7: standardize continuous one-hot categorical) #
# no dropping rows in VAL/TEST.
#  continuous NaNs impute with TRAIN mean
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


def standardize_fit_continuous_from_train(df_tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    #fit μ and std on TRAIN only (continuous block)
    Xc = df_tr[CONT_FEATURES].values.astype(float)
    # TRAIN can contain NaNs without dropping rows
    mu = np.nanmean(Xc, axis=0)
    std = np.nanstd(Xc, axis=0)
    std = np.where(std < EPS, 1.0, std)
    return mu, std


def build_full_features_q7(df: pd.DataFrame, mu_c: np.ndarray, std_c: np.ndarray) -> np.ndarray:
    #x = [x_continuous_std (8 dims), x_onehot (17 dims)] => d=25
    Xc = df[CONT_FEATURES].values.astype(float)

    # impute continuous NaNs with TRAIN mean
    Xc = np.where(np.isnan(Xc), mu_c.reshape(1, -1), Xc)

    # standardize continuous
    Xc_std = (Xc - mu_c.reshape(1, -1)) / std_c.reshape(1, -1)

    # one-hot categoricals
    Xcat = one_hot_encode_categoricals(df)

    return np.hstack([Xc_std, Xcat])


# DATA HELPERS #
def get_split_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    d = df[df["split"] == split_name].copy()
    return d


def ensure_incident_id(df: pd.DataFrame) -> pd.Series:
    if INCIDENT_ID_COL in df.columns:
        return df[INCIDENT_ID_COL]
    # for fallback use row index (deterministic)
    return pd.Series(df.index.astype(int), index=df.index, name=INCIDENT_ID_COL)


def infer_S_and_class_labels_from_train(y_tr: np.ndarray) -> tuple[int, list[int]]:
    #S is inferred from TRAIN labels and class_labels are [1..S]
    # to avoid breaking the required submission schema
    y_tr = y_tr.astype(int)
    if y_tr.size == 0:
        raise ValueError("Empty TRAIN labels: cannot infer S.")
    S = int(np.nanmax(y_tr))
    if S <= 0:
        raise ValueError(f"Invalid inferred S={S} from TRAIN.")
    class_labels = list(range(1, S + 1))
    return S, class_labels


# Q8 MAPPING  #
def build_majority_vote_mapping_and_probs(
    clusters_tr: np.ndarray,
    y_tr: np.ndarray,
    S: int,
) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    # g(q) = argmax_{k in {1,...,S}}  sum_{i in TRAIN} δ_{c_i^{(km)},q} δ_{K_i,k}

    clusters_tr = clusters_tr.astype(int)
    y_tr = y_tr.astype(int)

    K = int(np.max(clusters_tr)) + 1  # K = number of clusters = S
    if K != S:
        # keep a hard check because the statement fixes k=S.
        raise ValueError(f"k-means produced K={K} clusters but expected S={S}.")

    # count[q, k-1]
    count = np.zeros((S, S), dtype=int)
    for q, lab in zip(clusters_tr, y_tr):
        # ignore invalid labels
        if 1 <= int(lab) <= S:
            count[int(q), int(lab) - 1] += 1

    #fallback for empty clusters
    total = count.sum(axis=0).astype(float)
    prior = total / max(float(total.sum()), 1.0)
    if float(prior.sum()) <= 0:
        prior = np.ones(S, dtype=float) / float(S)

    mapping: dict[int, int] = {}
    cluster_probs: dict[int, np.ndarray] = {}

    for q in range(S):
        row = count[q].astype(float)
        n_q = float(row.sum())

        if n_q <= 0:
            probs = prior.copy()
        else:
            probs = row / n_q

        # g(q) = argmax_k probs[k-1]
        pred_k = int(np.argmax(probs)) + 1  # back to label space {1..S}
        mapping[q] = pred_k
        cluster_probs[q] = probs

    return mapping, cluster_probs


def predict_from_clusters(
    clusters: np.ndarray,
    mapping: dict[int, int],
    cluster_probs: dict[int, np.ndarray],
    S: int,
) -> tuple[np.ndarray, np.ndarray]:

     #returns y_hat: predicted_killer in {1..S}
     # P: (N,S) probability matrix
    clusters = clusters.astype(int)
    N = clusters.shape[0]

    y_hat = np.zeros(N, dtype=int)
    P = np.zeros((N, S), dtype=float)

    for i, q in enumerate(clusters):
        q = int(q)
        y_hat[i] = int(mapping[q])
        P[i] = cluster_probs[q]

    # numerical safety
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < EPS, 1.0, row_sums)
    P = P / row_sums
    return y_hat, P


def plot_test_scatter_pc1_pc2(Z_te_2d: np.ndarray, y_hat_te: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(Z_te_2d[:, 0], Z_te_2d[:, 1], c=y_hat_te, s=10, alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("TEST projected to PC1-PC2 (colored by k-means-based killer predictions)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Number of PCA components m (from elbow choice in Q7).",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=20,
        help="k-means n_init",
    )
    args = parser.parse_args()

    df = load_data()

    df_tr = get_split_df(df, "TRAIN")
    df_va = get_split_df(df, "VAL")
    df_te = get_split_df(df, "TEST")

    # TRAIN must contain labels for mapping
    if LABEL not in df_tr.columns:
        raise ValueError("TRAIN split does not contain killer_id. Cannot build majority-vote mapping.")

    # Read TRAIN labels
    y_tr = df_tr[LABEL].astype(float).values
    y_tr = np.where(np.isnan(y_tr), 0, y_tr).astype(int)

    S, CLASS_LABELS = infer_S_and_class_labels_from_train(y_tr)

    # PCA preprocessing like Q7 (fit on TRAIN only)
    mu_c, std_c = standardize_fit_continuous_from_train(df_tr)

    X_tr = build_full_features_q7(df_tr, mu_c, std_c)
    X_va = build_full_features_q7(df_va, mu_c, std_c)
    X_te = build_full_features_q7(df_te, mu_c, std_c)

    # m selection: require user to pass --m
    m = args.m
    if m is None:
        raise ValueError("m not provided. Run with: python scripts/Q8_kmeans.py --m <elbow_m>")

    m = int(max(1, min(int(m), X_tr.shape[1])))

    # PCA
    pca = PCA(n_components=m, random_state=0)
    Z_tr = pca.fit_transform(X_tr)   # TRAIN only
    Z_va = pca.transform(X_va)
    Z_te = pca.transform(X_te)

    # k-means with k=S
    kmeans = KMeans(n_clusters=S, n_init=int(args.n_init), random_state=SEED)
    clusters_tr = kmeans.fit_predict(Z_tr)   # 0..S-1
    clusters_va = kmeans.predict(Z_va)       # 0..S-1
    clusters_te = kmeans.predict(Z_te)       # 0..S-1

    # Mapping cluster = killer label using majority vote on TRAIN + probabilities
    mapping, cluster_probs = build_majority_vote_mapping_and_probs(
        clusters_tr=clusters_tr,
        y_tr=y_tr,
        S=S,
    )


    yhat_tr, P_tr = predict_from_clusters(clusters_tr, mapping, cluster_probs, S)
    yhat_va, P_va = predict_from_clusters(clusters_va, mapping, cluster_probs, S)
    yhat_te, P_te = predict_from_clusters(clusters_te, mapping, cluster_probs, S)

    #VAL accuracy
    if LABEL in df_va.columns and df_va[LABEL].notna().any():
        y_va = df_va[LABEL].astype(float).values
        y_va = np.where(np.isnan(y_va), 0, y_va).astype(int)
        acc_va = float(accuracy_score(y_va, yhat_va))
        cm_va = confusion_matrix(y_va, yhat_va, labels=CLASS_LABELS)

        (OUT_DIR / "acc_VAL.txt").write_text(f"VAL accuracy (kmeans->majority vote) = {acc_va:.6f}\n", encoding="utf-8")
        np.savetxt(OUT_DIR / "cm_VAL.csv", cm_va, delimiter=",", fmt="%d")
    else:
        (OUT_DIR / "acc_VAL.txt").write_text(
            "VAL labels not available in dataframe. Use score.py on exported VAL predictions if required.\n",
            encoding="utf-8",
        )

    # Save mapping + cluster probs
    lines = []
    lines.append(f"S = {S}")
    lines.append(f"m = {m}")
    lines.append("class_labels (fixed by schema) = " + str(CLASS_LABELS))
    lines.append("cluster q -> g(q) killer (majority vote on TRAIN):")
    for q in range(S):
        lines.append(f"  cluster {q} -> {mapping[q]}")
    (OUT_DIR / "cluster_to_killer_mapping.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    prob_lines = []
    prob_lines.append("Per-cluster probability vectors p_killer_k estimated from TRAIN counts (k=1..S):")
    for q in range(S):
        probs = cluster_probs[q]
        probs_str = ", ".join([f"p_killer_{k}={probs[k-1]:.6f}" for k in range(1, S + 1)])
        prob_lines.append(f"  cluster {q}: {probs_str}")
    (OUT_DIR / "cluster_probs.txt").write_text("\n".join(prob_lines) + "\n", encoding="utf-8")

    # --- Export VAL predictions (optional) ---
    val_ids = ensure_incident_id(df_va).values
    val_pred_df = pd.DataFrame({INCIDENT_ID_COL: val_ids, "predicted_killer": yhat_va.astype(int)})
    for k in range(1, S + 1):
        val_pred_df[f"p_killer_{k}"] = P_va[:, k - 1]
    val_pred_df.to_csv(OUT_DIR / "val_predictions.csv", index=False)


    test_ids = ensure_incident_id(df_te).values
    sub = pd.DataFrame({INCIDENT_ID_COL: test_ids, "predicted_killer": yhat_te.astype(int)})
    for k in range(1, S + 1):
        sub[f"p_killer_{k}"] = P_te[:, k - 1]
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    # TEST scatter in PC1-PC2 colored by final predicted labels
    # I use the first two PCs of the PCA fitted on TRAIN (same PCA object)
    if Z_te.shape[1] >= 2:
        Z_te_2d = Z_te[:, :2]
    else:
        # if m=1, pad with zeros
        Z_te_2d = np.hstack([Z_te, np.zeros((Z_te.shape[0], 1))])

    plot_test_scatter_pc1_pc2(
        Z_te_2d=Z_te_2d,
        y_hat_te=yhat_te,
        save_path=OUT_DIR / "test_scatter_pc1_pc2_colored_by_kmeans.png",
    )

    (OUT_DIR / "sanity.txt").write_text(
        "\n".join(
            [
                "Sanity checks:",
                " PCA fit on TRAIN only",
                " k-means fit on TRAIN only (latent space)",
                " cluster->killer mapping built using TRAIN labels only",
                " submission schema uses p_killer_1..p_killer_S exactly",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Q8 done.")
    print(f"S = {S} | class_labels = 1..S")
    print(f"PCA: fit on TRAIN only | m = {m}")
    print(f"KMeans: fit on TRAIN only | k = S = {S}")
    print(f"Saved results in: {OUT_DIR}")
    print(f"Saved submission: {OUT_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()
