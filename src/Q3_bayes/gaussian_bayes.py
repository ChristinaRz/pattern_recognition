import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, confusion_matrix


# PATH SETUP #
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # pattern_recognition/
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

# Import project utilities
from load_data import load_data
from plots import plot_confusion_matrix


# CONFIG #
LABEL = "killer_id"

# Continuous descriptors (dc = 8) exactly as Q2
FEATURES = [
    "hour_float",
    "latitude",
    "longitude",
    "victim_age",
    "temp_c",
    "humidity",
    "dist_precinct_km",
    "pop_density",
]

# Q2 parameters
Q2_DIR = PROJECT_ROOT / "results" / "Q2_mle"

# Q3 outputs
OUT_DIR = PROJECT_ROOT / "results" / "Q3_bayes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Small ridge for numerical stability
EPS = 1e-6

#  load Q2 params
def load_q2_parameters(q2_dir: Path):
    mus, sigmas, Nks = {}, {}, {}

    # Use *_mu.npy as index
    mu_files = sorted(q2_dir.glob("*_mu.npy"))
    if len(mu_files) == 0:
        raise FileNotFoundError(
            f"No Q2 parameter files found in {q2_dir}. "
            "Run Q2 first to create *_mu.npy, *_sigma.npy, *_Nk.npy."
        )

    for mu_file in mu_files:
        k = int(mu_file.stem.split("_")[0])

        mu = np.load(mu_file)
        Sigma = np.load(q2_dir / f"{k}_sigma.npy")
        Nk_arr = np.load(q2_dir / f"{k}_Nk.npy")
        Nk = int(Nk_arr.reshape(-1)[0])

        mus[k] = mu
        sigmas[k] = Sigma
        Nks[k] = Nk

    classes = sorted(mus.keys())
    return classes, mus, sigmas, Nks


#log Gaussian density
def logpdf_mvn(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    d = x.shape[0]

    # Stabilize covariance
    Sigma_stable = Sigma + EPS * np.eye(d)

    # log(det(Sigma)) via stable decomposition
    sign, logdet = np.linalg.slogdet(Sigma_stable)
    if sign <= 0:
        # If still not SPD add more ridge
        Sigma_stable = Sigma_stable + 1e-3 * np.eye(d)
        sign, logdet = np.linalg.slogdet(Sigma_stable)

    # (x-mu)^T Sigma^{-1} (x-mu)
    xc = x - mu
    sol = np.linalg.solve(Sigma_stable, xc)  # Sigma^{-1} xc
    quad = float(xc.T @ sol)

    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)



#  Bayes rule #
def predict_bayes(X: np.ndarray, classes, mus, sigmas, priors):
    y_pred = np.empty((X.shape[0],), dtype=int)

    for i in range(X.shape[0]):
        x = X[i]

        best_k = None
        best_score = -np.inf

        for k in classes:
            score = logpdf_mvn(x, mus[k], sigmas[k]) + np.log(priors[k])

            if score > best_score:
                best_score = score
                best_k = k

        y_pred[i] = best_k

    return y_pred



def main():
    df = load_data()

    #VAL
    df_val = df[df["split"] == "VAL"].copy()

    df_val = df_val.dropna(subset=FEATURES + [LABEL])
    # q2
    classes, mus, sigmas, Nks = load_q2_parameters(Q2_DIR)

    # class priors
    N_total = sum(Nks.values())
    priors = {k: (Nks[k] / N_total) for k in classes}

    X_val = df_val[FEATURES].values
    y_true = df_val[LABEL].astype(int).values

    y_pred = predict_bayes(X_val, classes, mus, sigmas, priors)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    with open(OUT_DIR / "acc_VAL.txt", "w", encoding="utf-8") as f:
        f.write(f"VAL accuracy = {acc:.6f}\n")

    np.savetxt(OUT_DIR / "cm_VAL.csv", cm, delimiter=",", fmt="%d")

    plot_confusion_matrix(
        cm=cm,
        class_labels=classes,
        save_path=OUT_DIR / "cm_VAL.png",
        title="Confusion Matrix (VAL) - Gaussian Bayes"
    )

    print(f"Q3 done. VAL accuracy = {acc:.4f}")
    print("Saved: results/Q3_bayes/acc_VAL.txt, cm_VAL.csv, cm_VAL.png")


if __name__ == "__main__":
    main()
