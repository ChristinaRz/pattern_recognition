import numpy as np
from pathlib import Path
import sys

from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # pattern_recognition/
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from load_data import load_data
from plots import plot_cov_heatmap, plot_ellipse_2d

#output
OUT_DIR = PROJECT_ROOT / "results" / "Q2_mle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data TRAIN only
df = load_data()
df_tr = df[df["split"] == "TRAIN"].copy()

FEATURES = [
    "hour_float",
    "latitude",
    "longitude",
    "victim_age",
    "temp_c",
    "humidity",
    "dist_precinct_km",
    "pop_density"
]
LABEL = "killer_id"

results = {}

#Return MLE mean and covariance divide by N
def mle_gaussian_params(X: np.ndarray):
    N = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    sigma = (Xc.T @ Xc) / N
    return mu, sigma

#Total log-likelihood under N(μ,Σ)
def loglik_gaussian(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):

    # Add ridge for numerical stability (MLE cov can be singular)
    eps = 1e-6
    sigma_stable = sigma + eps * np.eye(sigma.shape[0])

    return float(np.sum(
        multivariate_normal.logpdf(
            X,
            mean=mu,
            cov=sigma_stable,
            allow_singular=True
        )
    ))


for killer, g in df_tr.groupby(LABEL):

    X = g[FEATURES].dropna().values
    Nk = X.shape[0]
    if Nk < 2:
        continue

    # a) from-scratch MLE
    mu, sigma = mle_gaussian_params(X)

    # Save params
    np.save(OUT_DIR / f"{killer}_mu.npy", mu)
    np.save(OUT_DIR / f"{killer}_sigma.npy", sigma)
    np.save(OUT_DIR / f"{killer}_Nk.npy", np.array([Nk]))
    print("killer:", killer, "Nk:", Nk)


    # b) Verify log-likelihood vs trusted library (MLE)
    # EmpiricalCovariance uses MLE covariance
    lib = EmpiricalCovariance(assume_centered=False).fit(X)
    mu_lib = lib.location_
    sigma_lib = lib.covariance_

    ll_ours = loglik_gaussian(X, mu, sigma)
    ll_lib  = loglik_gaussian(X, mu_lib, sigma_lib)
    ll_diff = abs(ll_ours - ll_lib)

    # Store summary
    results[killer] = {
        "Nk": Nk,
        "mu": mu,
        "sigma": sigma,
        "ll_ours": ll_ours,
        "ll_lib": ll_lib,
        "ll_abs_diff": ll_diff
    }

    # c) Heatmap covariance
    plot_cov_heatmap(
        cov=sigma,
        feature_names=FEATURES,
        save_path=OUT_DIR / f"{killer}_cov_heatmap.png",
        title=f"Covariance heatmap (killer {killer})"
    )

    #  Ellipse in (lat, lon)
    X2_latlon = g[["latitude", "longitude"]].dropna().values
    if X2_latlon.shape[0] > 1:
        plot_ellipse_2d(
            X2=X2_latlon,
            save_path=OUT_DIR / f"{killer}_ellipse_lat_lon.png",
            xlabel="latitude",
            ylabel="longitude",
            title=f"Ellipse (lat, lon) containing TRAIN points (killer {killer})"
        )

    #  Ellipse in (lat, hour)
    X2_lathour = g[["latitude", "hour_float"]].dropna().values
    if X2_lathour.shape[0] > 1:
        plot_ellipse_2d(
            X2=X2_lathour,
            save_path=OUT_DIR / f"{killer}_ellipse_lat_hour.png",
            xlabel="latitude",
            ylabel="hour_float",
            title=f"Ellipse (lat, hour) containing TRAIN points (killer {killer})"
        )

# Save summary text
with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    for killer, res in results.items():
        f.write(f"Killer: {killer}\n")
        f.write(f"Nk: {res['Nk']}\n")
        f.write(f"mu: {res['mu']}\n")
        f.write(f"sigma:\n{res['sigma']}\n")
        f.write(f"loglik (ours): {res['ll_ours']:.6f}\n")
        f.write(f"loglik (lib) : {res['ll_lib']:.6f}\n")
        f.write(f"|diff|       : {res['ll_abs_diff']:.6e}\n\n")

print("Q2 done: MLE params + heatmaps + ellipses saved in results/Q2_mle/")
print("Q2(b) verification: log-likelihood differences written in summary.txt")
