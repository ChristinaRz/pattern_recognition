# Q1 – Exploratory Distributions #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture  # GMM (EM algorithm)
from pathlib import Path
import sys

# pattern_recognition root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# src directory
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from load_data import load_data
from plots import plot_histogram, plot_gaussian_vs_gmm, plot_2d_scatter


# folder saving figs
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = load_data()   # loading dataset

# TRAIN + VAL
df_ev = df[df["split"].isin(["TRAIN", "VAL"])].copy()



# HISTOGRAMS #
cols = ["hour_float", "victim_age", "latitude", "longitude"]

for c in cols:
    plot_histogram(
        data=df_ev[c],
        column_name=c,
        save_path=FIG_DIR / f"hist_{c}.png",
        bins=30
    )



#  Single Gaussian fit (MLE)  hour_float #
# hour_float
x = df_ev["hour_float"].dropna().values
mu = np.mean(x)                      # MLE mean
sigma = np.sqrt(np.mean((x - mu)**2))  # divided by N
#grid
xs = np.linspace(x.min(), x.max(), 500)
pdf_gauss = norm.pdf(xs, mu, sigma)


# Gaussian Mixture Model (3 components) #
# 2D input (N,1)
gmm = GaussianMixture(n_components=3, random_state=0)
# EM algorithm
gmm.fit(x.reshape(-1, 1))
# log-density
pdf_gmm = np.exp(gmm.score_samples(xs.reshape(-1, 1)))

# Histogram – Gaussian – GMM
plot_gaussian_vs_gmm(
    x=x,
    xs=xs,
    pdf_gauss=pdf_gauss,
    pdf_gmm=pdf_gmm,
    mu=mu,
    sigma=sigma,
    save_path=FIG_DIR / "hour_float_gaussian_vs_gmm.png"
)

# 2D plot hour_float with latitude
plot_2d_scatter(
    x=df_ev["latitude"],
    y=df_ev["hour_float"],
    xlabel="latitude",
    ylabel="hour_float",
    save_path=FIG_DIR / "hour_float_vs_latitude.png"
)


print("Q1 completed. Figures saved in /figures.")
