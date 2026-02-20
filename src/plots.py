import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(data, column_name, save_path, bins=30):
    plt.figure()
    plt.hist(data.dropna(),
             bins=bins,
             density=True,
             alpha=0.7)

    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# histogram + single Gaussian + GMM.
def plot_gaussian_vs_gmm(x, xs, pdf_gauss, pdf_gmm, mu, sigma, save_path):
    plt.figure()

    plt.hist(x,
             bins=30,
             density=True,
             alpha=0.5,
             label="Histogram")

    plt.plot(xs,
             pdf_gauss,
             linewidth=2,
             label=f"Gaussian (μ={mu:.2f}, σ={sigma:.2f})")

    plt.plot(xs,
             pdf_gmm,
             linewidth=2,
             label="GMM (3 components)")

    plt.xlabel("hour_float")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#2D scatter plot without labels
def plot_2d_scatter(x, y, xlabel, ylabel, save_path):
    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




def plot_cov_heatmap(cov, feature_names, save_path, title="Covariance heatmap"):
    cov = np.asarray(cov)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cov, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.yticks(range(len(feature_names)), feature_names)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# for Q2(c):
#compute μ^(2), Σ^(2) with MLE (/N)
#compute D_i^2 Mahalanobis
#set c_k = max D_i^2
#draw ellipse
def plot_ellipse_2d(X2, save_path, xlabel, ylabel, title="Ellipse"):
    X2 = np.asarray(X2)
    if X2.ndim != 2 or X2.shape[1] != 2 or X2.shape[0] < 2:
        return

    N = X2.shape[0]
    mu = X2.mean(axis=0)
    Xc = X2 - mu
    Sigma = (Xc.T @ Xc) / N

    # numerical stability
    eps = 1e-9
    Sigma = Sigma + eps * np.eye(2)

    Sigma_inv = np.linalg.inv(Sigma)
    D2 = np.einsum("ni,ij,nj->n", Xc, Sigma_inv, Xc)
    ck = float(np.max(D2))

    # ellipse param
    t = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])  # (2, T)

    # map unit circle
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    A = eigvecs @ np.diag(np.sqrt(eigvals))  # Σ^(1/2)

    ellipse = (mu.reshape(2,1) + np.sqrt(ck) * (A @ circle)).T  # (T,2)

    plt.figure(figsize=(6, 6))
    plt.scatter(X2[:,0], X2[:,1], s=10, alpha=0.5)
    plt.plot(ellipse[:,0], ellipse[:,1], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# confusion matrix plot (counts).
#cm: (K,K) numpy array
#class_labels: list of class ids (same order as cm)
def plot_confusion_matrix(cm, class_labels, save_path, title="Confusion Matrix"):
    cm = np.asarray(cm)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm, aspect="auto")
    plt.colorbar(im)
    plt.title(title)

    plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha="right")
    plt.yticks(range(len(class_labels)), class_labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pca_decision_regions(
    Z_train, y_train,  # TRAIN points in PCA-2D and their true labels
    grid_pred,         # 2D grid  predicted labels (shape [H,W])
    xlim, ylim,        # (xmin,xmax), (ymin,ymax)
    class_labels,
    save_path,
    title="Decision regions in PCA space"
):
    # decision regions in a 2D PCA plane.
    # Z_train: (N,2)
    # y_train: (N,)
    # grid_pred: (H,W) int labels
    Z_train = np.asarray(Z_train)
    y_train = np.asarray(y_train)

    xmin, xmax = xlim
    ymin, ymax = ylim

    plt.figure(figsize=(7, 6))
    # region map
    plt.imshow(
        grid_pred,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
        alpha=0.35
    )

    # overlay TRAIN points
    plt.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, s=10, alpha=0.75)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
