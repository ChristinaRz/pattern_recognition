Pattern Recognition – Who Is The Killer?

This project investigates multiclass killer identification in the Piraeus Vice dataset using statistical and machine learning methods.

The objective is to predict the most likely killer for each crime incident based on spatial, temporal and contextual features.

Project Structure
pattern_recognition/
│
├── data/
│   └── raw/                # Original dataset
│
├── src/                    # All implementation scripts (Q1–Q8)
│   ├── load_data.py
│   ├── gaussian_bayes.py
│   ├── linear_classifier.py
│   ├── svm_classifier.py
│   ├── mlp_classifier.py
│   ├── pca_linear.py
│   ├── kmeans_clustering.py
│   └── final_submission.py
│
├── results/                # Generated outputs and figures
├── report/                 # LaTeX report
├── submission.csv          # Final test predictions
└── README.md

Implemented Methods

The following modelling approaches were implemented and evaluated:

Gaussian Maximum Likelihood estimation

Multiclass Gaussian Bayes classifier

Linear discriminative classifier

Support Vector Machine (RBF kernel)

Multi-Layer Perceptron (MLP)

Principal Component Analysis (PCA)

k-means clustering in PCA latent space

Model selection was performed using a validation split.
No data leakage was introduced: preprocessing and model fitting were performed using TRAIN only.


Data Processing

Each incident is represented by:

8 continuous features (standardised)

17 one-hot encoded categorical variables

Total feature dimension: 25

Preprocessing steps (imputation, scaling, PCA fitting) are fitted exclusively on the TRAIN split and then applied to VAL and TEST.

Reproducing Results

From the project root:

Run any question script individually:
python src/gaussian_bayes.py
python src/svm_classifier.py
python src/mlp_classifier.py
Generated figures and tables are stored under results/ and are directly used in the report.

Reproducing the Final Submission

To generate the final prediction file: python src/final_submission.py
This script:

Constructs the 25-dimensional feature vector

Fits preprocessing on TRAIN only

Trains the selected MLP model

Produces submission.csv with:
incident_id
predicted_killer
p_killer_1 … p_killer_8

Built-in checks ensure:
Posterior probabilities sum to 1
No NaN or negative values
Predicted label equals argmax of probabilities

Validation Performance (Summary)
Model	VAL Accuracy
Gaussian Bayes	0.9050
Linear classifier	0.8664
SVM (RBF)	0.9436
MLP (final)	0.9426
PCA + k-means	0.8236

Non-linear discriminative models provided the strongest performance.

Notes
The repository excludes the virtual environment (venv/).
The project is fully reproducible from the provided scripts.
The report was written in LaTeX and all figures are programmatically generated.


