import pandas as pd
from pathlib import Path

#dataset
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "crimes.csv"

#loading crimes.csv and returning pandas DataFrame.
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df

#splitting dataset in partitions TRAIN, VAL, TEST
def split_data(df):
    if "split" not in df.columns:
        raise ValueError("Column 'split' not found in dataset.")

    train = df[df["split"] == "TRAIN"].copy()
    val   = df[df["split"] == "VAL"].copy()
    test  = df[df["split"] == "TEST"].copy()

    return train, val, test


if __name__ == "__main__":
    df = load_data()
    tr, va, te = split_data(df)

    print("Full dataset shape:", df.shape)
    print("TRAIN shape:", tr.shape)
    print("VAL shape:", va.shape)
    print("TEST shape:", te.shape)
