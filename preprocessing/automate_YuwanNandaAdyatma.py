"""
automate_YuwanNandaAdyatma.py
Skrip otomatisasi preprocessing Heart Disease Dataset.
Jalankan: python automate_YuwanNandaAdyatma.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
def load_data(filepath='heart.csv'):
    print("[1/4] Memuat dataset...")
    df = pd.read_csv(filepath)

    os.makedirs('heart_raw', exist_ok=True)
    df.to_csv('heart_raw/heart.csv', index=False)

    print(f"    Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ─────────────────────────────────────────
# 2. Validasi Data
# ─────────────────────────────────────────
def validate_data(df):
    print("[2/4] Validasi data...")

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    if missing > 0:
        df = df.dropna()
        print(f"    Missing values ditemukan dan dihapus: {missing}")
    else:
        print(f"    Missing values: 0 ✓")

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"    Duplikat ditemukan dan dihapus: {duplicates}")
    else:
        print(f"    Duplikat: 0 ✓")

    return df


# ─────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────
def preprocess(df):
    print("[3/4] Preprocessing...")

    X = df.drop('target', axis=1).values
    y = df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"    X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"    y_train distribusi: {np.bincount(y_train)}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# 4. Simpan Hasil
# ─────────────────────────────────────────
def save_data(X_train, X_test, y_train, y_test, output_dir='heart_preprocessing'):
    print("[4/4] Menyimpan hasil preprocessing...")

    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy',  X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy',  y_test)

    print(f"    File tersimpan di '{output_dir}/':")
    for f in os.listdir(output_dir):
        print(f"      - {f}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Automate Preprocessing — Heart Disease Dataset")
    print("=" * 50)

    df = load_data()
    df = validate_data(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    save_data(X_train, X_test, y_train, y_test)

    print("=" * 50)
    print("  Preprocessing selesai! Data siap dilatih.")
    print("=" * 50)


if __name__ == '__main__':
    main()
