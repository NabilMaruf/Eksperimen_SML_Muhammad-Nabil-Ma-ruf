"""
automate_Nabil.py
Preprocessing otomatis untuk dataset Heart Disease (heart.csv)

Output:
- output_dir/
  - X_train.csv, X_test.csv, y_train.csv, y_test.csv
  - preprocessing_pipeline.joblib
  - meta.json (info kolom & parameter)

Cara pakai (lokal):
python automate_Nabil.py --input heart.csv --output preprocessing/heart_preprocessed --test-size 0.2 --random-state 42
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessConfig:
    target_col: str = "target"
    test_size: float = 0.2
    random_state: int = 42


def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Menentukan kolom numerik vs kategorikal.
    Dataset Heart Disease biasanya sudah integer-coded untuk kategori.
    Kita treat sebagian kolom integer-coded sebagai kategorikal biar aman.
    """
    categorical_candidates = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    categorical_cols = [c for c in categorical_candidates if c in df.columns and c != target_col]
    numeric_cols = [c for c in df.columns if c not in categorical_cols and c != target_col]
    return numeric_cols, categorical_cols


def build_preprocess_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )


def preprocess_and_split(df: pd.DataFrame, config: PreprocessConfig):
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' tidak ditemukan. Kolom yang ada: {list(df.columns)}")

    df = df.drop_duplicates().copy()

    # handle missing sederhana (kalau ada)
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    X = df.drop(columns=[config.target_col])
    y = df[config.target_col].astype(int)

    numeric_cols, categorical_cols = infer_feature_types(df, config.target_col)
    preprocessor = build_preprocess_pipeline(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    preprocessor.fit(X_train)

    X_train_arr = preprocessor.transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    # nama kolom setelah transform
    feature_names = []
    feature_names.extend(numeric_cols)

    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(list(ohe.get_feature_names_out(categorical_cols)))

    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names)

    meta = {
        "target_col": config.target_col,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_features_after": len(feature_names),
        "n_rows_after_drop_duplicates": int(df.shape[0]),
        "train_rows": int(X_train_df.shape[0]),
        "test_rows": int(X_test_df.shape[0]),
        "class_balance_train": y_train.value_counts().to_dict(),
        "class_balance_test": y_test.value_counts().to_dict(),
    }

    return X_train_df, X_test_df, y_train, y_test, preprocessor, meta


def save_outputs(output_dir, X_train, X_test, y_train, y_test, preprocessor, meta):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessing_pipeline.joblib"))
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    config = PreprocessConfig(args.target_col, args.test_size, args.random_state)

    X_train, X_test, y_train, y_test, preprocessor, meta = preprocess_and_split(df, config)
    save_outputs(args.output, X_train, X_test, y_train, y_test, preprocessor, meta)

    print("✅ Preprocessing selesai.")
    print(f"Output: {args.output}")
    print(f"Fitur setelah preprocessing: {meta['n_features_after']}")
    print(f"Train: {meta['train_rows']} | Test: {meta['test_rows']}")


if __name__ == "__main__":
    main()
