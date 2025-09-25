"""
wine_quality_project.py

Wine Quality Prediction (RandomForest, SGDClassifier, SVC)
- EDA, preprocessing, training, evaluation
- Saves plots, trained models, and a model summary CSV to OUTPUT_DIR
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main(data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # 1. Basic cleaning & inspection
    print("\nMissing values per column:\n", df.isnull().sum())
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped duplicates: {before - df.shape[0]} rows removed")

    # 2. Create categorical target: low (<=5), medium (==6), high (>=7)
    # Use a deterministic mapping to avoid categorical casting warnings
    def label_quality(q):
        if q <= 5:
            return 'low'
        elif q == 6:
            return 'medium'
        else:
            return 'high'
    df['quality_label'] = df['quality'].apply(label_quality)

    print("\nQuality label distribution:")
    print(df['quality_label'].value_counts())

    # 3. Quick EDA plots (saved)
    sns.set(style="whitegrid")

    # 3a: quality counts
    plt.figure(figsize=(6,4))
    counts = df['quality_label'].value_counts().reindex(['low','medium','high'])
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Wine Quality Label Counts')
    plt.xlabel('Quality Label'); plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_label_counts.png'))
    plt.close()

    # 3b: Correlation matrix (numeric features)
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=False)
    plt.title('Correlation Matrix (numeric features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # 3c: Scatter example: alcohol vs quality (numeric)
    plt.figure(figsize=(6,4))
    jitter = np.random.normal(0, 0.05, size=df.shape[0])
    plt.scatter(df['alcohol'], df['quality'] + jitter, s=12, alpha=0.6)
    plt.xlabel('Alcohol'); plt.ylabel('Quality (numeric)')
    plt.title('Alcohol vs Quality (scatter)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alcohol_vs_quality.png'))
    plt.close()

    # 4. Prepare data for modeling
    # Select numeric feature columns and drop 'quality' and any ID column if present
    X = numeric.copy()
    for col in ['quality', 'Id']:  # drop if exist
        if col in X.columns:
            X = X.drop(columns=[col])
    y = df['quality_label'].astype(str)

    # Fill any numeric missing values with median (safe fallback)
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    # Align and drop any rows with missing label (shouldn't happen)
    data = pd.concat([X, y], axis=1).dropna()
    X = data[X.columns]
    y = data['quality_label']

    # 5. Train/test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nTrain/Test shapes:", X_train.shape, X_test.shape)

    # 6. Feature scaling (important for SGD & SVC)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))

    # 7. Define models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'sgd': SGDClassifier(max_iter=2000, tol=1e-3, random_state=42),
        'svc': SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        # Train using scaled features (consistent)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")
        print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
        results[name] = {'model': model, 'accuracy': acc, 'y_pred': y_pred}
        # Save model
        joblib.dump(model, os.path.join(output_dir, f"{name}.joblib"))

    # 8. Feature importance (RandomForest)
    rf = results['random_forest']['model']
    if hasattr(rf, 'feature_importances_'):
        fi = rf.feature_importances_
        feat_names = np.array(X.columns)
        idx = np.argsort(fi)[::-1]
        plt.figure(figsize=(8,6))
        sns.barplot(x=fi[idx], y=feat_names[idx])
        plt.title('Random Forest Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_feature_importances.png'))
        plt.close()

    # 9. Confusion matrix for best model
    best_name = max(results.items(), key=lambda kv: kv[1]['accuracy'])[0]
    best_pred = results[best_name]['y_pred']
    print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")

    labels = ['low','medium','high']
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label'); plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {best_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{best_name}.png'))
    plt.close()

    # 10. Save model summary CSV
    summary = pd.DataFrame([{'model': name, 'accuracy': results[name]['accuracy']} for name in results])
    summary.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)

    print("\nAll outputs saved to:", output_dir)
    print("Saved files:", os.listdir(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Quality Prediction pipeline")
    parser.add_argument("--data_path", type=str, default="winequality.csv", help="CSV file path")
    parser.add_argument("--output_dir", type=str, default="wine_quality_outputs", help="Directory to save outputs")
    args = parser.parse_args()
    main(args.data_path, args.output_dir)
