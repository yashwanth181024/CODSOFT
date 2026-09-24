import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)


def load_churn_data(filepath="churn_data"):
    dataset = None
    for ext in ("", ".csv", ".xlsx"):
        try:
            reader  = pd.read_excel if ext == ".xlsx" else pd.read_csv
            dataset = reader(filepath + ext)
            print(f"[OK] Loaded → {filepath}{ext}  ({len(dataset):,} rows, {dataset.shape[1]} cols)")
            break
        except FileNotFoundError:
            continue
    if dataset is None:
        raise FileNotFoundError(f"Cannot find {filepath}. Place it in the same folder as this notebook.")
    return dataset


def inspect_data(df):
    divider = "─" * 65
    print(divider)
    print("  DATASET OVERVIEW")
    print(divider)
    print(f"  Shape       : {df.shape}")
    print(f"  Nulls       : {df.isnull().sum().sum()}")
    print(f"  Dtypes      : {dict(df.dtypes.value_counts())}")
    print(divider)
    print(df.head(4).to_string())
    print(divider + "\n")


def find_target_column(df):
    candidates = ["Churn", "churn", "CHURN", "target", "Target", "label", "Label"]
    for col in candidates:
        if col in df.columns:
            print(f"[OK] Target column found → '{col}'")
            return col
    raise ValueError(f"No target column found. Columns: {list(df.columns)}")


def encode_and_clean(df, target_col):
    data = df.copy()

    drop_cols = ["customerID", "CustomerID", "customer_id", "ID", "id",
                 "Unnamed: 0", "RowNumber", "CustomerId", "Surname"]
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

    if data[target_col].dtype == object:
        data[target_col] = data[target_col].map(
            lambda v: 1 if str(v).strip().lower() in ("yes", "1", "true", "churn") else 0
        )

    for col in data.columns:
        if col == target_col:
            continue
        if data[col].dtype == object:
            data[col] = data[col].astype(str).str.strip()
            data[col] = data[col].replace(" ", np.nan)
            try:
                data[col] = pd.to_numeric(data[col])
            except (ValueError, TypeError):
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].fillna("Unknown"))

    for col in data.select_dtypes(include="number").columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)

    y = data[target_col].astype(int)
    X = data.drop(columns=[target_col])

    print(f"[OK] Encoding complete")
    print(f"     Features : {X.shape[1]}")
    print(f"     Samples  : {X.shape[0]:,}")
    print(f"     Features : {list(X.columns)}\n")
    return X, y


def engineer_churn_features(X):
    X = X.copy()

    if "tenure" in X.columns and "MonthlyCharges" in X.columns:
        X["lifetime_value"]   = X["tenure"] * X["MonthlyCharges"]
        X["avg_monthly_spend"] = X["MonthlyCharges"] / (X["tenure"] + 1)

    if "TotalCharges" in X.columns and "MonthlyCharges" in X.columns:
        X["charge_ratio"] = X["TotalCharges"] / (X["MonthlyCharges"] + 1)

    if "tenure" in X.columns:
        X["new_customer"]  = (X["tenure"] <= 6).astype(int)
        X["loyal_customer"] = (X["tenure"] >= 48).astype(int)

    print(f"[OK] Feature engineering done → {X.shape[1]} total features\n")
    return X


def show_churn_balance(y):
    total   = len(y)
    stayed  = int((y == 0).sum())
    churned = int((y == 1).sum())
    rate    = churned / total * 100

    print("─" * 65)
    print("  CHURN DISTRIBUTION")
    print("─" * 65)
    print(f"  Stayed   (0) : {stayed:>8,}   ({100 - rate:.2f} %)")
    print(f"  Churned  (1) : {churned:>8,}   ({rate:.2f} %)")
    print(f"  Churn Rate   : {rate:.2f} %\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(["Stayed", "Churned"], [stayed, churned],
            color=["#2196F3", "#F44336"], width=0.45, edgecolor="white")
    ax1.set_ylabel("Number of Customers")
    ax1.set_title("Customer Churn Count", fontweight="bold")
    ax1.set_ylim(0, stayed * 1.15)
    for i, v in enumerate([stayed, churned]):
        ax1.text(i, v + stayed * 0.02, f"{v:,}",
                 ha="center", fontsize=9, fontweight="bold")

    ax2.pie([stayed, churned],
            labels=["Stayed", "Churned"],
            colors=["#2196F3", "#F44336"],
            autopct="%1.2f%%",
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white"},
            textprops={"fontsize": 10})
    ax2.set_title("Churn Rate", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plot1_churn_distribution.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot1_churn_distribution.png\n")


def plot_feature_vs_churn(X, y, top_n=6):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()[:top_n]
    if not numeric_cols:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    combined = X.copy()
    combined["Churn"] = y.values

    for idx, col in enumerate(numeric_cols):
        stayed  = combined[combined["Churn"] == 0][col].dropna()
        churned = combined[combined["Churn"] == 1][col].dropna()
        axes[idx].hist(stayed,  bins=30, alpha=0.6, color="#2196F3", label="Stayed",  density=True)
        axes[idx].hist(churned, bins=30, alpha=0.6, color="#F44336", label="Churned", density=True)
        axes[idx].set_title(col, fontweight="bold")
        axes[idx].set_ylabel("Density")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3, axis="y")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Feature Distributions by Churn Status", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("plot2_feature_distributions.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot2_feature_distributions.png\n")


def train_churn_models(X_tr, X_te, y_tr, y_te):
    sample_wt = compute_sample_weight("balanced", y_tr)

    algo_dict = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=15,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, random_state=42
        ),
    }

    summary  = {}
    champion = {"name": None, "auc": 0, "model": None, "pred": None, "prob": None}

    for name, model in algo_dict.items():
        print(f"  Training → {name} …", end="  ", flush=True)

        if name == "Gradient Boosting":
            model.fit(X_tr, y_tr, sample_weight=sample_wt)
        else:
            model.fit(X_tr, y_tr, sample_weight=sample_wt)

        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]

        acc  = accuracy_score (y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec  = recall_score   (y_te, pred, zero_division=0)
        f1   = f1_score       (y_te, pred, zero_division=0)
        auc  = roc_auc_score  (y_te, prob)

        summary[name] = {"Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1 Score": f1, "ROC-AUC": auc}

        print(f"Done   |   ROC-AUC = {auc:.4f}")
        print(f"\n{'─'*65}")
        print(f"  {name} — Detailed Report")
        print(f"{'─'*65}")
        print(classification_report(y_te, pred,
              target_names=["Stayed", "Churned"], zero_division=0))

        if auc > champion["auc"]:
            champion.update(name=name, auc=auc, model=model, pred=pred, prob=prob)

    print(f"\n{'═'*65}")
    print(f"  BEST MODEL → {champion['name']}   (ROC-AUC = {champion['auc']:.4f})")
    print(f"{'═'*65}\n")

    return summary, champion


def plot_results(summary, y_te, champion, X_tr, y_tr, feature_names):
    algo_names  = list(summary.keys())
    bar_colours = ["#1565C0", "#2E7D32", "#E65100"]
    metrics     = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x_pos       = np.arange(len(metrics))
    bar_w       = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    for idx, algo in enumerate(algo_names):
        vals = [summary[algo][m] for m in metrics]
        bars = ax.bar(x_pos + idx * bar_w, vals, bar_w,
                      label=algo, color=bar_colours[idx],
                      alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Algorithm Comparison — All Metrics", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot3_model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot3_model_comparison.png")

    cm             = confusion_matrix(y_te, champion["pred"])
    tn, fp, fn, tp = cm.ravel()
    total_churn    = tp + fn

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1])

    ax_cm = fig.add_subplot(gs[0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                ax=ax_cm, linewidths=1,
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"],
                annot_kws={"size": 14, "weight": "bold"})
    ax_cm.set_xlabel("Predicted", fontweight="bold")
    ax_cm.set_ylabel("Actual",    fontweight="bold")
    ax_cm.set_title(f"Confusion Matrix  ({champion['name']})", fontweight="bold")

    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")
    info_text = (
        f"Business Interpretation\n"
        f"{'─' * 34}\n"
        f"Churners correctly identified  : {tp:,}\n"
        f"Churners missed                : {fn:,}\n"
        f"Churn detection rate           : {tp/total_churn*100:.1f} %\n\n"
        f"Loyal customers flagged wrongly: {fp:,}\n"
        f"False alarm rate               : {fp/(fp+tn)*100:.2f} %\n\n"
        f"Best ROC-AUC                   : {champion['auc']:.4f}"
    )
    ax_info.text(0.04, 0.55, info_text,
                 transform=ax_info.transAxes, fontsize=10,
                 verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.7",
                           facecolor="#E3F2FD", alpha=0.9))
    plt.tight_layout()
    plt.savefig("plot4_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot4_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_te, champion["prob"])
    fig, (ax_roc, ax_dist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_roc.plot(fpr, tpr, lw=2.5, color="#6A1B9A",
                label=f"ROC AUC = {champion['auc']:.4f}")
    ax_roc.fill_between(fpr, tpr, alpha=0.12, color="#6A1B9A")
    ax_roc.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=11)
    ax_roc.set_title(f"ROC Curve — {champion['name']}", fontweight="bold")
    ax_roc.legend(fontsize=10)
    ax_roc.grid(alpha=0.3)

    ax_dist.hist(champion["prob"][y_te == 0], bins=50, alpha=0.6,
                 color="#2196F3", density=True, label="Stayed")
    ax_dist.hist(champion["prob"][y_te == 1], bins=50, alpha=0.6,
                 color="#F44336", density=True, label="Churned")
    ax_dist.axvline(0.5, color="black", lw=2, ls="--", label="Threshold 0.5")
    ax_dist.set_xlabel("Predicted Churn Probability", fontsize=11)
    ax_dist.set_ylabel("Density", fontsize=11)
    ax_dist.set_title("Churn Score Distribution", fontweight="bold")
    ax_dist.legend(fontsize=10)
    ax_dist.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("plot5_roc_and_scores.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot5_roc_and_scores.png")

    if isinstance(champion["model"], (RandomForestClassifier, GradientBoostingClassifier)):
        rf_model = champion["model"]
    else:
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        sw = compute_sample_weight("balanced", y_tr)
        rf_model.fit(X_tr, y_tr, sample_weight=sw)

    imp = pd.Series(rf_model.feature_importances_, index=feature_names).nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind="barh", ax=ax, color="#1565C0", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Top 15 Feature Importances", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot6_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot6_feature_importance.png\n")


def print_final_table(summary):
    print("\n" + "═" * 70)
    print("  FINAL COMPARISON TABLE")
    print("═" * 70)
    header = f"{'Algorithm':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9} {'ROC-AUC':>9}"
    print(header)
    print("─" * 70)
    for algo, m in summary.items():
        print(f"{algo:<22} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} "
              f"{m['Recall']:>8.4f} {m['F1 Score']:>9.4f} {m['ROC-AUC']:>9.4f}")
    best = max(summary, key=lambda k: summary[k]["ROC-AUC"])
    print("─" * 70)
    print(f"\n  Best Model   : {best}")
    print(f"  Best ROC-AUC : {summary[best]['ROC-AUC']:.4f}")
    print("═" * 70)


if __name__ == "__main__":

    print("\n" + "═" * 70)
    print("  CODSOFT ML INTERNSHIP — TASK 3: CUSTOMER CHURN PREDICTION")
    print("═" * 70 + "\n")

    raw_df = load_churn_data("churn_data")

    inspect_data(raw_df)

    target_col = find_target_column(raw_df)

    X_raw, y = encode_and_clean(raw_df, target_col)

    X_engineered = engineer_churn_features(X_raw)

    show_churn_balance(y)

    plot_feature_vs_churn(X_engineered, y, top_n=6)

    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler   = StandardScaler()
    X_tr_sc  = pd.DataFrame(scaler.fit_transform(X_train), columns=X_engineered.columns)
    X_te_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_engineered.columns)

    print(f"[OK] Train : {len(X_tr_sc):,} rows")
    print(f"[OK] Test  : {len(X_te_sc):,} rows\n")

    results, best = train_churn_models(X_tr_sc, X_te_sc, y_train, y_test)

    plot_results(results, y_test, best, X_tr_sc, y_train, X_engineered.columns)

    print_final_table(results)

    print("\n  Output files:")
    for f in [
        "plot1_churn_distribution.png",
        "plot2_feature_distributions.png",
        "plot3_model_comparison.png",
        "plot4_confusion_matrix.png",
        "plot5_roc_and_scores.png",
        "plot6_feature_importance.png",
    ]:
        print(f"      • {f}")

    print("\n" + "═" * 70 + "\n")