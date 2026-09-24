import warnings
warnings.filterwarnings("ignore")

import pandas   as pd
import numpy    as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection     import train_test_split, StratifiedKFold
from sklearn.preprocessing       import StandardScaler
from sklearn.utils.class_weight  import compute_sample_weight

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)

def load_transactions(train_file, test_file):
    """
    Reads both CSV files and merges them into one dataframe.
    Tries multiple extensions so it works whether or not
    the user added '.csv' to the filename.
    """
    combined = None
    for ext in ("", ".csv", ".xlsx"):
        try:
            reader  = pd.read_excel if ext == ".xlsx" else pd.read_csv
            train   = reader(train_file + ext)
            test    = reader(test_file  + ext)
            combined = pd.concat([train, test], ignore_index=True)
            print(f"[OK] Loaded  →  {train_file}{ext}  ({len(train):,} rows)")
            print(f"[OK] Loaded  →  {test_file}{ext}   ({len(test):,} rows)")
            print(f"     Combined : {len(combined):,} total transactions\n")
            break
        except FileNotFoundError:
            continue

    if combined is None:
        raise FileNotFoundError(
            "Cannot find fraudTrain / fraudTest. "
            "Place them in the same folder as this notebook."
        )
    return combined

def quick_summary(df):
    """Prints a concise overview of the raw dataframe."""
    divider = "─" * 60
    print(divider)
    print("  DATASET OVERVIEW")
    print(divider)
    print(f"  Rows      : {df.shape[0]:,}")
    print(f"  Columns   : {df.shape[1]}")
    print(f"  Nulls     : {df.isnull().sum().sum()}")
    print(f"  Dtypes    : {dict(df.dtypes.value_counts())}")
    print(divider)
    print(df.head(3).to_string())
    print(divider + "\n")

def build_features(df):
    """
    Converts raw transaction columns into ML-ready numeric features.

    Why these specific features?
    ─ Fraudsters often transact far from the cardholder's home city
    ─ Unusually large or small amounts are suspicious
    ─ Late-night transactions have higher fraud probability
    ─ Less-populated areas have fewer verification resources
    """

    TARGET   = "is_fraud"
    label    = df[TARGET].copy()

    to_discard = [
        "Unnamed: 0", TARGET, "cc_num", "trans_num",
        "trans_date_trans_time", "merchant", "category",
        "first", "last", "gender", "street",
        "city", "state", "job", "dob"
    ]
    numeric = df.drop(columns=[c for c in to_discard if c in df.columns])
    numeric = numeric.select_dtypes(include="number").copy()

    if {"lat", "long", "merch_lat", "merch_long"}.issubset(numeric.columns):
        numeric["card_merchant_distance"] = np.sqrt(
            (numeric["lat"]  - numeric["merch_lat"]) ** 2 +
            (numeric["long"] - numeric["merch_long"]) ** 2
        )

    if "amt" in numeric.columns:
        avg_amt   = numeric["amt"].mean()
        std_amt   = numeric["amt"].std()
        numeric["amt_deviation"] = (numeric["amt"] - avg_amt) / (std_amt + 1e-9)
        numeric["high_amt_flag"] = (numeric["amt"] > avg_amt + 2 * std_amt).astype(int)

    if "unix_time" in numeric.columns:
        hour_of_day            = (numeric["unix_time"] // 3600) % 24
        numeric["night_flag"]  = hour_of_day.apply(
            lambda h: 1 if (h >= 22 or h <= 5) else 0
        )

    if "city_pop" in numeric.columns:
        low_pop_cutoff         = numeric["city_pop"].quantile(0.25)
        numeric["small_city"]  = (numeric["city_pop"] < low_pop_cutoff).astype(int)

    suspicion_cols = [
        c for c in ["amt_deviation", "card_merchant_distance",
                    "high_amt_flag", "night_flag", "small_city"]
        if c in numeric.columns
    ]
    numeric["suspicion_score"] = numeric[suspicion_cols].sum(axis=1)

    print(f"[OK] Features ready  →  {numeric.shape[1]} total columns")
    print(f"     Original numeric: {numeric.shape[1] - len(suspicion_cols) - 1}")
    print(f"     New engineered  : {len(suspicion_cols) + 1}")
    print(f"     Feature list    : {list(numeric.columns)}\n")

    return numeric, label

def show_class_balance(label):
    """
    Fraud datasets are heavily imbalanced.
    This function measures and plots that imbalance clearly.
    """
    total     = len(label)
    n_legit   = int((label == 0).sum())
    n_fraud   = int((label == 1).sum())
    ratio     = n_legit / n_fraud

    print("─" * 60)
    print("  CLASS DISTRIBUTION")
    print("─" * 60)
    print(f"  Legitimate  : {n_legit:>10,}   ({n_legit/total*100:.2f} %)")
    print(f"  Fraudulent  : {n_fraud:>10,}   ({n_fraud/total*100:.2f} %)")
    print(f"  Imbalance   : {ratio:.0f} legitimate for every 1 fraud")
    print(f"\n  Solution applied → class_weight='balanced'")
    print(f"  This makes each algorithm pay extra attention to the rare fraud class.\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(["Legitimate", "Fraudulent"],
            [n_legit, n_fraud],
            color=["#1A9850", "#D73027"],
            width=0.4, edgecolor="white")
    ax1.set_ylabel("Number of transactions")
    ax1.set_title("Class Count", fontweight="bold")
    ax1.set_ylim(0, n_legit * 1.12)
    for x, v in enumerate([n_legit, n_fraud]):
        ax1.text(x, v + n_legit * 0.01, f"{v:,}",
                 ha="center", fontsize=9, fontweight="bold")

    wedge_props = {"width": 0.5, "edgecolor": "white", "linewidth": 2}
    ax2.pie([n_legit, n_fraud],
            labels=["Legitimate", "Fraudulent"],
            colors=["#1A9850", "#D73027"],
            autopct="%1.2f %%",
            startangle=90,
            wedgeprops=wedge_props,
            textprops={"fontsize": 10})
    ax2.set_title("Class Proportion", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plot1_class_balance.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot1_class_balance.png\n")

def train_all_models(X_tr, X_te, y_tr, y_te):
    """
    Trains Logistic Regression, Decision Tree, and Random Forest.
    Returns a results dictionary and the single best model object.
    """

    sample_wt = compute_sample_weight("balanced", y_tr)

    algo_dict = {

        "Logistic Regression": LogisticRegression(
            C              = 0.5,          # regularisation strength
            max_iter       = 1000,
            class_weight   = "balanced",   # penalise fraud misses more
            solver         = "lbfgs",
            random_state   = 42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth        = 10,         # prevent overfitting
            min_samples_leaf = 20,         # no tiny leaves
            class_weight     = "balanced",
            random_state     = 42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators     = 200,        # 200 individual trees
            max_depth        = 10,
            min_samples_leaf = 20,
            class_weight     = "balanced",
            n_jobs           = -1,         # use all CPU cores
            random_state     = 42
        ),
    }

    summary  = {}
    champion = {"name": None, "auc": 0, "model": None,
                "pred": None, "prob": None}

    for algo_name, algo_obj in algo_dict.items():

        print(f"  Training → {algo_name} …", end="  ", flush=True)
        algo_obj.fit(X_tr, y_tr, sample_weight=sample_wt)

        predictions  = algo_obj.predict(X_te)
        fraud_probs  = algo_obj.predict_proba(X_te)[:, 1]

        acc  = accuracy_score (y_te, predictions)
        prec = precision_score(y_te, predictions, zero_division=0)
        rec  = recall_score   (y_te, predictions, zero_division=0)
        f1   = f1_score       (y_te, predictions, zero_division=0)
        auc  = roc_auc_score  (y_te, fraud_probs)

        summary[algo_name] = {
            "Accuracy"  : acc,
            "Precision" : prec,
            "Recall"    : rec,
            "F1 Score"  : f1,
            "ROC-AUC"   : auc,
        }

        print(f"Done   |   ROC-AUC = {auc:.4f}")

        print(f"\n{'─'*60}")
        print(f"  {algo_name} — Detailed Report")
        print(f"{'─'*60}")
        print(classification_report(
            y_te, predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        ))

        if auc > champion["auc"]:
            champion.update(
                name=algo_name, auc=auc, model=algo_obj,
                pred=predictions, prob=fraud_probs
            )

    print(f"\n{'═'*60}")
    print(f"  WINNER  →  {champion['name']}   (ROC-AUC = {champion['auc']:.4f})")
    print(f"{'═'*60}\n")

    return summary, champion

def visualise_results(summary, y_te, champion, X_tr, y_tr, feature_names):
    """Creates four diagnostic plots and saves them to disk."""

    algo_names  = list(summary.keys())
    bar_colours = ["#2166AC", "#4DAC26", "#D6604D"]

    metrics  = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x_pos    = np.arange(len(metrics))
    bar_w    = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, algo in enumerate(algo_names):
        vals = [summary[algo][m] for m in metrics]
        bars = ax.bar(x_pos + idx * bar_w, vals, bar_w,
                      label=algo, color=bar_colours[idx],
                      alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Algorithm Comparison — All Metrics", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot2_algorithm_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot2_algorithm_comparison.png")

    cm          = confusion_matrix(y_te, champion["pred"])
    tn, fp, fn, tp = cm.ravel()

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1])

    ax_cm = fig.add_subplot(gs[0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
                ax=ax_cm, linewidths=1,
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"],
                annot_kws={"size": 14, "weight": "bold"})
    ax_cm.set_xlabel("Predicted Label", fontweight="bold")
    ax_cm.set_ylabel("True Label",      fontweight="bold")
    ax_cm.set_title(f"Confusion Matrix  ({champion['name']})",
                    fontweight="bold")

    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")
    total_fraud = tp + fn
    info_text = (
        f"Business Interpretation\n"
        f"{'─' * 32}\n"
        f"Fraud transactions caught  : {tp:,}\n"
        f"Fraud transactions missed  : {fn:,}\n"
        f"Detection rate             : {tp/total_fraud*100:.1f} %\n\n"
        f"Legitimate marked as fraud : {fp:,}\n"
        f"False alarm rate           : {fp/(fp+tn)*100:.2f} %\n\n"
        f"Best ROC-AUC score         : {champion['auc']:.4f}"
    )
    ax_info.text(0.05, 0.55, info_text,
                 transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.7",
                           facecolor="#EFF3FF", alpha=0.9))
    plt.tight_layout()
    plt.savefig("plot3_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot3_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_te, champion["prob"])

    fig, (ax_roc, ax_dist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_roc.plot(fpr, tpr, lw=2.5, color="#762A83",
                label=f"ROC AUC = {champion['auc']:.4f}")
    ax_roc.fill_between(fpr, tpr, alpha=0.12, color="#762A83")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4,
                label="Random guess")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=11)
    ax_roc.set_title(f"ROC Curve — {champion['name']}", fontweight="bold")
    ax_roc.legend(fontsize=10)
    ax_roc.grid(alpha=0.3)

    ax_dist.hist(champion["prob"][y_te == 0], bins=70,
                 color="#1A9850", alpha=0.55, density=True,
                 label="Legitimate")
    ax_dist.hist(champion["prob"][y_te == 1], bins=70,
                 color="#D73027", alpha=0.55, density=True,
                 label="Fraud")
    ax_dist.axvline(0.5, color="black", lw=2, ls="--",
                    label="Decision boundary (0.5)")
    ax_dist.set_xlabel("Predicted Fraud Probability", fontsize=11)
    ax_dist.set_ylabel("Density", fontsize=11)
    ax_dist.set_title("Fraud Score Distribution", fontweight="bold")
    ax_dist.legend(fontsize=10)
    ax_dist.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("plot4_roc_and_scores.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot4_roc_and_scores.png")

    if isinstance(champion["model"], RandomForestClassifier):
        rf = champion["model"]
    else:
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        sw = compute_sample_weight("balanced", y_tr)
        rf.fit(X_tr, y_tr, sample_weight=sw)

    importance_series = (
        pd.Series(rf.feature_importances_, index=feature_names)
          .nlargest(15)
          .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    importance_series.plot(kind="barh", ax=ax, color="#2166AC",
                           alpha=0.85, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Top 15 Feature Importances  (Random Forest)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot5_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot5_feature_importance.png\n")

def print_summary_table(summary):
    """Prints a clean formatted comparison table."""
    print("\n" + "═" * 70)
    print("  FINAL COMPARISON TABLE")
    print("═" * 70)

    header = f"{'Algorithm':<22} {'Accuracy':>9} {'Precision':>10} "  \
             f"{'Recall':>8} {'F1-Score':>9} {'ROC-AUC':>9}"
    print(header)
    print("─" * 70)

    for algo, metrics in summary.items():
        row = (
            f"{algo:<22} "
            f"{metrics['Accuracy']:>9.4f} "
            f"{metrics['Precision']:>10.4f} "
            f"{metrics['Recall']:>8.4f} "
            f"{metrics['F1 Score']:>9.4f} "
            f"{metrics['ROC-AUC']:>9.4f}"
        )
        print(row)

    best_algo = max(summary, key=lambda k: summary[k]["ROC-AUC"])
    print("─" * 70)
    print(f"\n    Best algorithm  : {best_algo}")
    print(f"    Best ROC-AUC    : {summary[best_algo]['ROC-AUC']:.4f}")
    print("═" * 70)

if __name__ == "__main__":

    print("\n" + "═" * 70)
    print("  CODSOFT ML INTERNSHIP — TASK 2: CREDIT CARD FRAUD DETECTION")
    print("═" * 70 + "\n")

    raw_data = load_transactions("fraudTrain", "fraudTest")

    quick_summary(raw_data)

    X_all, y_all = build_features(raw_data)

    show_class_balance(y_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size    = 0.25,
        stratify     = y_all,
        random_state = 42
    )

    scaler    = StandardScaler()
    X_tr_sc   = pd.DataFrame(scaler.fit_transform(X_train), columns=X_all.columns)
    X_te_sc   = pd.DataFrame(scaler.transform (X_test ),    columns=X_all.columns)

    print(f"[OK] Train rows : {len(X_tr_sc):,}")
    print(f"[OK] Test  rows : {len(X_te_sc):,}\n")

    results, best = train_all_models(X_tr_sc, X_te_sc, y_train, y_test)

    visualise_results(results, y_test, best, X_tr_sc, y_train, X_all.columns)

    print_summary_table(results)

    print("\n  Output files saved in current directory:")
    for fname in [
        "plot1_class_balance.png",
        "plot2_algorithm_comparison.png",
        "plot3_confusion_matrix.png",
        "plot4_roc_and_scores.png",
        "plot5_feature_importance.png",
    ]:
        print(f"      • {fname}")

    print("\n" + "═" * 70 + "\n")