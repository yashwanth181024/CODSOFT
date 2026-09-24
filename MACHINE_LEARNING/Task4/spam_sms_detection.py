import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

plt.style.use("seaborn-v0_8-whitegrid")


def load_sms_data(filepath="spam"):
    df = None
    for ext in ("", ".csv", ".txt", ".xlsx"):
        try:
            if ext == ".xlsx":
                df = pd.read_excel(filepath + ext)
            elif ext == ".txt":
                df = pd.read_csv(filepath + ext, sep="\t", header=None,
                                 names=["label", "message"], encoding="latin-1")
            else:
                for enc in ("utf-8", "latin-1"):
                    try:
                        df = pd.read_csv(filepath + ext, encoding=enc)
                        break
                    except:
                        continue
            if df is not None:
                print(f"[OK] Loaded → {filepath}{ext}  ({len(df):,} rows)")
                break
        except FileNotFoundError:
            continue

    if df is None:
        raise FileNotFoundError(f"Cannot find {filepath}. Place it in the same folder.")
    return df


def identify_columns(df):
    label_candidates = ["label", "Label", "Category", "category", "class", "Class", "v1", "V1"]
    text_candidates  = ["message", "Message", "text", "Text", "sms", "SMS", "v2", "V2"]

    label_col = next((c for c in label_candidates if c in df.columns), df.columns[0])
    text_col  = next((c for c in text_candidates  if c in df.columns), df.columns[1])

    print(f"[OK] Label column → '{label_col}'")
    print(f"[OK] Text  column → '{text_col}'\n")
    return label_col, text_col


def encode_labels(df, label_col):
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df["target"] = df[label_col].map(
        lambda x: 1 if x in ("spam", "1", "true", "yes") else 0
    )
    return df


def clean_message(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def show_distribution(df):
    total = len(df)
    n_ham  = int((df["target"] == 0).sum())
    n_spam = int((df["target"] == 1).sum())

    print("─" * 60)
    print("  MESSAGE DISTRIBUTION")
    print("─" * 60)
    print(f"  Legitimate (Ham)  : {n_ham:>6,}  ({n_ham/total*100:.2f} %)")
    print(f"  Spam              : {n_spam:>6,}  ({n_spam/total*100:.2f} %)")
    print(f"  Total Messages    : {total:>6,}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(["Legitimate", "Spam"], [n_ham, n_spam],
            color=["#2196F3", "#F44336"], width=0.45, edgecolor="white")
    ax1.set_title("Message Count by Class", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, n_ham * 1.15)
    for i, v in enumerate([n_ham, n_spam]):
        ax1.text(i, v + n_ham * 0.02, f"{v:,}",
                 ha="center", fontsize=9, fontweight="bold")

    ax2.pie([n_ham, n_spam],
            labels=["Legitimate", "Spam"],
            colors=["#2196F3", "#F44336"],
            autopct="%1.2f%%", startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white"},
            textprops={"fontsize": 10})
    ax2.set_title("Spam Percentage", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plot1_distribution.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot1_distribution.png\n")


def show_message_length(df, text_col):
    df = df.copy()
    df["msg_length"] = df[text_col].apply(lambda x: len(str(x)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[df["target"] == 0]["msg_length"], bins=50,
            alpha=0.6, color="#2196F3", label="Legitimate", density=True)
    ax.hist(df[df["target"] == 1]["msg_length"], bins=50,
            alpha=0.6, color="#F44336", label="Spam", density=True)
    ax.set_xlabel("Message Length (characters)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Message Length Distribution by Class", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("plot2_message_length.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot2_message_length.png\n")


def build_tfidf(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features = 10000,
        ngram_range  = (1, 2),
        sublinear_tf = True,
        min_df       = 2,
        analyzer     = "word"
    )
    X_tr_vec = vectorizer.fit_transform(X_train)
    X_te_vec = vectorizer.transform(X_test)
    print(f"[OK] TF-IDF vocabulary size → {len(vectorizer.vocabulary_):,}")
    return X_tr_vec, X_te_vec, vectorizer


def train_models(X_tr, X_te, y_tr, y_te):
    algo_dict = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42
        ),
        "SVM (LinearSVC)": LinearSVC(
            C=1.0, max_iter=2000, class_weight="balanced",
            random_state=42
        ),
    }

    summary  = {}
    champion = {"name": None, "f1": 0, "model": None, "pred": None}

    for name, model in algo_dict.items():
        print(f"  Training → {name} …", end="  ", flush=True)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_te)
        acc  = accuracy_score (y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec  = recall_score   (y_te, pred, zero_division=0)
        f1   = f1_score       (y_te, pred, zero_division=0)

        summary[name] = {"Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1 Score": f1}

        print(f"Done   |   F1 = {f1:.4f}   Accuracy = {acc:.4f}")
        print(f"\n{'─'*60}")
        print(f"  {name} — Detailed Report")
        print(f"{'─'*60}")
        print(classification_report(y_te, pred,
              target_names=["Legitimate", "Spam"], zero_division=0))

        if f1 > champion["f1"]:
            champion.update(name=name, f1=f1, model=model, pred=pred)

    print(f"\n{'═'*60}")
    print(f"  BEST MODEL → {champion['name']}   (F1 = {champion['f1']:.4f})")
    print(f"{'═'*60}\n")

    return summary, champion


def plot_results(summary, y_te, champion):
    algo_names  = list(summary.keys())
    bar_colours = ["#1565C0", "#2E7D32", "#E65100"]
    metrics     = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x_pos       = np.arange(len(metrics))
    bar_w       = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, algo in enumerate(algo_names):
        vals = [summary[algo][m] for m in metrics]
        bars = ax.bar(x_pos + idx * bar_w, vals, bar_w,
                      label=algo, color=bar_colours[idx],
                      alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — All Metrics", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot3_model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot3_model_comparison.png")

    cm             = confusion_matrix(y_te, champion["pred"])
    tn, fp, fn, tp = cm.ravel()
    total_spam     = tp + fn

    fig, (ax_cm, ax_info) = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                ax=ax_cm, linewidths=1,
                xticklabels=["Legitimate", "Spam"],
                yticklabels=["Legitimate", "Spam"],
                annot_kws={"size": 14, "weight": "bold"})
    ax_cm.set_xlabel("Predicted", fontweight="bold")
    ax_cm.set_ylabel("Actual",    fontweight="bold")
    ax_cm.set_title(f"Confusion Matrix  ({champion['name']})", fontweight="bold")

    ax_info.axis("off")
    info_text = (
        f"Detection Summary\n"
        f"{'─' * 30}\n"
        f"Spam correctly caught    : {tp:,}\n"
        f"Spam missed              : {fn:,}\n"
        f"Spam detection rate      : {tp/total_spam*100:.1f} %\n\n"
        f"Legit flagged as spam    : {fp:,}\n"
        f"False alarm rate         : {fp/(fp+tn)*100:.2f} %\n\n"
        f"Best F1 Score            : {champion['f1']:.4f}"
    )
    ax_info.text(0.05, 0.55, info_text,
                 transform=ax_info.transAxes, fontsize=11,
                 verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.7",
                           facecolor="#E3F2FD", alpha=0.9))
    plt.tight_layout()
    plt.savefig("plot4_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("[OK] Saved → plot4_confusion_matrix.png\n")


def print_final_table(summary):
    print("\n" + "═" * 65)
    print("  FINAL COMPARISON TABLE")
    print("═" * 65)
    print(f"{'Algorithm':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
    print("─" * 65)
    for algo, m in summary.items():
        print(f"{algo:<22} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} "
              f"{m['Recall']:>8.4f} {m['F1 Score']:>9.4f}")
    best = max(summary, key=lambda k: summary[k]["F1 Score"])
    print("─" * 65)
    print(f"\n  Best Model   : {best}")
    print(f"  Best F1-Score: {summary[best]['F1 Score']:.4f}")
    print("═" * 65)


def predict_message(model, vectorizer, message):
    cleaned = clean_message(message)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    return "SPAM" if pred == 1 else "LEGITIMATE"


if __name__ == "__main__":

    print("\n" + "═" * 65)
    print("  CODSOFT ML INTERNSHIP — TASK 4: SPAM SMS DETECTION")
    print("═" * 65 + "\n")

    raw_df = load_sms_data("spam")

    label_col, text_col = identify_columns(raw_df)

    df = encode_labels(raw_df, label_col)

    show_distribution(df)

    show_message_length(df, text_col)

    df["clean_text"] = df[text_col].apply(clean_message)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["target"],
        test_size=0.2, random_state=42, stratify=df["target"]
    )

    print(f"[OK] Train : {len(X_train):,} messages")
    print(f"[OK] Test  : {len(X_test):,} messages\n")

    X_tr_vec, X_te_vec, vectorizer = build_tfidf(X_train, X_test)

    results, best = train_models(X_tr_vec, X_te_vec, y_train, y_test)

    plot_results(results, y_test, best)

    print_final_table(results)

    print("\n  Sample Predictions:")
    print("  " + "─" * 55)
    test_messages = [
        "Congratulations! You won a FREE iPhone. Click here to claim now!!!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account will be suspended. Call 0800 now to verify.",
        "Can you please send me the homework notes?",
    ]
    for msg in test_messages:
        result = predict_message(best["model"], vectorizer, msg)
        print(f"  {result}  →  {msg[:55]}...")

    print("\n  Output files:")
    for f in ["plot1_distribution.png", "plot2_message_length.png",
              "plot3_model_comparison.png", "plot4_confusion_matrix.png"]:
        print(f"      • {f}")

    print("\n" + "═" * 65 + "\n")