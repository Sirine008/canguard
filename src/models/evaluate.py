"""Full Phase 3 evaluation on the test split.

Produces ``reports/metrics.json`` (everything the dashboard's Evaluate tab
needs) and the evaluation figures. Covered:

* window-level precision / recall / F1 / ROC-AUC, mean ± std over seeds,
* per-attack-type results,
* unsupervised thresholds chosen on VAL: target 1 % FPR on normal windows
  (primary) and F1-optimal (reported for comparison),
* episode-level metrics: detection latency (median + p95) per attack type,
  episode recall, false alarms per hour of attack-free traffic,
* supervised confusion matrix + per-class metrics,
* feature ablation (ID-only vs timing-only vs all) for IF and LightGBM,
* leave-one-attack-type-out for the supervised model vs unsupervised recall,
* single-window inference latency (ms, CPU) per model.

Run::

    python -m src.models.evaluate
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    ATTACK_TYPES,
    EPISODE_GAP_S,
    EVAL_SEEDS,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    VAL_TARGET_FPR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from src.models.autoencoder import LSTMAutoencoder, score_sequences
from src.models.common import (
    FEATURE_SETS,
    gather_sequences,
    load_message_matrix,
    load_windows,
    select,
)
from src.models.isolation_forest import score_windows, train_isolation_forest
from src.models.supervised import (
    predict_attack_probability,
    predict_classes,
    train_lightgbm,
)

ATTACKS: tuple[str, ...] = ("dos", "fuzzy", "gear", "rpm")
MODEL_NAMES: tuple[str, ...] = ("isolation_forest", "lstm_autoencoder", "lightgbm")


# ---------------------------------------------------------------------------
# Score normalisation and thresholds
# ---------------------------------------------------------------------------

def normalise(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map raw scores to [0, 1] with constants learned on VAL."""
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


def norm_constants(val_scores: np.ndarray) -> tuple[float, float]:
    """Robust [0, 1] range: 1st and 99.9th percentile of VAL scores."""
    lo, hi = np.percentile(val_scores, [1, 99.9])
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


def threshold_at_fpr(normal_scores: np.ndarray,
                     target_fpr: float = VAL_TARGET_FPR) -> float:
    """Score threshold giving ``target_fpr`` on attack-free VAL windows."""
    return float(np.quantile(normal_scores, 1.0 - target_fpr))


def threshold_best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """F1-optimal threshold on the VAL split."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def window_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   scores: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def mean_std(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate(per_seed: list[dict]) -> dict:
    """Turn a list of per-seed metric dicts into {metric: {mean, std}}."""
    return {key: mean_std([entry[key] for entry in per_seed])
            for key in per_seed[0]}


# ---------------------------------------------------------------------------
# Episode-level evaluation
# ---------------------------------------------------------------------------

def build_episodes(test_messages: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    """Group injected frames into attack episodes per source."""
    episodes: dict[str, list[tuple[float, float]]] = {}
    for source in ATTACKS:
        times = test_messages.loc[
            (test_messages["source"] == source) & (test_messages["label"] == 1),
            "timestamp",
        ].to_numpy()
        if len(times) == 0:
            episodes[source] = []
            continue
        breaks = np.nonzero(np.diff(times) > EPISODE_GAP_S)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(times) - 1]])
        episodes[source] = [
            (float(times[a]), float(times[b])) for a, b in zip(starts, ends)
        ]
    return episodes


def merge_intervals(intervals: np.ndarray, slack: float = 0.1) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent [start, end] alarm intervals into events."""
    if len(intervals) == 0:
        return []
    order = np.argsort(intervals[:, 0])
    merged: list[list[float]] = []
    for start, end in intervals[order]:
        if merged and start <= merged[-1][1] + slack:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    return [(a, b) for a, b in merged]


def episode_metrics(
    test_windows: pd.DataFrame,
    predictions: np.ndarray,
    episodes: dict[str, list[tuple[float, float]]],
    attack_free_hours: float,
) -> dict:
    """Detection latency, episode recall and false alarms per hour."""
    out: dict = {"per_attack": {}}
    false_alarm_events = 0

    for source in ("normal",) + ATTACKS:
        mask = (test_windows["source"] == source).to_numpy()
        flagged = test_windows[mask & (predictions == 1)]
        intervals = flagged[["t_start", "t_end"]].to_numpy()
        events = merge_intervals(intervals)
        source_episodes = episodes.get(source, [])

        # False alarm events: no overlap with any episode (0.5 s tolerance)
        for ev_start, ev_end in events:
            hit = any(ev_start <= ep_end + 0.5 and ev_end >= ep_start - 0.5
                      for ep_start, ep_end in source_episodes)
            if not hit:
                false_alarm_events += 1

        if source == "normal":
            continue

        latencies: list[float] = []
        detected = 0
        flagged_sorted = flagged.sort_values("t_end")
        t_start_arr = flagged_sorted["t_start"].to_numpy()
        t_end_arr = flagged_sorted["t_end"].to_numpy()
        for ep_start, ep_end in source_episodes:
            overlap = (t_end_arr >= ep_start) & (t_start_arr <= ep_end)
            if overlap.any():
                detected += 1
                first_end = t_end_arr[overlap][0]
                latencies.append(max(0.0, float(first_end) - ep_start))
        n = len(source_episodes)
        out["per_attack"][source] = {
            "episodes": n,
            "episode_recall": detected / n if n else None,
            "latency_median_s": float(np.median(latencies)) if latencies else None,
            "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
        }

    out["false_alarms_per_hour"] = false_alarm_events / attack_free_hours
    return out


# ---------------------------------------------------------------------------
# Inference latency (single window, CPU)
# ---------------------------------------------------------------------------

def time_single_window(fn, inputs: list, repeats: int = 1) -> list[float]:
    """Milliseconds per single-window call."""
    samples: list[float] = []
    for _ in range(repeats):
        for x in inputs:
            start = time.perf_counter()
            fn(x)
            samples.append((time.perf_counter() - start) * 1e3)
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: PLR0915 - orchestration script
    rng = np.random.default_rng(EVAL_SEEDS[0])
    windows = load_windows()
    val = select(windows, "val").reset_index(drop=True)
    test = select(windows, "test").reset_index(drop=True)
    y_val = val["label"].to_numpy()
    y_test = test["label"].to_numpy()

    print("Loading message sequences ...")
    message_matrix = load_message_matrix()
    seq_val = gather_sequences(message_matrix, val["start_row"].to_numpy())
    seq_test = gather_sequences(message_matrix, test["start_row"].to_numpy())

    print("Building attack episodes from test messages ...")
    test_messages = pd.read_parquet(
        PROCESSED_DIR / "messages.parquet",
        columns=["timestamp", "label", "source", "split"],
    )
    test_messages = test_messages[test_messages["split"] == "test"]
    episodes = build_episodes(test_messages)
    total_hours = 0.0
    for source in ("normal",) + ATTACKS:
        m = test_messages[test_messages["source"] == source]["timestamp"]
        duration = float(m.max() - m.min())
        episode_time = sum(e - s for s, e in episodes.get(source, []))
        total_hours += (duration - episode_time) / 3600.0
    print(f"  episodes: " + ", ".join(
        f"{s}={len(episodes[s])}" for s in ATTACKS)
        + f" | attack-free test traffic: {total_hours:.2f} h")
    del test_messages

    results: dict = {name: {"per_seed": []} for name in MODEL_NAMES}
    roc_curves: dict = {}
    latency_samples: dict = {}
    episode_results: dict = {name: [] for name in MODEL_NAMES}
    per_attack_f1: dict = {name: {a: [] for a in ATTACKS} for name in MODEL_NAMES}
    per_attack_recall: dict = {name: {a: [] for a in ATTACKS} for name in MODEL_NAMES}
    thresholds_out: dict = {}
    norm_out: dict = {}
    confusion: list[list[int]] | None = None
    per_class_f1_seeds: list[dict] = []

    latency_rows = rng.choice(len(test), size=200, replace=False)

    for name in MODEL_NAMES:
        print(f"\n=== {name} ===")
        for seed in EVAL_SEEDS:
            if name == "isolation_forest":
                model = joblib.load(MODELS_DIR / f"isolation_forest_seed{seed}.joblib")
                raw_val = score_windows(model, val)
                raw_test = score_windows(model, test)
            elif name == "lstm_autoencoder":
                model = LSTMAutoencoder()
                model.load_state_dict(torch.load(
                    MODELS_DIR / f"lstm_ae_seed{seed}.pt", weights_only=True))
                model.eval()
                raw_val = score_sequences(model, seq_val)
                raw_test = score_sequences(model, seq_test)
            else:
                model = joblib.load(MODELS_DIR / f"lightgbm_seed{seed}.joblib")
                raw_val = predict_attack_probability(model, val)
                raw_test = predict_attack_probability(model, test)

            if name == "lightgbm":
                classes_test = predict_classes(model, test)
                preds = (classes_test != 0).astype(np.uint8)
                scores_test = raw_test
                scores_val = raw_val
                thr_fpr = thr_f1 = None
                if seed == EVAL_SEEDS[0]:
                    confusion = confusion_matrix(
                        test["attack_type"].astype(str).map(
                            {a: i for i, a in enumerate(ATTACK_TYPES)}
                        ).to_numpy(),
                        classes_test,
                        labels=list(range(len(ATTACK_TYPES))),
                    ).tolist()
                per_class = {}
                true_multi = test["attack_type"].astype(str).to_numpy()
                pred_multi = np.array(ATTACK_TYPES)[classes_test.astype(int)]
                for attack in ATTACKS:
                    per_class[attack] = float(f1_score(
                        true_multi == attack, pred_multi == attack,
                        zero_division=0))
                per_class_f1_seeds.append(per_class)
            else:
                lo, hi = norm_constants(raw_val)
                scores_val = normalise(raw_val, lo, hi)
                scores_test = normalise(raw_test, lo, hi)
                thr_fpr = threshold_at_fpr(scores_val[y_val == 0])
                thr_f1 = threshold_best_f1(scores_val, y_val)
                preds = (scores_test >= thr_fpr).astype(np.uint8)
                if seed == EVAL_SEEDS[0]:
                    norm_out[name] = {"lo": lo, "hi": hi}
                    thresholds_out[name] = {
                        "fpr_target": thr_fpr, "f1_optimal": thr_f1,
                        "f1_at_f1_optimal_test": float(f1_score(
                            y_test, scores_test >= thr_f1)),
                    }

            metrics = window_metrics(y_test, preds, scores_test)
            if thr_f1 is not None:
                metrics["f1_at_f1opt_threshold"] = float(
                    f1_score(y_test, scores_test >= thr_f1))
            results[name]["per_seed"].append(metrics)
            print(f"  seed {seed}: " + "  ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()))

            for attack in ATTACKS:
                mask = (test["source"] == attack).to_numpy()
                per_attack_f1[name][attack].append(float(
                    f1_score(y_test[mask], preds[mask], zero_division=0)))
                per_attack_recall[name][attack].append(float(
                    recall_score(y_test[mask], preds[mask], zero_division=0)))

            episode_results[name].append(
                episode_metrics(test, preds, episodes, total_hours))

            if seed == EVAL_SEEDS[0]:
                fpr, tpr, _ = roc_curve(y_test, scores_test)
                idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
                roc_curves[name] = {
                    "fpr": fpr[idx].round(5).tolist(),
                    "tpr": tpr[idx].round(5).tolist(),
                }
                # Single-window inference latency
                if name == "isolation_forest":
                    feats = test[FEATURE_SETS["all"]].to_numpy(np.float32)
                    inputs = [feats[i:i + 1] for i in latency_rows]
                    fn = model.decision_function
                elif name == "lstm_autoencoder":
                    inputs = [torch.from_numpy(seq_test[i:i + 1]) for i in latency_rows]
                    fn = lambda x: model(x)  # noqa: E731
                else:
                    feats = test[FEATURE_SETS["all"]].to_numpy(np.float32)
                    inputs = [feats[i:i + 1] for i in latency_rows]
                    fn = model.predict_proba
                with torch.no_grad():
                    samples = time_single_window(fn, inputs)
                latency_samples[name] = [round(s, 4) for s in samples]
                print(f"  latency/window: {np.mean(samples):.3f} ms "
                      f"(p95 {np.percentile(samples, 95):.3f})")

    # ------------------------------------------------------------------
    # Ablation: feature sets x (IF, LightGBM), per-attack F1
    # ------------------------------------------------------------------
    print("\n=== feature ablation ===")
    train_all = select(windows, "train")
    train_normal = select(windows, "train", label=0)
    val_all = select(windows, "val")
    ablation: dict = {}
    for feature_set in ("id", "timing", "all"):
        ablation[feature_set] = {}
        for model_name in ("isolation_forest", "lightgbm"):
            per_attack: dict = {a: [] for a in ATTACKS}
            for seed in EVAL_SEEDS:
                if model_name == "isolation_forest":
                    m = train_isolation_forest(train_normal, seed, feature_set)
                    raw_val = score_windows(m, val, feature_set)
                    raw_test = score_windows(m, test, feature_set)
                    lo, hi = norm_constants(raw_val)
                    thr = threshold_at_fpr(normalise(raw_val, lo, hi)[y_val == 0])
                    preds = (normalise(raw_test, lo, hi) >= thr).astype(np.uint8)
                else:
                    m = train_lightgbm(train_all, val_all, seed, feature_set)
                    preds = (predict_classes(m, test, feature_set) != 0
                             ).astype(np.uint8)
                for attack in ATTACKS:
                    mask = (test["source"] == attack).to_numpy()
                    per_attack[attack].append(float(f1_score(
                        y_test[mask], preds[mask], zero_division=0)))
            ablation[feature_set][model_name] = {
                a: mean_std(v) for a, v in per_attack.items()}
            print(f"  {feature_set:>6} | {model_name:<17} | " + "  ".join(
                f"{a}={np.mean(v):.3f}" for a, v in per_attack.items()))

    # ------------------------------------------------------------------
    # Leave-one-attack-type-out (supervised) vs unsupervised recall
    # ------------------------------------------------------------------
    print("\n=== leave-one-attack-type-out ===")
    loao: dict = {}
    for held_out in ATTACKS:
        recalls: list[float] = []
        for seed in EVAL_SEEDS:
            train_subset = train_all[~((train_all["source"] == held_out)
                                       & (train_all["label"] == 1))]
            val_subset = val_all[~((val_all["source"] == held_out)
                                   & (val_all["label"] == 1))]
            m = train_lightgbm(train_subset, val_subset, seed)
            mask = ((test["source"] == held_out) & (test["label"] == 1)).to_numpy()
            preds = (predict_classes(m, test[mask]) != 0).astype(np.uint8)
            recalls.append(float(np.mean(preds)))
        loao[held_out] = {
            "lightgbm_unseen_recall": mean_std(recalls),
            "isolation_forest_recall": mean_std(
                per_attack_recall["isolation_forest"][held_out]),
            "lstm_autoencoder_recall": mean_std(
                per_attack_recall["lstm_autoencoder"][held_out]),
        }
        print(f"  without {held_out}: lgbm recall on unseen attack "
              f"{np.mean(recalls):.3f} "
              f"(IF {np.mean(per_attack_recall['isolation_forest'][held_out]):.3f}, "
              f"AE {np.mean(per_attack_recall['lstm_autoencoder'][held_out]):.3f})")

    # ------------------------------------------------------------------
    # Assemble metrics.json
    # ------------------------------------------------------------------
    with open(MODELS_DIR / "train_times.json", encoding="utf-8") as f:
        train_times = json.load(f)

    def episode_summary(entries: list[dict]) -> dict:
        summary: dict = {"per_attack": {}, "false_alarms_per_hour": mean_std(
            [e["false_alarms_per_hour"] for e in entries])}
        for attack in ATTACKS:
            rows = [e["per_attack"][attack] for e in entries]
            summary["per_attack"][attack] = {
                "episodes": rows[0]["episodes"],
                "episode_recall": mean_std(
                    [r["episode_recall"] for r in rows if r["episode_recall"] is not None]),
                "latency_median_s": mean_std(
                    [r["latency_median_s"] for r in rows if r["latency_median_s"] is not None]),
                "latency_p95_s": mean_std(
                    [r["latency_p95_s"] for r in rows if r["latency_p95_s"] is not None]),
            }
        return summary

    metrics: dict = {
        "dataset": {
            "name": "HCRL Car-Hacking Dataset (Seo et al., 2018)",
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "split": "time-ordered 60/20/20 per file, no shuffling",
            "threshold_policy": f"chosen on val at {VAL_TARGET_FPR:.0%} FPR "
                                "on attack-free windows",
            "seeds": list(EVAL_SEEDS),
            "limitation": "Attack captures were likely recorded with the "
                          "vehicle stationary; normal traffic in attack files "
                          "differs from the driving normal_run capture "
                          "(domain shift).",
            "test_attack_free_hours": round(total_hours, 3),
        },
        "models": {},
        "ablation": ablation,
        "leave_one_attack_out": loao,
    }
    for name in MODEL_NAMES:
        metrics["models"][name] = {
            "window": aggregate(results[name]["per_seed"]),
            "per_seed": results[name]["per_seed"],
            "per_attack_f1": {a: mean_std(v)
                              for a, v in per_attack_f1[name].items()},
            "per_attack_recall": {a: mean_std(v)
                                  for a, v in per_attack_recall[name].items()},
            "episodes": episode_summary(episode_results[name]),
            "roc_curve": roc_curves[name],
            "latency_ms": {
                "mean": float(np.mean(latency_samples[name])),
                "p95": float(np.percentile(latency_samples[name], 95)),
                "samples": latency_samples[name],
            },
        }
    metrics["models"]["isolation_forest"]["train_time_s"] = mean_std(
        [train_times[str(s)]["isolation_forest_s"] for s in EVAL_SEEDS])
    metrics["models"]["lightgbm"]["train_time_s"] = mean_std(
        [train_times[str(s)]["lightgbm_s"] for s in EVAL_SEEDS])
    metrics["models"]["lstm_autoencoder"]["train_time_s"] = mean_std(
        [train_times[str(s)]["lstm_ae_s"] for s in EVAL_SEEDS])
    metrics["models"]["lightgbm"]["confusion_matrix"] = {
        "labels": list(ATTACK_TYPES), "rows_are_true": True,
        "counts": confusion,
    }
    metrics["models"]["lightgbm"]["per_class_f1"] = {
        a: mean_std([s[a] for s in per_class_f1_seeds]) for a in ATTACKS}
    metrics["models"]["isolation_forest"]["thresholds"] = thresholds_out.get(
        "isolation_forest")
    metrics["models"]["lstm_autoencoder"]["thresholds"] = thresholds_out.get(
        "lstm_autoencoder")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(MODELS_DIR / "score_norm.json", "w", encoding="utf-8") as f:
        json.dump({"norm": norm_out, "thresholds": thresholds_out}, f, indent=2)
    print(f"\nmetrics -> {REPORTS_DIR / 'metrics.json'}")

    make_figures(metrics)


def make_figures(metrics: dict) -> None:
    """Evaluation figures for the report (dashboard renders its own)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
    })
    model_labels = {"isolation_forest": "Isolation Forest",
                    "lstm_autoencoder": "LSTM-AE", "lightgbm": "LightGBM"}
    model_colors = {"isolation_forest": "#5b8def",
                    "lstm_autoencoder": "#7c5cd6", "lightgbm": "#2c9d94"}

    # F1 per model per attack with error bars
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(ATTACKS))
    width = 0.26
    for k, name in enumerate(MODEL_NAMES):
        f1 = metrics["models"][name]["per_attack_f1"]
        means = [f1[a]["mean"] for a in ATTACKS]
        stds = [f1[a]["std"] for a in ATTACKS]
        ax.bar(x + (k - 1) * width, means, width, yerr=stds, capsize=3,
               label=model_labels[name], color=model_colors[name])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [a.upper() if a in ("dos", "rpm") else a.capitalize()
                      for a in ATTACKS])
    ax.set_ylabel("F1 (window level, test)")
    ax.set_title("F1 per attack type, mean ± std over 3 seeds")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(FIGURES_DIR / "eval_f1_per_attack.png")
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for name in MODEL_NAMES:
        roc = metrics["models"][name]["roc_curve"]
        auc = metrics["models"][name]["window"]["roc_auc"]["mean"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"{model_labels[name]} "
                f"(AUC {auc:.3f})", color=model_colors[name], linewidth=1.6)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa1ab", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC, any-attack vs normal (test)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIGURES_DIR / "eval_roc_curves.png")
    plt.close(fig)

    # Confusion matrix (LightGBM)
    cm = np.array(metrics["models"]["lightgbm"]["confusion_matrix"]["counts"])
    labels = metrics["models"]["lightgbm"]["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    row_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=8,
                    color="white" if row_norm[i, j] > 0.6 else "#1c2333")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class (rows)")
    ax.set_title("LightGBM confusion matrix (test, seed 42)")
    ax.grid(False)
    fig.savefig(FIGURES_DIR / "eval_confusion_matrix.png")
    plt.close(fig)

    # Inference latency box plot
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [metrics["models"][name]["latency_ms"]["samples"]
            for name in MODEL_NAMES]
    box = ax.boxplot(data, tick_labels=[model_labels[n] for n in MODEL_NAMES],
                     showfliers=False, patch_artist=True)
    for patch, name in zip(box["boxes"], MODEL_NAMES):
        patch.set_facecolor(model_colors[name])
        patch.set_alpha(0.5)
    ax.set_ylabel("inference latency per window (ms)")
    ax.set_title("Single-window CPU inference latency")
    fig.savefig(FIGURES_DIR / "eval_latency_box.png")
    plt.close(fig)
    print(f"figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
