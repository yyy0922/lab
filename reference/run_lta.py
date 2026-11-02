"""Reproducible APK -> opcode 5-gram -> LTA experiment for the bundled APKs.

The original notebook expects a precomputed 5_gram.csv and uses the test set
for feature selection and early stopping. This runner rebuilds features from
the APKs and keeps a held-out test set untouched until final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Static Analysis"))
from infrastructure.map import bytecode_map  # noqa: E402


def opcode_ngrams(decoded: Path, n: int = 5) -> set[str]:
    found: set[str] = set()
    smali_dirs = sorted(p for p in decoded.iterdir() if p.is_dir() and p.name.startswith("smali"))
    for smali_dir in smali_dirs:
        for path in smali_dir.rglob("*.smali"):
            method: list[str] | None = None
            # Windows still applies MAX_PATH to ordinary file opens on some hosts.
            readable = Path("\\\\?\\" + str(path.resolve())) if sys.platform == "win32" else path
            with readable.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    token = line.strip().split(maxsplit=1)
                    if not token:
                        continue
                    if token[0] == ".method":
                        method = []
                    elif token[0] == ".end" and len(token) > 1 and token[1].startswith("method"):
                        if method is not None:
                            found.update("".join(method[i : i + n]) for i in range(len(method) - n + 1))
                        method = None
                    elif method is not None:
                        mapped = bytecode_map.get(token[0])
                        if mapped:
                            method.append(mapped)
    return found


def label_map(apk_dir: Path, label_root: Path) -> dict[str, int]:
    samples = {p.stem for p in apk_dir.glob("*.apk")}
    if not samples:
        raise ValueError(f"No APKs in {apk_dir}")
    by_class = {}
    for folder, label in (("benign", 0), ("malradar", 1)):
        for p in (label_root / folder).glob("*.apk"):
            if p.stem in by_class:
                raise ValueError(f"Ambiguous class for {p.stem}")
            by_class[p.stem] = label
    missing = samples - by_class.keys()
    if missing:
        raise ValueError(f"Missing directory labels: {sorted(missing)}")
    return {name: by_class[name] for name in sorted(samples)}


def extract(args: argparse.Namespace) -> None:
    apk_dir, label_root, out = map(Path, (args.apk_dir, args.label_root, args.out))
    out.mkdir(parents=True, exist_ok=True)
    labels = label_map(apk_dir, label_root)
    features: dict[str, set[str]] = {}
    manifest = []
    for index, (name, label) in enumerate(labels.items(), 1):
        apk = apk_dir / f"{name}.apk"
        digest = hashlib.sha256(apk.read_bytes()).hexdigest()
        if digest != name:
            raise ValueError(f"Filename is not SHA-256 of APK: {apk}")
        decoded = out / "decoded" / name
        if not decoded.exists():
            decoded.parent.mkdir(parents=True, exist_ok=True)
            cmd = [args.java, "-jar", str(ROOT / "Static Analysis" / "apktool.jar"),
                   "d", "-r", "-o", str(decoded), str(apk)]
            run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if run.returncode:
                print(run.stdout, file=sys.stderr)
                raise RuntimeError(f"apktool failed for {apk.name} with exit {run.returncode}")
        grams = opcode_ngrams(decoded, args.ngram)
        if not grams:
            raise ValueError(f"No opcode {args.ngram}-grams in {apk.name}")
        features[name] = grams
        manifest.append({"sha256": name, "apk": str(apk.resolve()), "label": label,
                         "label_source": f"{label_root.resolve()}/{'benign' if label == 0 else 'malradar'}",
                         "n_grams": len(grams)})
        print(f"[{index}/{len(labels)}] {name[:12]} label={label} grams={len(grams)}", flush=True)
    columns = sorted(set().union(*features.values()))
    with (out / f"{args.ngram}_gram.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sha256", *columns, "Class"])
        for name, label in labels.items():
            writer.writerow([name, *(int(c in features[name]) for c in columns), label])
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps({"samples": len(labels), "class_counts": dict(Counter(labels.values())),
                      "features": len(columns), "output": str(out.resolve())}), flush=True)


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    return {"accuracy": accuracy_score(y_true, pred),
            "precision_weighted": precision_score(y_true, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
            "confusion_matrix_true_rows": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()}


def select_features(x: np.ndarray, y: np.ndarray, population_size: int, generations: int,
                    seed: int) -> tuple[np.ndarray, list[float]]:
    """SVM wrapper GA matching notebook operators, scored by train-only 3-fold CV."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    rng = np.random.default_rng(seed)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cache: dict[bytes, float] = {}

    def score(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        key = np.packbits(mask).tobytes()
        if key not in cache:
            cache[key] = float(cross_val_score(SVC(kernel="linear", random_state=seed),
                                                x[:, mask], y, cv=folds, scoring="accuracy",
                                                n_jobs=1).mean())
        return cache[key]

    population = rng.integers(0, 2, (population_size, x.shape[1]), dtype=np.uint8).astype(bool)
    scores = np.array([score(p) for p in population])
    history = [float(scores.max())]
    for generation in range(generations):
        children = []
        while len(children) < population_size:
            contenders = rng.integers(0, population_size, 3)
            parent = population[contenders[np.argmax(scores[contenders])]].copy()
            if rng.random() < 0.5 and x.shape[1] > 2:
                other = population[rng.integers(population_size)]
                a, b = sorted(rng.choice(np.arange(1, x.shape[1]), 2, replace=False))
                parent[a:b] = other[a:b]
            if rng.random() < 0.2:
                parent ^= rng.random(x.shape[1]) < 0.1
            children.append(parent)
        population = np.array(children)
        scores = np.array([score(p) for p in population])
        history.append(float(scores.max()))
        print(f"GA generation {generation + 1}/{generations}: CV accuracy={history[-1]:.4f}", flush=True)
    return population[np.argmax(scores)], history


def train(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from torch import nn

    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    df = pd.read_csv(out / f"{args.ngram}_gram.csv")
    names = df.sha256.to_numpy()
    y = df.Class.to_numpy(dtype=np.int64)
    x = df.drop(columns=["sha256", "Class"]).to_numpy(dtype=np.float32)
    columns = df.columns[1:-1].to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Training requires both classes")
    trainval, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=args.seed)
    train_idx, val_idx = train_test_split(trainval, test_size=0.25, stratify=y[trainval], random_state=args.seed)
    mask, history = select_features(x[train_idx], y[train_idx], args.ga_population,
                                     args.ga_generations, args.seed)
    selected = columns[mask].tolist()
    (out / "selected_features.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    svm = SVC(kernel="linear", random_state=args.seed)
    svm.fit(x[trainval][:, mask], y[trainval])
    svm_result = metrics(y[test], svm.predict(x[test][:, mask]))

    class CausalConv(nn.Module):
        def __init__(self, source: int, target: int, dilation: int):
            super().__init__()
            self.left_padding = 2 * dilation
            self.conv = nn.Conv1d(source, target, 3, dilation=dilation)

        def forward(self, x):
            return self.conv(nn.functional.pad(x, (self.left_padding, 0)))

    class TCNBlock(nn.Module):
        """keras-tcn default 64 filters, kernel 3, six residual dilation blocks."""
        def __init__(self, width: int):
            super().__init__()
            self.blocks = nn.ModuleList()
            source = width
            for dilation in (1, 2, 4, 8, 16, 32):
                self.blocks.append(nn.ModuleDict({
                    "conv1": CausalConv(source, 64, dilation),
                    "conv2": CausalConv(64, 64, dilation),
                    "project": nn.Conv1d(source, 64, 1),
                }))
                source = 64

        def forward(self, sequence):
            z = sequence.transpose(1, 2)
            skips = []
            for block in self.blocks:
                transformed = torch.relu(block["conv2"](torch.relu(block["conv1"](z))))
                z = transformed + block["project"](z)
                skips.append(transformed)
            return torch.stack(skips).sum(0).transpose(1, 2)

    class KerasStyleAttention(nn.Module):
        """2 heads with key_dim=64 and output width=64 as in the notebook."""
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 128)
            self.key = nn.Linear(64, 128)
            self.value = nn.Linear(64, 128)
            self.output = nn.Linear(128, 64)

        def forward(self, query, key, value):
            def heads(x):
                return x.reshape(x.shape[0], x.shape[1], 2, 64).transpose(1, 2)
            q = heads(self.query(query))
            k = heads(self.key(key))
            v = heads(self.value(value))
            scores = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)
            context = (scores @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], 128)
            return self.output(context)

    class LTA(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.lstm = nn.LSTM(width, 64, batch_first=True)
            self.tcn = TCNBlock(width)
            self.attention = KerasStyleAttention()
            self.dropout = nn.Dropout(0.5)
            self.output = nn.Linear(192, 1)

        def forward(self, x):
            lstm, _ = self.lstm(x)
            lstm = self.dropout(lstm)
            tcn = self.dropout(self.tcn(x))
            attention = self.attention(lstm, tcn, tcn)
            return self.output(torch.cat((lstm, tcn, attention), dim=-1).flatten(1)).squeeze(-1)

    device = torch.device("cpu")
    model = LTA(int(mask.sum())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    tensor_x = torch.from_numpy(x[:, mask]).unsqueeze(1).to(device)
    tensor_y = torch.from_numpy(y.astype(np.float32)).to(device)
    best_loss, best_epoch, patience = float("inf"), 0, 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in np.array_split(np.random.permutation(train_idx), max(1, int(np.ceil(len(train_idx) / 64)))):
            optimizer.zero_grad()
            loss = loss_fn(model(tensor_x[batch]), tensor_y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(tensor_x[val_idx]), tensor_y[val_idx]))
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, patience = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0 or patience >= 10:
            print(f"epoch {epoch}: validation_loss={val_loss:.4f} best={best_loss:.4f}", flush=True)
        if patience >= 10:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "features": selected, "seed": args.seed}, out / "lta_model.pt")
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor_x[test])).numpy()
    prediction = (probs > 0.5).astype(int)
    lta_result = metrics(y[test], prediction)
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sha256", "true_label", "lta_probability", "lta_prediction", "svm_prediction"])
        for idx, prob, pred, svm_pred in zip(test, probs, prediction, svm.predict(x[test][:, mask])):
            writer.writerow([names[idx], y[idx], float(prob), int(pred), int(svm_pred)])

    # The notebook's second experiment: 10-fold LightGBM and SHAP. All folds
    # use train/validation samples; the held-out set is only scored afterward.
    sys.path.insert(0, str(ROOT / "vendor"))
    import lightgbm as lgb
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import StratifiedKFold

    selected_frame = pd.DataFrame(x[:, mask], columns=selected)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    train_rmse, val_rmse, fold_probs = [], [], []
    for fold, (tr, va) in enumerate(folds.split(trainval, y[trainval]), 1):
        tr_idx, va_idx = trainval[tr], trainval[va]
        boost = lgb.LGBMClassifier(objective="binary", n_estimators=100,
                                   num_leaves=7, min_child_samples=2,
                                   verbosity=-1, random_state=args.seed)
        boost.fit(selected_frame.iloc[tr_idx], y[tr_idx])
        tr_pred = boost.predict(selected_frame.iloc[tr_idx])
        va_pred = boost.predict(selected_frame.iloc[va_idx])
        train_rmse.append(float(np.sqrt(mean_squared_error(y[tr_idx], tr_pred))))
        val_rmse.append(float(np.sqrt(mean_squared_error(y[va_idx], va_pred))))
        fold_probs.append(boost.predict_proba(selected_frame.iloc[test])[:, 1])
        print(f"LightGBM fold {fold}/10: validation RMSE={val_rmse[-1]:.4f}", flush=True)
    boost_probs = np.mean(fold_probs, axis=0)
    boost_result = metrics(y[test], (boost_probs > 0.5).astype(int))
    final_boost = lgb.LGBMClassifier(objective="binary", n_estimators=100,
                                     num_leaves=7, min_child_samples=2,
                                     verbosity=-1, random_state=args.seed)
    final_boost.fit(selected_frame.iloc[trainval], y[trainval])
    explainer = shap.TreeExplainer(final_boost)
    explanation = explainer.shap_values(selected_frame.iloc[test])
    if isinstance(explanation, list):
        explanation = explanation[-1]
    explanation = np.asarray(explanation)
    if explanation.ndim == 3:
        explanation = explanation[:, :, -1]
    importance = np.abs(explanation).mean(axis=0)
    pd.DataFrame({"feature": selected, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(out / "shap_importance.csv", index=False)
    shap.summary_plot(explanation, selected_frame.iloc[test], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()
    report = {"seed": args.seed, "samples": len(y), "class_counts": dict(Counter(map(int, y))),
              "feature_count": x.shape[1], "selected_feature_count": len(selected),
              "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test),
                        "test_sha256": names[test].tolist()},
              "ga": {"population": args.ga_population, "generations": args.ga_generations,
                     "cv_best_by_generation": history},
              "training": {"max_epochs": args.epochs, "best_epoch": best_epoch,
                           "best_validation_loss": best_loss},
              "svm_test": svm_result, "lta_test": lta_result,
              "lightgbm": {"cv_train_rmse_mean": float(np.mean(train_rmse)),
                           "cv_validation_rmse_mean": float(np.mean(val_rmse)),
                           "test": boost_result,
                           "num_leaves": 7, "min_child_samples": 2},
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                          "device": str(device)}}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "train", "all"])
    parser.add_argument("--apk-dir", default=str(ROOT / "apk"))
    parser.add_argument("--label-root", default=str(ROOT.parent / "apk"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "bundled_46"))
    parser.add_argument("--java", default="java")
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ga-population", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.stage in ("extract", "all"):
        extract(args)
    if args.stage in ("train", "all"):
        train(args)


if __name__ == "__main__":
    main()
