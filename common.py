import json
import os
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "images-dataSAT.tar"
DATA = ROOT / "data" / "images_dataSAT"
MODE = os.environ.get("LAND_RUN_MODE", "quick")
SEED = 7331
SIZE = (64, 64)
BATCH = 16
CLASSES = ("class_0_non_agri", "class_1_agri")
MAX_PER_CLASS = 240 if MODE == "quick" else None
EPOCHS = 2 if MODE == "quick" else 8


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)


def ensure_data():
    if all((DATA / name).exists() and any((DATA / name).glob("*.jpg")) for name in CLASSES):
        return DATA
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Place the supplied image archive at {ARCHIVE}")
    with tarfile.open(ARCHIVE) as archive:
        members = archive.getmembers()
        allowed = {"images_dataSAT"} | {f"images_dataSAT/{name}" for name in CLASSES}
        if any(not (m.name.rstrip("/") in allowed or (m.name.startswith("images_dataSAT/") and m.name.endswith(".jpg") and len(Path(m.name).parts) == 3)) for m in members):
            raise ValueError("The image archive has an unexpected path")
        archive.extractall(ROOT / "data", filter="data")
    return DATA


def paths_and_labels(max_per_class=MAX_PER_CLASS):
    ensure_data()
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        files = sorted((DATA / name).glob("*.jpg"))
        if max_per_class is not None:
            rng = np.random.default_rng(SEED + label)
            files = [files[i] for i in sorted(rng.choice(len(files), min(max_per_class, len(files)), replace=False))]
        paths.extend(files)
        labels.extend([label] * len(files))
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)


def split_paths():
    paths, labels = paths_and_labels()
    train_paths, rest_paths, y_train, y_rest = train_test_split(paths, labels, test_size=0.30, random_state=SEED, stratify=labels)
    val_paths, test_paths, y_val, y_test = train_test_split(rest_paths, y_rest, test_size=0.50, random_state=SEED, stratify=y_rest)
    return (train_paths, y_train), (val_paths, y_val), (test_paths, y_test)


def load_array(paths):
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(np.asarray(im.convert("RGB").resize(SIZE), dtype=np.float32) / 255.0)
    return np.stack(images)


def evaluate_binary(y_true, probabilities):
    probabilities = np.asarray(probabilities).reshape(-1)
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "n_test": len(y_true),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-7, 1 - 1e-7))),
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
        "classification_report": classification_report(y_true, predicted, target_names=CLASSES, zero_division=0),
    }


def save_metrics(name, metrics):
    target = ROOT / "results" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"mode": MODE, "seed": SEED, **metrics}, indent=2) + "\n", encoding="utf-8")
    return target
