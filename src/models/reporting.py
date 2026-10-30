"""
reporting.py
------------
Registra configurações, tempos, status e integridade dos arquivos de cada execução,
além de salvar métricas, predições e matrizes de confusão do classificador final.
"""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.validator import write_json

def file_hash(path):
    digest = sha256()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

class RunReport:
    def __init__(self, base: Path, config: dict, root: Path):
        now = datetime.now(timezone.utc)
        self.directory = base / (now.strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = perf_counter()

        packages = {}
        for name in ["pandas", "numpy", "scikit-learn", "keras", "tensorflow", "tensorflow-cpu", "pyarrow", "Pillow", "joblib"]:
            try:
                packages[name] = version(name)

            except PackageNotFoundError:
                pass

        self.metadata = {"status": "running", "started_at": now.isoformat(),
                         "python": platform.python_version(), "packages": packages,
                         "code_sha256": {str(path.relative_to(root)): file_hash(path)
                                         for path in sorted((root / "src").rglob("*.py"))}}
        
        (self.directory / "models.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.flush()

    def flush(self):
        write_json(self.directory / "run.json", self.metadata)

    def finish(self, error=None):
        self.metadata.update(status="failed" if error else "completed",
                             elapsed_seconds=perf_counter() - self.started,
                             finished_at=datetime.now(timezone.utc).isoformat())
        
        if error:
            self.metadata["error"] = f"{type(error).__name__}: {error}"

        self.metadata["artifacts_sha256"] = {
            str(path.relative_to(self.directory)): file_hash(path)
            for path in sorted(self.directory.rglob("*")) if path.is_file() and path.name != "run.json"}
        
        self.flush()

def classification_results(directory, name, ids, true, probabilities, classes, seconds):
    import numpy as np

    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    true = np.asarray(true).astype(str)
    metrics = {"accuracy": float(accuracy_score(true, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
               "f1_macro": float(f1_score(true, predicted, labels=classes, average="macro", zero_division=0)),
               "inference_seconds": seconds,
               "milliseconds_per_sample": seconds * 1000 / len(true),
               "classification_report": classification_report(true, predicted, labels=classes,
                                                                output_dict=True, zero_division=0)}
    
    output = directory / "evaluation"
    output.mkdir(exist_ok=True)
    predictions = pd.DataFrame({"ticket_id": np.asarray(ids).astype(str), "y_true": true, "y_pred": predicted})

    for index, label in enumerate(classes):
        predictions[f"probability__{label}"] = probabilities[:, index]

    predictions.to_csv(output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
    confusion = pd.DataFrame(confusion_matrix(true, predicted, labels=classes), index=classes, columns=classes)
    confusion.index.name = "real / previsto"
    confusion.to_csv(output / f"{name}_confusion.csv", encoding="utf-8-sig")
    
    return metrics
