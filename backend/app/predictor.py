"""LightGBM fire-risk inference loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np


@dataclass
class FireRiskPredictor:
    """Self-contained predictor loaded from `fire_risk_bundle.joblib`."""

    model_version: str
    feature_columns: list[str]
    logit_offset: float
    risk_cutoffs: dict[str, float]
    model_str: str

    def __post_init__(self) -> None:
        self._booster = lgb.Booster(model_str=self.model_str)

    @classmethod
    def from_bundle(cls, bundle_path: Path) -> FireRiskPredictor:
        payload = joblib.load(bundle_path)
        return cls(
            model_version=str(payload["model_version"]),
            feature_columns=list(payload["feature_columns"]),
            logit_offset=float(payload["logit_offset"]),
            risk_cutoffs={k: float(v) for k, v in payload["risk_cutoffs"].items()},
            model_str=str(payload["model_str"]),
        )

    def _raw_to_prob(self, raw: np.ndarray) -> np.ndarray:
        z = np.clip(raw + self.logit_offset, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities `p in (0, 1)` for feature matrix `X`."""
        raw = self._booster.predict(np.asarray(X, dtype=np.float32), raw_score=True)
        return self._raw_to_prob(raw)

    def predict_with_stability(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(probs, stability)` where `stability in [0.2, 1.0]` measures tree-depth agreement."""
        X_arr = np.asarray(X, dtype=np.float32)
        total_trees = int(self._booster.num_trees())
        if total_trees < 3:
            probs = self.predict_probabilities(X_arr)
            return probs, np.ones(len(probs), dtype=np.float32)

        cuts = (max(1, total_trees // 2), max(1, (total_trees * 3) // 4), total_trees)
        probs_by_cut = [
            self._raw_to_prob(self._booster.predict(X_arr, raw_score=True, num_iteration=k))
            for k in cuts
        ]
        stacked = np.vstack(probs_by_cut)
        spread = np.max(stacked, axis=0) - np.min(stacked, axis=0)
        stability = np.clip(1.0 - (spread / 0.25), 0.2, 1.0).astype(np.float32)
        return probs_by_cut[-1], stability

    def categorize(self, prob: np.ndarray) -> np.ndarray:
        """Map probabilities to (`Extremo`, `Alto`, `Moderado`, `Bajo`)."""
        c = self.risk_cutoffs
        return np.where(
            prob >= c["Extremo"],
            "Extremo",
            np.where(
                prob >= c["Alto"],
                "Alto",
                np.where(prob >= c["Moderado"], "Moderado", "Bajo"),
            ),
        )
