"""
module_b_drift_predictor.py
Module B: Time-Series Parametric Drift Predictor and Early Rejection Engine.
Forecasts Value_168h using only 0h and 24h burn-in data.
Flags components for early chamber rejection (t=24h) if predicted drift rate exceeds safety slope.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.linear_model import BayesianRidge
import lightgbm as lgb


class DriftPredictor:
    """
    Time-Series Early Drift Predictor.
    Takes 0h and 24h measurements, engineers physical degradation features,
    and forecasts the 168h parametric value.
    Computes safety slope to make early pass/reject decisions at 24h.
    """

    def __init__(
        self,
        safety_slope: float = 0.15, # uA/h threshold for leakage
        datasheet_limit: float = 50.0,
        model_type: str = "lightgbm", # 'lightgbm' or 'bayesian_ridge'
        random_state: int = 42
    ):
        self.safety_slope = safety_slope
        self.datasheet_limit = datasheet_limit
        self.model_type = model_type
        self.random_state = random_state

        self.model = None
        self.lot_drift_stats: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = [
            "val_0h", "val_24h",
            "delta_24_0", "pct_drift_24_0", "drift_rate_24h",
            "iddq_0h", "iddq_24h", "delta_iddq_24_0",
            "tpd_0h", "tpd_24h", "delta_tpd_24_0",
            "z_delta_leak"
        ]

    def _engineer_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Engineer domain-specific physical degradation features available at t=24h."""
        data = df.copy()

        # Primary parameter deltas
        data["delta_24_0"] = data["val_24h"] - data["val_0h"]
        data["pct_drift_24_0"] = (data["delta_24_0"] / (data["val_0h"] + 1e-4)) * 100.0
        data["drift_rate_24h"] = data["delta_24_0"] / 24.0

        # Auxiliary deltas
        data["delta_iddq_24_0"] = data["iddq_24h"] - data["iddq_0h"]
        data["delta_tpd_24_0"] = data["tpd_24h"] - data["tpd_0h"]

        # Lot-level standard drift profiles
        if is_training:
            self.lot_drift_stats = {}
            for lot_id, grp in data.groupby("lot_id"):
                med_d = float(grp["delta_24_0"].median())
                q75, q25 = np.percentile(grp["delta_24_0"], [75, 25])
                iqr = q75 - q25
                robust_std = float(iqr / 1.349) if iqr > 1e-6 else float(grp["delta_24_0"].std() + 1e-6)
                self.lot_drift_stats[lot_id] = {"med_delta": med_d, "std_delta": robust_std}

        z_deltas = []
        for _, row in data.iterrows():
            lot_id = row["lot_id"]
            stats = self.lot_drift_stats.get(lot_id)
            if stats is None:
                stats = list(self.lot_drift_stats.values())[0]
            z = (row["delta_24_0"] - stats["med_delta"]) / (stats["std_delta"] + 1e-6)
            z_deltas.append(z)

        data["z_delta_leak"] = z_deltas
        return data

    def fit(self, train_df: pd.DataFrame) -> "DriftPredictor":
        """Train regression model on 0h and 24h features to forecast val_168h."""
        feats = self._engineer_features(train_df, is_training=True)
        X = feats[self.feature_names]
        y = train_df["val_168h"]

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1
            )
            self.model.fit(X, y)
        else:
            self.model = BayesianRidge()
            self.model.fit(X, y)

        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Forecasts val_168h and evaluates safety slope.
        Returns:
            pred_168h: Array of forecasted 168h values.
            rejection_flags: Array of binary flags (1 = Reject early at 24h, 0 = Safe to continue).
            annotated_df: DataFrame with predictions and safety metrics.
        """
        feats = self._engineer_features(df, is_training=False)
        X = feats[self.feature_names]

        pred_168h = self.model.predict(X)
        pred_168h = np.round(pred_168h, 4)

        # Safety Slope calculation: total predicted drift rate per hour over 168h
        # slope = (predicted_168h - val_0h) / 168.0
        predicted_drift_total = pred_168h - feats["val_0h"].values
        predicted_drift_slope = predicted_drift_total / 168.0

        # Early rejection triggers:
        # 1. Predicted drift slope exceeds safety critical threshold OR
        # 2. Predicted 168h value breaches static datasheet limit
        early_reject = (
            (predicted_drift_slope > self.safety_slope) |
            (pred_168h > self.datasheet_limit)
        ).astype(int)

        annotated_df = feats.copy()
        annotated_df["pred_val_168h"] = pred_168h
        annotated_df["pred_drift_total"] = np.round(predicted_drift_total, 4)
        annotated_df["pred_drift_slope"] = np.round(predicted_drift_slope, 4)
        annotated_df["early_rejection_flag"] = early_reject

        return pred_168h, early_reject, annotated_df

    def explain_drift_prediction(self, row: pd.Series) -> Dict[str, any]:
        """Generate plain-text explanation of 168h drift forecast and safety slope evaluation."""
        v0 = row["val_0h"]
        v24 = row["val_24h"]
        pred_168 = row.get("pred_val_168h", None)
        slope = row.get("pred_drift_slope", (pred_168 - v0) / 168.0 if pred_168 else 0.0)
        reject = row.get("early_rejection_flag", int(slope > self.safety_slope or pred_168 > self.datasheet_limit))

        delta_24 = v24 - v0
        pct_24 = (delta_24 / (v0 + 1e-4)) * 100.0

        explanation = {
            "component_id": row.get("component_id", "Unknown"),
            "val_0h": round(v0, 2),
            "val_24h": round(v24, 2),
            "delta_24h": round(delta_24, 2),
            "pct_drift_24h": round(pct_24, 2),
            "predicted_168h": round(pred_168, 2) if pred_168 is not None else None,
            "safety_slope_observed": round(slope, 4),
            "safety_slope_threshold": self.safety_slope,
            "early_rejection": bool(reject),
            "summary": (
                f"Component {row.get('component_id', 'IC')} drifted from {v0:.2f} µA at 0h to {v24:.2f} µA at 24h ({pct_24:+.1f}%). "
                f"Model forecasts 168h value of {pred_168:.2f} µA (drift slope: {slope:.4f} µA/h vs limit {self.safety_slope:.4f} µA/h). "
                f"Verdict: {'REJECT AT 24H (Saves 144h chamber time)' if reject else 'PASS / CONTINUE SCREENING'}."
            )
        }
        return explanation
