"""
module_a_outlier_detector.py
Module A: Dynamic Outlier Detection System for Space-Grade Semiconductor Components.
Implements:
 1. AEC-Q001 Dynamic Part Average Testing (DPAT) with robust lot statistics.
 2. Multivariate Isolation Forest on lot-normalized features (leakage, Iddq, propagation delay).
 3. Cost-Sensitive Threshold Tuning heavily penalizing False Negatives (Cost_FN >> Cost_FP).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import IsolationForest


class DynamicOutlierDetector:
    """
    Dynamic Part Average Testing (DPAT) & Multivariate Anomaly Detector.
    Catches latent defects that pass absolute static datasheet limits (e.g. 45uA vs 50uA limit)
    when the lot distribution average is ~10uA.
    """

    def __init__(
        self,
        k_sigma: float = 3.5,
        contamination: float = 0.08,
        cost_ratio_fn_fp: float = 10.0,
        random_state: int = 42
    ):
        """
        Args:
            k_sigma: Number of robust standard deviations for DPAT upper/lower limits (AEC-Q001 standard).
            contamination: Expected anomaly contamination rate in wafer lots.
            cost_ratio_fn_fp: Relative penalty cost of a False Negative vs False Positive.
        """
        self.k_sigma = k_sigma
        self.contamination = contamination
        self.cost_ratio_fn_fp = cost_ratio_fn_fp
        self.random_state = random_state

        self.lot_stats: Dict[str, Dict[str, float]] = {}
        self.iso_forest: Optional[IsolationForest] = None
        self.optimal_threshold: float = 0.5
        self.feature_cols = ["z_val_0h", "z_iddq_0h", "z_tpd_0h"]

    def _compute_robust_stats(self, series: pd.Series) -> Tuple[float, float]:
        """Compute robust location (median) and scale (Normalized IQR / MAD) to avoid outlier skew."""
        med = float(series.median())
        q75, q25 = np.percentile(series, [75, 25])
        iqr = q75 - q25
        # Standard deviation equivalent for Gaussian distribution: IQR / 1.349
        robust_std = float(iqr / 1.349) if iqr > 1e-6 else float(series.std() + 1e-6)
        return med, robust_std

    def _extract_lot_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Calculates lot-relative normalized features (Z-scores) using robust lot statistics."""
        data = df.copy()

        if is_training:
            self.lot_stats = {}
            for lot_id, group in data.groupby("lot_id"):
                v_med, v_std = self._compute_robust_stats(group["val_0h"])
                i_med, i_std = self._compute_robust_stats(group["iddq_0h"])
                t_med, t_std = self._compute_robust_stats(group["tpd_0h"])
                
                # AEC-Q001 DPAT Limits
                dpat_upper = v_med + self.k_sigma * v_std
                dpat_lower = max(0.0, v_med - self.k_sigma * v_std)

                self.lot_stats[lot_id] = {
                    "v_med": v_med, "v_std": v_std,
                    "i_med": i_med, "i_std": i_std,
                    "t_med": t_med, "t_std": t_std,
                    "dpat_upper": dpat_upper,
                    "dpat_lower": dpat_lower
                }

        # Calculate Z-scores relative to component's lot
        z_v, z_i, z_t = [], [], []
        dpat_flag = []
        dpat_upper_col, dpat_lower_col = [], []

        for _, row in data.iterrows():
            lot_id = row["lot_id"]
            stats = self.lot_stats.get(lot_id)
            if stats is None:
                # Fallback to global statistics if an unseen lot occurs
                stats = list(self.lot_stats.values())[0]

            zv = (row["val_0h"] - stats["v_med"]) / (stats["v_std"] + 1e-6)
            zi = (row["iddq_0h"] - stats["i_med"]) / (stats["i_std"] + 1e-6)
            zt = (row["tpd_0h"] - stats["t_med"]) / (stats["t_std"] + 1e-6)

            z_v.append(zv)
            z_i.append(zi)
            z_t.append(zt)

            # DPAT single-parameter statistical check
            is_dpat_outlier = int((row["val_0h"] > stats["dpat_upper"]) or (row["val_0h"] < stats["dpat_lower"]))
            dpat_flag.append(is_dpat_outlier)
            dpat_upper_col.append(stats["dpat_upper"])
            dpat_lower_col.append(stats["dpat_lower"])

        data["z_val_0h"] = z_v
        data["z_iddq_0h"] = z_i
        data["z_tpd_0h"] = z_t
        data["dpat_outlier_0h"] = dpat_flag
        data["dpat_limit_upper"] = dpat_upper_col
        data["dpat_limit_lower"] = dpat_lower_col

        return data

    def fit(self, train_df: pd.DataFrame, y_true: Optional[pd.Series] = None) -> "DynamicOutlierDetector":
        """
        Fit DPAT bounds and multivariate Isolation Forest.
        If ground truth y_true is available, optimizes decision threshold for Cost(FN) = 10 * Cost(FP).
        """
        features_df = self._extract_lot_features(train_df, is_training=True)
        X = features_df[self.feature_cols]

        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150
        )
        self.iso_forest.fit(X)

        # Raw anomaly score: higher = more anomalous (normalized 0 to 1)
        raw_scores = -self.iso_forest.score_samples(X)
        self.min_score = float(raw_scores.min())
        self.max_score = float(raw_scores.max())
        norm_scores = (raw_scores - self.min_score) / (self.max_score - self.min_score + 1e-6)

        # Cost-sensitive threshold optimization if labels provided
        if y_true is not None:
            best_cost = float("inf")
            best_t = 0.5
            candidate_thresholds = np.linspace(0.1, 0.9, 81)
            y_arr = y_true.values

            for t in candidate_thresholds:
                pred = (norm_scores >= t).astype(int)
                fn = np.sum((y_arr == 1) & (pred == 0))
                fp = np.sum((y_arr == 0) & (pred == 1))
                # Heavy penalty for False Negatives (catastrophic field escape)
                cost = self.cost_ratio_fn_fp * fn + 1.0 * fp
                if cost < best_cost:
                    best_cost = cost
                    best_t = t
            self.optimal_threshold = best_t
        else:
            self.optimal_threshold = 0.5

        return self

    def predict_scores(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Returns:
            anomaly_scores: Continuous anomaly score [0, 1]
            flags: Binary flag (1 = Outlier, 0 = In-Spec Normal)
            annotated_df: DataFrame with DPAT limits and Z-scores
        """
        features_df = self._extract_lot_features(df, is_training=False)
        X = features_df[self.feature_cols]

        raw_scores = -self.iso_forest.score_samples(X)
        norm_scores = (raw_scores - self.min_score) / (self.max_score - self.min_score + 1e-6)
        norm_scores = np.clip(norm_scores, 0.0, 1.0)

        # Component is flagged if:
        # 1. DPAT univariate limit exceeded (AEC-Q001 rule) OR
        # 2. Multivariate Isolation Forest anomaly score exceeds cost-tuned threshold
        ml_flags = (norm_scores >= self.optimal_threshold).astype(int)
        combined_flags = np.maximum(features_df["dpat_outlier_0h"].values, ml_flags)

        features_df["anomaly_score_0h"] = np.round(norm_scores, 4)
        features_df["module_a_flag"] = combined_flags

        return norm_scores, combined_flags, features_df

    def explain_component_0h(self, row: pd.Series) -> Dict[str, any]:
        """Generate human-readable QA audit explanation for why component was flagged at 0h."""
        lot_id = row["lot_id"]
        stats = self.lot_stats.get(lot_id, list(self.lot_stats.values())[0])

        val = row["val_0h"]
        lot_med = stats["v_med"]
        lot_std = stats["v_std"]
        z_score = (val - lot_med) / (lot_std + 1e-6)
        static_limit = row.get("static_limit", 50.0)

        is_static_pass = val <= static_limit
        is_dpat_violation = val > stats["dpat_upper"] or val < stats["dpat_lower"]

        explanation = {
            "component_id": row.get("component_id", "Unknown"),
            "lot_id": lot_id,
            "val_0h": val,
            "lot_median": round(lot_med, 2),
            "lot_robust_std": round(lot_std, 2),
            "z_score": round(z_score, 2),
            "static_limit": static_limit,
            "dpat_upper_limit": round(stats["dpat_upper"], 2),
            "static_pass": is_static_pass,
            "dpat_violation": is_dpat_violation,
            "summary": (
                f"Component {row.get('component_id', 'IC')} measures {val:.2f} µA. "
                f"While {'PASSING' if is_static_pass else 'FAILING'} static limit ({static_limit:.1f} µA), "
                f"it is a {z_score:+.1f}σ anomaly relative to Lot {lot_id} median ({lot_med:.2f} ± {lot_std:.2f} µA). "
                f"Dynamic screening limit: {stats['dpat_upper']:.2f} µA."
            )
        }
        return explanation
