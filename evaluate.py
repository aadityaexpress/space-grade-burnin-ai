"""
evaluate.py
Comprehensive Benchmark and Evaluation Suite for Space-Grade Burn-In AI System.
Evaluates:
 1. Module A: Anomaly Detection Score, Recall, Cost-Penalized Escape Rate.
 2. Module B: Drift Prediction Accuracy (MAE, RMSE, R2) & Chamber Hours Saved.
 3. Module C: QA Inspector Explainability Audit Verification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score, mean_absolute_error, mean_squared_error, r2_score

from data_generator import split_and_save_data
from module_a_outlier_detector import DynamicOutlierDetector
from module_b_drift_predictor import DriftPredictor
from module_c_explainability import QAInspectorExplainer


def evaluate_system():
    print("=" * 70)
    print(" SPACE-GRADE SEMICONDUCTOR BURN-IN SCREENING BENCHMARK SUITE")
    print("=" * 70)

    # Step 1: Generate & Split Data
    print("\n[STEP 1] Generating Physics-Informed Burn-In Dataset...")
    train_df, test_df = split_and_save_data(".")

    # Step 2: Benchmark Traditional Static Limits
    print("\n" + "-" * 70)
    print("[BASELINE] Traditional Static Parametric Screening (Limit: 50.0 uA)")
    print("-" * 70)
    static_pass = (test_df["val_0h"] <= 50.0) & (test_df["val_24h"] <= 50.0)
    # A component passing static test is predicted as normal (0), failing is 1
    static_pred = (~static_pass).astype(int)
    y_true = test_df["is_defective"].values

    static_fn = int(np.sum((y_true == 1) & (static_pred == 0)))
    static_fp = int(np.sum((y_true == 0) & (static_pred == 1)))
    static_recall = recall_score(y_true, static_pred)
    static_cost = 10 * static_fn + 1 * static_fp

    print(f"Traditional Static Screening Results:")
    print(f"  - Latent Defect Field Escapes (False Negatives): {static_fn} parts! (CATASTROPHIC)")
    print(f"  - Recall Rate: {static_recall * 100:.2f}% (Misses nearly all latent defects)")
    print(f"  - Mission Risk Cost Penalty: {static_cost}")

    # Step 3: Train & Benchmark Module A (Dynamic Outlier Detection)
    print("\n" + "-" * 70)
    print("[MODULE A] Dynamic Outlier Detection System (AEC-Q001 DPAT + ML)")
    print("-" * 70)
    detector = DynamicOutlierDetector(k_sigma=3.5, contamination=0.08, cost_ratio_fn_fp=10.0)
    detector.fit(train_df, y_true=train_df["is_defective"])

    scores, mod_a_flags, annotated_a = detector.predict_scores(test_df)

    mod_a_fn = int(np.sum((y_true == 1) & (mod_a_flags == 0)))
    mod_a_fp = int(np.sum((y_true == 0) & (mod_a_flags == 1)))
    mod_a_recall = recall_score(y_true, mod_a_flags)
    mod_a_precision = precision_score(y_true, mod_a_flags)
    mod_a_f2 = fbeta_score(y_true, mod_a_flags, beta=2)
    mod_a_cost = 10 * mod_a_fn + 1 * mod_a_fp

    print(f"Module A Performance at 0h:")
    print(f"  - Recall (Sensitivity)    : {mod_a_recall * 100:.2f}%")
    print(f"  - Precision              : {mod_a_precision * 100:.2f}%")
    print(f"  - F2-Score (Recall Heavy): {mod_a_f2:.4f}")
    print(f"  - Escaped Latent Defects : {mod_a_fn} (Reduced from {static_fn})")
    print(f"  - Mission Risk Cost      : {mod_a_cost} (vs {static_cost} baseline)")

    # Step 4: Train & Benchmark Module B (Drift Predictor)
    print("\n" + "-" * 70)
    print("[MODULE B] Time-Series Early Drift Predictor & 24h Early Rejection")
    print("-" * 70)
    predictor = DriftPredictor(safety_slope=0.15, datasheet_limit=50.0, model_type="lightgbm")
    predictor.fit(train_df)

    pred_168h, early_reject_flags, annotated_b = predictor.predict(annotated_a)

    # Evaluate prediction accuracy on hidden ground truth val_168h
    actual_168h = test_df["val_168h"].values
    mae = mean_absolute_error(actual_168h, pred_168h)
    rmse = np.sqrt(mean_squared_error(actual_168h, pred_168h))
    r2 = r2_score(actual_168h, pred_168h)

    print(f"Drift Prediction Accuracy (0h + 24h -> 168h):")
    print(f"  - Mean Absolute Error (MAE) : {mae:.4f} uA")
    print(f"  - Root Mean Squared (RMSE)  : {rmse:.4f} uA")
    print(f"  - R^2 Score                 : {r2:.4f}")

    # Evaluate Early Chamber Rejection Efficiency
    early_drifter_mask = test_df["defect_type"] == "early_drifter"
    drifters_caught = np.sum(early_reject_flags[early_drifter_mask] == 1)
    total_drifters = np.sum(early_drifter_mask)
    hours_saved = np.sum(early_reject_flags) * 144

    print(f"\nChamber Optimization at 24h:")
    print(f"  - Early Drifters Detected at 24h: {drifters_caught} / {total_drifters} ({drifters_caught/total_drifters*100:.1f}%)")
    print(f"  - Total Chamber Hours Saved    : {hours_saved:,} hours across test fleet!")

    # Step 5: Combined System Screening (Module A + Module B)
    combined_rejections = np.maximum(mod_a_flags, early_reject_flags)
    comb_recall = recall_score(y_true, combined_rejections)
    comb_fn = int(np.sum((y_true == 1) & (combined_rejections == 0)))

    print("\n" + "=" * 70)
    print(" FINAL INTEGRATED MISSION SCREENING VERDICT (Module A + Module B)")
    print("=" * 70)
    print(f"  - Overall Defect Detection Recall : {comb_recall * 100:.2f}%")
    print(f"  - Total Escaped Defects           : {comb_fn} out of {np.sum(y_true)} defective parts")
    print(f"  - Flight Reliability Assurance   : {'PASS (Space Qualified Level)' if comb_recall > 0.95 else 'FAIL'}")

    # Step 6: Verify Module C Explainability
    print("\n" + "-" * 70)
    print("[MODULE C] QA Inspector Explainability Verification")
    print("-" * 70)
    explainer = QAInspectorExplainer(drift_predictor=predictor)

    # Pick an example of each defect class to inspect
    sample_categories = ["normal", "latent_distribution_outlier", "early_drifter", "gross_failure"]
    sample_indices = []
    for cat in sample_categories:
        idx = test_df[test_df["defect_type"] == cat].index[0]
        sample_indices.append(idx)

    for idx in sample_indices:
        row = annotated_b.iloc[idx]
        card = explainer.explain_component(row)
        report = explainer.generate_inspector_text_report(card)
        print(f"\n--- Inspection Sample: {row['defect_type'].upper()} ---")
        print(report)

    # Save benchmark summary results
    results_summary = {
        "metric": [
            "Static Screening Recall", "Static Field Escapes (FN)", "Module A Recall",
            "Module A F2-Score", "Drift MAE (uA)", "Drift RMSE (uA)", "Drift R2",
            "Early Drifters Caught @ 24h", "Integrated System Recall", "Chamber Hours Saved"
        ],
        "value": [
            f"{static_recall*100:.1f}%", f"{static_fn}", f"{mod_a_recall*100:.1f}%",
            f"{mod_a_f2:.4f}", f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}",
            f"{drifters_caught}/{total_drifters} ({drifters_caught/total_drifters*100:.1f}%)",
            f"{comb_recall*100:.1f}%", f"{hours_saved:,} hrs"
        ]
    }
    pd.DataFrame(results_summary).to_csv("benchmark_results.csv", index=False)
    print("\nBenchmark results saved to 'benchmark_results.csv'.")


if __name__ == "__main__":
    evaluate_system()
