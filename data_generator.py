"""
data_generator.py
Physics-informed Burn-In Parametric Time-Series Generator for Space-Grade ICs.
Simulates environmental stress screening (ESS) at 125°C across wafer lots.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

# Datasheet static specification limits (Absolute Maximum Ratings)
STATIC_LIMITS = {
    "I_leak_uA": 50.0,   # Datasheet limit: 50 uA
    "Iddq_mA": 8.0,      # Datasheet limit: 8.0 mA
    "t_pd_ns": 25.0      # Datasheet limit: 25.0 ns
}

# Safety slope thresholds for early rejection at t=24h (rate of drift per hour)
SAFETY_SLOPES = {
    "I_leak_uA": 0.15,   # uA/h drift slope limit
    "Iddq_mA": 0.025,    # mA/h drift slope limit
    "t_pd_ns": 0.05      # ns/h drift slope limit
}


def generate_burnin_dataset(
    n_samples: int = 5000,
    n_lots: int = 6,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic time-series burn-in parametric data mimicking space-grade IC screening.
    Timepoints: 0h, 24h, 96h, 168h at 125°C.
    Defect Categories:
      - Normal: Standard Arrhenius thermal aging (+2% to +6% over 168h)
      - Latent Distribution Outlier (AEC-Q001 target): In-spec (<50uA) but extreme lot outlier (35-46uA)
      - Early Drifter: Normal at 0h, accelerated degradation (TDDB/electromigration) breaching limits by 168h
      - Gross Failure: Immediate defect exceeding static limits
    """
    np.random.seed(random_seed)

    lots = [f"LOT_{chr(65 + i)}" for i in range(n_lots)]
    # Each lot has intrinsic wafer fab baseline variations
    lot_profiles = {
        lot: {
            "I_leak_base": np.random.uniform(8.5, 12.0),
            "I_leak_std": np.random.uniform(1.2, 1.8),
            "Iddq_base": np.random.uniform(2.1, 2.7),
            "Iddq_std": np.random.uniform(0.18, 0.28),
            "t_pd_base": np.random.uniform(11.0, 13.5),
            "t_pd_std": np.random.uniform(0.6, 0.9),
        }
        for lot in lots
    }

    records = []
    component_counter = 1000

    # Defect distribution weights: Normal (87%), Latent Outlier (5%), Early Drifter (6%), Gross Failure (2%)
    defect_classes = ["normal", "latent_distribution_outlier", "early_drifter", "gross_failure"]
    defect_weights = [0.87, 0.05, 0.06, 0.02]

    for _ in range(n_samples):
        lot_id = np.random.choice(lots)
        wafer_id = f"W{np.random.randint(1, 25):02d}"
        comp_id = f"IC_{component_counter}"
        component_counter += 1

        category = np.random.choice(defect_classes, p=defect_weights)
        profile = lot_profiles[lot_id]

        # Simulate for Leakage Current (I_leak_uA) - Primary parameter
        leak_base = profile["I_leak_base"]
        leak_std = profile["I_leak_std"]
        limit = STATIC_LIMITS["I_leak_uA"]

        if category == "normal":
            # Normal Gaussian lot distribution
            v0 = np.clip(np.random.normal(leak_base, leak_std), 3.0, 18.0)
            # Sub-linear thermal aging: delta(t) = a * sqrt(t) + small noise
            aging_factor = np.random.uniform(0.15, 0.35)
            v24 = v0 + aging_factor * np.sqrt(24) + np.random.normal(0, 0.15)
            v96 = v0 + aging_factor * np.sqrt(96) + np.random.normal(0, 0.25)
            v168 = v0 + aging_factor * np.sqrt(168) + np.random.normal(0, 0.35)
            is_defective = 0

        elif category == "latent_distribution_outlier":
            # Latent DPAT outlier: In-spec (<50uA), but high outlier relative to lot (e.g., 34 to 47 uA)
            # Crucial requirement: It passes static limit (50 uA) but is a ~15 sigma lot anomaly!
            v0 = np.random.uniform(32.0, 46.5)
            aging_factor = np.random.uniform(0.20, 0.50)
            v24 = v0 + aging_factor * np.sqrt(24) + np.random.normal(0, 0.3)
            v96 = v0 + aging_factor * np.sqrt(96) + np.random.normal(0, 0.5)
            v168 = v0 + aging_factor * np.sqrt(168) + np.random.normal(0, 0.8)
            is_defective = 1

        elif category == "early_drifter":
            # Starts completely normal at 0h! Indistinguishable at t=0
            v0 = np.clip(np.random.normal(leak_base, leak_std), 7.0, 14.0)
            # TDDB / Electromigration accelerated power-law drift
            # At 24h, slight to moderate anomaly (+15% to +40%)
            accel_rate = np.random.uniform(0.04, 0.09)
            v24 = v0 + accel_rate * (24 ** 1.35) + np.random.normal(0, 0.2)
            v96 = v0 + accel_rate * (96 ** 1.35) + np.random.normal(0, 0.5)
            v168 = v0 + accel_rate * (168 ** 1.35) + np.random.normal(0, 1.0)
            is_defective = 1

        else: # gross_failure
            # Exceeds static limit immediately or early
            v0 = np.random.uniform(52.0, 85.0)
            v24 = v0 + np.random.uniform(2.0, 10.0)
            v96 = v24 + np.random.uniform(5.0, 20.0)
            v168 = v96 + np.random.uniform(10.0, 30.0)
            is_defective = 1

        # Correlated auxiliary parameters: Iddq (mA) and Propagation Delay (ns)
        # Quiescent current correlates with leakage
        iddq_0 = profile["Iddq_base"] + (v0 - leak_base) * 0.08 + np.random.normal(0, profile["Iddq_std"])
        iddq_24 = iddq_0 + (v24 - v0) * 0.06 + np.random.normal(0, 0.03)
        iddq_96 = iddq_0 + (v96 - v0) * 0.06 + np.random.normal(0, 0.05)
        iddq_168 = iddq_0 + (v168 - v0) * 0.06 + np.random.normal(0, 0.08)

        # Propagation delay increases with degradation
        tpd_0 = profile["t_pd_base"] + (v0 - leak_base) * 0.15 + np.random.normal(0, profile["t_pd_std"])
        tpd_24 = tpd_0 + (v24 - v0) * 0.12 + np.random.normal(0, 0.1)
        tpd_96 = tpd_0 + (v96 - v0) * 0.12 + np.random.normal(0, 0.2)
        tpd_168 = tpd_0 + (v168 - v0) * 0.12 + np.random.normal(0, 0.3)

        # Static screening result at 0h and 24h
        static_pass_0h = int(v0 <= STATIC_LIMITS["I_leak_uA"])
        static_pass_24h = int(v24 <= STATIC_LIMITS["I_leak_uA"])

        records.append({
            "component_id": comp_id,
            "lot_id": lot_id,
            "wafer_id": wafer_id,
            # Primary parameter: I_leak (uA)
            "val_0h": round(float(v0), 4),
            "val_24h": round(float(v24), 4),
            "val_96h": round(float(v96), 4),
            "val_168h": round(float(v168), 4),
            # Auxiliary parameters
            "iddq_0h": round(float(iddq_0), 4),
            "iddq_24h": round(float(iddq_24), 4),
            "tpd_0h": round(float(tpd_0), 4),
            "tpd_24h": round(float(tpd_24), 4),
            # Static check
            "static_limit": STATIC_LIMITS["I_leak_uA"],
            "static_pass_0h": static_pass_0h,
            "static_pass_24h": static_pass_24h,
            # Labels
            "defect_type": category,
            "is_defective": is_defective
        })

    df = pd.DataFrame(records)
    return df


def split_and_save_data(output_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates dataset, splits into Train (60%) and Test (40%), and saves CSVs.
    In Test set, val_168h is retained for ground-truth hidden evaluation.
    """
    df = generate_burnin_dataset(n_samples=6000, n_lots=6, random_seed=42)
    
    # Stratified split by defect_type to ensure balanced evaluation
    train_dfs, test_dfs = [], []
    for defect, group in df.groupby("defect_type"):
        shuffled = group.sample(frac=1.0, random_state=42)
        n_train = int(len(shuffled) * 0.6)
        train_dfs.append(shuffled.iloc[:n_train])
        test_dfs.append(shuffled.iloc[n_train:])

    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(f"{output_dir}/burnin_full_dataset.csv", index=False)
    train_df.to_csv(f"{output_dir}/burnin_train.csv", index=False)
    test_df.to_csv(f"{output_dir}/burnin_test.csv", index=False)

    print(f"Dataset generated successfully:")
    print(f"  Total records: {len(df)}")
    print(f"  Train set:     {len(train_df)}")
    print(f"  Test set:      {len(test_df)}")
    print(f"  Defect distribution:\n{df['defect_type'].value_counts(normalize=True).round(4) * 100}%")

    return train_df, test_df


if __name__ == "__main__":
    split_and_save_data(".")
