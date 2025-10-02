import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_rel

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

CSV_PATH = "data/curriculum_evaluation_results_PPO.csv"
ALPHA = 0.05


def safe_ttest(group1: pd.Series, group2: pd.Series, paired: bool = False):
    """Return t-statistic and p-value, handling empty groups."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan
    if paired:
        min_len = min(len(group1), len(group2))
        return ttest_rel(group1.iloc[:min_len], group2.iloc[:min_len])
    return ttest_ind(group1, group2, equal_var=False)


def main():
    df = pd.read_csv(CSV_PATH)

    tests = {}

    # 1. Seen vs Unseen environments
    seen = df[df["evaluated_env"] == "seen"]["average_success_rate"].dropna()
    unseen = df[df["evaluated_env"] == "unseen"]["average_success_rate"].dropna()
    t_stat, p_val = safe_ttest(seen, unseen)
    tests["Seen vs Unseen"] = (t_stat, p_val)

    # 2. Sparse vs Step-Cost rewards
    sparse = df[df["reward_type"] == "sparse"]["average_success_rate"].dropna()
    step_cost = df[df["reward_type"] == "step_cost"]["average_success_rate"].dropna()
    t_stat, p_val = safe_ttest(sparse, step_cost)
    tests["Sparse vs Step-Cost"] = (t_stat, p_val)

    # ------------------------------------------------------------------
    # Purge sparse rows from further analysis as requested
    # ------------------------------------------------------------------
    df = df[df["reward_type"] != "sparse"]

    # 3. Complete vs Incomplete instructions (on step_cost data only)
    complete = df[df["instruction_type"] == "complete"]["average_success_rate"].dropna()
    incomplete = df[df["instruction_type"] == "incomplete"]["average_success_rate"].dropna()
    t_stat, p_val = safe_ttest(complete, incomplete)
    tests["Complete vs Incomplete"] = (t_stat, p_val)


    # Collect p-values and apply FDR correction
    p_values = [p for (_, p) in tests.values()]

    if multipletests is None:
        print("statsmodels not installed; printing raw p-values only. Install statsmodels for FDR correction.")
        adjusted_p = [np.nan] * len(p_values)
    else:
        _, adj_p_values, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")
        adjusted_p = adj_p_values

    print(f"\nBenjamini-Hochberg FDR Correction (alpha={ALPHA})")
    print("Comparison                         t-stat     raw p-val    adj p-val   Significant?")
    print("----------------------------------------------------------------------------------")
    for (name, (t_stat, p_val)), adj in zip(tests.items(), adjusted_p):
        sig = "YES" if not np.isnan(adj) and adj < ALPHA else "NO"
        print(f"{name:<33} {t_stat:>8.3f}   {p_val:>10.3e}   {adj:>10.3e}   {sig}")


if __name__ == "__main__":
    main() 