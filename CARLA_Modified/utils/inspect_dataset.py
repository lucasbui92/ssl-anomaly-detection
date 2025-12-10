# Created by: Thuan Anh Bui (2025)
# Description: Inspect dataset for reporting.

import pandas as pd

def main():
    # Read the Smart Grid dataset
    df = pd.read_csv(f"./datasets/SmartGrid/smart_grid_major.csv") 

    # ---- Total number of rows ----
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")

    # ---- Rows with Overload Condition ----
    overload = df[df["Overload Condition"] == 1]
    overload_count = len(overload)
    overload_pct = (overload_count / total_rows) * 100
    print(f"Overload Condition: {overload_count:,} rows ({overload_pct:.2f}%)")

    # ---- Rows with Transformer Fault ----
    faults = df[df["Transformer Fault"] == 1]
    faults_count = len(faults)
    faults_pct = (faults_count / total_rows) * 100
    print(f"Transformer Fault: {faults_count:,} rows ({faults_pct:.2f}%)")

    # ---- Rows where EITHER occurs ----
    either = df[(df["Overload Condition"] == 1) | (df["Transformer Fault"] == 1)]
    either_count = len(either)
    either_pct = (either_count / total_rows) * 100
    print(f"Either Overload OR Transformer Fault: {either_count:,} rows ({either_pct:.2f}%)")

    # ---- Rows where BOTH occur ----
    both = df[(df["Overload Condition"] == 1) & (df["Transformer Fault"] == 1)]
    both_count = len(both)
    both_pct = (both_count / total_rows) * 100
    print(f"Both Overload + Transformer Fault: {both_count:,} rows ({both_pct:.2f}%)")

    # ---- Normal rows: both columns = 0 ----
    normal = df[(df["Overload Condition"] == 0) & (df["Transformer Fault"] == 0)]
    normal_count = len(normal)
    normal_pct = (normal_count / total_rows) * 100
    print(f"Normal rows (both = 0): {normal_count:,} rows ({normal_pct:.2f}%)")


if __name__ == '__main__':
    main()
