import pandas as pd
import numpy as np

# ----------------------------
# Load Data
# ----------------------------

auto = pd.read_csv("vehicle_log.csv")
manual = pd.read_csv("Manual_Intersection_ Analysis.csv")

# ----------------------------
# Clean Manual Data
# ----------------------------

manual = manual.iloc[1:].copy()
manual.rename(columns={"Unnamed: 0":"minute"}, inplace=True)

manual["minute"] = (
    manual["minute"]
    .astype(str)
    .str.extract(r'(\d+):')[0]
)

manual["minute"] = pd.to_numeric(manual["minute"], errors="coerce")
manual = manual.dropna(subset=["minute"])
manual["minute"] = manual["minute"].astype(int)

movement_cols = [str(i) for i in range(1,13)]
manual = manual[["minute"] + movement_cols]

manual = manual.melt(
    id_vars=["minute"],
    value_vars=movement_cols,
    var_name="movement_id",
    value_name="manual_count"
)

manual["movement_id"] = manual["movement_id"].astype(int)
manual["manual_count"] = manual["manual_count"].astype(int)

# ----------------------------
# Process Automated Log
# ----------------------------

auto["minute"] = (auto["timestamp"] // 60).astype(int)

auto_grouped = (
    auto.groupby(["minute","movement_id"])
        .size()
        .reset_index(name="auto_count")
)

# ----------------------------
# Merge
# ----------------------------

merged = pd.merge(
    manual,
    auto_grouped,
    on=["minute","movement_id"],
    how="left"
)

merged["auto_count"] = merged["auto_count"].fillna(0).astype(int)

# ----------------------------
# Error Metrics
# ----------------------------

merged["abs_error"] = abs(merged["auto_count"] - merged["manual_count"])

merged["pct_error"] = np.where(
    merged["manual_count"] > 0,
    merged["abs_error"] / merged["manual_count"],
    0
)

merged["accuracy"] = 1 - merged["pct_error"]

# ----------------------------
# Summary Stats
# ----------------------------

overall_mae = merged["abs_error"].mean()
overall_mape = merged["pct_error"].mean() * 100
overall_accuracy = merged["accuracy"].mean() * 100

print("\n========= OVERALL PERFORMANCE =========")
print(f"Mean Absolute Error (MAE): {overall_mae:.3f}")
print(f"Mean Absolute % Error (MAPE): {overall_mape:.2f}%")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

# ----------------------------
# Per-Movement Accuracy
# ----------------------------

movement_accuracy = (
    merged.groupby("movement_id")
           .agg(
               manual_total=("manual_count","sum"),
               auto_total=("auto_count","sum"),
               mean_accuracy=("accuracy","mean"),
               total_error=("abs_error","sum")
           )
           .reset_index()
)

movement_accuracy["accuracy_%"] = movement_accuracy["mean_accuracy"] * 100

print("\n========= PER-MOVEMENT ACCURACY =========")
print(movement_accuracy[["movement_id","manual_total","auto_total","accuracy_%","total_error"]])

# ----------------------------
# Save Results
# ----------------------------

merged.to_csv("accuracy_detailed.csv", index=False)
movement_accuracy.to_csv("accuracy_by_movement.csv", index=False)

print("\nSaved: accuracy_detailed.csv")
print("Saved: accuracy_by_movement.csv")