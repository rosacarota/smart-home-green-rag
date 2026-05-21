import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_FILE = Path("data/processed/dataset_candidates_rag.csv")
OUTPUT_SORTED_FILE = Path("data/processed/dataset_candidates_rag_sorted.csv")
OUTPUT_GROUP_PREFIX = "dataset_group"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE, sep=";")

# Controllo che la colonna esista
if "rule_scope" not in df.columns:
    print("Colonne trovate nel file:")
    print(df.columns.tolist())
    raise ValueError("La colonna 'rule_scope' non è presente nel dataset.")

# =========================
# CLEAN rule_scope
# =========================
df["rule_scope"] = df["rule_scope"].astype(str).str.strip()

# =========================
# CUSTOM ORDER
# =========================
scope_priority = {
    "core_smart_home": 0,
    "context_aware_smart_home": 1,
    "smart_home_related": 2
}

df["scope_order"] = df["rule_scope"].map(scope_priority).fillna(99)

# =========================
# SORT DATAFRAME
# =========================
df_sorted = df.sort_values(
    by=["scope_order", "rule_scope"],
    ascending=[True, True]
).reset_index(drop=True)

# Rimuove la colonna tecnica
df_sorted = df_sorted.drop(columns=["scope_order"])

# =========================
# SAVE SORTED DATASET
# =========================
df_sorted.to_csv(OUTPUT_SORTED_FILE, sep=";", index=False, encoding="utf-8")

print(f"Dataset ordinato salvato in: {OUTPUT_SORTED_FILE}")

# =========================
# PRINT SUMMARY
# =========================
print("\nConteggio per rule_scope:")
print(df_sorted["rule_scope"].value_counts())