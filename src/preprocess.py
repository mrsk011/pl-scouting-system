from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/fbref_PL_2024-25.csv")
PROCESSED_DATA_PATH = Path("data/processed/players_per90.csv")

# Raw counting stats that will be aggregated and converted to per-90 rates
COUNTING_COLS = [
    "MP", "Starts", "Min",
    "Gls", "Ast", "G-PK", "PK", "PKatt",
    "CrdY", "CrdR",
    "xG", "npxG", "xAG",
    "PrgC", "PrgP", "PrgR",
]

# Non-redundant features used for clustering and similarity
# Avoids double-counting cols like G+A or npxG+xAG alongside their components
MODEL_SOURCE_COLS = [
    "G-PK", "Ast", "npxG", "xAG",
    "PrgC", "PrgP", "PrgR", "CrdY",
]

def load_and_clean(
        raw_path: Path = RAW_DATA_PATH,
        min_minutes: int = 600,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # Drop repeated header rows if present & remove Goalkeepers
    df = df[df["Player"] != "Player"].copy()
    df = df[~df["Pos"].astype(str).str.contains("GK", na=False)].copy()

    # Convert counting columns safely (handles commas like 1,250)
    for col in COUNTING_COLS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0.0)

    df["Age"] = pd.to_numeric(
        df["Age"].astype(str).str.split("-").str[0],
        errors="coerce",
    )

    # Aggregate mid-season transfers (same player appearing for 2 clubs)
    agg_rules = {
        "Nation": "first",
        "Pos": "first",
        "Squad": lambda s: " / ".join(dict.fromkeys(s.dropna())),
        "Age": "max",
        **{col: "sum" for col in COUNTING_COLS},
    }
    df = df.groupby(["Player", "Born"], as_index=False).agg(agg_rules)

    # Filter out players with low playing time that distort per-90 rates
    df = df[df["Min"] >= min_minutes].copy()
    df["90s"] = (df["Min"] / 90.0).round(2)

    # Extract primary position (e.g., "MF,FW" -> "MF")
    df["Primary_Pos"] = df["Pos"].astype(str).str.split(",").str[0].str.strip()

    # 6. Recompute per-90 stats from aggregated totals (accurate for transfers)
    for col in MODEL_SOURCE_COLS:
        df[f"{col}_per90"] = (df[col] / df["90s"]).round(3)

    df = df.sort_values("Min", ascending=False).reset_index(drop=True)
    return df

def main() -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df = load_and_clean()
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    per90_cols = [f"{c}_per90" for c in MODEL_SOURCE_COLS]
    print(f"Saved {len(cleaned_df)} outfield players to {PROCESSED_DATA_PATH}")
    print("\nFeature columns:", per90_cols)
    print("\nTop 5 players by minutes:")
    print(cleaned_df[["Player", "Squad", "Primary_Pos", "90s", "npxG_per90", "PrgP_per90"]].head())


if __name__ == "__main__":
    main()
