"""Prepare a recent Ghaziabad APMC training series from Kaggle and Agmarknet CSVs."""

import argparse
from pathlib import Path

import pandas as pd


CROPS = {
    "green chilli": "Green Chilli",
    "green chili": "Green Chilli",
    "chilli": "Green Chilli",
    "chili": "Green Chilli",
    "mirchi": "Green Chilli",
    "potato": "Potato",
    "aloo": "Potato",
    "rice": "Rice",
    "paddy": "Rice",
    "chawal": "Rice",
    "tomato": "Tomato",
    "tamatar": "Tomato",
    "wheat": "Wheat",
    "gehun": "Wheat",
}
MARKET_ALIASES = {"ghaziabad", "ghaziabad apmc"}
OUTPUT_COLUMNS = [
    "State", "District", "Market", "Commodity", "Variety", "Grade",
    "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price",
]


def read_source(path: Path, as_of: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    required = set(OUTPUT_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame["Arrival_Date"] = pd.to_datetime(
        frame["Arrival_Date"], format="mixed", dayfirst=False, errors="coerce"
    )
    frame["_market_key"] = frame["Market"].astype(str).str.strip().str.casefold()
    frame["_crop_key"] = frame["Commodity"].astype(str).str.strip().str.casefold()
    frame["Commodity"] = frame["_crop_key"].map(CROPS)
    frame["Modal_Price"] = pd.to_numeric(
        frame["Modal_Price"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    state = frame["State"].astype(str).str.strip().str.casefold()
    district = frame["District"].astype(str).str.strip().str.casefold()
    frame = frame[
        state.isin({"up", "uttar pradesh"})
        & district.eq("ghaziabad")
        & frame["_market_key"].isin(MARKET_ALIASES)
        & frame["Commodity"].isin(set(CROPS.values()))
        & frame["Arrival_Date"].notna()
        & (frame["Arrival_Date"] <= as_of)
        & frame["Modal_Price"].gt(0)
    ].copy()

    frame["Market"] = "Ghaziabad"
    return frame[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True, type=Path)
    parser.add_argument("--agmarknet", required=True, type=Path)
    parser.add_argument("--as-of-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output", default="ghaziabad_best_mandi_2018_2026.csv", type=Path)
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of_date).normalize()
    historical = read_source(args.kaggle, as_of)
    recent = read_source(args.agmarknet, as_of)

    # Where the official feed has a price for a date/crop, prefer it over the
    # Kaggle value for that same date. Keep different varieties from the feed.
    official_keys = set(zip(recent["Arrival_Date"], recent["Commodity"]))
    historical = historical[
        ~historical.apply(
            lambda row: (row["Arrival_Date"], row["Commodity"]) in official_keys,
            axis=1,
        )
    ]
    merged = pd.concat([historical, recent], ignore_index=True)
    merged["Arrival_Date"] = merged["Arrival_Date"].dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates().sort_values(["Commodity", "Arrival_Date"])
    merged.to_csv(args.output, index=False)

    print(f"Saved {len(merged):,} rows: {args.output}")
    print("Market series latest dates:")
    for crop, group in merged.groupby("Commodity"):
        print(f"  {crop}: {group['Arrival_Date'].max()} ({len(group):,} rows)")


if __name__ == "__main__":
    main()
