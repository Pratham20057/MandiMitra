
import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def _load_metadata_and_prices(
    meta_path: str = "model_metadata.json",
    prices_path: str = "latest_prices.csv",
    full_csv_path: str = "ghaziabad_5crops_2018_2026.csv",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Loads model metadata, the latest prices buffer (400 days), and the full historical dataset.
    """
    if not os.path.exists(meta_path):
        meta_candidates = list(Path(".").glob("**/model_metadata.json"))
        if meta_candidates:
            meta_path = str(meta_candidates[0])
        else:
            raise FileNotFoundError(
                f"Metadata file '{meta_path}' not found. Please run train.py first."
            )

    if not os.path.exists(prices_path):
        price_candidates = list(Path(".").glob("**/latest_prices.csv"))
        if price_candidates:
            prices_path = str(price_candidates[0])
        else:
            raise FileNotFoundError(
                f"Latest prices file '{prices_path}' not found. Please run train.py first."
            )

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    latest_df = pd.read_csv(prices_path)
    latest_df["date"] = pd.to_datetime(latest_df["date"])

    full_df = None
    if os.path.exists(full_csv_path):
        full_df = pd.read_csv(full_csv_path)
        date_col = [c for c in full_df.columns if "date" in c.lower()][0]
        crop_col = [c for c in full_df.columns if any(k in c.lower() for k in ["crop", "commodity"])][0]
        price_col = [c for c in full_df.columns if "modal" in c.lower() or "price" in c.lower()][0]
        full_df["dt"] = pd.to_datetime(full_df[date_col], dayfirst=True)
        full_df = full_df.rename(columns={crop_col: "crop", price_col: "modal_price"})
        full_df = full_df.dropna(subset=["dt", "modal_price"])
        full_df = full_df[full_df["modal_price"] > 0].sort_values("dt")

    return metadata, latest_df, full_df


def _build_inference_features(price_series: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    """
    Builds feature vector at target date using historical price series.
    """
    idx = price_series.index
    feats = pd.DataFrame(index=idx)
    p = price_series

    # 1. Lagged returns: np.log(P_t / P_{t-k})
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 28, 56]:
        feats[f"ret_lag_{lag}"] = np.log(p / p.shift(lag))

    # 2. Distance from rolling means (7, 14, 30, 60, 90 days)
    for w in [7, 14, 30, 60, 90]:
        rm = p.rolling(w, min_periods=max(3, w // 4)).mean()
        feats[f"dist_mean_{w}"] = (p - rm) / rm

    # 3. Rolling volatility
    log_ret_1 = np.log(p / p.shift(1))
    feats["vol_14"] = log_ret_1.rolling(14, min_periods=5).std()
    feats["vol_30"] = log_ret_1.rolling(30, min_periods=7).std()

    # 4. Distance from 30-day high and low
    r_max_30 = p.rolling(30, min_periods=5).max()
    r_min_30 = p.rolling(30, min_periods=5).min()
    feats["dist_high_30"] = (p - r_max_30) / r_max_30
    feats["dist_low_30"] = (p - r_min_30) / r_min_30
    feats["range_pos_30"] = (p - r_min_30) / (r_max_30 - r_min_30 + 1e-6)

    # 5. Day-of-year sine and cosine
    doy = idx.dayofyear
    feats["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feats["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # 6. Month & Weekday
    feats["month"] = idx.month
    feats["weekday"] = idx.weekday

    # 7. Price change vs one year ago
    feats["ret_1y"] = np.log(p / p.shift(365))

    return feats[feature_names]


def predict(
    crop: str,
    days: int,
    as_of_date: str = None,
    current_price: float = None
) -> dict:
    """
    Predicts the modal price of a specified crop after 1, 7, or 14 days.

    Parameters
    ----------
    crop : str
        Name of the commodity (e.g., 'Wheat', 'Rice', 'Potato', 'Tomato', 'Green Chilli').
        Case-insensitive.
    days : int
        Forecast horizon in days. Must be one of [1, 7, 14].
    as_of_date : str, optional
        Reference date in 'YYYY-MM-DD' or 'today'.
        If None, defaults to the latest date available in latest_prices.csv.
    current_price : float, optional
        Custom starting price for as_of_date. If None, uses the recorded/latest price.

    Returns
    -------
    dict
        Forecast details containing:
        {
            crop: str,
            current_price: float,
            current_date: str,
            days: int,
            target_date: str,
            predicted_price: float,
            low: float,
            high: float,
            change_amount: float,
            change_percent: float,
            advice: str ("Sell now" / "Wait" / "Hold, uncertain"),
            confidence: str ("High" / "Medium" / "Low")
        }
    """
    # 1. Validate days
    valid_horizons = [1, 7, 14]
    if days not in valid_horizons:
        raise ValueError(
            f"Invalid horizon 'days={days}'. Must be one of {valid_horizons}."
        )

    # 2. Load metadata and prices
    metadata, latest_df, full_df = _load_metadata_and_prices()

    # 3. Validate crop (case-insensitive matching)
    available_crops = sorted(latest_df["crop"].unique().tolist())
    crop_lookup = {c.lower(): c for c in available_crops}
    crop_normalized = crop.strip().lower()

    if crop_normalized not in crop_lookup:
        raise ValueError(
            f"Unknown crop '{crop}'. Available crops are: {available_crops}."
        )

    canonical_crop = crop_lookup[crop_normalized]
    meta_key = f"{canonical_crop}_{days}d"

    if meta_key not in metadata:
        raise ValueError(
            f"No trained model metadata found for key '{meta_key}'."
        )

    meta = metadata[meta_key]

    # Handle 'today' keyword
    is_today_query = False
    if as_of_date is not None and as_of_date.strip().lower() == "today":
        as_of_date = datetime.date.today().strftime("%Y-%m-%d")
        is_today_query = True
    elif as_of_date is not None and pd.to_datetime(as_of_date).date() == datetime.date.today():
        is_today_query = True

    actual_future_price = None

    # 4. Extract price series
    if is_today_query:
        # Today's date forecasting
        today_dt = pd.to_datetime(datetime.date.today())

        # Pull historical base from full_df or latest_df
        if full_df is not None:
            c_raw = full_df[full_df["crop"] == canonical_crop].sort_values("dt")
            daily = c_raw.groupby("dt")["modal_price"].mean().sort_index()
        else:
            crop_df = latest_df[latest_df["crop"] == canonical_crop].sort_values("date").set_index("date")
            daily = crop_df["modal_price"]

        last_recorded_price = float(daily.iloc[-1])
        base_price = float(current_price) if current_price is not None else last_recorded_price

        # --- BUG FIX ---
        # The dataset may end months before today (e.g. April 2026, today Oct 2026).
        # interpolate(limit=7) only fills up to 7-day gaps, leaving the entire
        # April-to-October period as NaN. That makes ALL lag features NaN,
        # so the model ignores the price input and always returns the same answer.
        #
        # Fix: build a complete daily calendar up to today, first interpolating
        # short gaps (<=7 days) within the historical data, then FORWARD-FILLING
        # the remaining gap up to today using the last known price as a flat
        # baseline. Finally, override today's row with the user-supplied price.
        # This gives the model realistic non-NaN lag features to work with.

        full_calendar = pd.date_range(daily.index.min(), today_dt, freq="D")
        # Step 1: reindex and fill short historical gaps (<=7 days)
        price_series = daily.reindex(full_calendar).interpolate(method="time", limit=7)
        # Step 2: forward-fill any remaining NaN (long gaps or months without data)
        #         using the last known recorded price as a flat carry-forward
        price_series = price_series.ffill()
        # Step 3: override today's date with the actual/user-supplied price
        price_series.loc[today_dt] = base_price

        current_date = today_dt
        current_price = base_price

    elif as_of_date is not None and full_df is not None:
        # Custom historical date
        c_raw = full_df[full_df["crop"] == canonical_crop].sort_values("dt")
        daily = c_raw.groupby("dt")["modal_price"].mean().sort_index()
        med = daily.rolling(15, min_periods=3, center=True).median().bfill().ffill()
        daily[(daily > 2.5 * med) | (daily < 0.4 * med)] = np.nan
        full_calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        full_series = daily.reindex(full_calendar).interpolate(method="time", limit=7)

        valid_dates = full_series.dropna().index
        if len(valid_dates) == 0:
            raise ValueError(f"No valid price history for {canonical_crop}.")

        parsed_target_asof = pd.to_datetime(as_of_date)
        valid_before = valid_dates[valid_dates <= parsed_target_asof]
        if len(valid_before) > 0:
            target_asof = valid_before[-1]
        else:
            target_asof = valid_dates[0]

        price_series = full_series.loc[:target_asof]
        current_date = target_asof
        current_price = float(current_price) if current_price is not None else float(price_series.loc[current_date])

        target_future_dt = current_date + pd.Timedelta(days=days)
        if target_future_dt in full_series.index and pd.notna(full_series.loc[target_future_dt]):
            actual_future_price = round(float(full_series.loc[target_future_dt]), 2)
    else:
        # Default: latest available date in latest_prices.csv
        crop_df = latest_df[latest_df["crop"] == canonical_crop].sort_values("date").set_index("date")
        full_idx = pd.date_range(crop_df.index.min(), crop_df.index.max(), freq="D")
        price_series = crop_df["modal_price"].reindex(full_idx).interpolate(method="time", limit=7)

        current_date = price_series.dropna().index[-1]
        current_price = float(current_price) if current_price is not None else float(price_series.loc[current_date])

        target_future_dt = current_date + pd.Timedelta(days=days)
        if target_future_dt in price_series.index and pd.notna(price_series.loc[target_future_dt]):
            actual_future_price = round(float(price_series.loc[target_future_dt]), 2)

    target_date = current_date + pd.Timedelta(days=days)
    use_baseline = meta.get("use_baseline", False)
    confidence = meta.get("confidence", "Low")
    residual_quantiles = meta.get("residual_quantiles", {})

    # 5. Make prediction
    if use_baseline:
        strategy = meta.get("best_baseline_strategy", "stays_same")
        if strategy == "moving_average" and not is_today_query:
            window = max(days, 7)
            recent_mean = price_series.dropna().tail(window).mean()
            predicted_price = float(recent_mean)
        else:
            predicted_price = current_price

        p_q10 = residual_quantiles.get("price_q10", -0.05 * current_price)
        p_q90 = residual_quantiles.get("price_q90", 0.05 * current_price)
        low = max(0.0, predicted_price + p_q10)
        high = max(low, predicted_price + p_q90)

    else:
        model_file = meta.get("model_file")
        if not model_file or not os.path.exists(model_file):
            alt_path = os.path.join("models", os.path.basename(model_file))
            if os.path.exists(alt_path):
                model_file = alt_path
            else:
                raise FileNotFoundError(f"Model file '{model_file}' could not be located.")

        model = joblib.load(model_file)
        feature_names = meta.get("feature_list", [])

        # Build feature vector at current_date
        all_features = _build_inference_features(price_series, feature_names)
        X_latest = all_features.loc[[current_date]]

        pred_log_change = float(model.predict(X_latest)[0])
        predicted_price = current_price * np.exp(pred_log_change)

        log_q10 = residual_quantiles.get("log_q10", -0.05)
        log_q90 = residual_quantiles.get("log_q90", 0.05)
        low = max(0.0, current_price * np.exp(pred_log_change + log_q10))
        high = max(low, current_price * np.exp(pred_log_change + log_q90))

    predicted_price = round(predicted_price, 2)
    low = round(min(low, predicted_price), 2)
    high = round(max(high, predicted_price), 2)
    current_price = round(current_price, 2)

    change_amount = round(predicted_price - current_price, 2)
    change_percent = round((predicted_price - current_price) / current_price * 100, 2)

    # 6. Actionable advice
    if confidence == "Low":
        advice = "Hold, uncertain"
    elif confidence in ["Medium", "High"]:
        if change_percent >= 1.5:
            advice = "Wait"
        elif change_percent <= -1.5:
            advice = "Sell now"
        else:
            advice = "Hold, uncertain"
    else:
        advice = "Hold, uncertain"

    output = {
        "crop": canonical_crop,
        "current_price": current_price,
        "current_date": current_date.strftime("%Y-%m-%d"),
        "days": days,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "predicted_price": predicted_price,
        "low": low,
        "high": high,
        "change_amount": change_amount,
        "change_percent": change_percent,
        "advice": advice,
        "confidence": confidence,
    }

    if actual_future_price is not None:
        output["actual_future_price"] = actual_future_price
        act_chg = round((actual_future_price - current_price) / current_price * 100, 2)
        output["actual_change_percent"] = act_chg
        pred_dir = np.sign(predicted_price - current_price)
        act_dir = np.sign(actual_future_price - current_price)
        output["direction_correct"] = bool(pred_dir == act_dir)

    return output


def print_forecast_card(res: dict):
    """Prints an ASCII forecast card."""
    sign = "+" if res["change_amount"] >= 0 else ""
    print("\n" + "=" * 65)
    print(f"  MANDI PRICE FORECAST: {res['crop'].upper()} ({res['days']}-DAY HORIZON)")
    print("=" * 65)
    print(f"  As of Date:      {res['current_date']}")
    print(f"  Current Price:   Rs. {res['current_price']:.2f} / quintal")
    print(f"  Target Date:     {res['target_date']}")
    print(f"  Forecast Price:  Rs. {res['predicted_price']:.2f} / quintal")
    print(f"  Expected Change: {sign}Rs. {res['change_amount']:.2f} ({sign}{res['change_percent']:.2f}%)")
    print(f"  Expected Range:  [Rs. {res['low']:.2f} - Rs. {res['high']:.2f}] (80% interval)")
    print(f"  Confidence:      {res['confidence']}")
    print(f"  Action Advice:   >>> {res['advice'].upper()} <<<")

    if "actual_future_price" in res:
        act_sign = "+" if res["actual_change_percent"] >= 0 else ""
        correct_str = "YES (Matched Market)" if res.get("direction_correct") else "NO"
        print("-" * 65)
        print(f"  [HISTORICAL BACKTEST CHECK]")
        print(f"  Actual Price:    Rs. {res['actual_future_price']:.2f} ({act_sign}{res['actual_change_percent']:.2f}%)")
        print(f"  Direction Match: {correct_str}")
    print("=" * 65 + "\n")


def interactive_mode():
    """Interactive CLI prompting for crop, horizon, date, and optional price."""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    print("\n" + "=" * 65)
    print("  INTERACTIVE MANDI PRICE FORECASTER")
    print(f"  (System Local Date: {today_str})")
    print("=" * 65)
    crops = ["Green Chilli", "Potato", "Rice", "Tomato", "Wheat"]
    print("Available Crops:")
    for i, c in enumerate(crops, 1):
        print(f"  [{i}] {c}")

    while True:
        choice = input("\nSelect crop number (1-5) or name: ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            crop = crops[int(choice) - 1]
            break
        elif choice.lower() in [c.lower() for c in crops]:
            crop = choice
            break
        print("Invalid crop selection. Please choose 1 to 5 or type the crop name.")

    while True:
        h_choice = input("Select horizon in days (1, 7, or 14): ").strip()
        if h_choice in ["1", "7", "14"]:
            days = int(h_choice)
            break
        print("Invalid horizon. Must be 1, 7, or 14.")

    date_choice = input(
        f"Enter date (YYYY-MM-DD), type 'today' for today ({today_str}), or press ENTER for latest data: "
    ).strip()
    as_of_date = date_choice if date_choice else None

    price_input = input("Enter today's current mandi price in Rs. (or press ENTER to use default): ").strip()
    custom_price = float(price_input) if price_input else None

    try:
        res = predict(crop, days, as_of_date=as_of_date, current_price=custom_price)
        print_forecast_card(res)
    except Exception as e:
        print(f"\nError calculating prediction: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mandi Crop Price Forecasting System CLI"
    )
    parser.add_argument(
        "--crop", type=str, default=None,
        help="Crop name (e.g., Tomato, Wheat, Potato, Rice, Green Chilli)"
    )
    parser.add_argument(
        "--days", type=int, default=None, choices=[1, 7, 14],
        help="Horizon in days: 1, 7, or 14"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Reference date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--today", action="store_true",
        help="Forecast starting from TODAY's current calendar date"
    )
    parser.add_argument(
        "--price", type=float, default=None,
        help="Optional current market price to forecast from"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Launch interactive terminal prompt"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    as_of = "today" if args.today else args.date

    # Specific crop & horizon provided via CLI
    if args.crop is not None and args.days is not None:
        res = predict(crop=args.crop, days=args.days, as_of_date=as_of, current_price=args.price)
        print_forecast_card(res)
        return

    # If --today is passed with --days
    if args.today and args.days is not None:
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        print("=" * 80)
        print(f"TODAY'S FORECASTS: As of {today_str} -> Horizon: {args.days} Days")
        print("=" * 80)
        crops = ["Green Chilli", "Potato", "Rice", "Tomato", "Wheat"]
        for c in crops:
            res = predict(c, args.days, as_of_date="today", current_price=args.price)
            sign = "+" if res["change_amount"] >= 0 else ""
            print(
                f"  {c:12s} | Today: Rs.{res['current_price']:.2f} -> Target ({res['target_date']}): Rs.{res['predicted_price']:.2f} "
                f"({sign}{res['change_amount']:.2f}, {sign}{res['change_percent']:.2f}%) | "
                f"Range: [Rs.{res['low']:.2f} - Rs.{res['high']:.2f}] | Advice: {res['advice']:15s} | Conf: {res['confidence']}"
            )
        print("=" * 80)
        return

    # Default: Run full demonstration across all crops
    print("=" * 80)
    print("DEMONSTRATION: Mandi Price Forecasts for Farmers (Latest Available Date)")
    print("=" * 80)
    print("Tip: Run `python predict.py --today --days 7` to forecast 7 days from today!\n")

    crops = ["Green Chilli", "Potato", "Rice", "Tomato", "Wheat"]
    for c in crops:
        print(f"--- Crop: {c} ---")
        for d in [1, 7, 14]:
            res = predict(c, d)
            sign = "+" if res["change_amount"] >= 0 else ""
            print(
                f"  [{d:2d} Days | Target: {res['target_date']}] "
                f"Current: Rs.{res['current_price']:.2f} -> Forecast: Rs.{res['predicted_price']:.2f} "
                f"({sign}{res['change_amount']:.2f}, {sign}{res['change_percent']:.2f}%) | "
                f"Range: [Rs.{res['low']:.2f} - Rs.{res['high']:.2f}] | "
                f"Advice: {res['advice']:15s} | Confidence: {res['confidence']}"
            )
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
