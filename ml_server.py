
import os
import sys
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = BASE_DIR / "model_metadata.json"
LATEST_PRICES_PATH = BASE_DIR / "latest_prices.csv"
FULL_CSV_PATH = BASE_DIR / "model_history.csv"

app = Flask(__name__)
CORS(app)

# Global in-memory cache loaded once at startup
METADATA = {}
LOADED_MODELS = {}
LATEST_PRICES_DF = None
FULL_HISTORY_DF = None

SUPPORTED_CITY = "Ghaziabad"
SUPPORTED_CROPS = ["Green Chilli", "Potato", "Rice", "Tomato", "Wheat"]
VALID_HORIZONS = [1, 7, 14]


def load_artifacts_at_startup():
    """Loads metadata, serialized models, and price series into memory once."""
    global METADATA, LOADED_MODELS, LATEST_PRICES_DF, FULL_HISTORY_DF

    print("=" * 60)
    print("Loading Mandi ML Model Artifacts into Memory at Startup...")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    with open(METADATA_PATH, "r") as f:
        METADATA = json.load(f)

    # Preload all 15 models into memory
    loaded_count = 0
    for meta_key, meta_info in METADATA.items():
        if not meta_info.get("use_baseline", False):
            m_path = BASE_DIR / meta_info.get("model_file")
            if not m_path.exists():
                m_path = MODELS_DIR / os.path.basename(meta_info.get("model_file"))
            if m_path.exists():
                LOADED_MODELS[meta_key] = joblib.load(m_path)
                loaded_count += 1
            else:
                print(f"Warning: Model file missing for {meta_key}: {m_path}")
        else:
            # Baseline model: marked for fast baseline execution
            LOADED_MODELS[meta_key] = "BASELINE"

    print(f"Pre-loaded {loaded_count} active ML models into memory.")

    # Preload latest_prices.csv
    if LATEST_PRICES_PATH.exists():
        LATEST_PRICES_DF = pd.read_csv(LATEST_PRICES_PATH)
        LATEST_PRICES_DF["date"] = pd.to_datetime(LATEST_PRICES_DF["date"])
        print(f"Pre-loaded latest_prices.csv ({len(LATEST_PRICES_DF)} rows).")

    # Preload full CSV history for robust multi-year lag extraction
    if FULL_CSV_PATH.exists():
        f_df = pd.read_csv(FULL_CSV_PATH)
        date_col = [c for c in f_df.columns if "date" in c.lower()][0]
        crop_col = [c for c in f_df.columns if any(k in c.lower() for k in ["crop", "commodity"])][0]
        price_col = [c for c in f_df.columns if "modal" in c.lower() or "price" in c.lower()][0]
        f_df["dt"] = pd.to_datetime(
            f_df[date_col], format="mixed", dayfirst=False, errors="coerce"
        )
        f_df = f_df.rename(columns={crop_col: "crop", price_col: "modal_price"})
        f_df = f_df.dropna(subset=["dt", "modal_price"])
        FULL_HISTORY_DF = f_df[f_df["modal_price"] > 0].sort_values("dt")
        print(f"Pre-loaded historical dataset ({len(FULL_HISTORY_DF)} rows).")

    print("All ML artifacts successfully initialized in memory.")
    print("=" * 60)


def build_feature_vector(price_series: pd.Series, feature_names: list) -> pd.DataFrame:
    """
    Constructs the exact 25 technical, volatility, calendar, and return features
    used during training. Avoids lookahead leakage.
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

    # 4. Distance from 30-day high and low + range position
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


def get_last_known_price(canonical_crop: str) -> float:
    """Retrieves the most recent verified price for a crop from the local buffer."""
    if LATEST_PRICES_DF is not None:
        sub = LATEST_PRICES_DF[LATEST_PRICES_DF["crop"] == canonical_crop]
        if not sub.empty:
            return round(float(sub["modal_price"].iloc[-1]), 2)
    return 1000.0


def predict_crop_price(
    crop: str,
    days: int,
    current_price: float = None,
    city: str = "Ghaziabad",
    mandi: str = "Ghaziabad Mandi",
    current_price_date: str = None,
) -> dict:
    """
    Executes the trained ML model inference using in-memory models.
    """
    # 1. City Check
    if city.strip().lower() != SUPPORTED_CITY.lower():
        return {
            "success": False,
            "error_type": "CITY_NOT_SUPPORTED",
            "message": f"City '{city}' is not supported yet. Only Ghaziabad Mandi is currently supported."
        }

    # 2. Days Check
    if days not in VALID_HORIZONS:
        return {
            "success": False,
            "error_type": "INVALID_DAYS",
            "message": f"Invalid horizon '{days}' days. Supported horizons are 1, 7, or 14 days."
        }

    # 3. Crop Check
    crop_lookup = {c.lower(): c for c in SUPPORTED_CROPS}
    c_norm = crop.strip().lower()
    if c_norm not in crop_lookup:
        return {
            "success": False,
            "error_type": "CROP_NOT_SUPPORTED",
            "message": f"Crop '{crop}' is not supported. Supported crops are: {SUPPORTED_CROPS}."
        }
    canonical_crop = crop_lookup[c_norm]

    meta_key = f"{canonical_crop}_{days}d"
    if meta_key not in METADATA:
        return {
            "success": False,
            "error_type": "MODEL_METADATA_MISSING",
            "message": f"Model metadata missing for {meta_key}."
        }
    meta = METADATA[meta_key]
    model_market = meta.get("market") or "Ghaziabad"
    training_data_through = meta.get("last_training_date")

    # 4. Resolve Starting Price (Baseline)
    price_source = "government_api"
    price_as_of = current_price_date
    if current_price is None or current_price <= 0:
        current_price = get_last_known_price(canonical_crop)
        price_source = "fallback_last_known"
        if LATEST_PRICES_DF is not None:
            crop_prices = LATEST_PRICES_DF[LATEST_PRICES_DF["crop"] == canonical_crop]
            if not crop_prices.empty and not price_as_of:
                price_as_of = crop_prices["date"].max().strftime("%Y-%m-%d")
    else:
        current_price = round(float(current_price), 2)

    # 5. Build price series with current_price placed at today
    today_dt = pd.to_datetime(datetime.date.today())
    target_date = today_dt + pd.Timedelta(days=days)

    if FULL_HISTORY_DF is not None:
        c_raw = FULL_HISTORY_DF[FULL_HISTORY_DF["crop"] == canonical_crop].sort_values("dt")
        daily = c_raw.groupby("dt")["modal_price"].mean().sort_index()
    else:
        crop_sub = LATEST_PRICES_DF[LATEST_PRICES_DF["crop"] == canonical_crop].sort_values("date").set_index("date")
        daily = crop_sub["modal_price"]

    daily_series = daily.copy()
    daily_series.loc[today_dt] = current_price

    full_calendar = pd.date_range(daily_series.index.min(), today_dt, freq="D")
    p_series = daily_series.reindex(full_calendar).interpolate(method="time", limit=7).ffill()
    p_series.loc[today_dt] = current_price

    use_baseline = meta.get("use_baseline", False)
    confidence = meta.get("confidence", "Low")
    quantiles = meta.get("residual_quantiles", {})

    # 6. Prediction
    if use_baseline:
        # Conservative persistence / moving-average baseline
        strategy = meta.get("best_baseline_strategy", "stays_same")
        if strategy == "moving_average":
            window = max(days, 7)
            predicted_price = float(p_series.dropna().tail(window).mean())
        else:
            predicted_price = current_price

        p_q10 = quantiles.get("price_q10", -0.05 * current_price)
        p_q90 = quantiles.get("price_q90", 0.05 * current_price)
        low = max(0.0, predicted_price + p_q10)
        high = max(low, predicted_price + p_q90)
    else:
        model = LOADED_MODELS.get(meta_key)
        feature_names = meta.get("feature_list", [])

        # Build feature vector at today_dt
        all_feats = build_feature_vector(p_series, feature_names)
        X_today = all_feats.loc[[today_dt]]

        pred_log_ret = float(model.predict(X_today)[0])
        predicted_price = current_price * np.exp(pred_log_ret)

        log_q10 = quantiles.get("log_q10", -0.05)
        log_q90 = quantiles.get("log_q90", 0.05)
        low = max(0.0, current_price * np.exp(pred_log_ret + log_q10))
        high = max(low, current_price * np.exp(pred_log_ret + log_q90))

    predicted_price = round(predicted_price, 2)
    low = round(min(low, predicted_price), 2)
    high = round(max(high, predicted_price), 2)

    net_gain_loss_rs = round(predicted_price - current_price, 2)
    net_gain_loss_pct = round((net_gain_loss_rs / current_price) * 100, 2)

    # 7. Actionable Recommendation with One-Line Reason
    # Thresholds: ±1.5% for actionable action when confident
    if confidence == "Low":
        if use_baseline:
            recommendation = "Hold, uncertain"
            reason = f"Short-term price is anchored by market stability (MSP/procurement); expect minimal fluctuation over the next {days} days."
        else:
            recommendation = "Hold, uncertain"
            reason = f"High market volatility detected with low model confidence. Waiting or selling in small batches is advised."
    else:
        if net_gain_loss_pct >= 1.5:
            recommendation = f"Wait {days} days"
            reason = f"Price is projected to increase by Rs. {abs(net_gain_loss_rs):.2f} (+{net_gain_loss_pct:.2f}%) over the next {days} days due to positive momentum."
        elif net_gain_loss_pct <= -1.5:
            recommendation = "Sell now"
            reason = f"Price is projected to drop by Rs. {abs(net_gain_loss_rs):.2f} ({net_gain_loss_pct:.2f}%) over the next {days} days. Selling immediately prevents losses."
        else:
            recommendation = f"Wait {days} days" if net_gain_loss_rs >= 0 else "Sell now"
            reason = f"Expected price movement is marginal ({net_gain_loss_pct:+.2f}%) over {days} days."

    disclaimer = (
        "Disclaimer: Mandi price predictions are statistical estimates derived from historical market trends and machine learning models. "
        "Actual market prices may vary due to weather events, supply shocks, and policy changes. Not financial advice."
    )

    return {
        "success": True,
        "city": city,
        "mandi": mandi,
        "crop": canonical_crop,
        "days": days,
        "current_date": today_dt.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "unit": "Rs/quintal",
        "current_price": current_price,
        "price_source": price_source,
        "price_as_of": price_as_of,
        "model_market": model_market,
        "training_data_through": training_data_through,
        "predicted_price": predicted_price,
        "net_gain_loss_rs": net_gain_loss_rs,
        "net_gain_loss_pct": net_gain_loss_pct,
        "low_estimate": low,
        "high_estimate": high,
        "recommendation": recommendation,
        "reason": reason,
        "confidence": confidence,
        "model_used": "LightGBM ML Model" if not use_baseline else "Market Persistence Baseline",
        "disclaimer": disclaimer
    }


# --- REST API Endpoints ---

@app.route("/health", methods=["GET"])
def health_check():
    """Health status and model inventory."""
    return jsonify({
        "status": "healthy",
        "service": "MandiMantra ML Inference Service",
        "supported_city": SUPPORTED_CITY,
        "supported_crops": SUPPORTED_CROPS,
        "valid_horizons": VALID_HORIZONS,
        "models_in_memory": len(LOADED_MODELS)
    }), 200


@app.route("/crops", methods=["GET"])
def get_crops():
    """List supported crops and horizons."""
    return jsonify({
        "city": SUPPORTED_CITY,
        "crops": SUPPORTED_CROPS,
        "horizons": VALID_HORIZONS,
        "units": "Rs/quintal"
    }), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Main prediction endpoint.
    Accepts JSON:
    {
        "city": "Ghaziabad",
        "mandi": "Ghaziabad Mandi",
        "crop": "Wheat",
        "days": 7,
        "current_price": 2600.0 (optional)
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error_type": "INVALID_JSON",
            "message": "Request body must be a valid JSON object."
        }), 400

    city = data.get("city", "Ghaziabad")
    mandi = data.get("mandi", "Ghaziabad Mandi")
    crop = data.get("crop")
    days = data.get("days")
    current_price = data.get("current_price")
    current_price_date = data.get("current_price_date")

    if not crop:
        return jsonify({
            "success": False,
            "error_type": "MISSING_CROP",
            "message": "Field 'crop' is required."
        }), 400

    if days is None:
        return jsonify({
            "success": False,
            "error_type": "MISSING_DAYS",
            "message": "Field 'days' is required (must be 1, 7, or 14)."
        }), 400

    try:
        days = int(days)
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error_type": "INVALID_DAYS",
            "message": f"'days' must be an integer (1, 7, or 14). Received: {days}"
        }), 400

    if current_price is not None:
        try:
            current_price = float(current_price)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error_type": "INVALID_PRICE",
                "message": f"'current_price' must be a valid numeric price. Received: {current_price}"
            }), 400

    result = predict_crop_price(
        crop=crop,
        days=days,
        current_price=current_price,
        city=city,
        mandi=mandi,
        current_price_date=current_price_date,
    )

    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


# Initialize models into memory at startup
load_artifacts_at_startup()

if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 5001))
    print(f"Starting ML Server on http://localhost:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
