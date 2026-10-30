# Mandi Crop Price Forecasting System (Ghaziabad Mandi)

An end-to-end Machine Learning price-forecasting solution developed specifically for Indian mandi market dynamics. This system forecasts the modal market price for farmers across **1-day, 7-day, and 14-day** horizons, empowering growers to decide whether to sell immediately or wait for higher expected returns.

---

## 1. Project Files & Directory Structure

```text
MandiModel/
├── ghaziabad_5crops_2018_2026.csv   # Source dataset (Ghaziabad mandi, 5 crops, 2018-2026)
├── train.py                         # Complete data prep, validation, baseline benchmark & training pipeline
├── predict.py                       # Standalone inference module exposing predict(crop, days)
├── accuracy_report.csv              # Full model vs. baseline benchmark report with direction accuracy & CV
├── model_metadata.json              # Trained metadata, feature lists, residual quantiles, and baseline flags
├── latest_prices.csv                # Most recent 400 days of cleaned prices per crop (for lag calculations)
├── README_ML.md                     # Documentation, usage guide, and plain-English evaluation
└── models/                          # Serialized trained model files (1 per crop & horizon)
    ├── green_chilli_1d.joblib
    ├── green_chilli_7d.joblib
    ├── green_chilli_14d.joblib
    ├── potato_1d.joblib
    ├── potato_7d.joblib
    ├── potato_14d.joblib
    ├── rice_1d.joblib
    ├── rice_7d.joblib
    ├── rice_14d.joblib
    ├── tomato_1d.joblib
    ├── tomato_7d.joblib
    ├── tomato_14d.joblib
    ├── wheat_1d.joblib
    ├── wheat_7d.joblib
    └── wheat_14d.joblib
```

---

## 2. How to Call `predict()`

The `predict(crop: str, days: int) -> dict` function runs **offline in sub-second time without retraining**. It automatically calculates required rolling features and returns actionable farmer guidance.

### Quick Example (Forecasting 7 Days from Today)
```python
from predict import predict

# Predict price 7 days from today's calendar date
result = predict(crop="Tomato", days=7, as_of_date="today")
print(result)

# Or pass today's current market price if you know it
result = predict(crop="Wheat", days=7, as_of_date="today", current_price=2600.0)
print(result)
```

### CLI Terminal Commands for Today's Forecast:
```bash
# Check all 5 crops 7 days from today:
python predict.py --today --days 7

# Check a specific crop 7 days from today:
python predict.py --crop Tomato --days 7 --today

# Forecast with your own starting mandi price:
python predict.py --crop Wheat --days 7 --today --price 2600
```

### Sample Output Dictionary
```python
{
    "crop": "Tomato",
    "current_price": 1131.00,
    "current_date": "2026-04-10",
    "days": 7,
    "target_date": "2026-04-17",
    "predicted_price": 1143.35,
    "low": 1029.60,
    "high": 1360.11,
    "change_amount": 12.35,
    "change_percent": 1.09,
    "advice": "Hold, uncertain",
    "confidence": "Low"
}
```

### Return Keys Explained
- `crop`: Standardized commodity name.
- `current_price`: Most recent observed mandi modal price (₹ / quintal).
- `current_date`: Date of the latest available observation.
- `days`: Forecast horizon (`1`, `7`, or `14`).
- `target_date`: Projected future date (`current_date + days`).
- `predicted_price`: Expected modal price at `target_date`.
- `low`: 10th percentile expected bound (derived from validation residuals).
- `high`: 90th percentile expected bound (derived from validation residuals).
- `change_amount`: Expected monetary gain/loss per quintal (`predicted_price - current_price`).
- `change_percent`: Expected percentage gain/loss.
- `advice`: Actionable farmer recommendation:
  - `"Wait"`: Projected price rise $\ge +1.5\%$ with confident signal (farmer gains by waiting).
  - `"Sell now"`: Projected price drop $\le -1.5\%$ with confident signal (sell today to prevent losses).
  - `"Hold, uncertain"`: Small price movement or high market uncertainty / baseline default.
- `confidence`: Confidence rating based on rigorous out-of-sample testing:
  - `"High"`: Model beats baseline by $\ge 20\%$ with $\ge 65\%$ direction accuracy.
  - `"Medium"`: Model beats baseline by $5\% - 20\%$.
  - `"Low"`: Model improvement under $5\%$ or defaults to safety baseline.

### Input Validation & Safety
- **Invalid Horizon**: Passing any integer other than `1`, `7`, or `14` raises a `ValueError`.
- **Unknown Crop**: Passing an invalid or unsupported crop name raises a clear `ValueError` listing all available crops.
- **Case-Insensitive**: Supports `"wheat"`, `"Wheat"`, `"WHEAT"`, `"tomato"`, etc.

---

## 3. How to Retrain When New Data Arrives

When new mandi market quotes are appended or updated:

1. Replace or append the new records to `ghaziabad_5crops_2018_2026.csv` (or your updated CSV file).
2. Run the training script:
   ```bash
   python train.py
   ```
3. What happens automatically:
   - Detects the date, commodity, and modal price columns automatically.
   - Cleans the series, eliminates spurious quotes outside $0.4\times - 2.5\times$ the 15-day rolling median, and fills short gaps ($\le 7$ days).
   - Re-runs strict time-ordered out-of-sample backtesting (last 12 months) and walk-forward cross-validation.
   - Benchmarks against Baseline A (persistence) and Baseline B (moving average).
   - Generates updated model files in `models/`, updates `model_metadata.json`, regenerates `latest_prices.csv` (latest 400 days), and outputs an updated `accuracy_report.csv`.

---

## 4. Accuracy Table & Plain-English Findings

The model was evaluated using a strict **embargoed time-based split** (trained on historical data up to 12 months prior, with an embargo gap equal to the horizon $h$ to ensure zero lookahead leakage, tested exclusively on the subsequent 12 months).

### Benchmark Summary Table

| Crop | Horizon | Model MAPE | Best Baseline MAPE | Best Baseline Name | Improvement % | Direction Accuracy | Decision | Confidence |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Green Chilli** | 1d | 3.07% | 3.05% | Baseline A (Stays same) | -0.5% | 61.4% | Use Baseline | Low |
| **Green Chilli** | 7d | 10.09% | 10.45% | Baseline A (Stays same) | **+3.4%** | 42.9% | **ML Model** | Low |
| **Green Chilli** | 14d | 16.70% | 18.53% | Baseline A (Stays same) | **+9.9%** | **75.8%** | **ML Model** | **Medium** |
| **Potato** | 1d | 1.85% | 1.81% | Baseline A (Stays same) | -2.3% | 59.6% | Use Baseline | Low |
| **Potato** | 7d | 4.33% | 3.84% | Baseline A (Stays same) | -12.8% | 50.5% | Use Baseline | Low |
| **Potato** | 14d | 8.15% | 7.03% | Baseline A (Stays same) | -15.9% | 55.8% | Use Baseline | Low |
| **Rice** | 1d | 2.92% | 2.92% | Baseline A (Stays same) | -0.1% | 43.9% | Use Baseline | Low |
| **Rice** | 7d | 7.02% | 6.40% | Baseline B (7d Moving Avg) | -9.6% | 62.2% | Use Baseline | Low |
| **Rice** | 14d | 8.37% | 8.38% | Baseline A (Stays same) | **+0.1%** | 53.5% | **ML Model** | Low |
| **Tomato** | 1d | 3.12% | 3.21% | Baseline A (Stays same) | **+2.7%** | 62.2% | **ML Model** | Low |
| **Tomato** | 7d | 9.47% | 9.76% | Baseline A (Stays same) | **+3.0%** | **68.3%** | **ML Model** | Low |
| **Tomato** | 14d | 20.47% | 15.13% | Baseline A (Stays same) | -35.3% | 60.5% | Use Baseline | Low |
| **Wheat** | 1d | 0.56% | 0.56% | Baseline A (Stays same) | -0.5% | 48.1% | Use Baseline | Low |
| **Wheat** | 7d | 1.41% | 1.15% | Baseline A (Stays same) | -22.9% | 49.0% | Use Baseline | Low |
| **Wheat** | 14d | 1.95% | 2.26% | Baseline A (Stays same) | **+13.8%** | **72.6%** | **ML Model** | **Medium** |

---

### Plain-Words Interpretation of Results

1. **Why do baselines win on short horizons for staple grains (Wheat and Rice 1d & 7d)?**
   - In India, wheat and paddy rice are subject to government Minimum Support Prices (MSP) and steady procurement. Day-to-day volatility in Ghaziabad mandi is exceptionally low (daily MAPE $\approx 0.5\% - 1.2\%$).
   - On a 1-day or 7-day basis, simply assuming *"the price tomorrow will be identical to today"* yields nearly unbeatable accuracy ($0.56\%$ error for Wheat). Attempting to fit complex models on micro-fluctuations introduces noise. The system honestly identifies this and sets `use_baseline = True`.
2. **Where Machine Learning delivers high value (14-day horizons & volatile perishables):**
   - Over a 14-day horizon, multi-day seasonal and cyclical momentum kicks in.
   - For **Wheat (14d)**, the ML model beats the baseline by **+13.8%**, achieving an outstanding **72.6% direction accuracy** (and $80.8\%$ in walk-forward folds).
   - For **Green Chilli (14d)**, the model achieves **75.8% direction accuracy** and beats the baseline by **+9.9%** ($16.7\%$ MAPE vs. $18.5\%$ for baseline).
   - For **Tomato (1d & 7d)**, the model successfully captures supply elasticity rebounds, achieving **68.3% direction accuracy** at 7 days.
3. **Potato Notice:**
   - Potato prices in cold-storage clusters undergo prolonged stable holding cycles. The baseline outperforms gradient boosted models during flat storage months. `predict.py` safely falls back to the baseline strategy so farmers are never misled by false signals.
