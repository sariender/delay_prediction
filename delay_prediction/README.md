# Delay Prediction Pipeline

This folder contains the offline pipeline used to predict public-transport arrival delays and generate the calibrated February lookup used by the robust router.

## Files

| File | Purpose |
|---|---|
| `weather_extraction.py` | Prepares weather features for training and February prediction. |
| `special_days.py` | Creates holiday and special-day features. |
| `train_calibration_test_gbt_model.py` | Trains and evaluates the point-delay model using the train/calibration/test design, then builds and validates residual-based quantiles. |
| `final_model_training.py` | Trains the final point-delay model on all labeled historical data available before February. |
| `from_timetable_to_lookup.py` | Generates February predictions and calibrated quantiles for the router lookup table. |
| `data_viz.py` | Produces figures for model interpretation, calibration validation, and routing explanation. |

## Pipeline

```text
Feature preparation
→ Train/calibration/test evaluation
→ Residual-based quantile calibration
→ Final model training
→ February lookup generation
→ Router uses P(delay ≤ X)
