#!/usr/bin/env python3
"""
Train Nutrition Model (Per-100g, Robust & Lightweight)

- Extracts grams from text safely (decimals, kg/g/mg/oz/lb)
- Normalizes labels to per-100g
- Filters implausible outliers
- Uses RobustScaler on outputs (less outlier-sensitive)
- Uses Ridge (tiny, fast) + compact TF-IDF
"""

import re
import json
import math
import os
import joblib
import numpy as np
import pandas as pd

from typing import Tuple, Optional, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

# ---------------------------
# Quantity parsing utilities
# ---------------------------

_WEIGHT_UNITS = {
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "mg": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
    "oz": 28.349523125,
    "ounce": 28.349523125,
    "ounces": 28.349523125,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
}

# volume units are tricky without a density table; we ignore them on purpose
_VOLUME_UNITS = {"cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons"}

# Match: number (int/decimal) + optional space + unit
# e.g., "200g", "200 g", "0.5 kg", "1 oz", "2.0 pounds"
_QTY_REGEX = re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)')

def extract_quantity_grams(text: str) -> Tuple[int, str]:
    """
    Extract a plausible weight (in grams) from text. If none found, default to 100g.
    - Supports decimals and weight units (kg/g/mg/oz/lb).
    - Ignores volume units (cup/tbsp/tsp...) to avoid wrong conversions.
    - Clamps result to [10g, 2000g] for stability; otherwise fallback to 100g.
    Returns: (grams:int, clean_text:str-without-the-matched-quantity)
    """
    s = text.lower().strip()
    grams_candidates: List[float] = []
    spans_to_remove: List[Tuple[int, int]] = []

    for m in _QTY_REGEX.finditer(s):
        unit = m.group("unit")
        if unit not in _WEIGHT_UNITS and unit not in _VOLUME_UNITS:
            continue
        if unit in _VOLUME_UNITS:
            # ignore volumes; too ambiguous without density
            continue

        try:
            val = float(m.group("num"))
        except Exception:
            continue

        grams = val * _WEIGHT_UNITS[unit]
        grams_candidates.append(grams)
        spans_to_remove.append(m.span())

    if grams_candidates:
        # choose the largest plausible weight mention (often the main serving)
        grams_float = max(grams_candidates)
        # clamp to safe range; else fallback
        if grams_float < 10 or grams_float > 2000 or math.isnan(grams_float) or math.isinf(grams_float):
            grams = 100
            spans_to_remove = []  # don't remove anything in this case
        else:
            grams = int(round(grams_float))
    else:
        grams = 100  # default

    # remove matched qty tokens from text (longest span first to keep indices valid)
    if spans_to_remove:
        spans_to_remove.sort(key=lambda x: x[1]-x[0], reverse=True)
        s_list = list(s)
        for start, end in spans_to_remove:
            for i in range(start, end):
                s_list[i] = ' '
        s = ''.join(s_list)

    # collapse whitespace and strip
    clean_text = re.sub(r'\s+', ' ', s).strip()
    return grams, clean_text

# ---------------------------
# Data loading & cleaning
# ---------------------------

def load_training_data(path: str = "training_data.json"):
    if not os.path.exists(path):
        print(f"Error: {path} not found!")
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        training_examples = json.load(f)

    print(f"Loaded {len(training_examples)} training examples")
    texts_clean: List[str] = []
    labels_per100g: List[List[float]] = []

    dropped_qty = 0
    dropped_outlier = 0

    # plausible per-100g ranges (hard caps)
    MAXS = {
        "calories": 1200.0,  # kcal/100g (very fatty foods ~900)
        "protein":  120.0,   # g/100g (whey isolate ~90)
        "fat":      120.0,   # g/100g (pure fats ~100)
        "carbs":    180.0,   # g/100g (starches/sugars)
    }

    for ex in training_examples:
        raw_text = ex.get("input_text", "")
        nutr = ex.get("nutrition", {})
        if not raw_text or not nutr:
            continue

        grams, clean_text = extract_quantity_grams(raw_text)

        # safety: if grams is somehow invalid, fallback to 100
        if grams <= 0 or grams > 20000:
            grams = 100
            dropped_qty += 1  # track anomaly

        # Convert to per-100g from the example's given-grams nutrition
        # scale = 100 / grams
        try:
            scale = 100.0 / float(grams)
        except ZeroDivisionError:
            scale = 1.0
            dropped_qty += 1

        cals = float(nutr.get("calories", 0.0)) * scale
        prot = float(nutr.get("protein", 0.0)) * scale
        fat  = float(nutr.get("fat", 0.0)) * scale
        carb = float(nutr.get("carbs", 0.0)) * scale

        # filter out negative / NaN / inf and extreme outliers beyond hard caps
        bad = (
            any(map(lambda v: v is None or math.isnan(v) or math.isinf(v), [cals, prot, fat, carb])) or
            cals < 0 or prot < 0 or fat < 0 or carb < 0 or
            cals > MAXS["calories"] or prot > MAXS["protein"] or
            fat  > MAXS["fat"] or carb > MAXS["carbs"]
        )
        if bad:
            dropped_outlier += 1
            continue

        texts_clean.append(clean_text)
        labels_per100g.append([cals, prot, fat, carb])

    labels_df = pd.DataFrame(labels_per100g, columns=["calories", "protein", "fat", "carbs"])

    kept = len(labels_df)
    print(f"Cleaned dataset: kept={kept}, dropped_qty={dropped_qty}, dropped_outlier={dropped_outlier}")
    if kept == 0:
        print("All examples were dropped after cleaning. Check your training_data.json format.")
        return None, None

    # quick stats to sanity-check
    desc = labels_df.describe().T[["min", "50%", "max", "mean"]]
    print("\nPer-100g label stats after cleaning:")
    print(desc.to_string(float_format=lambda x: f"{x:,.2f}"))

    return texts_clean, labels_df

# ---------------------------
# Training
# ---------------------------

def train_model(texts: List[str], labels_df: pd.DataFrame):
    print("\n" + "="*50)
    print("TRAINING PER-100G MODEL (RIDGE + RobustScaler)")
    print("="*50)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels_df, test_size=0.2, random_state=42, shuffle=True
    )

    # Compact TF-IDF; ignore numeric tokens to avoid quantity leakage
    print("\nCreating text vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=1200,
        ngram_range=(1, 1),
        lowercase=True,
        stop_words="english",
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',  # letters only
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"Vectorized features: {X_train_vec.shape[1]} dimensions")

    # Robust scaling on outputs (uses median/IQR; less outlier-sensitive than StandardScaler)
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled  = y_scaler.transform(y_test)

    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(X_train_vec, y_train_scaled)
    print("Model training completed!")

    # Evaluation (inverse scale + clip to plausible ranges)
    y_pred_scaled = model.predict(X_test_vec)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # clip predictions to plausible per-100g ranges
    CLIPS = np.array([[0, 0, 0, 0],
                      [1200, 120, 120, 180]], dtype=float)
    y_pred = np.minimum(np.maximum(y_pred, CLIPS[0]), CLIPS[1])

    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print("\nPerformance on test set (per-100g):")
    print(f"- MAE:  {mae:.2f}")
    print(f"- RMSE: {rmse:.2f}")

    nutrients = ["calories", "protein", "fat", "carbs"]
    print("\nPer-nutrient MAE (per-100g):")
    for i, n in enumerate(nutrients):
        err = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        print(f"- {n.capitalize()}: {err:.2f}")

    return model, vectorizer, y_scaler

# ---------------------------
# Inference helpers
# ---------------------------

def predict_for_text(model, vectorizer, y_scaler, text: str):
    grams, clean_text = extract_quantity_grams(text)
    X = vectorizer.transform([clean_text])
    pred_scaled = model.predict(X)
    per100g = y_scaler.inverse_transform(pred_scaled)[0]

    # clip to plausible ranges
    per100g = np.clip(per100g, [0, 0, 0, 0], [1200, 120, 120, 180])

    final = per100g * (grams / 100.0)
    return grams, clean_text, per100g, final

def test_examples(model, vectorizer, y_scaler):
    print("\n" + "="*50)
    print("TESTING MODEL WITH EXAMPLES")
    print("="*50)

    samples = [
        "200 grams of chicken breast",
        "100g of rice, cooked",
        "150 grams of apple",
        "300 grams of broccoli",
        "250g of salmon fillet",
        "100 grams of hummus, commercial",
    ]
    for s in samples:
        grams, clean_text, per100g, final = predict_for_text(model, vectorizer, y_scaler, s)
        print(f"\nInput: {s}")
        print(f"Parsed: grams={grams}, text='{clean_text}'")
        print("Prediction (per-100g): "
              f"{per100g[0]:.1f} kcal | {per100g[1]:.1f} g P | {per100g[2]:.1f} g F | {per100g[3]:.1f} g C")
        print(f"Scaled to {grams} g: "
              f"{final[0]:.1f} kcal | {final[1]:.1f} g P | {final[2]:.1f} g F | {final[3]:.1f} g C")

# ---------------------------
# Save
# ---------------------------

def save_all(model, vectorizer, y_scaler):
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    joblib.dump(model, "nutrition_model.pkl")
    joblib.dump(vectorizer, "text_vectorizer.pkl")
    joblib.dump(y_scaler, "output_scaler.pkl")
    print("Saved:")
    print("- nutrition_model.pkl")
    print("- text_vectorizer.pkl")
    print("- output_scaler.pkl")

# ---------------------------
# Save to onnx
# ---------------------------
def save_models(model, vectorizer, output_scaler):
    """
    Save trained models, vocab, and export to ONNX with scaler parameters
    """
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    import joblib, json, os, numpy as np
    
    # Save sklearn objects (for debugging in Python)
    joblib.dump(model, "nutrition_model.pkl")
    joblib.dump(vectorizer, "text_vectorizer.pkl")
    joblib.dump(output_scaler, "output_scaler.pkl")
    print("Saved:")
    print("- nutrition_model.pkl")
    print("- text_vectorizer.pkl")
    print("- output_scaler.pkl")

    # --- Export vocabulary for C++ preprocessing ---
    vocab_path = "vectorizer_vocab.json"
    vocab = {word: int(idx) for word, idx in vectorizer.vocabulary_.items()}
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"- {vocab_path} (TF-IDF vocabulary)")

    # --- NEW: Export RobustScaler parameters ---
    scaler_params = {
        "center_": output_scaler.center_.tolist(),  # median values
        "scale_": output_scaler.scale_.tolist(),    # IQR-based scaling factors
        "nutrient_order": ["calories", "protein", "fat", "carbs"]
    }
    
    scaler_path = "output_scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"- {scaler_path} (RobustScaler parameters)")
    
    print(f"Scaler centers (medians): {scaler_params['center_']}")
    print(f"Scaler scales (IQR): {scaler_params['scale_']}")

    # Try exporting Ridge model to ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        n_features = len(vectorizer.get_feature_names_out())
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open("nutrition_model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        print("✅ Exported Ridge model to nutrition_model.onnx")
        print("⚠️  IMPORTANT: ONNX model outputs SCALED values. Use output_scaler_params.json to inverse transform!")
    except ImportError:
        print("⚠️ ONNX export skipped: install skl2onnx with `pip install skl2onnx onnx`")
    except Exception as e:
        print(f"ONNX export failed: {e}")



# ---------------------------
# Main
# ---------------------------

def main():
    print("="*60)
    print("NUTRITION MODEL TRAINER (PER-100G, ROBUST)")
    print("="*60)

    texts, labels_df = load_training_data("../cleaned_data/training_data.json")
    if texts is None or labels_df is None:
        return

    model, vectorizer, y_scaler = train_model(texts, labels_df)
    test_examples(model, vectorizer, y_scaler)
    save_all(model, vectorizer, y_scaler)
    save_models(model, vectorizer, y_scaler)

if __name__ == "__main__":
    main()
