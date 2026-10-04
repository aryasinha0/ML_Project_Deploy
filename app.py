# ============================================================
# SURVIVOR DETECTION API - WITH WEB UI
# Model: Logistic + ExtraTrees Blend (83.85% OOF Accuracy)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for

# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model():
    """Load the trained Logistic + ExtraTrees blend model."""
    with open("models/model.pkl", "rb") as f:
        model_config = pickle.load(f)
    
    with open("models/feature_config.pkl", "rb") as f:
        feature_config = pickle.load(f)
    
    print(f"✓ Model loaded: {model_config.get('blend_name', 'Blend')}")
    print(f"✓ OOF Accuracy: {model_config.get('oof_accuracy', 'N/A')}")
    print(f"✓ Threshold: {model_config.get('threshold', 0.5)}")
    
    return model_config, feature_config


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """Apply the same feature engineering as during training."""
    df = df.copy()
    df = df.drop(columns=["PassengerId", "PassengerName"], errors="ignore")

    if "TicketTier" in df.columns:
        df["TicketTier"] = pd.to_numeric(df["TicketTier"], errors="coerce").round()

    if "RelativesAboard" in df.columns:
        df["RelativesAboard"] = pd.to_numeric(df["RelativesAboard"], errors="coerce").clip(lower=0)

    if "ParentsChildren" in df.columns:
        df["ParentsChildren"] = pd.to_numeric(df["ParentsChildren"], errors="coerce").clip(lower=0)

    if "RelativesAboard" in df.columns and "ParentsChildren" in df.columns:
        df["TotalFamily"] = df["RelativesAboard"] + df["ParentsChildren"]
        df["FamilySize"] = df["TotalFamily"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df.loc[df["Age"] < 0, "Age"] = np.nan

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()
        df["Gender"] = df["Gender"].replace({"male": "Male", "m": "Male", "female": "Female", "f": "Female"})

    if "Title" in df.columns:
        df["Title"] = df["Title"].astype(str).str.strip()
        common_titles = ["Mr", "Miss", "Mrs", "Master"]
        df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")

    if "CLass" in df.columns:
        ticket = df["CLass"].astype(str)
        df["TicketNumber"] = pd.to_numeric(ticket.str.extract(r"(\d+)$")[0], errors="coerce")
        df["TicketNumberMissing"] = df["TicketNumber"].isna().astype(int)
        df["TicketPrefix"] = ticket.str.replace(r"\d+", "", regex=True).str.replace(r"[\W_]+", "", regex=True).str.upper().str.strip()
        df["TicketPrefix"] = df["TicketPrefix"].replace("", "NONE")

        prefix_counts = df["TicketPrefix"].value_counts()
        common_prefixes = prefix_counts[prefix_counts >= 3].index
        df["TicketPrefix"] = df["TicketPrefix"].where(df["TicketPrefix"].isin(common_prefixes), "OTHER")

        df = df.drop(columns=["CLass"])

    if "Berth" in df.columns:
        berth = df["Berth"].astype(str)
        df["BerthKnown"] = (~berth.isin(["nan", "NaN", "", "None"])).astype(int)
        df["Deck"] = berth.str[0].str.upper()
        df["Deck"] = df["Deck"].replace(["N", "NAN"], "Unknown")
        df["NumCabins"] = berth.str.split().str.len()

    if "TicketCost" in df.columns:
        df["TicketCost"] = pd.to_numeric(df["TicketCost"], errors="coerce")
        df.loc[df["TicketCost"] < 0, "TicketCost"] = np.nan

        if "FamilySize" in df.columns:
            df["FarePerPerson"] = df["TicketCost"] / df["FamilySize"].replace(0, np.nan)

    return df


# ============================================================
# 3. PREDICTION FUNCTION
# ============================================================

def predict_survival(input_data, model_config, feature_config):
    """Make survival predictions using Logistic + ExtraTrees blend."""
    
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    input_processed = create_features(input_data)
    
    features = feature_config.get("core_features", [])
    threshold = model_config.get("threshold", 0.50)
    models = model_config["models"]
    weights = model_config["weights"]
    
    probabilities = 0.0
    
    for model_name, weight in weights.items():
        model = models[model_name]
        prob = model.predict_proba(input_processed[features])[:, 1]
        probabilities += weight * prob
    
    predictions = (probabilities >= threshold).astype(int)
    
    return {
        "probability": round(float(probabilities[0]), 4),
        "prediction": int(predictions[0]),
        "survived": bool(predictions[0] == 1)
    }


# ============================================================
# 4. FLASK APP
# ============================================================

app = Flask(__name__)
app.secret_key = "survivor-detection-secret-key-2024"

# Load model at startup
try:
    model_config, feature_config = load_model()
    MODEL_STATUS = "ready"
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    model_config = None
    feature_config = None
    MODEL_STATUS = "error"


@app.route("/", methods=["GET"])
def home():
    """Home page with prediction form."""
    return render_template("index.html", model_status=MODEL_STATUS)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction form submission."""
    if MODEL_STATUS != "ready":
        flash("Model not loaded. Please check server logs.", "error")
        return redirect(url_for("home"))
    
    try:
        # Get form data
        form_data = {
            "Age": float(request.form.get("Age", 0)) if request.form.get("Age") else None,
            "Gender": request.form.get("Gender", ""),
            "Title": request.form.get("Title", ""),
            "TicketTier": float(request.form.get("TicketTier", 0)) if request.form.get("TicketTier") else None,
            "RelativesAboard": float(request.form.get("RelativesAboard", 0)) if request.form.get("RelativesAboard") else None,
            "ParentsChildren": float(request.form.get("ParentsChildren", 0)) if request.form.get("ParentsChildren") else None,
            "TicketCost": float(request.form.get("TicketCost", 0)) if request.form.get("TicketCost") else None,
            "BoardingPort": request.form.get("BoardingPort", ""),
            "Berth": request.form.get("Berth", ""),
            "CLass": request.form.get("CLass", "")
        }
        
        # Remove None values
        form_data = {k: v for k, v in form_data.items() if v is not None and v != ""}
        
        if not form_data:
            flash("Please fill in at least some passenger details.", "warning")
            return redirect(url_for("home"))
        
        # Make prediction
        result = predict_survival(form_data, model_config, feature_config)
        
        return render_template(
            "result.html",
            input_data=form_data,
            prediction=result,
            model_info={
                "type": "Logistic + ExtraTrees",
                "threshold": 0.50,
                "oof_accuracy": 0.8385
            }
        )
    
    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "error")
        return redirect(url_for("home"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for JSON predictions."""
    if MODEL_STATUS != "ready":
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        result = predict_survival(data, model_config, feature_config)
        
        return jsonify({
            "success": True,
            "input": data,
            "prediction": result,
            "model_info": {
                "type": "Logistic + ExtraTrees",
                "threshold": 0.50,
                "oof_accuracy": 0.8385
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if MODEL_STATUS == "ready" else "unhealthy",
        "model_loaded": MODEL_STATUS == "ready",
        "model_type": "Logistic + ExtraTrees Blend"
    })


# ============================================================
# 5. RUN APP
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    print("=" * 70)
    print("SURVIVOR DETECTION API - WEB UI")
    print("=" * 70)
    print(f"Model: Logistic + ExtraTrees Blend")
    print(f"OOF Accuracy: 0.8385")
    print(f"Threshold: 0.50")
    print(f"Running on: http://localhost:{port}")
    print(f"Debug: {debug}")
    print("=" * 70)
    
    app.run(host="0.0.0.0", port=port, debug=debug)