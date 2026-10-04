
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURE_CONFIG_PATH = os.path.join(BASE_DIR, "feature_config.pkl")

with open(MODEL_PATH, "rb") as f:
    model_config = pickle.load(f)

with open(FEATURE_CONFIG_PATH, "rb") as f:
    feature_config = pickle.load(f)

logistic_model = model_config["models"]["Logistic"]
extra_trees_model = model_config["models"]["ExtraTrees"]
LOGISTIC_WEIGHT = model_config["weights"]["Logistic"]
EXTRA_TREES_WEIGHT = model_config["weights"]["ExtraTrees"]
THRESHOLD = model_config["threshold"]
CORE_FEATURES = feature_config["core_features"]

def create_features(df):
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
        df["Gender"] = df["Gender"].astype("string").str.lower().str.strip()
        df["Gender"] = df["Gender"].replace({"male": "Male", "m": "Male", "female": "Female", "f": "Female"})

    if "Title" in df.columns:
        df["Title"] = df["Title"].astype("string").str.strip()
        common_titles = ["Mr", "Miss", "Mrs", "Master"]
        df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")

    if "CLass" in df.columns:
        ticket = df["CLass"].astype("string")
        df["TicketNumber"] = pd.to_numeric(ticket.str.extract(r"(\d+)$")[0], errors="coerce")
        df["TicketNumberMissing"] = df["TicketNumber"].isna().astype(int)
        df["TicketPrefix"] = ticket.str.replace(r"\d+", "", regex=True).str.replace(r"[\W_]+", "", regex=True).str.upper().str.strip()
        df["TicketPrefix"] = df["TicketPrefix"].replace("", "NONE")
        prefix_counts = df["TicketPrefix"].value_counts()
        common_prefixes = prefix_counts[prefix_counts >= 3].index
        df["TicketPrefix"] = df["TicketPrefix"].where(df["TicketPrefix"].isin(common_prefixes), "OTHER")
        df = df.drop(columns=["CLass"])

    if "Berth" in df.columns:
        berth = df["Berth"].astype("string")
        df["BerthKnown"] = (~berth.isna() & berth.ne("")).astype(int)
        df["Deck"] = berth.str[0].str.upper()
        df["Deck"] = df["Deck"].fillna("Unknown").replace(["N", "NAN"], "Unknown")
        df["NumCabins"] = berth.str.split().str.len()

    if "TicketCost" in df.columns:
        df["TicketCost"] = pd.to_numeric(df["TicketCost"], errors="coerce")

        df.loc[df["TicketCost"] < 0, "TicketCost"] = np.nan

        if "FamilySize" in df.columns:
            df["FarePerPerson"] = df["TicketCost"] / df["FamilySize"].replace(0, np.nan)

    return df

def to_number(value):
    if value is None or value.strip() == "":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan

def predict_survival(data):
    df = pd.DataFrame([data])
    df = create_features(df)
    X = df[CORE_FEATURES]

    logistic_probability = logistic_model.predict_proba(X)[0][1]
    extra_trees_probability = extra_trees_model.predict_proba(X)[0][1]

    final_probability = LOGISTIC_WEIGHT * logistic_probability + EXTRA_TREES_WEIGHT * extra_trees_probability
    prediction = int(final_probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "probability": final_probability,
        "logistic_probability": logistic_probability,
        "extra_trees_probability": extra_trees_probability
    }

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            data = {
                "PassengerId": request.form.get("PassengerId", ""),
                "PassengerName": request.form.get("PassengerName", ""),
                "Age": to_number(request.form.get("Age", "")),
                "Gender": request.form.get("Gender", ""),
                "Title": request.form.get("Title", ""),
                "TicketTier": to_number(request.form.get("TicketTier", "")),
                "RelativesAboard": to_number(request.form.get("RelativesAboard", "")),
                "ParentsChildren": to_number(request.form.get("ParentsChildren", "")),
                "CLass": request.form.get("CLass", ""),
                "Berth": request.form.get("Berth", ""),
                "TicketCost": to_number(request.form.get("TicketCost", "")),
                "BoardingPort": request.form.get("BoardingPort", "")
            }

            result = predict_survival(data)
            result["probability_percent"] = round(result["probability"] * 100, 2)
            result["logistic_percent"] = round(result["logistic_probability"] * 100, 2)
            result["extra_trees_percent"] = round(result["extra_trees_probability"] * 100, 2)

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

