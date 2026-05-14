from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import ast
import shap
# =========================================================
# PATHS + LOAD MODELS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

predictor_scaler = joblib.load(
    os.path.join(BASE_DIR, "models/scaler.pkl")
)

model = joblib.load(
    os.path.join(BASE_DIR, "models/spotify_popularity_model.pkl")
)

app = FastAPI()

explainer = shap.TreeExplainer(model)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    os.path.join(BASE_DIR, "processed_data/spotify_cleaned.csv")
)

# =========================================================
# FEATURES
# =========================================================

genre_cols = [c for c in df.columns if c.startswith("genre_")]
era_cols = [c for c in df.columns if c.startswith("era_")]

LABELS = {
    0: "Low Popularity",
    1: "Medium Popularity",
    2: "High Popularity"
}

# =========================================================
# RECOMMENDER FEATURES (AUDIO ONLY)
# =========================================================

RECOMMENDER_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "valence",
    "tempo",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "key",
    "mode",
    "duration_ms"
]

# =========================================================
# PREDICTION FEATURES
# =========================================================

PREDICTION_FEATURES = [

    "danceability",
    "energy",
    "loudness",
    "valence",
    "tempo",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "key",
    "mode",
    "duration_ms",
    "avg_artist_popularity",

    *genre_cols,
    *era_cols
]

# =========================================================
# RECOMMENDER SCALER + MODEL
# =========================================================

recommender_scaler = StandardScaler()

X_scaled = recommender_scaler.fit_transform(
    df[RECOMMENDER_FEATURES]
)

nn_model = NearestNeighbors(
    n_neighbors=10,
    metric="cosine"
)

nn_model.fit(X_scaled)

# =========================================================
# REQUEST MODELS
# =========================================================

class RecommendationFeatures(BaseModel):
    danceability: float
    energy: float
    loudness: float
    valence: float
    tempo: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    key: float
    mode: float
    duration_ms: float


class PredictionFeatures(BaseModel):
    danceability: float
    energy: float
    loudness: float
    valence: float
    tempo: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    key: float
    mode: float
    duration_ms: float
    avg_artist_popularity: float
    genre: str
    year: int

# =========================================================
# ERA HELPERS
# =========================================================

def year_to_era(year: int):

    if year < 1980:
        return "classic"

    elif year < 2000:
        return "retro"

    elif year < 2010:
        return "early_streaming"

    else:
        return "modern"


def encode_era(year: int):

    era = year_to_era(year)

    return [
        1 if era == "classic" else 0,
        1 if era == "retro" else 0,
        1 if era == "early_streaming" else 0,
        1 if era == "modern" else 0
    ]


def encode_genre(genre: str):

    return [
        1 if f"genre_{genre}" == col else 0
        for col in genre_cols
    ]

# =========================================================
# RECOMMEND ENDPOINT
# =========================================================

@app.post("/recommend")
def recommend(song: RecommendationFeatures):

    input_vector = np.array([[
        song.danceability,
        song.energy,
        song.loudness,
        song.valence,
        song.tempo,
        song.speechiness,
        song.acousticness,
        song.instrumentalness,
        song.liveness,
        song.key,
        song.mode,
        song.duration_ms
    ]])

    input_scaled = recommender_scaler.transform(
        input_vector
    )

    distances, indices = nn_model.kneighbors(
        input_scaled
    )

    recommended = df.iloc[indices[0]][["name", "artists", "popularity"]].copy()

    recommended["artists"] = recommended["artists"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    recommended["similarity_score"] = (
        (1 - distances[0])*100
    )

    return recommended.to_dict(
        orient="records"
    )

# =========================================================
# PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
def predict(song: PredictionFeatures):

    genre_vector = encode_genre(song.genre)

    era_vector = encode_era(song.year)

    input_vector = np.array([[
        song.danceability,
        song.energy,
        song.loudness,
        song.valence,
        song.tempo,
        song.speechiness,
        song.acousticness,
        song.instrumentalness,
        song.liveness,
        song.key,
        song.mode,
        song.duration_ms,
        song.avg_artist_popularity,
        *genre_vector,
        *era_vector
    ]])

    input_scaled = predictor_scaler.transform(
        input_vector
    )

    probs = model.predict_proba(
        input_scaled
    )[0]

    pred_class = model.predict(
        input_scaled
    )[0]

    # -----------------------------------
    # SHAP EXPLANATION
    # -----------------------------------

    shap_values = explainer.shap_values(input_scaled)

    feature_impacts = []

    # multiclass handling
    class_shap_values = shap_values[:, :, int(pred_class)][0]

    for feature, impact in zip(PREDICTION_FEATURES, class_shap_values):

        feature_impacts.append({
            "feature": feature,
            "impact": round(float(impact), 4)
        })

    feature_impacts = sorted(
    feature_impacts,
    key=lambda x: abs(x["impact"]),
    reverse=True
)
    return {

        "Prediction label": LABELS[int(pred_class)],

        "Probabilities": {
            "Low Popularity %": round(float(probs[0]) * 100, 2),
            "Medium Popularity %": round(float(probs[1]) * 100, 2),
            "High popularity %": round(float(probs[2]) * 100, 2)
        },

        "Top Feature Impacts": feature_impacts[:10]
    }