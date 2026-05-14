import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("../processed_data/spotify_cleaned.csv")

# =========================================================
# FEATURES USED FOR SIMILARITY (IMPORTANT)
# =========================================================

FEATURES = [
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
# =========================================================
# BUILD NEAREST NEIGHBORS MODEL
# =========================================================

nn_model = NearestNeighbors(
    n_neighbors=11,
    metric="cosine"
)

nn_model.fit(X_scaled)

# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def get_recommendations(song_index: int, n=10):

    distances, indices = nn_model.kneighbors([X_scaled[song_index]])

    recommended_indices = indices[0][1:]  # skip itself

    results = df.iloc[recommended_indices][
        ["name", "artists", "popularity"]
    ].copy()

    results["similarity_score"] = 1 - distances[0][1:]

    return results.to_dict(orient="records")


# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    recs = get_recommendations(song_index=0, n=10)

    for r in recs:
        print(r)