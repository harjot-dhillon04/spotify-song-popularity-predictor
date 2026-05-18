# 📈 Spotify Song Popularity Predictor & Recommendation API

An end-to-end machine learning system that predicts **song popularity** and provides **music recommendations based on audio feature similarity**.

The system is deployed as a production-grade REST API using **FastAPI**, hosted on an **AWS EC2 instance**, with trained model artifacts stored in **Amazon S3** and loaded dynamically at runtime.

The machine learning pipeline was trained using a dataset containing **550,000+ Spotify songs across 10 genres**, enabling large-scale popularity prediction and recommendation generation based on audio characteristics.

---

# 🚀 Features

## 🎵 Machine Learning Prediction
- Predicts Spotify song popularity based on audio features, genre, and artist popularity
- Trained using supervised learning models:
  - 🌲 Random Forest
  - 🚀 XGBoost
- Feature preprocessing using **StandardScaler**
- Best model selected based on validation performance

---

## 🎧 Music Recommendation System
- Recommends songs based on **similarity of audio features**
- Uses feature-space comparison to find closest matches in dataset
- Returns **top-N most similar songs along with the popularity score** for each song
- Enables Spotify-like discovery experience

### How it works:
- User provides audio feature vector
- System compares input against dataset feature space
- Computes similarity using distance-based methods
- Returns closest matching songs

---

## 🌐 REST API (FastAPI)
- Real-time inference via `/predict`
- Recommendation engine via `/recommend`
- Auto-generated Swagger documentation (`/docs`)

---

## ☁️ Cloud-Based Model Storage (S3)
- Model artifacts stored in **Amazon S3**
  - Trained ML model (`.pkl`)
  - Feature scaler (`.pkl`)
- EC2 instance downloads artifacts using `boto3` at startup

---

## ⚙️ Cloud Deployment (AWS EC2)
- Hosted on Ubuntu EC2 instance
- Exposed publicly via port `8000`
- Runs inside isolated Python virtual environment (`venv`)

---

# 📊 Dataset

The project was trained using a large Spotify dataset containing:

- 🎵 **550,000+ songs**
- 🎼 **10 music genres**
- 📈 Popularity metrics
- 🎚 Audio features such as:
  - Danceability
  - Energy
  - Tempo
  - Loudness
  - Acousticness
  - Valence
  - Speechiness
  - Instrumentalness
  - Liveness

The dataset enables both:
- Soong popularity prediction
- Audio-feature-based recommendation generation

## 🔗 Dataset Source

```text
https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres
```

---

# 🧠 Machine Learning Models

The system evaluates multiple supervised learning models:

- 🌲 Random Forest  
- 🚀 XGBoost  
- 📊 StandardScaler for feature normalization

---

## 🧪 Model Performance

The classification model was evaluated using a held-out test set (72,629 samples). Both Random Forest and XGBoost were trained and compared, and performance was found to be broadly similar. Random Forest was selected for deployment due to stability and comparable performance.

### 🏆 Final Model: Random Forest

### 📊 Test Set Results

- Accuracy: **0.65**
- Macro F1-score: **0.51**
- Weighted F1-score: **0.62**

---

# 🧠 System Architecture

```text
User (Browser / Python / App)
        ↓
FastAPI Backend (AWS EC2)
        ↓
S3 Model Loader (boto3)
        ↓
Preprocessing (StandardScaler)
        ↓
ML Model (Random Forest / XGBoost)
        ↓
Prediction OR Recommendation Engine
        ↓
JSON Response
```

---

# ⚙️ How to Run

This project can be run locally or deployed on an AWS EC2 instance. The system requires access to model artifacts stored in **Amazon S3**.

---

## 1. Clone the Repository

```bash
git clone https://github.com/<harjot-dhillon04>/spotify-song-popularity-predictor.git
cd spotify-song-popularity-predictor
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# or
venv\Scripts\activate      # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure AWS Access (Required for Model Loading)

The application loads trained ML artifacts from **S3 at startup**.

### Option A — EC2 (Recommended)
If running on EC2:
- Attach an **IAM Role** with S3 read permissions
- No need to manually store AWS keys

### Option B — Local Setup
Configure AWS CLI:

```bash
aws configure
```

Or set environment variables:

```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="ca-central-1"
```

---

## 5. Required Files

The following artifacts must exist in S3:

- Trained ML model (`spotify_popularity_model.pkl`)
- Feature scaler (`scaler.pkl`)
- Any preprocessing objects

These are automatically downloaded at runtime by the FastAPI app.

---

## 6. Run the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 7. Access the API

Once running:

### API Root
```text
http://<EC2_PUBLIC_IP>:8000
```

### Swagger Docs
```text
http://<EC2_PUBLIC_IP>:8000/docs
```

---

# 🔮 Prediction Endpoint

## POST `/predict`

### Request

```json
{
  "danceability": 0.72,
  "energy": 0.81,
  "loudness": -5.4,
  "valence": 0.67,
  "tempo": 124.5,
  "speechiness": 0.08,
  "acousticness": 0.14,
  "instrumentalness": 0.0,
  "liveness": 0.11,
  "key": 5,
  "mode": 1,
  "duration_ms": 214000,
  "avg_artist_popularity": 76,
  "genre": "Pop",
  "year": 2022
}
```

### Response

```json
{
  "prediction_label": "High Popularity",

  "probabilities": {
    "low_popularity_percent": 4.8,
    "medium_popularity_percent": 27.3,
    "high_popularity_percent": 67.9
  },

  "top_feature_impacts": [
    {
      "feature": "avg_artist_popularity",
      "impact": 0.341
    },
    {
      "feature": "danceability",
      "impact": 0.037
    },
    {
      "feature": "energy",
      "impact": 0.021
    },
    {
      "feature": "duration_ms",
      "impact": -0.018
    },
    {
      "feature": "genre_Pop",
      "impact": 0.015
    }
  ]
}
```

---

# 🎧 Recommendation Endpoint

## POST `/recommend`

### Request

```json
{
  "danceability": 0.74,
  "energy": 0.82,
  "loudness": -5.1,
  "valence": 0.61,
  "tempo": 128.0,
  "speechiness": 0.07,
  "acousticness": 0.12,
  "instrumentalness": 0.0,
  "liveness": 0.15,
  "key": 1,
  "mode": 1,
  "duration_ms": 210000
}
```

### Response

```json
[
  {
    "name": "Recommended Song 1",
    "artists": [
      "Artist A"
    ],
    "popularity": 73,
    "similarity_score": 91.4
  },
  {
    "name": "Recommended Song 2",
    "artists": [
      "Artist B"
    ],
    "popularity": 58,
    "similarity_score": 90.8
  },
  {
    "name": "Recommended Song 3",
    "artists": [
      "Artist C"
    ],
    "popularity": 41,
    "similarity_score": 89.7
  }
]
```

---

# 🛠 Tech Stack

## Backend
- FastAPI
- Uvicorn

## Machine Learning
- scikit-learn
- XGBoost
- Random Forest
- pandas
- joblib

## Cloud Infrastructure
- AWS EC2 (compute)
- AWS S3 (model storage)
- boto3 (S3 integration)

---

# 📁 Project Structure

```text
spotify-song-popularity-predictor/
│
├── api/
│   └── main.py
│
├── models/                  
│
├── processed_data/          
│
├── requirements.txt
└── README.md
```

---

# ⚙️ How It Works

## 1. Model Training
- Models trained on Spotify dataset
- Multiple algorithms evaluated (Random Forest, XGBoost)

## 2. Model Storage
- Serialized model + scaler uploaded to **S3**

## 3. Deployment Flow
- EC2 server starts FastAPI app
- Downloads artifacts from S3
- Loads ML model into memory
- Serves:
  - `/predict`
  - `/recommend`

---

# ☁️ Deployment Notes

- EC2 Ubuntu instance
- Virtual environment (`venv`) for dependencies
- IAM permissions required for S3 access
- Port `8000` exposed via security group
- Model loaded at startup from S3

---

# 🧠 Skills Demonstrated

- End-to-end ML system design
- Recommendation systems (audio feature similarity search)
- REST API development (FastAPI)
- AWS cloud deployment (EC2 + S3)
- ML model training (Random Forest, XGBoost)
- Model serialization and serving
- Backend engineering fundamentals

---

# 📌 Summary

This project demonstrates:

- Production-style ML system deployment
- Real-time prediction and recommendation APIs
- Cloud deployment using AWS EC2
- S3-based model artifact storage
- Backend engineering and ML infrastructure fundamentals
