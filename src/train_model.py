import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    features = []

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features.extend(np.mean(mel, axis=1))

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    # Tonnetz
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    features.extend(np.mean(tonnetz, axis=1))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    # RMSE
    rmse = librosa.feature.rms(y=y)
    features.append(np.mean(rmse))

    return features

# 3. LOAD & PROCESS TRAIN DATA
train_df = pd.read_csv("dataset/train.csv")  # Contains 'filename' & 'label'

X = []
y = []

for i, row in train_df.iterrows():
    path = f"dataset/audios_train/{row['filename']}"
    feats = extract_features(path)
    X.append(feats)
    y.append(row["label"])

X = pd.DataFrame(X)
y = pd.Series(y)

print("Train features shape:", X.shape)

#TRAIN-TEST SPLIT & HYPERPARAMETER TUNING
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=5,
    verbose=1, n_jobs=-1, random_state=42
)

random_search.fit(X_train, y_train)

print("Best Params:", random_search.best_params_)

#FINAL TRAINING WITH BEST PARAMS
final_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
final_model.fit(X, y)

# Save model for reuse
joblib.dump(final_model, "final_model.pkl")

# LOAD TEST DATA & EXTRACT FEATURES
test_df = pd.read_csv("dataset/test.csv")  
filenames = test_df["filename"].tolist()

test_features = []

for fname in filenames:
    path = f"dataset/audios_test/{fname}"
    feats = extract_features(path)
    test_features.append(feats)

test_features_df = pd.DataFrame(test_features)
print("Test features shape:", test_features_df.shape)

#PREDICT ON TEST DATA
predictions = final_model.predict(test_features_df)

# Prepare submission
submission = pd.DataFrame({
    "filename": filenames,
    "label": predictions
})

submission.to_csv("submission.csv", index=False)