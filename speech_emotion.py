import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract features from audio
def extract_features(file_path):
    # Load audio file
    signal, sr = librosa.load(file_path, sr=22050)
    
    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=signal, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, spectral_contrast])

# Load dataset
dataset_path =  r"C:\Users\itaiy\Downloads\archive\Ravdess\audio_speech_actors_01-24"
emotion_labels = []
features = [] 

# Iterate through dataset
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            try:
                # Extract features and save label
                features.append(extract_features(file_path))
                emotion = file.split("-")[2]  # Extract emotion from filename (specific to RAVDESS)
                emotion_labels.append(emotion)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert to DataFrame
features_df = pd.DataFrame(features)
features_df['label'] = emotion_labels

# Encode labels
encoder = LabelEncoder()
features_df['label'] = encoder.fit_transform(features_df['label'])

# Split data
X = features_df.iloc[:, :-1].values
y = features_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

import joblib

# Save model
joblib.dump(model, 'ser_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Test with a new audio file
def predict_emotion(file_path):
    # Load and preprocess audio
    features = extract_features(file_path).reshape(1, -1)
    features = scaler.transform(features)
    
    # Predict emotion
    emotion_label = model.predict(features)[0]
    emotion_name = encoder.inverse_transform([emotion_label])[0]
    return emotion_name

# Example
test_audio_path = "path_to_test_audio.wav"
print(f"Predicted Emotion: {predict_emotion(test_audio_path)}")