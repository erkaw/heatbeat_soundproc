import os
import glob

import numpy as np
import pandas as pd
import librosa
from python_speech_features import mfcc
import joblib
from tensorflow.keras.models import load_model

# ============================
# Parameters (same as notebook)
# ============================
parameters = {
    'maxdur': 5,        # seconds
    'downfactor': 50,   # (not used here, but kept for reference)
    'target_fs': 1000,  # Hz
    'lowcut': 50.0,
    'highcut': 450.0,
    'order': 4,
}

# Label mapping from the notebook
label_to_int = {
    'artifact': 0,
    'normal': 1,
    'murmur': 2,
    'extrastole': 3,
    'extrahls': 4,
}
int_to_label = {v: k for k, v in label_to_int.items()}

# ============================
# Audio loading & preprocessing
# ============================

def load_and_preprocess_audio(path, params):
    """
    Load raw WAV, resample to target_fs, and fix duration to maxdur seconds
    (crop or zero-pad).
    """
    y, sr = librosa.load(path, sr=None, mono=True)

    target_fs = params['target_fs']
    if sr != target_fs:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_fs)
        sr = target_fs

    # Fix duration
    max_len = int(params['maxdur'] * sr)
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")

    # Ensure float32
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    return y, sr

# ============================
# Feature extraction (MFCC)
# ============================

def extract_features(signal, sr, n_mfcc=13, n_fft=256, hop_length=64,
                     n_mels=40, max_pad_len=500):
    """
    Approximation of the extract_features() from the notebook using
    python_speech_features.mfcc.
    Output shape: (n_mfcc, max_pad_len)
    """
    signal = signal.astype(np.float32)

    # python_speech_features uses winlen/winstep in *seconds*
    winlen = n_fft / sr
    winstep = hop_length / sr

    mfccs = mfcc(
        signal,
        samplerate=sr,
        winlen=winlen,
        winstep=winstep,
        numcep=n_mfcc,
        nfilt=n_mels,
        nfft=n_fft
    )  # (num_frames, num_cepstra)

    # transpose to (n_mfcc, time_frames)
    mfccs = mfccs.T

    # Pad or truncate along time axis
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs

def build_feature_array(audio_dir):
    """
    For every .wav in audio_dir:
      - load + preprocess
      - extract MFCC features

    Returns:
      filenames: list[str]
      X_mfccs : np.ndarray (N, n_mfcc, max_pad_len)
    """
    wav_paths = sorted(
        glob.glob(os.path.join(audio_dir, "*.wav"))
    )

    if not wav_paths:
        raise RuntimeError(f"No .wav files found in {audio_dir}")

    mfcc_list = []
    filenames = []

    for path in wav_paths:
        print(f"[INFO] Processing {path}")
        y, sr = load_and_preprocess_audio(path, parameters)
        feats = extract_features(y, sr)
        mfcc_list.append(feats)
        filenames.append(os.path.basename(path))

    X_mfccs = np.stack(mfcc_list, axis=0)  # (N, n_mfcc, max_pad_len)
    print("[INFO] Feature array shape:", X_mfccs.shape)
    return filenames, X_mfccs

# ============================
# Model loading
# ============================

def load_models(model_dir="model"):
    """
    Load CNN (.keras), RandomForest (.joblib), SVM (.joblib)
    from the given directory.
    """
    cnn_path = os.path.join(model_dir, "cnn.keras")
    rf_path = os.path.join(model_dir, "random_forest.joblib")
    svm_path = os.path.join(model_dir, "svm.joblib")

    print("[INFO] Loading models...")
    cnn_model = load_model(cnn_path)
    rf_model = joblib.load(rf_path)
    svm_model = joblib.load(svm_path)
    print("[INFO] Models loaded.")

    return cnn_model, rf_model, svm_model

# ============================
# Inference
# ============================

def run_inference(unlabelled_dir="unlabelled", model_dir="model",
                  output_csv="unlabelled_predictions.csv"):
    # 1. Build features
    filenames, X_mfccs = build_feature_array(unlabelled_dir)

    # CNN expects shape (N, n_mfcc, n_time_frames, 1)
    X_cnn = np.expand_dims(X_mfccs, axis=-1)

    # RF & SVM expect flattened features (N, n_mfcc * n_time_frames)
    N, n_mfcc, n_time = X_mfccs.shape
    X_flat = X_mfccs.reshape(N, n_mfcc * n_time)

    print("[INFO] X_cnn shape:", X_cnn.shape)
    print("[INFO] X_flat shape:", X_flat.shape)

    # 2. Load models
    cnn_model, rf_model, svm_model = load_models(model_dir)

    # 3. Predict
    print("[INFO] Predicting with CNN...")
    cnn_probs = cnn_model.predict(X_cnn)
    cnn_pred_int = np.argmax(cnn_probs, axis=1)

    print("[INFO] Predicting with Random Forest...")
    rf_pred_int = rf_model.predict(X_flat)

    print("[INFO] Predicting with SVM...")
    svm_pred_int = svm_model.predict(X_flat)

    # 4. Decode int -> label
    def decode_labels(int_arr):
        return [int_to_label[int(x)] for x in int_arr]

    cnn_labels = decode_labels(cnn_pred_int)
    rf_labels = decode_labels(rf_pred_int)
    svm_labels = decode_labels(svm_pred_int)

    # 5. Build DataFrame + save
    df = pd.DataFrame({
        "filename": filenames,
        "cnn_label": cnn_labels,
        "rf_label": rf_labels,
        "svm_label": svm_labels,
    })

    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved predictions to {output_csv}")
    print(df)

if __name__ == "__main__":
    run_inference(unlabelled_dir="unlabelled", model_dir="model")
