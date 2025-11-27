"""train.py
Extracted training commands for RandomForest, SVM and CNN.

This module provides functions to train models from MFCC features (as in the
notebook) and save the trained models and evaluation metrics to `models/`.

Usage from notebook:

  from train import train_and_save_random_forest, train_and_save_svm, train_and_save_cnn

  # RF / SVM expect MFCC arrays shaped (n_samples, n_mfcc, n_time)
  rf_res = train_and_save_random_forest(X_train_mfccs, X_test_mfccs, y_train, y_test)
  svm_res = train_and_save_svm(X_train_mfccs, X_test_mfccs, y_train, y_test)

  # CNN expects arrays with channel dim (n_samples, n_mfcc, n_time, 1)
  cnn_res = train_and_save_cnn(X_train_cnn, X_test_cnn, y_train, y_test)

When run as a script, the module will attempt to load NumPy files from the
current working directory: `X_train_mfccs.npy`, `X_test_mfccs.npy`,
`y_train.npy`, `y_test.npy` and run all trainers. This is optional convenience
for quick CLI testing.
"""

from pathlib import Path
import json
import os
from typing import Dict, Tuple

import numpy as np
import joblib
import librosa

from model import (
	reshape_mfccs_for_classifier,
	expand_dims_for_cnn,
	train_random_forest,
	train_svm,
	build_cnn,
	train_cnn,
	evaluate_classifier,
)


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def extract_mfcc_librosa(y: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 256,
						 hop_length: int = 64, max_pad_len: int = 500) -> np.ndarray:
	"""Extract MFCCs using librosa and pad/truncate to `max_pad_len` frames.

	Returns array shaped (n_mfcc, max_pad_len).
	"""
	mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
	if mfccs.shape[1] < max_pad_len:
		pad_width = max_pad_len - mfccs.shape[1]
		mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
	else:
		mfccs = mfccs[:, :max_pad_len]
	return mfccs


def load_wav_and_extract_mfcc(path: Path, target_sr: int = 1000, n_mfcc: int = 13,
							  n_fft: int = 256, hop_length: int = 64, max_pad_len: int = 500) -> np.ndarray:
	"""Load a WAV file with librosa (resampled to `target_sr`) and return MFCCs."""
	y, sr = librosa.load(str(path), sr=target_sr, mono=True)
	return extract_mfcc_librosa(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, max_pad_len=max_pad_len)


def _save_metrics(name: str, metrics: Dict, out_dir: Path = MODELS_DIR) -> None:
	path = out_dir / f"{name}_metrics.json"
	# Convert numpy arrays (like confusion matrix) into lists
	serializable = {}
	for k, v in metrics.items():
		try:
			json.dumps(v)
			serializable[k] = v
		except Exception:
			# fallback for numpy arrays etc
			if isinstance(v, np.ndarray):
				serializable[k] = v.tolist()
			else:
				serializable[k] = str(v)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(serializable, f, indent=2)


def train_and_save_random_forest(X_train_mfccs: np.ndarray, X_test_mfccs: np.ndarray,
								 y_train: np.ndarray, y_test: np.ndarray,
								 model_name: str = "random_forest") -> Dict:
	"""Train RandomForest on MFCCs (3D arrays) and save model + metrics.

	Returns a dict with evaluation results.
	"""
	X_train = reshape_mfccs_for_classifier(X_train_mfccs)
	X_test = reshape_mfccs_for_classifier(X_test_mfccs)

	clf = train_random_forest(X_train, y_train)

	res = evaluate_classifier(clf, X_test, y_test)

	out_path = MODELS_DIR / f"{model_name}.joblib"
	joblib.dump(clf, out_path)
	_save_metrics(model_name, res)
	return res


def train_and_save_svm(X_train_mfccs: np.ndarray, X_test_mfccs: np.ndarray,
					   y_train: np.ndarray, y_test: np.ndarray,
					   model_name: str = "svm") -> Dict:
	X_train = reshape_mfccs_for_classifier(X_train_mfccs)
	X_test = reshape_mfccs_for_classifier(X_test_mfccs)

	clf = train_svm(X_train, y_train)

	res = evaluate_classifier(clf, X_test, y_test)

	out_path = MODELS_DIR / f"{model_name}.joblib"
	joblib.dump(clf, out_path)
	_save_metrics(model_name, res)
	return res


def train_and_save_cnn(X_train_cnn: np.ndarray, X_test_cnn: np.ndarray,
					   y_train: np.ndarray, y_test: np.ndarray,
					   model_name: str = "cnn", epochs: int = 50, batch_size: int = 32) -> Dict:
	"""Train Keras CNN and save the model and metrics.

	`X_train_cnn` expected shape: (n_samples, n_mfcc, n_time, channels)
	"""
	if X_train_cnn.ndim != 4:
		raise ValueError("X_train_cnn must have 4 dims (n_samples, n_mfcc, n_time, channels)")

	num_classes = int(np.max(y_train) - np.min(y_train) + 1)
	input_shape = X_train_cnn.shape[1:]

	model = build_cnn(input_shape, num_classes)

	# Use a small validation split if explicit test not desired here
	history = train_cnn(model, X_train_cnn, y_train, X_val=X_test_cnn, y_val=y_test,
						epochs=epochs, batch_size=batch_size)

	# Predict on test
	y_pred_proba = model.predict(X_test_cnn)
	y_pred = np.argmax(y_pred_proba, axis=1)

	from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, zero_division=0)
	cm = confusion_matrix(y_test, y_pred)

	res = {
		"accuracy": float(acc),
		"report": report,
		"confusion_matrix": cm,
		"y_pred": y_pred.tolist(),
		"history": {k: [float(x) for x in v] for k, v in history.history.items()},
	}

	model_dir = MODELS_DIR / model_name
	model_dir.mkdir(parents=True, exist_ok=True)
	# Save in TF SavedModel format
	model.save(str(model_dir))

	_save_metrics(model_name, res)
	return res


def _try_load_npy(name: str):
	p = Path(name)
	if p.exists():
		return np.load(p, allow_pickle=True)
	return None


def run_all_if_npy_exist():
	"""Convenience runner: if required npy files found, run all training pipelines."""
	X_train = _try_load_npy("X_train_mfccs.npy")
	X_test = _try_load_npy("X_test_mfccs.npy")
	y_train = _try_load_npy("y_train.npy")
	y_test = _try_load_npy("y_test.npy")

	if X_train is None or X_test is None or y_train is None or y_test is None:
		print("Required .npy files not present in cwd. Skipping CLI run.")
		return

	print("Training RandomForest...")
	rf_res = train_and_save_random_forest(X_train, X_test, y_train, y_test)
	print("RF done.")

	print("Training SVM...")
	svm_res = train_and_save_svm(X_train, X_test, y_train, y_test)
	print("SVM done.")

	# Prepare CNN inputs
	try:
		X_train_cnn = expand_dims_for_cnn(X_train)
		X_test_cnn = expand_dims_for_cnn(X_test)
		print("Training CNN (this may take a while)...")
		cnn_res = train_and_save_cnn(X_train_cnn, X_test_cnn, y_train, y_test)
		print("CNN done.")
	except Exception as e:
		print(f"Skipping CNN: {e}")


if __name__ == "__main__":
	run_all_if_npy_exist()

