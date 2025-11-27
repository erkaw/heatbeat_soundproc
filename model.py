"""model.py
Reusable model builders and training/evaluation helpers extracted from the notebook.

Functions:
- reshape_mfccs_for_classifier
- expand_dims_for_cnn
- train_random_forest
- train_svm
- build_cnn
- train_cnn
- evaluate_classifier

This file intentionally keeps responsibilities small so the notebook can import
these utilities instead of containing long training blocks.
"""

from typing import Optional, Tuple, Dict

import numpy as np

# Classical ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def reshape_mfccs_for_classifier(X: np.ndarray) -> np.ndarray:
	"""Reshape MFCC array (n_samples, n_mfcc, n_time) -> (n_samples, n_mfcc * n_time).

	Caller should ensure the input is a 3-D numpy array.
	"""
	if X.ndim != 3:
		raise ValueError("Expected X with shape (n_samples, n_mfcc, n_time)")
	n_samples, n_mfcc, n_time = X.shape
	return X.reshape(n_samples, n_mfcc * n_time)


def expand_dims_for_cnn(X: np.ndarray) -> np.ndarray:
	"""Add channel dimension for CNN input: (n_samples, n_mfcc, n_time) -> (n_samples, n_mfcc, n_time, 1)"""
	if X.ndim != 3:
		raise ValueError("Expected X with shape (n_samples, n_mfcc, n_time)")
	return np.expand_dims(X, axis=-1)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, *, n_estimators: int = 100,
						random_state: int = 42, n_jobs: int = -1) -> RandomForestClassifier:
	"""Train a RandomForestClassifier on already-reshaped feature vectors.

	X_train is expected to be 2-D (n_samples, n_features).
	"""
	clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
	clf.fit(X_train, y_train)
	return clf


def train_svm(X_train: np.ndarray, y_train: np.ndarray, *, kernel: str = 'linear', random_state: int = 42) -> SVC:
	"""Train an SVM classifier (expects 2-D features)."""
	clf = SVC(kernel=kernel, random_state=random_state)
	clf.fit(X_train, y_train)
	return clf


def build_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
	"""Return a small compiled CNN suitable for MFCC inputs.

	`input_shape` should be (n_mfcc, n_time, channels) e.g. (13, 500, 1)
	"""
	model = Sequential([
		Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
		MaxPooling2D((2, 2)),
		Conv2D(64, (3, 3), activation='relu'),
		MaxPooling2D((2, 2)),
		Flatten(),
		Dense(128, activation='relu'),
		Dropout(0.5),
		Dense(num_classes, activation='softmax')
	])

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model


def train_cnn(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
			    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
			    epochs: int = 50, batch_size: int = 32, verbose: int = 1) -> tf.keras.callbacks.History:
	"""Train a compiled Keras model. X inputs should already have channel dim.

	If validation data is provided it will be used during training.
	"""
	if X_val is not None and y_val is not None:
		history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
							batch_size=batch_size, verbose=verbose)
	else:
		history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return history


def evaluate_classifier(clf, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
	"""Compute accuracy, classification report and confusion matrix for a classifier.

	`clf` can be an sklearn classifier or any object with a `predict` method.
	Returns a dict with keys: `accuracy`, `report`, `confusion_matrix`, `y_pred`.
	"""
	y_pred = clf.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, zero_division=0)
	cm = confusion_matrix(y_test, y_pred)
	return {
		'accuracy': acc,
		'report': report,
		'confusion_matrix': cm,
		'y_pred': y_pred
	}


__all__ = [
	'reshape_mfccs_for_classifier', 'expand_dims_for_cnn',
	'train_random_forest', 'train_svm', 'build_cnn', 'train_cnn', 'evaluate_classifier'
]

