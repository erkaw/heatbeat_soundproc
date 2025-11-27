import os
import numpy as np
import librosa
from python_speech_features import mfcc as psf_mfcc
import joblib
from tensorflow.keras.models import load_model

import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ======================================================
# Konfigurasi / parameter (samain dengan notebook-mu)
# ======================================================
parameters = {
    "maxdur": 5,          # seconds
    "target_fs": 1000,    # Hz (resample target)
    "n_mfcc": 13,
    "n_mels": 40,
    "n_fft": 256,
    "hop_length": 64,
    "max_pad_len": 500,
}

# Label mapping (urutkan sesuai training-mu)
CLASS_NAMES = ["artifact", "normal", "murmur", "extrastole", "extrahls"]


# ======================================================
# Audio loading & preprocessing
# ======================================================
def load_and_preprocess_audio(path, params):
    """
    Load WAV, resample ke target_fs, dan fix durasi ke maxdur detik.
    """
    y, sr = librosa.load(path, sr=None, mono=True)

    target_fs = params["target_fs"]
    if sr != target_fs:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_fs)
        sr = target_fs

    max_len = int(params["maxdur"] * sr)
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")

    return y.astype(np.float32), sr


def extract_mfcc_psf(signal, sr, params):
    """
    MFCC untuk model (python_speech_features).
    Output: (n_mfcc, max_pad_len)
    """
    n_mfcc = params["n_mfcc"]
    n_mels = params["n_mels"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    max_pad_len = params["max_pad_len"]

    winlen = n_fft / sr
    winstep = hop_length / sr

    mfcc_feat = psf_mfcc(
        signal,
        samplerate=sr,
        winlen=winlen,
        winstep=winstep,
        numcep=n_mfcc,
        nfilt=n_mels,
        nfft=n_fft,
    )  # (frames, n_mfcc)

    mfcc_feat = mfcc_feat.T  # (n_mfcc, frames)

    if mfcc_feat.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc_feat.shape[1]
        mfcc_feat = np.pad(mfcc_feat, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc_feat = mfcc_feat[:, :max_pad_len]

    return mfcc_feat


def extract_mfcc_librosa_for_plot(signal, sr, n_mfcc=13):
    """
    MFCC versi librosa hanya untuk visualisasi heatmap.
    """
    mfcc_feat = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc_feat  # (n_mfcc, frames)


# ======================================================
# Load models
# ======================================================
def load_models(model_dir="model"):
    cnn_path = os.path.join(model_dir, "cnn.keras")
    rf_path = os.path.join(model_dir, "random_forest.joblib")
    svm_path = os.path.join(model_dir, "svm.joblib")

    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"Tidak menemukan {cnn_path}")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Tidak menemukan {rf_path}")
    if not os.path.exists(svm_path):
        raise FileNotFoundError(f"Tidak menemukan {svm_path}")

    cnn_model = load_model(cnn_path)
    rf_model = joblib.load(rf_path)
    svm_model = joblib.load(svm_path)

    return cnn_model, rf_model, svm_model


# ======================================================
# Inference untuk satu file
# ======================================================
def predict_single_file(
    file_path, cnn_model, rf_model, svm_model, params=parameters
):
    # 1. preprocess
    signal, sr = load_and_preprocess_audio(file_path, params)

    # 2. features untuk model
    mfcc_psf = extract_mfcc_psf(signal, sr, params)  # (n_mfcc, max_pad_len)
    mfcc_plot = extract_mfcc_librosa_for_plot(signal, sr, n_mfcc=params["n_mfcc"])

    # 3. bentuk input untuk model
    # CNN: (1, n_mfcc, time, 1)
    X_cnn = np.expand_dims(mfcc_psf, axis=(0, -1))  # (1, n_mfcc, T, 1)

    # RF & SVM: flatten (1, n_mfcc * time)
    n_mfcc, T = mfcc_psf.shape
    X_flat = mfcc_psf.reshape(1, n_mfcc * T)

    # 4. prediction CNN (softmax probabilitas)
    cnn_probs = cnn_model.predict(X_cnn, verbose=0)[0]  # (n_classes,)

    # 5. prediction RF & SVM (pakai predict_proba kalau ada)
    def get_probs(clf, X):
        try:
            probs = clf.predict_proba(X)[0]
        except Exception:
            # fallback: hard prediction -> one-hot
            pred = clf.predict(X)[0]
            n_classes = len(CLASS_NAMES)
            probs = np.zeros(n_classes, dtype=float)
            if int(pred) < n_classes:
                probs[int(pred)] = 1.0
        return probs

    rf_probs = get_probs(rf_model, X_flat)
    svm_probs = get_probs(svm_model, X_flat)

    # 6. samakan panjangnya dgn CLASS_NAMES (jaga-jaga)
    def align_probs(probs):
        n_class = len(CLASS_NAMES)
        if len(probs) < n_class:
            probs = np.pad(probs, (0, n_class - len(probs)))
        elif len(probs) > n_class:
            probs = probs[:n_class]
        return probs

    cnn_probs = align_probs(cnn_probs)
    rf_probs = align_probs(rf_probs)
    svm_probs = align_probs(svm_probs)

    return {
        "signal": signal,
        "sr": sr,
        "mfcc_plot": mfcc_plot,
        "cnn_probs": cnn_probs,
        "rf_probs": rf_probs,
        "svm_probs": svm_probs,
    }


# ======================================================
# GUI Tkinter
# ======================================================
class AudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Inference GUI")

        # Load models once
        try:
            self.cnn_model, self.rf_model, self.svm_model = load_models("model")
        except Exception as e:
            messagebox.showerror("Error load model", str(e))
            raise

        # ==== TOP: button & label file ====
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.btn_load = tk.Button(
            top_frame, text="Pilih WAV...", command=self.load_file
        )
        self.btn_load.pack(side=tk.LEFT)

        self.lbl_file = tk.Label(top_frame, text="Tidak ada file", anchor="w")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        # ==== MIDDLE: Matplotlib Figure ====
        plot_frame = tk.Frame(root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ==== BOTTOM: Probabilities ====
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.text_probs = tk.Text(bottom_frame, height=10, width=80)
        self.text_probs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(bottom_frame, command=self.text_probs.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_probs.config(yscrollcommand=scrollbar.set)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Pilih file WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not file_path:
            return

        self.lbl_file.config(text=os.path.basename(file_path))

        try:
            result = predict_single_file(
                file_path, self.cnn_model, self.rf_model, self.svm_model
            )
        except Exception as e:
            messagebox.showerror("Error saat inference", str(e))
            return

        self.update_plots(result["signal"], result["sr"], result["mfcc_plot"])
        self.update_probs(
            result["cnn_probs"], result["rf_probs"], result["svm_probs"]
        )

    def update_plots(self, signal, sr, mfcc_feat):
        # Clear figure
        self.fig.clear()

        # Waveform
        ax1 = self.fig.add_subplot(1, 2, 1)
        t = np.arange(len(signal)) / sr
        ax1.plot(t, signal)
        ax1.set_title("Waveform")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")

        # MFCC heatmap
        ax2 = self.fig.add_subplot(1, 2, 2)
        im = ax2.imshow(
            mfcc_feat,
            aspect="auto",
            origin="lower",
        )
        ax2.set_title("MFCC Heatmap")
        ax2.set_xlabel("Frames")
        ax2.set_ylabel("MFCC Coefficients")

        self.fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_probs(self, cnn_probs, rf_probs, svm_probs):
        self.text_probs.delete("1.0", tk.END)

        def format_probs(model_name, probs):
            lines = [f"{model_name}:"]
            total = np.sum(probs)
            if total <= 0:
                total = 1.0
            for cls, p in zip(CLASS_NAMES, probs):
                pct = 100.0 * p / total
                lines.append(f"  {cls:12s}: {pct:6.2f}%")
            best_idx = int(np.argmax(probs))
            lines.append(f"  -> Prediksi: {CLASS_NAMES[best_idx]}")
            lines.append("")
            return "\n".join(lines)

        text = ""
        text += format_probs("CNN", cnn_probs)
        text += format_probs("Random Forest", rf_probs)
        text += format_probs("SVM", svm_probs)

        self.text_probs.insert(tk.END, text)


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioGUI(root)
    root.mainloop()
