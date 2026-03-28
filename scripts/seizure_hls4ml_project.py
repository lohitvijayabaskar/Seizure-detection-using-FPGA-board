"""
=============================================================================
Epileptic Seizure Onset Detection with hls4ml — FPGA Deployment
=============================================================================
Based on:
  - MIT Thesis: "Application of ML to Epileptic Seizure Onset Detection"
    (Shoeb, 2009) — SVM + EEG spectral/spatial features
  - hls4ml Tutorial Parts 1–8

Architecture:
  - Input: 16 EEG spectral/spatial features (8 frequency bands × 2 channels)
  - Network: 3-layer MLP (64 → 32 → 32 → 2), QKeras quantized + pruned
  - Output: Binary classification [non-seizure, seizure]
  - Target FPGA: Xilinx Artix-7 (Vivado 2019.2 compatible)
  - Backend: Vivado HLS

Usage:
  python seizure_hls4ml_project.py

Requirements:
  pip install tensorflow==2.6 hls4ml qkeras tensorflow-model-optimization
  scikit-learn numpy matplotlib
  (Vivado 2019.2 must be on PATH for synthesis)
=============================================================================
"""

# =============================================================================
# PART 0 — Imports and Environment Setup
# =============================================================================
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)

# --- Point to Vivado 2019.2 ---
# Update this path to match your Vivado installation
VIVADO_PATH = r"C:\Xilinx\Vivado\2019.2"   # Windows host
# In WSL, this might be: /mnt/c/Xilinx/Vivado/2019.2
# If running from WSL:
# VIVADO_PATH = "/mnt/c/Xilinx/Vivado/2019.2"

if os.path.exists(VIVADO_PATH):
    os.environ['PATH'] = os.path.join(VIVADO_PATH, 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['XILINX_VIVADO'] = VIVADO_PATH
    print(f"[INFO] Vivado 2019.2 found at {VIVADO_PATH}")
else:
    print(f"[WARN] Vivado not found at {VIVADO_PATH}. HLS build/synthesis will be skipped.")

# FPGA part — Artix-7 (xc7a35t is on the Basys3; adjust for your board)
# Common choices:
#   Basys3:       xc7a35tcpg236-1
#   Nexys A7-100: xc7a100tcsg324-1
#   ZedBoard:     xc7z020clg484-1
#   PYNQ-Z2:      xc7z020clg400-1
FPGA_PART = 'xc7a100tcsg324-1'   # <-- Change to match YOUR board

# =============================================================================
# PART 1 — Dataset: Synthetic EEG-like features (thesis-inspired)
# =============================================================================
# The thesis uses 8-band filterbank energies across N=18 EEG channels.
# For a compact FPGA design we use 8 bands × 2 channels = 16 features,
# matching the hls4ml tutorial input dimension exactly.
#
# Feature vector layout (per Shoeb 2009, Section 3.2):
#   Features 0–7:  Spectral band energies (0.5–24 Hz) for channel 1
#   Features 8–15: Spectral band energies (0.5–24 Hz) for channel 2
#
# Here we generate a synthetic dataset that mimics the EEG statistics.
# Replace this section with real CHB-MIT or your own EEG features.

print("\n" + "="*60)
print("PART 1: Generating synthetic EEG feature dataset")
print("="*60)

N_SAMPLES  = 10000   # total samples
N_FEATURES = 16      # 8 bands × 2 channels
N_CLASSES  = 2       # 0=non-seizure, 1=seizure
SEIZURE_RATIO = 0.2  # 20% seizure (class imbalance, as in clinical data)

def generate_eeg_features(n_samples, n_features=16, seizure_ratio=0.2, seed=42):
    """
    Generate synthetic EEG spectral band features.
    Seizure class has elevated energy in lower bands (0.5–12 Hz, bands 0–4).
    Non-seizure class has energy concentrated in alpha/beta bands.
    """
    rng = np.random.RandomState(seed)
    n_seizure = int(n_samples * seizure_ratio)
    n_non     = n_samples - n_seizure

    # Non-seizure: energy peaks in bands 3–5 (alpha/beta, ~8–16 Hz)
    X_non = rng.randn(n_non, n_features) * 0.3
    X_non[:, 3:6] += rng.exponential(1.5, (n_non, 3))   # alpha/beta
    X_non[:, 11:14] += rng.exponential(1.2, (n_non, 3)) # channel 2 alpha

    # Seizure: energy peaks in bands 0–3 (delta/theta/alpha, 0.5–12 Hz)
    X_seiz = rng.randn(n_seizure, n_features) * 0.3
    X_seiz[:, 0:4] += rng.exponential(2.5, (n_seizure, 4))   # delta/theta
    X_seiz[:, 8:12] += rng.exponential(2.2, (n_seizure, 4))  # channel 2

    X = np.vstack([X_non, X_seiz])
    y = np.array([0]*n_non + [1]*n_seizure)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

X_raw, y_raw = generate_eeg_features(N_SAMPLES, N_FEATURES, SEIZURE_RATIO)
print(f"Dataset shape: X={X_raw.shape}, y={y_raw.shape}")
print(f"Class distribution: non-seizure={np.sum(y_raw==0)}, seizure={np.sum(y_raw==1)}")

# Train/test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=seed, stratify=y_raw
)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test       = scaler.transform(X_test)
X_test       = np.ascontiguousarray(X_test)   # required by hls4ml

# One-hot encode
y_train_cat = to_categorical(y_train_val, N_CLASSES)
y_test_cat  = to_categorical(y_test,      N_CLASSES)
classes     = np.array(['non_seizure', 'seizure'])

# Save for later parts
os.makedirs('seizure_project', exist_ok=True)
np.save('seizure_project/X_train_val.npy', X_train_val)
np.save('seizure_project/X_test.npy',      X_test)
np.save('seizure_project/y_train_val.npy', y_train_cat)
np.save('seizure_project/y_test.npy',      y_test_cat)
np.save('seizure_project/classes.npy',     classes)
print("Data saved to seizure_project/")

# =============================================================================
# PART 2 — Baseline Keras Model (Part 1 + 2 of tutorial)
# =============================================================================
print("\n" + "="*60)
print("PART 2: Training baseline Keras model")
print("="*60)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

def build_baseline_model(input_dim=16, n_classes=2):
    """
    3-layer MLP matching hls4ml tutorial architecture.
    Compact for FPGA: 64→32→32 neurons with ReLU activations.
    """
    model = Sequential([
        Dense(64, input_shape=(input_dim,), name='fc1',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)),
        Activation('relu', name='relu1'),
        Dense(32, name='fc2',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)),
        Activation('relu', name='relu2'),
        Dense(32, name='fc3',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)),
        Activation('relu', name='relu3'),
        Dense(n_classes, name='output',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)),
        Activation('softmax', name='softmax'),
    ])
    return model

model_base = build_baseline_model(N_FEATURES, N_CLASSES)
model_base.summary()

os.makedirs('seizure_project/model_1', exist_ok=True)
callbacks_base = [
    ModelCheckpoint('seizure_project/model_1/best_model.h5',
                    save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=20, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
]

model_base.compile(optimizer=Adam(lr=1e-4),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

hist_base = model_base.fit(
    X_train_val, y_train_cat,
    batch_size=512, epochs=100,
    validation_split=0.2,
    callbacks=callbacks_base,
    verbose=1,
)

from sklearn.metrics import accuracy_score, roc_auc_score
y_pred_base = model_base.predict(X_test)
acc_base = accuracy_score(np.argmax(y_test_cat, 1), np.argmax(y_pred_base, 1))
auc_base = roc_auc_score(y_test_cat[:, 1], y_pred_base[:, 1])
print(f"\nBaseline Keras Accuracy: {acc_base:.4f}  |  AUC: {auc_base:.4f}")

# =============================================================================
# PART 3 — Pruning (Part 3 of tutorial)
# =============================================================================
print("\n" + "="*60)
print("PART 3: Pruning (75% sparsity)")
print("="*60)

from tensorflow_model_optimization.python.core.sparsity.keras import (
    prune, pruning_callbacks, pruning_schedule
)
from tensorflow_model_optimization.sparsity.keras import strip_pruning

# Rebuild fresh model to wrap with pruning
model_to_prune = build_baseline_model(N_FEATURES, N_CLASSES)
model_to_prune.compile(optimizer=Adam(lr=1e-4),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

pruning_params = {
    "pruning_schedule": pruning_schedule.ConstantSparsity(
        0.75, begin_step=500, frequency=100
    )
}
model_pruned = prune.prune_low_magnitude(model_to_prune, **pruning_params)
model_pruned.compile(optimizer=Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

os.makedirs('seizure_project/model_2', exist_ok=True)
callbacks_prune = [
    pruning_callbacks.UpdatePruningStep(),
    ModelCheckpoint('seizure_project/model_2/best_model.h5',
                    save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=20, restore_best_weights=True),
]

model_pruned.fit(
    X_train_val, y_train_cat,
    batch_size=512, epochs=100,
    validation_split=0.2,
    callbacks=callbacks_prune,
    verbose=1,
)

model_pruned = strip_pruning(model_pruned)
y_pred_pruned = model_pruned.predict(X_test)
acc_pruned = accuracy_score(np.argmax(y_test_cat, 1), np.argmax(y_pred_pruned, 1))
print(f"Pruned Model Accuracy: {acc_pruned:.4f}")

# Check sparsity
w = model_pruned.layers[0].weights[0].numpy()
sparsity = np.sum(w == 0) / w.size
print(f"Layer 'fc1' sparsity: {sparsity*100:.1f}%")

# =============================================================================
# PART 4 — QKeras Quantization (Part 4 of tutorial)
# =============================================================================
print("\n" + "="*60)
print("PART 4: QKeras quantization (6-bit weights + activations)")
print("="*60)

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

def build_qkeras_model(input_dim=16, n_classes=2):
    """
    Quantized version: 6-bit weights/biases, 6-bit ReLU activations.
    This maps directly to ap_fixed<6,0> in HLS — very resource-efficient.
    """
    model = Sequential([
        QDense(64, input_shape=(input_dim,), name='fc1',
               kernel_quantizer=quantized_bits(6, 0, alpha=1),
               bias_quantizer=quantized_bits(6, 0, alpha=1),
               kernel_initializer='lecun_uniform',
               kernel_regularizer=l1(0.0001)),
        QActivation(activation=quantized_relu(6), name='relu1'),
        QDense(32, name='fc2',
               kernel_quantizer=quantized_bits(6, 0, alpha=1),
               bias_quantizer=quantized_bits(6, 0, alpha=1),
               kernel_initializer='lecun_uniform',
               kernel_regularizer=l1(0.0001)),
        QActivation(activation=quantized_relu(6), name='relu2'),
        QDense(32, name='fc3',
               kernel_quantizer=quantized_bits(6, 0, alpha=1),
               bias_quantizer=quantized_bits(6, 0, alpha=1),
               kernel_initializer='lecun_uniform',
               kernel_regularizer=l1(0.0001)),
        QActivation(activation=quantized_relu(6), name='relu3'),
        Dense(n_classes, name='output',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)),
        Activation('softmax', name='softmax'),
    ])
    return model

model_q = build_qkeras_model(N_FEATURES, N_CLASSES)

# Wrap with pruning
pruning_params_q = {
    "pruning_schedule": pruning_schedule.ConstantSparsity(
        0.75, begin_step=500, frequency=100
    )
}
model_q = prune.prune_low_magnitude(model_q, **pruning_params_q)
model_q.compile(optimizer=Adam(lr=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

os.makedirs('seizure_project/model_3', exist_ok=True)
callbacks_q = [
    pruning_callbacks.UpdatePruningStep(),
    ModelCheckpoint('seizure_project/model_3/best_model.h5',
                    save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=25, restore_best_weights=True),
]

model_q.fit(
    X_train_val, y_train_cat,
    batch_size=512, epochs=150,
    validation_split=0.2,
    callbacks=callbacks_q,
    verbose=1,
)

model_q = strip_pruning(model_q)
model_q.save('seizure_project/model_3/qkeras_model.h5')

y_pred_q = model_q.predict(X_test)
acc_q = accuracy_score(np.argmax(y_test_cat, 1), np.argmax(y_pred_q, 1))
auc_q = roc_auc_score(y_test_cat[:, 1], y_pred_q[:, 1])
print(f"QKeras Pruned Model Accuracy: {acc_q:.4f}  |  AUC: {auc_q:.4f}")

# =============================================================================
# PART 5 — hls4ml Conversion with Vivado Backend (Parts 1–4 of tutorial)
# =============================================================================
print("\n" + "="*60)
print("PART 5: Converting to hls4ml (Vivado backend)")
print("="*60)

import hls4ml

# Load real trained model instead of synthetic
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)
model_q = tf.keras.models.load_model(
    'seizure_project/model_real/qkeras_real_model.h5',
    custom_objects=co
)
print("Real EEG model loaded")

import hls4ml
config = hls4ml.utils.config_from_keras_model(model_q, granularity='name')
config['Model'] = {}
config['Model']['Part'] = FPGA_PART
config['Model']['ClockPeriod'] = 10
config['Model']['IOType'] = 'io_parallel'
config['Model']['Backend'] = 'Vivado'
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 1

hls_model = hls4ml.converters.convert_from_keras_model(
    model_q,
    hls_config=config,
    output_dir='seizure_project/model_real/hls4ml_prj',
)
print("hls4ml project generated!")


# =============================================================================
# PART 6 — C-Simulation (compile + predict)
# =============================================================================
print("\n" + "="*60)
print("PART 6: C-simulation (compile + predict)")
print("="*60)

try:
    hls_model.compile()   # requires g++ / ap_cint headers (works on Linux/WSL)
    y_hls = hls_model.predict(X_test)
    acc_hls = accuracy_score(np.argmax(y_test_cat, 1), np.argmax(y_hls, 1))
    print(f"hls4ml C-sim Accuracy: {acc_hls:.4f}")
except Exception as e:
    print(f"[WARN] C-sim skipped: {e}")
    print("  → Run from WSL or Linux where ap_cint headers are available.")
    acc_hls = None

# =============================================================================
# PART 7 — HLS Build (C-synthesis) and Vivado Synthesis
# =============================================================================
print("\n" + "="*60)
print("PART 7: HLS Build (C-synthesis → RTL)")
print("="*60)

try:
    # csim=False — skip re-running C-sim here; synth=True runs HLS synthesis
    hls_model.build(
        csim=False,     # set True if you want C-sim in HLS flow
        synth=True,     # run C-synthesis (generates RTL)
        cosim=False,    # set True for RTL co-simulation (slow)
        export=True,    # export IP core for Vivado block design
    )
    print("HLS synthesis complete. Check:")
    print("  seizure_project/model_3/hls4ml_prj/myproject_prj/solution1/syn/report/")
    hls4ml.report.read_vivado_report('seizure_project/model_3/hls4ml_prj')
except Exception as e:
    print(f"[WARN] HLS build skipped: {e}")
    print("  → Make sure Vivado 2019.2 is on PATH and XILINX_VIVADO is set.")

# =============================================================================
# PART 8 — Results Summary and Comparison
# =============================================================================
print("\n" + "="*60)
print("PART 8: Results Summary")
print("="*60)

print(f"\n  Baseline Keras accuracy  : {acc_base:.4f}  AUC: {auc_base:.4f}")
print(f"  Pruned Keras accuracy    : {acc_pruned:.4f}")
print(f"  QKeras (pruned+quant)    : {acc_q:.4f}  AUC: {auc_q:.4f}")
if acc_hls is not None:
    print(f"  hls4ml C-sim accuracy    : {acc_hls:.4f}")

# ---- ROC curve plot ----
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(7, 7))
for y_pred, label, ls in [
    (y_pred_base, 'Baseline Keras',       '-'),
    (y_pred_pruned, 'Pruned Keras',        '--'),
    (y_pred_q,     'QKeras Pruned+Quant', '-.'),
]:
    fpr, tpr, _ = roc_curve(y_test_cat[:, 1], y_pred[:, 1])
    auc_val = roc_auc_score(y_test_cat[:, 1], y_pred[:, 1])
    ax.plot(fpr, tpr, ls=ls, lw=2, label=f'{label} (AUC={auc_val:.3f})')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Seizure Detection ROC Curve')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('seizure_project/roc_curve.png', dpi=150)
print("\nROC curve saved to seizure_project/roc_curve.png")

# ---- Model size / resource estimate ----
total_params = model_q.count_params()
bits_per_param = 6  # 6-bit quantization
model_size_kb = (total_params * bits_per_param) / (8 * 1024)
print(f"\n  QKeras model total parameters : {total_params}")
print(f"  Estimated weight storage      : {model_size_kb:.2f} KB @ 6-bit")
print(f"  (75% pruned → ~{model_size_kb*0.25:.2f} KB effective)")

print("""
=============================================================================
NEXT STEPS FOR FPGA DEPLOYMENT (Vivado 2019.2 + WSL)
=============================================================================

1. Run this script from WSL (Ubuntu):
   $ python seizure_hls4ml_project.py

2. Verify FPGA_PART and VIVADO_PATH at the top of this file match your board.

3. After HLS synthesis, open Vivado 2019.2 and create a new project:
   - Add the exported IP: seizure_project/model_3/hls4ml_prj/
   - Create a Block Design, add the NN IP + AXI connections
   - Run Implementation → Generate Bitstream

4. For PYNQ boards, use VivadoAccelerator backend instead:
     backend='VivadoAccelerator'
     board='pynq-z2'    # or 'pynq-z1'

5. Resource report location (after build):
   seizure_project/model_3/hls4ml_prj/myproject_prj/solution1/syn/report/

6. To reduce resource usage:
   - Increase ReuseFactor (e.g., 4 or 8) in config
   - Reduce bit width (try ap_fixed<8,4> globally)
   - Further increase pruning sparsity (0.85 or 0.90)

7. CHB-MIT real EEG dataset:
   - Download: https://physionet.org/content/chbmit/1.0.0/
   - Use MNE-Python to extract 8-band filterbank features
   - Replace generate_eeg_features() with real feature extraction
=============================================================================
""")
