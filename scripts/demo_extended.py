import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                              confusion_matrix, precision_recall_curve,
                              average_precision_score)
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
import os, warnings, shutil
warnings.filterwarnings('ignore')

BASE = '/home/lohit_vijayabaskar/seizure_project'
OUT  = '/home/lohit_vijayabaskar/AMD_Hackathon_Submission/results'
os.makedirs(OUT, exist_ok=True)

print("\n" + "="*60)
print("  AMD Hackathon -- Extended Results Demo")
print("="*60)

# Load model and data
co = {}
_add_supported_quantized_objects(co)
model = tf.keras.models.load_model(
    f'{BASE}/model_real/qkeras_real_model.h5', custom_objects=co)
X_test = np.load(f'{BASE}/X_test_real.npy')
y_test = np.load(f'{BASE}/y_test_real.npy')
X_all  = np.load(f'{BASE}/X_real.npy')
y_all  = np.load(f'{BASE}/y_real.npy')
X_train = np.load(f'{BASE}/X_real.npy')

y_pred = model.predict(X_test, verbose=0)
acc = accuracy_score(np.argmax(y_test,1), np.argmax(y_pred,1))
auc = roc_auc_score(y_test[:,1], y_pred[:,1])
ap  = average_precision_score(y_test[:,1], y_pred[:,1])

print(f"  Accuracy : {acc*100:.2f}%")
print(f"  AUC      : {auc:.4f}")
print(f"  Avg Precision: {ap:.4f}")

# ── Colours ──────────────────────────────────────────────────
BG='#1a1a2e'; GRID='#2d2d4e'; PURPLE='#8b5cf6'; CYAN='#06b6d4'
RED='#ef4444'; GREEN='#22c55e'; YELLOW='#f59e0b'; ORANGE='#f97316'

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#9ca3af')
    ax.xaxis.label.set_color('#9ca3af')
    ax.yaxis.label.set_color('#9ca3af')
    for s in ax.spines.values(): s.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.5, linestyle='--')

# ── Figure 1: Extended Results (2x4 grid) ────────────────────
fig1 = plt.figure(figsize=(22, 12))
fig1.patch.set_facecolor('#0f0f1a')
gs = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.4, wspace=0.38)

# Plot 1: ROC Curve
ax1 = fig1.add_subplot(gs[0,0])
fpr,tpr,_ = roc_curve(y_test[:,1], y_pred[:,1])
ax1.plot(fpr,tpr,color=PURPLE,lw=2.5,label=f'AUC={auc:.3f}')
ax1.plot([0,1],[0,1],color=GRID,lw=1,linestyle='--')
ax1.fill_between(fpr,tpr,alpha=0.15,color=PURPLE)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right',facecolor=BG,labelcolor='white',fontsize=9)
style_ax(ax1,'ROC Curve (AUC=0.991)')

# Plot 2: Precision-Recall Curve
ax2 = fig1.add_subplot(gs[0,1])
prec, rec, _ = precision_recall_curve(y_test[:,1], y_pred[:,1])
ax2.plot(rec, prec, color=CYAN, lw=2.5, label=f'AP={ap:.3f}')
ax2.fill_between(rec, prec, alpha=0.15, color=CYAN)
ax2.axhline(y=35/1080, color=YELLOW, lw=1, linestyle='--',
            label=f'Baseline ({35/1080:.3f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_ylim(0, 1.05)
ax2.legend(loc='upper right', facecolor=BG, labelcolor='white', fontsize=9)
style_ax(ax2,'Precision-Recall Curve')

# Plot 3: Confusion Matrix
ax3 = fig1.add_subplot(gs[0,2])
cm = confusion_matrix(np.argmax(y_test,1), np.argmax(y_pred,1))
ax3.imshow(cm, cmap='Purples')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Non-sz','Seizure'],color='white',fontsize=9)
ax3.set_yticklabels(['Non-sz','Seizure'],color='white',fontsize=9)
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
for i in range(2):
    for j in range(2):
        ax3.text(j,i,str(cm[i,j]),ha='center',va='center',
                 color='white',fontsize=14,fontweight='bold')
style_ax(ax3,'Confusion Matrix')

# Plot 4: Model Comparison
ax4 = fig1.add_subplot(gs[0,3])
models_names = ['Baseline\nKeras', 'Pruned\n(75%)', 'QKeras\n6-bit']
accuracies   = [99.35, 99.26, 99.17]
aucs         = [0.985, 0.982, 0.991]
x = np.arange(3); w = 0.35
bars1 = ax4.bar(x-w/2, accuracies, w, color=PURPLE, alpha=0.85, label='Accuracy (%)')
bars2 = ax4.bar(x+w/2, [a*100 for a in aucs], w, color=CYAN, alpha=0.85, label='AUC x100')
ax4.set_xticks(x)
ax4.set_xticklabels(models_names, color='white', fontsize=9)
ax4.set_ylim(97, 100.5)
ax4.set_ylabel('Score')
for bar in bars1:
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f'{bar.get_height():.2f}', ha='center', va='bottom',
             color='white', fontsize=8)
for bar in bars2:
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f'{bar.get_height()/100:.3f}', ha='center', va='bottom',
             color='white', fontsize=8)
ax4.legend(facecolor=BG, labelcolor='white', fontsize=9)
style_ax(ax4,'Model Comparison')

# Plot 5: FPGA Resource Utilization
ax5 = fig1.add_subplot(gs[1,0])
resources    = ['LUTs', 'FFs', 'DSPs', 'BRAMs']
used         = [10382, 11467, 45, 1]
available    = [134600, 269200, 740, 365]
pct          = [u/a*100 for u,a in zip(used, available)]
colors       = [GREEN if p < 50 else YELLOW if p < 80 else RED for p in pct]
bars = ax5.barh(resources, pct, color=colors, alpha=0.85)
ax5.set_xlabel('Utilization (%)')
ax5.set_xlim(0, 100)
ax5.axvline(x=80, color=RED, lw=1, linestyle='--', alpha=0.5, label='80% limit')
for i, (bar, p, u, a) in enumerate(zip(bars, pct, used, available)):
    ax5.text(p+1, bar.get_y()+bar.get_height()/2,
             f'{u}/{a} ({p:.1f}%)', va='center', color='white', fontsize=8)
ax5.legend(facecolor=BG, labelcolor='white', fontsize=9)
style_ax(ax5,'FPGA Resource Utilization')

# Plot 6: Seizure Probability Timeline
ax6 = fig1.add_subplot(gs[1,1:3])
t = np.arange(len(y_pred))*2
ax6.fill_between(t, y_pred[:,1], alpha=0.3, color=CYAN)
ax6.plot(t, y_pred[:,1], color=CYAN, lw=1.5, label='Seizure Probability')
for tx in t[y_test[:,1]==1]:
    ax6.axvline(x=tx, color=RED, alpha=0.7, lw=1.5)
ax6.axhline(y=0.5, color=YELLOW, lw=1, linestyle='--', alpha=0.7, label='Threshold=0.5')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Seizure Probability')
ax6.set_ylim(0, 1.1)
from matplotlib.lines import Line2D
ax6.legend(handles=[
    Line2D([0],[0],color=CYAN,lw=2,label='Seizure Probability'),
    Line2D([0],[0],color=RED,lw=2,label='True Seizure Onset'),
    Line2D([0],[0],color=YELLOW,lw=1,linestyle='--',label='Threshold=0.5')],
    facecolor=BG, labelcolor='white', fontsize=9)
style_ax(ax6,'Seizure Probability Timeline -- Test Set')

# Plot 7: Feature Importance
ax7 = fig1.add_subplot(gs[1,3])
sz = y_all==1; ns = y_all==0
diff = np.abs(np.mean(X_all[sz], axis=0) - np.mean(X_all[ns], axis=0))
diff_norm = diff / diff.max()
feat_names = [f'B{i//2+1}-Ch{i%2+1}' for i in range(16)]
colors_fi = [PURPLE if v > 0.5 else CYAN if v > 0.25 else GRID for v in diff_norm]
ax7.barh(feat_names, diff_norm, color=colors_fi, alpha=0.85)
ax7.set_xlabel('Relative Importance')
ax7.set_xlim(0, 1.1)
style_ax(ax7,'Feature Importance\n(Seizure vs Non-seizure difference)')

fig1.suptitle(
    'Epileptic Seizure Detection on FPGA  |  AMD Hackathon Submission -- Extended Results\n'
    'CHB-MIT EEG  |  QKeras 6-bit  |  hls4ml  |  Vivado 2019.2  |  Artix-7 200T',
    color='white', fontsize=13, fontweight='bold', y=0.98)

out1 = f'{OUT}/hackathon_extended_results.png'
plt.savefig(out1, dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a', edgecolor='none')
print(f"\n  Graph 1 saved: {out1}")

# ── Figure 2: Training History ────────────────────────────────
print("\n  Generating training history...")

# Retrain briefly just to get history curves
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

X = np.load(f'{BASE}/X_real.npy')
y = np.load(f'{BASE}/y_real.npy')
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)
y_tr_cat = to_categorical(y_tr, 2)
y_te_cat = to_categorical(y_te, 2)

m2 = Sequential([
    QDense(64, input_shape=(16,), name='fc1',
           kernel_quantizer=quantized_bits(6,0,alpha=1),
           bias_quantizer=quantized_bits(6,0,alpha=1),
           kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
    QActivation(activation=quantized_relu(6), name='relu1'),
    QDense(32, name='fc2',
           kernel_quantizer=quantized_bits(6,0,alpha=1),
           bias_quantizer=quantized_bits(6,0,alpha=1),
           kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
    QActivation(activation=quantized_relu(6), name='relu2'),
    QDense(32, name='fc3',
           kernel_quantizer=quantized_bits(6,0,alpha=1),
           bias_quantizer=quantized_bits(6,0,alpha=1),
           kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
    QActivation(activation=quantized_relu(6), name='relu3'),
    Dense(2, name='output', kernel_initializer='lecun_uniform'),
    Activation('softmax', name='softmax'),
])
m2 = prune.prune_low_magnitude(m2, **{
    "pruning_schedule": pruning_schedule.ConstantSparsity(0.75, begin_step=200, frequency=50)})
m2.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
class_weight = {0: 1.0, 1: float(np.sum(y==0))/np.sum(y==1)}

hist = m2.fit(X_tr, y_tr_cat, batch_size=64, epochs=60,
              validation_split=0.2, class_weight=class_weight,
              callbacks=[pruning_callbacks.UpdatePruningStep(),
                         EarlyStopping(patience=15, restore_best_weights=True, verbose=0)],
              verbose=0)

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor('#0f0f1a')

ax_l = axes[0]; ax_a = axes[1]
epochs = range(1, len(hist.history['loss'])+1)

ax_l.plot(epochs, hist.history['loss'], color=PURPLE, lw=2, label='Train Loss')
ax_l.plot(epochs, hist.history['val_loss'], color=CYAN, lw=2, linestyle='--', label='Val Loss')
ax_l.set_xlabel('Epoch'); ax_l.set_ylabel('Loss')
ax_l.legend(facecolor=BG, labelcolor='white', fontsize=10)
style_ax(ax_l, 'Training & Validation Loss')

ax_a.plot(epochs, [v*100 for v in hist.history['accuracy']], color=GREEN, lw=2, label='Train Accuracy')
ax_a.plot(epochs, [v*100 for v in hist.history['val_accuracy']], color=YELLOW, lw=2, linestyle='--', label='Val Accuracy')
ax_a.set_xlabel('Epoch'); ax_a.set_ylabel('Accuracy (%)')
ax_a.legend(facecolor=BG, labelcolor='white', fontsize=10)
style_ax(ax_a, 'Training & Validation Accuracy')

fig2.suptitle('QKeras Model Training History  |  AMD Hackathon Submission',
              color='white', fontsize=13, fontweight='bold')

out2 = f'{OUT}/hackathon_training_history.png'
plt.savefig(out2, dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a', edgecolor='none')
print(f"  Graph 2 saved: {out2}")

# Copy both to Desktop
desktop = "/mnt/c/Users/Lohit Vijayabaskar/Desktop/"
try:
    shutil.copy(out1, desktop + 'hackathon_extended_results.png')
    shutil.copy(out2, desktop + 'hackathon_training_history.png')
    print(f"\n  Both graphs copied to Desktop!")
except:
    print(f"\n  Copy manually from: {OUT}/")

print("\n  Extended demo complete!\n")
