import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
import os, warnings, shutil
warnings.filterwarnings('ignore')

BASE = '/home/lohit_vijayabaskar/seizure_project'
OUT  = '/home/lohit_vijayabaskar/AMD_Hackathon_Submission/results'
os.makedirs(OUT, exist_ok=True)

print("\n" + "="*60)
print("  AMD Hackathon -- Seizure Detection on FPGA")
print("="*60)

co = {}
_add_supported_quantized_objects(co)
model = tf.keras.models.load_model(
    f'{BASE}/model_real/qkeras_real_model.h5', custom_objects=co)
X_test = np.load(f'{BASE}/X_test_real.npy')
y_test = np.load(f'{BASE}/y_test_real.npy')
X_all  = np.load(f'{BASE}/X_real.npy')
y_all  = np.load(f'{BASE}/y_real.npy')

y_pred = model.predict(X_test, verbose=0)
acc = accuracy_score(np.argmax(y_test,1), np.argmax(y_pred,1))
auc = roc_auc_score(y_test[:,1], y_pred[:,1])
print(f"\n  Accuracy : {acc*100:.2f}%")
print(f"  AUC      : {auc:.4f}")
print(f"  Latency  : 0.27 us (FPGA @ 100 MHz)")

fig = plt.figure(figsize=(18,12))
fig.patch.set_facecolor('#0f0f1a')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
BG='#1a1a2e'; GRID='#2d2d4e'; PURPLE='#8b5cf6'; CYAN='#06b6d4'
RED='#ef4444'; GREEN='#22c55e'; YELLOW='#f59e0b'

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#9ca3af')
    ax.xaxis.label.set_color('#9ca3af')
    ax.yaxis.label.set_color('#9ca3af')
    for s in ax.spines.values(): s.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.5, linestyle='--')

ax1 = fig.add_subplot(gs[0,0])
fpr,tpr,_ = roc_curve(y_test[:,1], y_pred[:,1])
ax1.plot(fpr,tpr,color=PURPLE,lw=2.5,label=f'AUC={auc:.3f}')
ax1.plot([0,1],[0,1],color=GRID,lw=1,linestyle='--')
ax1.fill_between(fpr,tpr,alpha=0.1,color=PURPLE)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right',facecolor=BG,labelcolor='white',fontsize=9)
style_ax(ax1,'ROC Curve -- Seizure Detection')

ax2 = fig.add_subplot(gs[0,1:])
t = np.arange(len(y_pred))*2
ax2.fill_between(t, y_pred[:,1], alpha=0.3, color=CYAN)
ax2.plot(t, y_pred[:,1], color=CYAN, lw=1.5)
for tx in t[y_test[:,1]==1]: ax2.axvline(x=tx, color=RED, alpha=0.7, lw=1.5)
ax2.axhline(y=0.5, color=YELLOW, lw=1, linestyle='--', alpha=0.7)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Seizure Probability')
ax2.set_ylim(0,1.1)
from matplotlib.lines import Line2D
ax2.legend(handles=[
    Line2D([0],[0],color=CYAN,lw=2,label='Seizure Probability'),
    Line2D([0],[0],color=RED,lw=2,label='True Seizure Onset'),
    Line2D([0],[0],color=YELLOW,lw=1,linestyle='--',label='Threshold=0.5')],
    facecolor=BG,labelcolor='white',fontsize=9)
style_ax(ax2,'Seizure Probability Timeline -- Test Set')

ax3 = fig.add_subplot(gs[1,0])
cm = confusion_matrix(np.argmax(y_test,1), np.argmax(y_pred,1))
ax3.imshow(cm, cmap='Purples')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Non-seizure','Seizure'],color='white',fontsize=9)
ax3.set_yticklabels(['Non-seizure','Seizure'],color='white',fontsize=9)
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
for i in range(2):
    for j in range(2):
        ax3.text(j,i,str(cm[i,j]),ha='center',va='center',
                 color='white',fontsize=14,fontweight='bold')
style_ax(ax3,'Confusion Matrix')

ax4 = fig.add_subplot(gs[1,1])
sz = y_all==1; ns = y_all==0
sm = np.mean(X_all[sz,:8],axis=0)
nm = np.mean(X_all[ns,:8],axis=0)
x = np.arange(8); w=0.35
ax4.bar(x-w/2, nm, w, color=CYAN, alpha=0.8, label='Non-seizure')
ax4.bar(x+w/2, sm, w, color=RED, alpha=0.8, label='Seizure')
ax4.set_xticks(x)
ax4.set_xticklabels([f'B{i+1}' for i in range(8)])
ax4.set_xlabel('Frequency Band')
ax4.set_ylabel('Mean Energy')
ax4.legend(facecolor=BG,labelcolor='white',fontsize=9)
style_ax(ax4,'EEG Band Energy: Seizure vs Non-Seizure')

ax5 = fig.add_subplot(gs[1,2])
ax5.set_facecolor(BG); ax5.axis('off')
ax5.set_title('FPGA Pipeline Summary',color='white',fontsize=11,fontweight='bold',pad=10)
steps = [
    ("EEG Input","16 features (8 bands x 2 ch)",CYAN),
    ("QKeras MLP","64->32->32, 6-bit, 75% pruned",PURPLE),
    ("hls4ml","C++ HLS firmware",YELLOW),
    ("FPGA","Artix-7 200T, 0.27us latency",GREEN),
    ("Output","Seizure / Non-seizure | 99.17%",RED)
]
for i,(t2,d,c) in enumerate(steps):
    y2 = 9-i*1.8
    ax5.add_patch(plt.Rectangle((0.3,y2-0.5),9.4,1.2,
        facecolor=c,alpha=0.15,edgecolor=c,linewidth=1.5))
    ax5.text(5,y2+0.1,t2,ha='center',va='center',
             color=c,fontsize=10,fontweight='bold')
    ax5.text(5,y2-0.2,d,ha='center',va='center',
             color='#d1d5db',fontsize=8)
ax5.set_xlim(0,10); ax5.set_ylim(0,10)

fig.suptitle(
    'Epileptic Seizure Detection on FPGA  |  AMD Hackathon Submission\n'
    'CHB-MIT EEG  |  QKeras 6-bit  |  hls4ml  |  Vivado 2019.2',
    color='white',fontsize=13,fontweight='bold',y=0.98)

out_path = f'{OUT}/hackathon_results.png'
plt.savefig(out_path,dpi=150,bbox_inches='tight',
            facecolor='#0f0f1a',edgecolor='none')
print(f"\n  Graph saved: {out_path}")

desktop = "/mnt/c/Users/Lohit Vijayabaskar/Desktop/hackathon_results.png"
try:
    shutil.copy(out_path, desktop)
    print(f"  Also on Desktop: hackathon_results.png")
except Exception as e:
    print(f"  Copy manually from: {out_path}")
print("\n  Demo complete!\n")
