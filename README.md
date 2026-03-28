# Epileptic Seizure Detection on FPGA
## AMD Hackathon Submission - Round 1
### Team: Lohit Vijayabaskar & Aryaman

---

## Project Overview

A real-time epileptic seizure detector implemented on a Xilinx FPGA using
machine learning. The neural network is trained on real EEG recordings from
the CHB-MIT dataset and compiled to FPGA hardware using hls4ml and Vivado HLS.

---

## Quick Start (Run the Demo)

### Requirements
- Ubuntu / WSL2 (Windows Subsystem for Linux)
- Anaconda or Miniconda
- Python 3.10

### Step 1 - Create and activate environment
    conda create -n seizure_demo python=3.10 -y
    conda activate seizure_demo

### Step 2 - Install dependencies
    pip install tensorflow==2.12.0
    pip install qkeras
    pip install tensorflow-model-optimization
    pip install hls4ml==0.4.0
    pip install numpy==1.23.5
    pip install scikit-learn
    pip install matplotlib
    pip install mne

### Step 3 - Set up data paths
Edit the BASE variable at the top of scripts/demo.py to point to
the location of your data folder:

    BASE = '/path/to/AMD_Hackathon_Submission/data'

Also update OUT to point to your results folder:

    OUT = '/path/to/AMD_Hackathon_Submission/results'

### Step 4 - Run the basic demo
    python3 scripts/demo.py

### Step 5 - Run the extended demo (more graphs)
    python3 scripts/demo_extended.py

Output graphs will be saved to the results/ folder.

---

## Folder Structure

    AMD_Hackathon_Submission/
    |-- models/
    |   |-- qkeras_real_model.h5       <- Trained on real CHB-MIT EEG (USE THIS)
    |   `-- qkeras_synthetic_model.h5  <- Trained on synthetic data
    |
    |-- data/
    |   |-- X_real.npy                 <- EEG features (5397 windows x 16 features)
    |   |-- y_real.npy                 <- Labels (0=non-seizure, 1=seizure)
    |   |-- X_test_real.npy            <- Test features (1080 windows)
    |   `-- y_test_real.npy            <- Test labels one-hot encoded
    |
    |-- bitstreams/
    |   |-- myproject_real.bit         <- FPGA bitstream (real EEG model) *
    |   `-- myproject_synthetic.bit    <- FPGA bitstream (synthetic model)
    |
    |-- results/
    |   |-- hackathon_results.png          <- Main results graph
    |   |-- hackathon_extended_results.png <- Extended 8-panel results
    |   `-- hackathon_training_history.png <- Training loss and accuracy curves
    |
    |-- scripts/
    |   |-- demo.py                    <- Basic demo (5 graphs)
    |   |-- demo_extended.py           <- Extended demo (8 graphs + training)
    |   `-- seizure_hls4ml_project.py  <- Full training pipeline
    |
    `-- README.md                      <- This file

* Program myproject_real.bit onto a Nexys A7-100T board for Round 2

---

## Key Results

    Metric                  Value
    ----------------------  ----------
    Overall Accuracy        99.17%
    AUC Score               0.9913
    Average Precision       0.5576
    FPGA Inference Latency  0.27 us
    Clock Speed             100 MHz
    DSP Slices Used         45 / 740
    LUTs Used               10382 / 134600
    BRAMs Used              1 / 365
    Model Precision         6-bit quantized
    Model Sparsity          75% pruned

---

## Pipeline

    Real EEG Data (CHB-MIT Dataset)
            |
            v
    Feature Extraction
    - 2-second sliding windows at 256 Hz
    - 8-band filterbank (0.5 to 24.5 Hz)
    - 2 EEG channels -> 16 features per window
            |
            v
    QKeras Model Training
    - Architecture: 64 -> 32 -> 32 -> 2
    - 6-bit weight quantization
    - 75% weight pruning
    - Class weighting (1:153 seizure ratio)
            |
            v
    hls4ml Conversion
    - Keras model -> C++ HLS firmware
    - Vivado HLS backend
            |
            v
    Vivado HLS C-Synthesis
    - RTL generation (VHDL)
    - Target: Xilinx Artix-7 200T
            |
            v
    Vivado Implementation
    - Synthesis, Place and Route
    - Timing closure at 100 MHz
            |
            v
    FPGA Bitstream (.bit file)
    - Ready for deployment on Nexys A7

---

## Dataset

CHB-MIT Scalp EEG Database (PhysioNet)
- Patient: chb01 (pediatric epilepsy)
- Files used: chb01_01.edf, chb01_03.edf, chb01_04.edf
- Total windows: 5397 (2-second epochs)
- Seizure windows: 35
- Non-seizure windows: 5362
- Train/Test split: 80/20

Seizure annotations from chb01-summary.txt:
- chb01_03.edf: seizure at 2996-3036 seconds
- chb01_04.edf: seizure at 1467-1494 seconds

---

## Model Architecture

Based on: MIT Thesis "Application of ML to Epileptic Seizure Onset Detection"
          (Shoeb, 2009)

    Input (16 features)
        |
        v
    QDense(64) + QReLU (6-bit)
        |
        v
    QDense(32) + QReLU (6-bit)
        |
        v
    QDense(32) + QReLU (6-bit)
        |
        v
    Dense(2) + Softmax
        |
        v
    Output: [P(non-seizure), P(seizure)]

---

## Tools and Versions

    Tool                        Version   Purpose
    --------------------------  --------  ---------------------------
    TensorFlow                  2.12.0    Model training
    QKeras                      Latest    6-bit quantization
    tensorflow-model-optim      Latest    75% weight pruning
    hls4ml                      0.4.0     Keras to HLS C++ conversion
    Vivado HLS                  2019.2    C++ to RTL synthesis
    Vivado                      2019.2    Place, Route, Bitstream
    MNE-Python                  Latest    EEG feature extraction
    scikit-learn                Latest    Metrics and preprocessing
    NumPy                       1.23.5    Data handling
    Matplotlib                  Latest    Visualization

---

## Round 2 Plans (FPGA Board Deployment)

1. Program myproject_real.bit onto Nexys A7-100T board
   - Open Vivado Hardware Manager
   - Auto Connect to board
   - Program Device with myproject_real.bit

2. Hardware interface
   - Feed 16 EEG features through fc1_input_V port (256-bit bus)
   - Read output from layer13_out_0_V (non-seizure score)
   - Read output from layer13_out_1_V (seizure score)
   - ap_clk: 100 MHz clock input
   - ap_rst: active high reset

3. Real-time pipeline
   - EEG amplifier -> ADC -> FPGA GPIO
   - 2-second epoch accumulation
   - 8-band filterbank in firmware
   - Seizure alert via LED or UART output

4. Recommended board
   - Nexys A7-100T (~$265) matches xc7a100t part exactly
   - PYNQ-Z2 (~$165) for Python-based deployment

---

## References

1. Shoeb, A. H. (2009). Application of Machine Learning to Epileptic Seizure
   Onset Detection and Treatment. MIT PhD Thesis.

2. Duarte, J. et al. (2018). Fast inference of deep neural networks in FPGAs
   for particle physics. JINST 13 P07027. (hls4ml paper)

3. Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
   Circulation 101(23). (CHB-MIT dataset)

4. Courbariaux, M. et al. (2016). Binarized Neural Networks.
   NeurIPS 2016. (Quantization reference)

---

## Contact

Lohit Vijayabaskar
AMD Hackathon 2026 - Round 1 Submission
