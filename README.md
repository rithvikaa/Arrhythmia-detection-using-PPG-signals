# 🫀 Arrhythmia Detection Using PPG Signals

This project focuses on **detecting cardiac arrhythmia using Photoplethysmography (PPG) signals**.  
It includes dataset preprocessing, signal filtering, and a deep learning model combining **ResNet, BiGRU, and Attention mechanisms** for classification.

---

## 📂 Project Structure

```
├── PPG_Dataset_under_25MB.csv   # Optimized dataset (<25MB)
├── aer.py                      # Model training & evaluation script
├── best_arrhythmia_model.pth   # Saved best model (generated after training)
└── README.md                   # Project documentation
```

---

## 📊 Dataset

- **File:** `PPG_Dataset_under_25MB.csv`
- **Type:** Photoplethysmography (PPG) time-series data
- **Labels:** Arrhythmia classes (e.g., MI / Normal)
- **Optimization:**  
  - Empty columns removed  
  - Numeric data optimized  
  - Dataset reduced to meet **<25MB upload constraints**

This makes the dataset suitable for:
- GitHub uploads  
- Assignments  
- Model prototyping  

---

## ⚙️ Preprocessing Steps

The following preprocessing is applied in `aer.py`:

1. **Smoothing** using Savitzky–Golay filter  
2. **Bandpass filtering** (0.5–8 Hz) to remove noise  
3. **Standardization** using `StandardScaler`  
4. **Reshaping** for PyTorch CNN input  

---

## 🧠 Model Architecture

The arrhythmia detection model consists of:

- **1D Convolution Layer**
- **Residual Blocks (ResNet)**
- **Bidirectional GRU**
- **Attention Layer**
- **Fully Connected Output Layer**

This architecture captures:
- Local signal patterns (CNN)
- Temporal dependencies (BiGRU)
- Important time steps (Attention)

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install pandas numpy scipy scikit-learn torch tqdm
```

### 2️⃣ Run the Model
```bash
python aer.py
```

---

## 📈 Training Details

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Batch Size:** 32  
- **Train/Test Split:** 80/20 (stratified)  
- **Epochs:** 2 (can be increased)

The model with the **best validation accuracy** is automatically saved as:
```
best_arrhythmia_model.pth
```

---

## 🧪 Output

After training, the script:
- Evaluates the model on test data
- Prints **true vs predicted labels**
- Indicates whether **arrhythmia is detected** for each sample

---

## 🎯 Use Cases

- Academic projects  
- Biomedical signal processing  
- Deep learning experimentation  
- Arrhythmia classification research  

---

## ✍️ Author

**Rithvika**
