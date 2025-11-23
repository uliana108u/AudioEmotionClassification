# Audio Emotion Classification

A deep learning project for classifying emotions from audio speech using the RAVDESS dataset. The system achieves **92% accuracy** in classifying 8 different emotions from speech audio.

## Project Overview

This project implements a deep learning pipeline for audio emotion classification using:

- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Emotions**: 6 emotional categories (neutral, calm, happy, sad, angry, fearful)
- **Features**: MFCC, Mel-spectrogram, spectral, and temporal features
- **Model**: Advanced DNN with regularization and dropout
- **Accuracy**: 92% on test set

## Project Structure

```
AudioEmotionClassification/
│
├── data/
│   └── raw/                  # RAVDESS dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        
│   ├── feature_extractor.py  
│   ├── models/               
│   │   ├── __init__.py
│   │   ├── dnn_model.py
│   │   ├── cnn_model.py
│   │   └── lstm_model.py
│   ├── training/             
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/           
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/                
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
│
├── models/                   
│   ├── final_model.h5        # best model 
│   └── checkpoints/
│
├── configs/                  # Configuration files
│   └── default.yaml
│
├── requirements.txt
├── setup.py
├── main.py                   # Main training script
├── predict.py               
└── README.md
```

## Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation

1. Download the RAVDESS dataset and place it in `data/raw/`

3. Run training:
```bash
python main.py --config configs/default.yaml --experiment baseline
```

### Inference

Use the trained model to predict emotions from audio files:

```bash
python predict.py --audio path/to/audio.wav --model models/final_model.h5
```

## Configuration

The project uses YAML configuration files. Key settings in `configs/default.yaml`:

```yaml
data:
  sampling_rate: 22050
  duration: 3.0
  emotions:
    '01': 'neutral'
    '02': 'calm'
    '03': 'happy'
    '04': 'sad'
    '05': 'angry'
    '06': 'fearful'

features:
  n_mfcc: 40
  n_mels: 64
  include_delta: true

model:
  name: "advanced_dnn"
  hidden_layers: [256, 128, 64]
  dropout_rates: [0.4, 0.4, 0.3]

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0005
```

## Model Architecture

The system uses a sophisticated feature extraction pipeline and deep neural network:

### Feature Extraction
- **MFCCs**: 40 coefficients with delta and delta-delta features
- **Mel-spectrograms**: 64-band log-scaled spectrograms
- **Spectral features**: Centroid, rolloff, bandwidth
- **Temporal features**: ZCR, RMS energy
- **Chroma features**: 12-dimensional chromagram

### Neural Network
- **Architecture**: Feedforward DNN with 3 hidden layers
- **Regularization**: L2 regularization and dropout
- **Optimization**: Adam optimizer with learning rate scheduling
- **Prevention**: Early stopping and batch normalization

## Results

### Overall Performance
- **Accuracy**: 92%
- **Macro Average F1-Score**: 93%
- **Weighted Average F1-Score**: 92%

### Detailed Classification Report

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Neutral | 0.95      | 0.95 | 0.95 | 37 |
| Calm    | 0.97 | 0.95 | 0.96 | 37 |
| happy   | 0.80 | 0.89 | 0.85 | 37 |
| Sad     | 0.92 | 0.95 | 0.93 | 37 |
| Angry   | 1.00 | 1.00 | 1.00 | 18 |
| Fearful | 0.94 | 0.84 | 0.89 | 37 |

**Total Samples**: 203

## Performance Analysis

### Strengths
- **Excellent performance** on most emotion classes (85-100% F1-score)
- **Balanced performance** across different emotions
- **Robust feature engineering** capturing both spectral and temporal patterns
- **Effective regularization** preventing overfitting

### Areas for Improvement
- Class 2 (happy) shows slightly lower performance (85% F1-score)
- Class imbalance in Class 4 (angry) (only 18 samples)
- Potential for improving recall in Class 5


## Technical Details

### Data Preprocessing
- Audio resampling to 22.05 kHz
- Fixed-length segmentation (3 seconds)
- Pre-emphasis filtering
- Feature normalization using StandardScaler

### Training Strategy
- Stratified k-fold cross-validation
- Class weighting for imbalanced data
- Learning rate scheduling
- Early stopping with patience


## Output Files

After training, the following files are generated:

- `models/final_model.h5` - Trained model weights
- `models/preprocessor.pkl` - Feature scaler and label encoder
- `results/training_history.json` - Training metrics per epoch
- `results/training_plots.png` - Accuracy and loss curves
- `results/results_summary.txt` - Comprehensive training report
- `confusion_matrix.png` - Classification confusion matrix
