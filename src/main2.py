import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Enhanced feature extraction
def extract_features(file_path, n_mfcc=40, n_mels=64, hop_length=512):
    """
    Extract comprehensive audio features
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0)  # Fixed duration

        # Ensure consistent length
        if len(audio) < sr * 3:
            audio = np.pad(audio, (0, max(0, sr * 3 - len(audio))))
        else:
            audio = audio[:sr * 3]

        features = []

        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)

        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_mean = np.mean(mel_spec, axis=1)
        mel_std = np.std(mel_spec, axis=1)
        features.extend(mel_mean)
        features.extend(mel_std)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        features.extend(chroma_mean)
        features.extend(chroma_std)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)

        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.extend(np.mean(spectral_contrast, axis=1))

        # Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
        features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])

        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)
        features.extend([np.mean(rms), np.std(rms)])

        return np.array(features)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


# Data augmentation functions
def augment_audio(audio, sr):
    """Apply audio augmentation"""
    augmented = []

    # Original
    augmented.append(audio)

    # Time stretching
    stretched_fast = librosa.effects.time_stretch(audio, rate=1.2)
    stretched_slow = librosa.effects.time_stretch(audio, rate=0.8)

    # Ensure same length
    min_len = min(len(audio), len(stretched_fast), len(stretched_slow))
    augmented.append(stretched_fast[:min_len])
    augmented.append(stretched_slow[:min_len])

    # Pitch shifting
    pitched_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    pitched_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
    augmented.append(pitched_up[:min_len])
    augmented.append(pitched_down[:min_len])

    # Add noise
    noise = np.random.normal(0, 0.005, audio.shape)
    augmented.append(audio + noise)

    return augmented


# Enhanced model architecture
def create_advanced_model(input_shape, num_classes):
    """
    Create an advanced neural network for emotion classification
    """
    model = keras.Sequential([
        # Input layer
        layers.Dense(512, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_cnn_model(input_shape, num_classes):
    """
    CNN model for spectrogram-like features
    """
    model = keras.Sequential([
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Main execution
def main():
    # Configuration
    DATA_PATH = "./data"
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    print("Loading and extracting features...")

    # Load and extract features
    features = []
    labels = []
    file_paths = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion_code = file.split('-')[2]

                if emotion_code in emotions:
                    feature = extract_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotions[emotion_code])
                        file_paths.append(file_path)

    print(f"Extracted features from {len(features)} files")

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Convert to arrays
    X = np.array(features)
    y = np.array(labels_encoded)

    print(f"Feature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Classes: {le.classes_}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    print("\nStarting cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n--- Fold {fold + 1} ---")

        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create and compile model
        model = create_advanced_model(X_train.shape[1], len(le.classes_))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_accuracy)
        print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")

    print(f"\nCross-validation completed!")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

    # Train final model on all data
    print("\nTraining final model on all data...")

    final_model = create_advanced_model(X_scaled.shape[1], len(le.classes_))
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = final_model.fit(
        X_scaled, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Save model
    final_model.save('emotion_classification_advanced.h5')
    print("Model saved as 'emotion_classification_advanced.h5'")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

