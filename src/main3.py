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


# Improved feature extraction with better normalization
def extract_robust_features(file_path, n_mfcc=40, n_mels=64, hop_length=512):
    """
    Extract robust audio features with better preprocessing
    """
    try:
        # Load audio with better parameters
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0)

        # Pre-emphasis to enhance high frequencies
        audio = librosa.effects.preemphasis(audio)

        # Ensure consistent length
        target_length = sr * 3
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='reflect')
        else:
            audio = audio[:target_length]

        features = []

        # MFCC with delta features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))
        features.extend(np.mean(mfcc_delta2, axis=1))
        features.extend(np.std(mfcc_delta2, axis=1))

        # Mel-spectrogram (log scale)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.mean(mel_spec_db, axis=1))
        features.extend(np.std(mel_spec_db, axis=1))

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)

        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
        features.extend([np.mean(zcr), np.std(zcr)])

        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)
        features.extend([np.mean(rms), np.std(rms)])

        # Harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        features.extend([np.mean(harmonic), np.std(harmonic)])
        features.extend([np.mean(percussive), np.std(percussive)])

        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(tempo)

        return np.array(features)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


# Simplified and more robust model architecture
def create_simpler_model(input_shape, num_classes):
    """
    Create a simpler but more effective model to prevent overfitting
    """
    model = keras.Sequential([
        # Input layer with L2 regularization
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),  # Increased dropout

        # First hidden layer
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Second hidden layer
        layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Third hidden layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Alternative: 1D CNN model
def create_1d_cnn_model(input_shape, num_classes):
    """
    1D CNN model that might work better for audio features
    """
    model = keras.Sequential([
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        # First conv block
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        # Second conv block
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),

        # Third conv block
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),

        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Class weight calculation for imbalanced data
def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced datasets
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


# Advanced training with better callbacks
def get_training_callbacks():
    """
    Returns comprehensive callbacks for better training
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]


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

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion_code = file.split('-')[2]

                if emotion_code in emotions:
                    feature = extract_robust_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotions[emotion_code])

    print(f"Extracted features from {len(features)} files")

    # Convert to arrays
    X = np.array(features)
    y = np.array(labels)

    # Check for class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for emotion, count in zip(unique, counts):
        print(f"{emotion}: {count}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Calculate class weights
    class_weights = calculate_class_weights(y_encoded)
    print("Class weights:", class_weights)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature shape: {X_scaled.shape}")

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Try simpler model first
    print("\nCreating model...")
    model = create_simpler_model(X_train.shape[1], len(le.classes_))

    # Custom optimizer with lower learning rate
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0005,  # Reduced learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    # Train with class weights
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,  # Try smaller batch size
        class_weight=class_weights,
        callbacks=get_training_callbacks(),
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history.get('lr', [0.001] * len(history.history['loss'])), label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save the model
    model.save('emotion_classification_fixed.h5')
    print("Model saved as 'emotion_classification_fixed.h5'")


# Alternative approach: Try different models
def experiment_with_models(X, y):
    """
    Experiment with different model architectures
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

