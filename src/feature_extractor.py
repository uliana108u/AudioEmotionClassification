import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def extract_features(self, file_paths, labels):
        """Extract features from all audio files"""
        features = []
        valid_labels = []

        for i, file_path in enumerate(file_paths):
            if i % 100 == 0:
                print(f"Processing file {i}/{len(file_paths)}")

            feature = self.extract_features_from_file(file_path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(labels[i])

        # Convert to arrays
        features = np.array(features)

        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(valid_labels)

        # Fit scaler and transform features
        features_scaled = self.scaler.fit_transform(features)
        self.is_fitted = True

        print(f"Extracted features from {len(features)} files")
        print(f"Feature shape: {features_scaled.shape}")

        return features_scaled, labels_encoded

    def extract_features_from_file(self, file_path):
        """Extract features from single audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(
                file_path,
                sr=self.config.data.sampling_rate,
                duration=self.config.data.duration
            )

            # Ensure consistent length
            target_length = int(self.config.data.sampling_rate * self.config.data.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='reflect')
            else:
                audio = audio[:target_length]

            features = self._extract_all_features(audio, sr)
            return features

        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return None

    def _extract_all_features(self, audio, sr):
        """Extract all audio features"""
        features = []

        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr,
            n_mfcc=self.config.features.n_mfcc,
            hop_length=self.config.features.hop_length
        )
        features.extend(self._get_statistics(mfcc))

        # Add delta features if configured
        if self.config.features.include_delta:
            mfcc_delta = librosa.feature.delta(mfcc)
            features.extend(self._get_statistics(mfcc_delta))

        if self.config.features.include_delta_delta:
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features.extend(self._get_statistics(mfcc_delta2))

        # Add other features (mel-spectrogram, spectral, temporal)
        features.extend(self._extract_mel_features(audio, sr))
        features.extend(self._extract_spectral_features(audio, sr))
        features.extend(self._extract_temporal_features(audio, sr))

        return np.array(features)

    def _get_statistics(self, feature_matrix):
        """Get mean and std of feature matrix"""
        return [np.mean(feature_matrix, axis=1), np.std(feature_matrix, axis=1)]

    def _extract_mel_features(self, audio, sr):
        """Extract mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=self.config.features.n_mels,
            hop_length=self.config.features.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return self._get_statistics(mel_spec_db)

    def _extract_spectral_features(self, audio, sr):
        """Extract spectral features"""
        features = []

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.config.features.hop_length
        )
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.config.features.hop_length
        )
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

        return features

    def _extract_temporal_features(self, audio, sr):
        """Extract temporal features"""
        features = []

        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.config.features.hop_length
        )
        features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])

        rms = librosa.feature.rms(y=audio, hop_length=self.config.features.hop_length)
        features.extend([np.mean(rms), np.std(rms)])

        return features

    def save_preprocessor(self, filepath):
        if self.is_fitted:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder
                }, f)

    def load_preprocessor(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.is_fitted = True
