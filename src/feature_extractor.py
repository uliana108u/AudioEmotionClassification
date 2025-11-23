import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.expected_feature_length = None

    def extract_features(self, file_paths, labels):
        features = []
        valid_labels = []
        valid_paths = []

        print("Extracting features from audio files...")

        for i, file_path in enumerate(file_paths):
            if i % 50 == 0:
                print(f"Processing {i}/{len(file_paths)}...")

            feature = self.extract_features_from_file(file_path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(labels[i])
                valid_paths.append(file_path)

        if len(features) == 0:
            raise ValueError("No features extracted! Check your audio files.")

        feature_lengths = [len(f) for f in features]
        print(
            f"Feature lengths: min={min(feature_lengths)}, max={max(feature_lengths)}, mean={np.mean(feature_lengths):.1f}")

        # Find the most common feature length
        unique_lengths, counts = np.unique(feature_lengths, return_counts=True)
        most_common_length = unique_lengths[np.argmax(counts)]
        print(f"Most common feature length: {most_common_length} (count: {max(counts)})")

        # Filter features to only include those with the most common length
        filtered_features = []
        filtered_labels = []
        inconsistent_files = []

        for i, (feature, label, file_path) in enumerate(zip(features, valid_labels, valid_paths)):
            if len(feature) == most_common_length:
                filtered_features.append(feature)
                filtered_labels.append(label)
            else:
                inconsistent_files.append((file_path, len(feature)))

        if inconsistent_files:
            print(f"Removed {len(inconsistent_files)} files with inconsistent feature lengths:")
            for file_path, length in inconsistent_files[:5]:  # Show first 5
                print(f"   - {os.path.basename(file_path)}: {length} features")
            if len(inconsistent_files) > 5:
                print(f"   ... and {len(inconsistent_files) - 5} more")

        features_array = np.array(filtered_features)
        self.expected_feature_length = most_common_length

        print(f"Final feature array shape: {features_array.shape}")

        labels_encoded = self.label_encoder.fit_transform(filtered_labels)

        print("Scaling features...")
        features_scaled = self.scaler.fit_transform(features_array)
        self.is_fitted = True

        print(f"Successfully processed {len(features_scaled)} files")
        print(f"Feature shape: {features_scaled.shape}")
        print(f"Labels: {len(labels_encoded)} samples")

        return features_scaled, labels_encoded

    def extract_features_from_file(self, file_path):
        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.config.data.sampling_rate,
                duration=self.config.data.duration
            )

            target_length = int(self.config.data.sampling_rate * self.config.data.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]

            audio = librosa.effects.preemphasis(audio)

            features = self._extract_all_features(audio, sr)
            return features

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            return None

    def _extract_all_features(self, audio, sr):
        features = []

        # MFCC features - FIXED DIMENSION
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr,
            n_mfcc=self.config.features.n_mfcc,
            hop_length=self.config.features.hop_length
        )
        features.extend(np.mean(mfcc, axis=1))  # n_mfcc features
        features.extend(np.std(mfcc, axis=1))  # n_mfcc features

        # Delta features - FIXED DIMENSION
        if getattr(self.config.features, 'include_delta', True):
            mfcc_delta = librosa.feature.delta(mfcc)
            features.extend(np.mean(mfcc_delta, axis=1))  # n_mfcc features
            features.extend(np.std(mfcc_delta, axis=1))  # n_mfcc features

        # Delta-delta features - FIXED DIMENSION
        if getattr(self.config.features, 'include_delta_delta', True):
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features.extend(np.mean(mfcc_delta2, axis=1))  # n_mfcc features
            features.extend(np.std(mfcc_delta2, axis=1))  # n_mfcc features

        # Mel-spectrogram features - FIXED DIMENSION
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=getattr(self.config.features, 'n_mels', 64),
            hop_length=self.config.features.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.mean(mel_spec_db, axis=1))  # n_mels features
        features.extend(np.std(mel_spec_db, axis=1))  # n_mels features

        # Spectral features - FIXED DIMENSION (always 2 each)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.config.features.hop_length
        )
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.config.features.hop_length
        )
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=self.config.features.hop_length
        )
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

        # Temporal features - FIXED DIMENSION (always 2 each)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.config.features.hop_length
        )
        features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])

        rms = librosa.feature.rms(y=audio, hop_length=self.config.features.hop_length)
        features.extend([np.mean(rms), np.std(rms)])

        # Chroma features - FIXED DIMENSION (always 24)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=self.config.features.hop_length)
        features.extend(np.mean(chroma, axis=1))  # 12 features
        features.extend(np.std(chroma, axis=1))  # 12 features

        return np.array(features)

    def calculate_expected_feature_length(self):
        n_mfcc = getattr(self.config.features, 'n_mfcc', 40)
        n_mels = getattr(self.config.features, 'n_mels', 64)

        total_length = 0

        # MFCC: mean + std
        total_length += n_mfcc * 2

        # Delta features
        if getattr(self.config.features, 'include_delta', True):
            total_length += n_mfcc * 2

        # Delta-delta features
        if getattr(self.config.features, 'include_delta_delta', True):
            total_length += n_mfcc * 2

        # Mel-spectrogram: mean + std
        total_length += n_mels * 2

        # Spectral features: 2 each for centroid, rolloff, bandwidth
        total_length += 2 * 3

        # Temporal features: 2 each for ZCR, RMS
        total_length += 2 * 2

        # Chroma features: mean + std (12 each)
        total_length += 12 * 2

        return total_length

    def save_preprocessor(self, filepath):
        if self.is_fitted:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'expected_feature_length': self.expected_feature_length
                }, f)

    def load_preprocessor(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.expected_feature_length = data.get('expected_feature_length')
            self.is_fitted = True

