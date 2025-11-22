import argparse
import numpy as np
import librosa
import tensorflow as tf
from src.feature_extractor import FeatureExtractor
from src.utils.config import Config


class EmotionPredictor:
    def __init__(self, model_path, config_path='configs/default.yaml'):
        self.config = Config(config_path)
        self.model = tf.keras.models.load_model(model_path)
        self.feature_extractor = FeatureExtractor(self.config)
        self.label_encoder = self.feature_extractor.label_encoder

    def predict(self, audio_path):
        """Predict emotion from audio file"""
        features = self.feature_extractor.extract_features_from_file(audio_path)
        if features is None:
            return None

        # Reshape for model prediction
        features = features.reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        emotion = self.label_encoder.inverse_transform([predicted_class])[0]

        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'probabilities': {
                emotion: float(prob) for emotion, prob in
                zip(self.label_encoder.classes_, prediction[0])
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Predict emotion from audio file')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/final_model.h5',
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    predictor = EmotionPredictor(args.model, args.config)
    result = predictor.predict(args.audio)

    if result:
        print(f"🎭 Predicted Emotion: {result['emotion']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print("\nProbabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.2%}")
    else:
        print("❌ Failed to process audio file")


if __name__ == "__main__":
    main()

