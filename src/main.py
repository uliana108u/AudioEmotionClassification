import argparse
import yaml
import os
import sys

# Add src to path
sys.path.append('src')

from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Classification Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='default',
                        help='Experiment name')
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)

    print("🚀 Starting Audio Emotion Classification Training")
    print(f"📁 Experiment: {args.experiment}")

    # Load and preprocess data
    print("📊 Loading data...")
    data_loader = DataLoader(config)
    file_paths, labels = data_loader.load_data()

    # Extract features
    print("🎵 Extracting features...")
    feature_extractor = FeatureExtractor(config)
    features, labels_encoded = feature_extractor.extract_features(file_paths, labels)

    # Create model
    print("🧠 Creating model...")
    model = ModelFactory.create_model(config, input_shape=features.shape[1])

    # Train model
    print("🏋️ Training model...")
    trainer = Trainer(config)
    history = trainer.train(model, features, labels_encoded)

    # Evaluate model
    print("📈 Evaluating model...")
    trainer.evaluate(model, features, labels_encoded)

    # Save final model
    model_path = config.paths.final_model
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")


if __name__ == "__main__":
    main()

