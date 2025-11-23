import argparse
import os
import sys
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.config import Config
import numpy as np
import json
import matplotlib.pyplot as plt

sys.path.append('src')


def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Classification Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='default',
                        help='Experiment name')
    args = parser.parse_args()

    config = Config(args.config)

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)

    print("Starting Audio Emotion Classification Training")
    print(f"Experiment: {args.experiment}")

    try:
        print("Loading data...")
        data_loader = DataLoader(config)
        file_paths, labels = data_loader.load_data()

        print(f"Sample files: {file_paths[:3]}")
        print(f"Sample labels: {labels[:3]}")

        print("Extracting features...")
        feature_extractor = FeatureExtractor(config)

        expected_length = feature_extractor.calculate_expected_feature_length()
        print(f"Expected feature vector length: {expected_length}")

        features, labels_encoded = feature_extractor.extract_features(file_paths, labels)

        print(f"Final feature matrix shape: {features.shape}")
        print(f"Labels shape: {labels_encoded.shape}")
        print(f"Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
        print(f"Feature mean: {np.mean(features):.3f}, std: {np.std(features):.3f}")

        if np.isnan(features).any():
            print("WARNING: Features contain NaN values!")
            # Replace NaN with 0
            features = np.nan_to_num(features)

        print("Creating model...")
        model = ModelFactory.create_model(config, input_shape=features.shape[1])

        print("Training model...")
        trainer = Trainer(config)
        history = trainer.train(model, features, labels_encoded)

        print("\nSaving training history...")

        # Convert numpy float32 to Python float for JSON serialization
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(value) for value in values]

        with open('training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)

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

        print("Training history saved to 'training_history.json' and 'training_history.png'")

        print("Evaluating model...")
        trainer.evaluate(model, features, labels_encoded)

        model_path = config.paths.final_model
        model.save(model_path)
        print(f"Model saved to {model_path}")

        feature_extractor.save_preprocessor('models/preprocessor.pkl')
        print("Preprocessor saved")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

