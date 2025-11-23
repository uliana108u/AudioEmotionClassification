import os
import numpy as np
from typing import List, Tuple


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.emotions = config.get_emotions_dict()

    def load_data(self) -> Tuple[List[str], List[str]]:
        file_paths = []
        labels = []

        data_path = self.config.data.data_path

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        print(f"Searching for audio files in: {data_path}")

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)

                    # Extract emotion code from filename
                    # RAVDESS filename format: 03-01-06-01-02-01-12.wav
                    try:
                        parts = file.split('-')
                        if len(parts) >= 3:
                            emotion_code = parts[2]

                            # Check if emotion code exists in emotions dictionary
                            if emotion_code in self.emotions:
                                file_paths.append(file_path)
                                labels.append(self.emotions[emotion_code])
                            else:
                                print(f"Skipping {file}: emotion code {emotion_code} not in configured emotions")
                        else:
                            print(f"Skipping {file}: invalid filename format")

                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                        continue

        if len(file_paths) == 0:
            raise ValueError("No audio files found! Check your data path and file structure.")

        print(f"Loaded {len(file_paths)} audio files")
        print(f"Emotions distribution: {self.get_class_distribution(labels)}")

        return file_paths, labels

    def get_class_distribution(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

