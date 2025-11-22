import os
import numpy as np
from typing import List, Tuple


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.emotions = config.data.emotions

    def load_data(self) -> Tuple[List[str], List[str]]:
        """Load audio file paths and corresponding labels"""
        file_paths = []
        labels = []

        data_path = self.config.data.data_path

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    emotion_code = file.split('-')[2]

                    if emotion_code in self.emotions:
                        file_paths.append(file_path)
                        labels.append(self.emotions[emotion_code])

        print(f"Loaded {len(file_paths)} audio files")
        return file_paths, labels

    def get_class_distribution(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

