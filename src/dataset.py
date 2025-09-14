class EmotionDataset(Dataset):
    def __init__(self, data_dir, target_sample_rate=16000, max_length=16000 * 4):
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.audio_files = []
        self.labels = []
        self.max_length = max_length

        # Iterate over each speaker folder and load audio files
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):  # Ensure we only load WAV files
                    # Extract emotion label from the filename
                    # RAVDESS filename format: 03-01-06-01-02-01-12.wav
                    # Emotion is encoded in the third field (06 in this example)
                    emotion_code = int(file.split('-')[2])
                    emotion_label = emotion_classes[emotion_code - 1]  # RAVDESS codes start from 1
                    self.audio_files.append(os.path.join(root, file))
                    self.labels.append(emotion_classes.index(emotion_label))
