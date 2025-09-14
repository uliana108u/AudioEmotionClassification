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

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            # waveform, sample_rate = torchaudio.load(self.audio_files[idx]
            waveform, sample_rate = sf.read(self.audio_files[idx])
            print(type(waveform))  # Should be <class 'numpy.ndarray'>
            print(type(sample_rate))  # Should be <class 'int'>
            assert not isinstance(waveform, tuple), "Audio became a tuple unexpectedly!"
            # waveform = torch.from_numpy(waveform).float()

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)
            # Ensure proper shape [1, seq_len]
            waveform = waveform.mean(dim=0, keepdim=True)  # Mix down to mono if stereo
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)

            # Truncate/pad to fixed length
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            else:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return waveform.squeeze(0), self.labels[idx]  # [seq_len], label

        except Exception as e:
            print(f"Error loading {self.audio_files[idx]}: {e}")
            return torch.zeros(self.max_length), -1  # Return dummy data
