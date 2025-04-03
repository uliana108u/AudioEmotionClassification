import os
import zipfile
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Define the emotion classes (RAVDESS has 8 emotions)
emotion_classes = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']


# Function to unzip the dataset
def unzip_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset extracted to {extract_to}")


# Define the dataset class
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


# Define the model
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Wav2Vec2 expects [batch, seq_len]
        outputs = self.wav2vec(x)
        features = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(features)


# Custom collate function to handle variable-length waveforms
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    waveforms = torch.stack(waveforms)  # [batch, seq_len]
    waveforms = waveforms
    labels = torch.tensor(labels)
    print(labels)
    return waveforms, labels


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for waveforms, labels in dataloader:
        print(waveforms.shape, labels.shape)
        if len(waveforms) == 0:  # Skip empty batches
            continue
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in dataloader:
            if len(waveforms) == 0:  # Skip empty batches
                continue
            waveforms = waveforms.to(device)
            print("labels ", len(labels))
            labels = labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), correct / total


# Main function
def main():
    file_path = "data/Actor_19/03-02-04-02-02-01-19.wav"
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
    else:
        print("File exists.")
    # Path to the zipped RAVDESS dataset
    zip_path = 'Audio_Song_Actors_01-24.zip'
    extract_to = 'data'

    # # Unzip the dataset
    # if not os.path.exists(extract_to):
    #     os.makedirs(extract_to)
    # unzip_dataset(zip_path, extract_to)

    # Load dataset
    dataset = EmotionDataset(extract_to)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    # Initialize model, loss, and optimizer
    model = EmotionClassifier(num_classes=len(emotion_classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    # Use AdamW with weight decay
    optimizer = optim.AdamW([
        {'params': model.wav2vec.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    # # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
    # # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Only unfreeze top 3 layers
        for param in model.wav2vec.parameters():
            param.requires_grad = False

        for i in range(-3, 0):  # Unfreeze last 3 layers
            for param in model.wav2vec.encoder.layers[i].parameters():
                param.requires_grad = True

        train_loss = train(model, dataloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dataloader, criterion, device)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


if __name__ == '__main__':
    main()
