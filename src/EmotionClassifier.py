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
