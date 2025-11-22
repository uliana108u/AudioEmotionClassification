import tensorflow as tf
from tensorflow.keras import layers


class LSTMModel:
    @staticmethod
    def create(config, input_shape):
        model = tf.keras.Sequential()

        # Reshape for LSTM (sequence_length, features)
        # Assuming we want to treat features as time steps
        model.add(layers.Reshape((input_shape, 1), input_shape=(input_shape,)))

        # First LSTM layer
        model.add(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
        model.add(layers.BatchNormalization())

        # Second LSTM layer
        model.add(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3))
        model.add(layers.BatchNormalization())

        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        # Output layer
        model.add(layers.Dense(config.model.num_classes, activation='softmax'))

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model