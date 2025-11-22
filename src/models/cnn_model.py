import tensorflow as tf
from tensorflow.keras import layers, regularizers


class CNNModel:
    @staticmethod
    def create(config, input_shape):
        model = tf.keras.Sequential()

        # Reshape for CNN input
        model.add(layers.Reshape((input_shape, 1), input_shape=(input_shape,)))

        # First Conv Block
        model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(0.3))

        # Second Conv Block
        model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(0.4))

        # Third Conv Block
        model.add(layers.Conv1D(256, 3, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(0.5))

        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(64, activation='relu'))
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

