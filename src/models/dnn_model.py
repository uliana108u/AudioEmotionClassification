import tensorflow as tf
from tensorflow.keras import layers, regularizers


class DNNModel:
    @staticmethod
    def create(config, input_shape):
        model = tf.keras.Sequential()

        # Input layer
        model.add(layers.Dense(
            config.model.hidden_layers[0],
            activation='relu',
            kernel_regularizer=regularizers.l2(config.model.l2_regularization),
            input_shape=(input_shape,)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config.model.dropout_rates[0]))

        # Hidden layers
        for i, (units, dropout_rate) in enumerate(
                zip(config.model.hidden_layers[1:], config.model.dropout_rates[1:])
        ):
            model.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(config.model.l2_regularization)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

        # Output layer
        model.add(layers.Dense(config.model.num_classes, activation='softmax'))

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model