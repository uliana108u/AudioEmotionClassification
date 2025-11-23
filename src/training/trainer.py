import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from .callbacks import get_callbacks
from src.evaluation.metrics import EvaluationMetrics


class Trainer:
    def __init__(self, config):
        self.config = config
        self.history = None

    def train(self, model, features, labels):
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        class_weights = self._calculate_class_weights(y_train)

        callbacks = get_callbacks(self.config)

        print(f"Training on {X_train.shape[0]} samples")
        print(f"Validating on {X_val.shape[0]} samples")

        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate(self, model, features, labels):
        _, X_test, _, y_test = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Additional metrics
        evaluator = EvaluationMetrics(model, X_test, y_test)
        evaluator.print_classification_report()
        evaluator.plot_confusion_matrix()

        return test_accuracy, test_loss

    def _calculate_class_weights(self, y):
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

