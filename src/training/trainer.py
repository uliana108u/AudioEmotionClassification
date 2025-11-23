import numpy as np
import json
import matplotlib.pyplot as plt
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

    def save_training_history(self, history, filepath='training_history.json'):
        history_dict = {}
        for key, values in history.history.items():
            # Convert each value to Python native float
            history_dict[key] = [float(value) for value in values]

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"Training history saved to {filepath}")
        return history_dict

    def plot_training_history(self, history, save_path='training_history.png'):
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Training plots saved to {save_path}")

    def print_training_summary(self, history):
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)

        # Convert to Python native types for printing
        final_train_acc = float(history.history['accuracy'][-1])
        final_val_acc = float(history.history['val_accuracy'][-1])
        final_train_loss = float(history.history['loss'][-1])
        final_val_loss = float(history.history['val_loss'][-1])

        # Best epoch results
        best_val_acc_epoch = np.argmax(history.history['val_accuracy'])
        best_val_acc = float(history.history['val_accuracy'][best_val_acc_epoch])
        best_val_loss = float(history.history['val_loss'][best_val_acc_epoch])

        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_val_acc_epoch + 1})")
        print(f"Validation Loss at Best Accuracy: {best_val_loss:.4f}")
        print("=" * 50)

    def evaluate(self, model, features, labels):
        _, X_test, _, y_test = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        test_accuracy = float(test_accuracy)
        test_loss = float(test_loss)

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        evaluator = EvaluationMetrics(model, X_test, y_test)
        evaluator.print_classification_report()
        evaluator.plot_confusion_matrix()

        return test_accuracy, test_loss

    def _calculate_class_weights(self, y):
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

