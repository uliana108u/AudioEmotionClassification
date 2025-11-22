import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationMetrics:
    def __init__(self, model, X_test, y_test, label_encoder=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.label_encoder = label_encoder

    def print_classification_report(self):
        """Print detailed classification report"""
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        target_names = (self.label_encoder.classes_
                        if self.label_encoder else None)

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_classes,
                                    target_names=target_names))

    def plot_confusion_matrix(self, save_path='confusion_matrix.png'):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(self.y_test, y_pred_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=(self.label_encoder.classes_
                                 if self.label_encoder else range(cm.shape[1])),
                    yticklabels=(self.label_encoder.classes_
                                 if self.label_encoder else range(cm.shape[0])))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()