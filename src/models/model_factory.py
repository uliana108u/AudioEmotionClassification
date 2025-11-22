from .cnn_model import CNNModel
from .dnn_model import DNNModel
from .lstm_model import LSTMModel


class ModelFactory:
    @staticmethod
    def create_model(config, input_shape):
        model_type = config.model.name.lower()

        if model_type == "cnn":
            return CNNModel.create(config, input_shape)
        elif model_type == "lstm":
            return LSTMModel.create(config, input_shape)
        elif model_type in ["dnn", "advanced_dnn"]:
            return DNNModel.create(config, input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")