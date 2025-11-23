import yaml
from types import SimpleNamespace


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        # Convert dict to object attributes but keep emotions as dict
        self._create_namespace(config_dict)

    def _create_namespace(self, config_dict, parent=None):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # For 'emotions', keep it as a dictionary
                if key == 'emotions':
                    setattr(parent if parent else self, key, value)
                else:
                    # Create nested namespace for other dicts
                    nested_ns = SimpleNamespace()
                    self._create_namespace(value, nested_ns)
                    setattr(parent if parent else self, key, nested_ns)
            else:
                setattr(parent if parent else self, key, value)

    def get_emotions_dict(self):
        """Get emotions as a dictionary"""
        return self.data.emotions if hasattr(self.data, 'emotions') else {}

    def __str__(self):
        return str(self.__dict__)