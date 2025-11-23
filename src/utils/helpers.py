import os
import logging

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def setup_logging(log_file='training.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
