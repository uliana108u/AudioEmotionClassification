from setuptools import setup, find_packages

setup(
    name="audio-emotion-classification",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "librosa>=0.9.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "PyYAML>=6.0",
    ],
    author="Uliana",
    description="Audio Emotion Classification using RAVDESS dataset",
    python_requires=">=3.8",
)

