"""Shared pytest fixtures for testing infrastructure."""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import numpy as np
import torch
import yaml
import json


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for tests."""
    temp_file_path = temp_dir / "test_file.tmp"
    temp_file_path.touch()
    yield temp_file_path


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """Generate sample audio data for testing."""
    # Generate 1 second of sine wave at 22050 Hz
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def sample_spectrogram() -> np.ndarray:
    """Generate sample spectrogram data for testing."""
    # Create a simple spectrogram-like array
    return np.random.randn(1025, 87).astype(np.float32)


@pytest.fixture
def sample_mel_spectrogram() -> np.ndarray:
    """Generate sample mel spectrogram data for testing."""
    # Create a simple mel spectrogram-like array
    return np.random.randn(128, 87).astype(np.float32)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "params": {
                "sample_rate": 22050,
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128
            }
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "epochs": 100
        },
        "data": {
            "dataset_path": "/path/to/dataset",
            "validation_split": 0.2
        }
    }


@pytest.fixture
def sample_config_file(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary config file."""
    config_file = temp_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def sample_json_file(temp_dir: Path) -> Path:
    """Create a temporary JSON file with sample data."""
    json_file = temp_dir / "data.json"
    sample_data = {
        "items": [
            {"id": 1, "name": "test1", "value": 10.5},
            {"id": 2, "name": "test2", "value": 20.3},
            {"id": 3, "name": "test3", "value": 15.7}
        ],
        "metadata": {
            "total": 3,
            "created": "2024-01-01T00:00:00Z"
        }
    }
    with open(json_file, 'w') as f:
        json.dump(sample_data, f)
    return json_file


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Generate sample tensor for testing."""
    return torch.randn(4, 3, 32, 32)  # Batch of 4, 3 channels, 32x32


@pytest.fixture
def sample_audio_tensor() -> torch.Tensor:
    """Generate sample audio tensor for testing."""
    # Generate batch of audio samples
    return torch.randn(2, 22050)  # Batch of 2, 1 second at 22050 Hz


@pytest.fixture
def sample_embedding() -> torch.Tensor:
    """Generate sample embedding tensor for testing."""
    return torch.randn(1, 512)  # Single embedding of size 512


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Generate sample labels for testing."""
    return torch.tensor([0, 1, 2, 1, 0])


@pytest.fixture
def sample_dataset_csv(temp_dir: Path) -> Path:
    """Create a temporary CSV file with sample dataset."""
    csv_file = temp_dir / "dataset.csv"
    csv_content = """filename,caption,duration
audio1.wav,"A dog barking in the park",3.5
audio2.wav,"Classical piano music",120.0
audio3.wav,"Rain falling on leaves",45.2
audio4.wav,"Electric guitar solo",180.5
"""
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.pt"
    # Create a simple mock checkpoint
    checkpoint = {
        "model_state_dict": {"layer1.weight": torch.randn(10, 5)},
        "optimizer_state_dict": {},
        "epoch": 42,
        "loss": 0.123,
        "config": {"model_type": "test_model"}
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_batch_size() -> int:
    """Provide sample batch size for testing."""
    return 4


@pytest.fixture
def sample_sequence_length() -> int:
    """Provide sample sequence length for testing."""
    return 128


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


# Markers for different test categories
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning")
]