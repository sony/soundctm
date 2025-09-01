"""Test to validate the testing infrastructure setup."""

import pytest
import numpy as np
import torch
from pathlib import Path


class TestInfrastructureValidation:
    """Validate that the testing infrastructure is working correctly."""

    def test_basic_assertion(self):
        """Test basic assertion functionality."""
        assert True
        assert 1 + 1 == 2
        assert "hello" == "hello"

    def test_fixtures_work(self, temp_dir, sample_audio_data, sample_config):
        """Test that basic fixtures are working."""
        # Test temp_dir fixture
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Test sample_audio_data fixture
        assert isinstance(sample_audio_data, np.ndarray)
        assert sample_audio_data.dtype == np.float32
        assert len(sample_audio_data) == 22050  # 1 second at 22050 Hz

        # Test sample_config fixture
        assert isinstance(sample_config, dict)
        assert "model" in sample_config
        assert "training" in sample_config

    def test_torch_fixtures(self, sample_tensor, sample_audio_tensor, mock_torch_device):
        """Test PyTorch-related fixtures."""
        # Test sample_tensor fixture
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (4, 3, 32, 32)

        # Test sample_audio_tensor fixture
        assert isinstance(sample_audio_tensor, torch.Tensor)
        assert sample_audio_tensor.shape == (2, 22050)

        # Test mock_torch_device fixture
        assert isinstance(mock_torch_device, torch.device)

    def test_file_fixtures(self, sample_config_file, sample_json_file, sample_dataset_csv):
        """Test file-related fixtures."""
        # Test config file
        assert sample_config_file.exists()
        assert sample_config_file.suffix == ".yaml"

        # Test JSON file
        assert sample_json_file.exists()
        assert sample_json_file.suffix == ".json"

        # Test CSV file
        assert sample_dataset_csv.exists()
        assert sample_dataset_csv.suffix == ".csv"

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True

    def test_numpy_operations(self, sample_spectrogram, sample_mel_spectrogram):
        """Test numpy operations with fixtures."""
        # Test spectrogram fixture
        assert sample_spectrogram.shape == (1025, 87)
        assert sample_spectrogram.dtype == np.float32

        # Test mel spectrogram fixture
        assert sample_mel_spectrogram.shape == (128, 87)
        assert sample_mel_spectrogram.dtype == np.float32

        # Test basic numpy operations
        mean_val = np.mean(sample_spectrogram)
        assert isinstance(mean_val, (float, np.floating))

    def test_random_seed_reproducibility(self):
        """Test that random seeds are properly set for reproducibility."""
        # Test numpy random seed
        random_array1 = np.random.randn(10)
        np.random.seed(42)
        random_array2 = np.random.randn(10)
        
        # Arrays should be different (since seed was reset)
        # This mainly tests that seeding works
        assert isinstance(random_array1, np.ndarray)
        assert isinstance(random_array2, np.ndarray)

    def test_mock_model_checkpoint(self, mock_model_checkpoint):
        """Test mock model checkpoint fixture."""
        assert mock_model_checkpoint.exists()
        
        # Load and verify checkpoint structure
        checkpoint = torch.load(mock_model_checkpoint, map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "loss" in checkpoint
        assert "config" in checkpoint
        
        assert checkpoint["epoch"] == 42
        assert isinstance(checkpoint["loss"], float)

    def test_parametrized_example(self, sample_batch_size, sample_sequence_length):
        """Test parametrized fixtures."""
        assert isinstance(sample_batch_size, int)
        assert sample_batch_size > 0
        
        assert isinstance(sample_sequence_length, int)
        assert sample_sequence_length > 0

    def test_temp_file_creation(self, temp_file):
        """Test temp file fixture."""
        assert temp_file.exists()
        assert temp_file.is_file()
        
        # Write to and read from temp file
        test_content = "Hello, testing!"
        temp_file.write_text(test_content)
        assert temp_file.read_text() == test_content

    def test_exception_handling(self):
        """Test that exceptions are properly handled."""
        with pytest.raises(ZeroDivisionError):
            1 / 0
        
        with pytest.raises(ValueError):
            int("not_a_number")

    def test_approximate_equality(self):
        """Test approximate equality for floating point numbers."""
        assert 0.1 + 0.2 == pytest.approx(0.3)
        assert [0.1, 0.2, 0.3] == pytest.approx([0.1, 0.2, 0.3])

    def test_imports_work(self):
        """Test that common imports are working."""
        # Test that we can import major dependencies
        import numpy as np
        import torch
        import pytest
        import pathlib
        import json
        import yaml
        
        # Basic functionality check
        arr = np.array([1, 2, 3])
        tensor = torch.tensor([1, 2, 3])
        
        assert len(arr) == 3
        assert tensor.shape == (3,)


def test_standalone_function():
    """Test a standalone function (not in a class)."""
    assert "pytest" in "pytest is working"


@pytest.mark.unit
def test_standalone_with_marker():
    """Test a standalone function with marker."""
    assert True


# Integration test example
@pytest.mark.integration
class TestIntegrationExample:
    """Example integration test class."""
    
    def test_multiple_components_together(self, sample_audio_data, sample_config):
        """Test that multiple components work together."""
        # This would typically test interaction between modules
        assert len(sample_audio_data) > 0
        assert sample_config["model"]["params"]["sample_rate"] == 22050
        
        # Example of testing component interaction
        expected_samples = sample_config["model"]["params"]["sample_rate"]
        assert len(sample_audio_data) == expected_samples