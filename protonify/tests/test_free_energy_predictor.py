"""Unit tests for the FreeEnergyPredictor class."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from protonify.FreeEnergyPredictor import FreeEnergyPredictor
from protonify import get_model_path, MODEL_PATH_ENV_VAR, DEFAULT_MODEL_NAME


# Path to the model in the package
PACKAGE_MODEL_PATH = Path(__file__).parent.parent / "models" / DEFAULT_MODEL_NAME


def get_test_model_path():
    """Get model path for testing, skip if not available or cannot be loaded."""
    if PACKAGE_MODEL_PATH.exists():
        os.environ[MODEL_PATH_ENV_VAR] = str(PACKAGE_MODEL_PATH)
        # Try to actually load the model to verify it works
        try:
            torch.load(str(PACKAGE_MODEL_PATH), map_location='cpu', weights_only=False)
        except Exception as e:
            pytest.skip(f"Model exists but cannot be loaded: {e}")
        return str(PACKAGE_MODEL_PATH)
    try:
        return get_model_path()
    except Exception as e:
        pytest.skip(f"Model not available for testing: {e}")


class TestFreeEnergyPredictorDevice:
    """Tests for device selection."""

    def test_init_sets_device_cpu_when_no_cuda(self):
        """Test CPU is used when CUDA not available."""
        with patch('protonify.FreeEnergyPredictor.UniMolModel') as MockModel, \
             patch('torch.cuda.is_available', return_value=False):
            predictor = FreeEnergyPredictor('dummy_path', use_gpu=True)
            assert predictor.device.type == 'cpu'

    def test_init_sets_device_cuda_when_available(self):
        """Test CUDA is used when available and requested."""
        with patch('protonify.FreeEnergyPredictor.UniMolModel') as MockModel, \
             patch('torch.cuda.is_available', return_value=True):
            predictor = FreeEnergyPredictor('dummy_path', use_gpu=True)
            assert predictor.device.type == 'cuda'

    def test_init_sets_device_cpu_when_not_requested(self):
        """Test CPU is used when GPU not requested."""
        with patch('protonify.FreeEnergyPredictor.UniMolModel') as MockModel, \
             patch('torch.cuda.is_available', return_value=False):
            predictor = FreeEnergyPredictor('dummy_path', use_gpu=False)
            assert predictor.device.type == 'cpu'


class TestFreeEnergyPredictorPrediction:
    """Tests for prediction functionality."""

    def test_predict_single_returns_dict(self):
        """Test predict_single returns a dictionary."""
        model_path = get_test_model_path()
        predictor = FreeEnergyPredictor(model_path, batch_size=1, remove_hs=False)

        result = predictor.predict_single("CCO")
        assert isinstance(result, dict)
        assert "CCO" in result
        assert isinstance(result["CCO"], float)

    def test_predict_batch_returns_dict(self):
        """Test predict returns dictionary for batch."""
        model_path = get_test_model_path()
        predictor = FreeEnergyPredictor(model_path, batch_size=2, remove_hs=False)

        smiles_list = ["CCO", "CC(=O)O"]
        result = predictor.predict(smiles_list)

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_predictions_are_finite(self):
        """Test that predictions are finite numbers."""
        model_path = get_test_model_path()
        predictor = FreeEnergyPredictor(model_path, batch_size=1, remove_hs=False)

        result = predictor.predict_single("CCO")
        energy = result["CCO"]

        assert not np.isnan(energy)
        assert not np.isinf(energy)

    def test_different_molecules_different_energies(self):
        """Test that different molecules get different energies."""
        model_path = get_test_model_path()
        predictor = FreeEnergyPredictor(model_path, batch_size=2, remove_hs=False)

        result1 = predictor.predict_single("CCO")
        result2 = predictor.predict_single("CC(=O)O")

        # Different molecules should have different energies
        assert result1["CCO"] != result2["CC(=O)O"]