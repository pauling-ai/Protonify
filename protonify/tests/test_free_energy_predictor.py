"""Unit tests for the FreeEnergyPredictor class."""

import pytest
from unittest.mock import patch, MagicMock
import torch
from protonify.FreeEnergyPredictor import FreeEnergyPredictor


def test_init_sets_device_cpu_when_no_cuda():
    with patch('protonify.FreeEnergyPredictor.UniMolModel') as MockModel:
        predictor = FreeEnergyPredictor('dummy_path', use_gpu=True)
        assert predictor.device.type == 'cpu'


def test_init_sets_device_cuda_when_available():
    with patch('protonify.FreeEnergyPredictor.UniMolModel') as MockModel, \
         patch('torch.cuda.is_available', return_value=True):
        predictor = FreeEnergyPredictor('dummy_path', use_gpu=True)
        assert predictor.device.type == 'cuda'