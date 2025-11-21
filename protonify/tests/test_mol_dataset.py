import numpy as np
import pytest
from protonify.mol_dataset import MolDataset

def test_mol_dataset_initialization():
    # Test with simple data and no labels
    data = [1, 2, 3, 4, 5]
    dataset = MolDataset(data)
    assert len(dataset) == 5
    assert dataset.label.shape == (5, 1)
    assert np.all(dataset.label == 0)

    # Test with data and custom labels
    labels = np.array([[1], [2], [3], [4], [5]])
    dataset = MolDataset(data, labels)
    assert len(dataset) == 5
    assert np.array_equal(dataset.label, labels)

def test_mol_dataset_getitem():
    # Test with simple data
    data = [1, 2, 3, 4, 5]
    dataset = MolDataset(data)
    
    # Test single item access
    item, label = dataset[0]
    assert item == 1
    assert label == 0
    
    # Test multiple items
    for i, (item, label) in enumerate(dataset):
        assert item == data[i]
        assert label == 0

def test_mol_dataset_with_complex_data():
    # Test with more complex data (list of lists)
    data = [[1, 2], [3, 4], [5, 6]]
    labels = np.array([[1], [2], [3]])
    dataset = MolDataset(data, labels)
    
    assert len(dataset) == 3
    for i, (item, label) in enumerate(dataset):
        assert item == data[i]
        assert label == labels[i]

def test_mol_dataset_edge_cases():
    # Test with empty dataset
    with pytest.raises(IndexError):
        dataset = MolDataset([])
        _ = dataset[0]
    
    # Test with single item
    dataset = MolDataset([1])
    assert len(dataset) == 1
    item, label = dataset[0]
    assert item == 1
    assert label == 0

def test_mol_dataset_label_shape():
    # Test with different label shapes
    data = [1, 2, 3]
    
    # Test with 2D labels
    labels_2d = np.array([[1], [2], [3]])
    dataset = MolDataset(data, labels_2d)
    assert dataset.label.shape == (3, 1)
    
    # Test with 1D labels (should maintain shape)
    labels_1d = np.array([1, 2, 3])
    dataset = MolDataset(data, labels_1d)
    assert dataset.label.shape == (3,)  # Accept 1D shape
    assert np.array_equal(dataset.label, labels_1d)  # Values should be unchanged 