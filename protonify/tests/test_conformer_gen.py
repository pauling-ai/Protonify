"""Unit tests for the ConformerGen class and related functions."""

import pytest
import numpy as np
from protonify.conformer_gen import ConformerGen, inner_smi2coords, InvalidSmilesError


def test_conformer_gen_init():
    """Test initialization of ConformerGen with default and custom parameters."""
    gen = ConformerGen()
    assert gen.seed == 42
    assert gen.max_atoms == 256
    assert gen.mode == 'fast'
    # Custom params
    gen2 = ConformerGen(seed=123, max_atoms=10, mode='fast', remove_hs=True)
    assert gen2.seed == 123
    assert gen2.max_atoms == 10
    assert gen2.mode == 'fast'
    assert gen2.remove_hs is True


def test_single_process_valid_smiles():
    """Test conformer generation for a valid SMILES string."""
    gen = ConformerGen()
    smiles = 'CCO'  # Ethanol
    result = gen.single_process(smiles)
    assert isinstance(result, dict)
    assert 'src_tokens' in result
    assert 'src_coord' in result
    assert result['src_tokens'].size > 0
    assert result['src_coord'].shape[1] == 3


def test_single_process_with_different_modes():
    """Test conformer generation with different modes."""
    # Fast mode (default)
    gen_fast = ConformerGen(mode='fast')
    result = gen_fast.single_process('CCO')
    assert isinstance(result, dict)
    assert 'src_tokens' in result

    # Heavy mode
    gen_heavy = ConformerGen(mode='heavy')
    result = gen_heavy.single_process('CCO')
    assert isinstance(result, dict)
    assert 'src_tokens' in result


def test_inner_smi2coords_removes_hs():
    """Test that hydrogens are removed when remove_hs=True."""
    atoms, coords, charges = inner_smi2coords('CCO', remove_hs=True)
    assert 'H' not in atoms
    assert len(atoms) == len(coords) == len(charges)


def test_inner_smi2coords_keeps_hs():
    """Test that hydrogens are present when remove_hs=False."""
    atoms, coords, charges = inner_smi2coords('CCO', remove_hs=False)
    assert 'H' in atoms
    assert len(atoms) == len(coords) == len(charges)


def test_inner_smi2coords_invalid_smiles():
    """Test that invalid SMILES raises InvalidSmilesError."""
    with pytest.raises(InvalidSmilesError):
        inner_smi2coords('not_a_smiles', remove_hs=True) 