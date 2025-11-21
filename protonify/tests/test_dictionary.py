"""Unit tests for the Dictionary class."""

import io
import pytest
from protonify.dictionary import Dictionary


def test_dictionary_init_and_basic_methods():
    d = Dictionary()
    # Test special symbols
    assert d.bos_word in d.specials
    assert d.pad_word in d.specials
    assert d.eos_word in d.specials
    assert d.unk_word in d.specials
    # Add and retrieve symbol
    idx = d.add_symbol('C')
    assert d.index('C') == idx
    assert d[idx] == 'C'
    assert 'C' in d
    # Test __len__
    assert len(d) > 0
    # Test unknown symbol returns unk index
    unk_idx = d.unk()
    assert d.index('not_in_dict') == unk_idx


def test_add_symbol_overwrite():
    d = Dictionary()
    idx1 = d.add_symbol('O', n=1)
    idx2 = d.add_symbol('O', n=5, overwrite=True)
    assert idx1 == idx2
    assert d.count[idx1] == 5


def test_vec_index():
    d = Dictionary()
    d.add_symbol('A')
    d.add_symbol('B')
    arr = ['A', 'B', 'not_in_dict']
    idxs = d.vec_index(arr)
    assert idxs[0] == d.index('A')
    assert idxs[1] == d.index('B')
    assert idxs[2] == d.unk()


def test_load_from_str():
    s = """
C 10
O 5
N 2
"""
    d = Dictionary.load_from_str(s)
    assert d.index('C') != d.unk()
    assert d.index('O') != d.unk()
    assert d.index('N') != d.unk()
    assert d.count[d.index('C')] == 10
    assert d.count[d.index('O')] == 5
    assert d.count[d.index('N')] == 2


def test_add_from_lines_duplicate():
    s = """
C 1
C 2 #overwrite
"""
    d = Dictionary.load_from_str(s)
    assert d.count[d.index('C')] == 2


def test_load_from_file(tmp_path):
    content = "H 3\nO 2\n"
    file_path = tmp_path / "dict.txt"
    file_path.write_text(content)
    d = Dictionary.load(str(file_path))
    assert d.index('H') != d.unk()
    assert d.count[d.index('H')] == 3
    assert d.index('O') != d.unk()
    assert d.count[d.index('O')] == 2 