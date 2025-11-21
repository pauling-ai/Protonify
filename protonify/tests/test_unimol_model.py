import pytest
import torch
import numpy as np
import os
from protonify.unimol_model import (
    UniMolModel, ClassificationHead, NonLinearHead, GaussianLayer,
    pad_1d_tokens, pad_2d, pad_coords, gaussian, molecule_architecture
)
from protonify.FreeEnergyPredictor import FreeEnergyPredictor

# Get the path to the pretrained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "t_dwar_v_novartis_a_b.pt")

@pytest.fixture
def sample_smiles():
    """Create sample SMILES for testing"""
    return ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F"]

@pytest.fixture
def predictor():
    """Create a FreeEnergyPredictor instance"""
    return FreeEnergyPredictor(MODEL_PATH, batch_size=1, remove_hs=False)

def test_unimol_model_forward(sample_smiles, predictor):
    """Test UniMolModel forward pass using FreeEnergyPredictor"""
    # Test single prediction
    result = predictor.predict_single(sample_smiles[0])
    assert isinstance(result, dict)
    assert sample_smiles[0] in result
    assert isinstance(result[sample_smiles[0]], float)
    
    # Test batch prediction
    results = predictor.predict(sample_smiles)
    assert isinstance(results, dict)
    # Check that we got at least one valid prediction
    assert len(results) > 0
    # Check that all predictions are valid floats
    assert all(isinstance(energy, float) for energy in results.values())
    # Check that the predictions are reasonable (not NaN or Inf)
    assert all(not np.isnan(energy) and not np.isinf(energy) for energy in results.values())

def test_unimol_model_batch_collate(sample_smiles, predictor):
    """Test UniMolModel batch collate function using FreeEnergyPredictor"""
    # Test that we can predict with different batch sizes
    # For batch_size=1, we expect only the first molecule to be processed
    predictor.batch_size = 1
    results_single = predictor.predict(sample_smiles)
    assert len(results_single) == 1  # Only first molecule
    assert sample_smiles[0] in results_single
    assert isinstance(results_single[sample_smiles[0]], float)
    assert not np.isnan(results_single[sample_smiles[0]]) and not np.isinf(results_single[sample_smiles[0]])
    
    # For batch_size=2, we expect both molecules to be processed
    predictor.batch_size = 2
    results_batch = predictor.predict(sample_smiles)
    assert len(results_batch) == 2  # Both molecules
    assert all(smiles in results_batch for smiles in sample_smiles)
    assert all(isinstance(energy, float) for energy in results_batch.values())
    assert all(not np.isnan(energy) and not np.isinf(energy) for energy in results_batch.values())

@pytest.fixture
def model_args():
    """Create model arguments for testing"""
    class Args:
        def __init__(self):
            self.encoder_layers = 2  # Reduced for testing
            self.encoder_embed_dim = 64  # Reduced for testing
            self.encoder_ffn_embed_dim = 128
            self.encoder_attention_heads = 4
            self.dropout = 0.1
            self.emb_dropout = 0.1
            self.attention_dropout = 0.1
            self.activation_dropout = 0.0
            self.pooler_dropout = 0.0
            self.max_seq_len = 512
            self.activation_fn = "gelu"
            self.pooler_activation_fn = "tanh"
            self.post_ln = False
            self.backbone = "transformer"
            self.kernel = "gaussian"
            self.delta_pair_repr_norm_loss = -1.0
    return Args()

def test_classification_head():
    """Test ClassificationHead module"""
    input_dim = 64
    inner_dim = 32
    num_classes = 2
    batch_size = 4
    
    head = ClassificationHead(
        input_dim=input_dim,
        inner_dim=inner_dim,
        num_classes=num_classes,
        activation_fn="gelu",
        pooler_dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    output = head(x)
    assert output.shape == (batch_size, num_classes)
    
    # Test with different activation functions
    for activation in ["relu", "gelu", "tanh"]:
        head = ClassificationHead(
            input_dim=input_dim,
            inner_dim=inner_dim,
            num_classes=num_classes,
            activation_fn=activation,
            pooler_dropout=0.1
        )
        output = head(x)
        assert output.shape == (batch_size, num_classes)

def test_nonlinear_head():
    """Test NonLinearHead module"""
    input_dim = 64
    out_dim = 32
    batch_size = 4
    
    head = NonLinearHead(
        input_dim=input_dim,
        out_dim=out_dim,
        activation_fn="gelu"
    )
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    output = head(x)
    assert output.shape == (batch_size, out_dim)
    
    # Test with hidden dimension
    hidden_dim = 48
    head = NonLinearHead(
        input_dim=input_dim,
        out_dim=out_dim,
        activation_fn="gelu",
        hidden=hidden_dim
    )
    output = head(x)
    assert output.shape == (batch_size, out_dim)

def test_gaussian_layer():
    """Test GaussianLayer module"""
    K = 8  # Reduced for testing
    edge_types = 10
    batch_size = 2
    seq_len = 4
    
    layer = GaussianLayer(K=K, edge_types=edge_types)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, seq_len)
    edge_type = torch.randint(0, edge_types, (batch_size, seq_len, seq_len))
    output = layer(x, edge_type)
    
    assert output.shape == (batch_size, seq_len, seq_len, K)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_gaussian_function():
    """Test gaussian function"""
    x = torch.tensor([-1.0, 0.0, 1.0])
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    
    output = gaussian(x, mean, std)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Test symmetry around mean
    assert torch.allclose(output[0], output[2], atol=1e-6)

def test_padding_functions():
    """Test padding utility functions"""
    # Test pad_1d_tokens
    values = [torch.tensor([1, 2, 3]), torch.tensor([1, 2])]
    pad_idx = 0
    padded = pad_1d_tokens(values, pad_idx)
    assert padded.shape == (2, 3)
    assert torch.all(padded[1, 2:] == pad_idx)
    
    # Test pad_2d
    values = [torch.randn(2, 2), torch.randn(1, 1)]
    padded = pad_2d(values, pad_idx)
    assert padded.shape == (2, 2, 2)
    assert torch.all(padded[1, 1:, 1:] == pad_idx)
    
    # Test pad_coords
    values = [torch.randn(2, 3), torch.randn(1, 3)]
    padded = pad_coords(values, pad_idx)
    assert padded.shape == (2, 2, 3)
    assert torch.all(padded[1, 1:, :] == pad_idx)

def create_dummy_state_dict(model):
    """Create a dummy state dict with random weights matching model structure"""
    state_dict = {}
    
    # Embedding layers
    state_dict["embed_tokens.weight"] = torch.randn_like(model.embed_tokens.weight)
    state_dict["embed_charges.weight"] = torch.randn_like(model.embed_charges.weight)
    
    # Encoder layers
    for i in range(model.args.encoder_layers):
        prefix = f"encoder.layers.{i}"
        # Self attention
        state_dict[f"{prefix}.self_attn.in_proj.weight"] = torch.randn_like(model.encoder.layers[i].self_attn.in_proj.weight)
        state_dict[f"{prefix}.self_attn.in_proj.bias"] = torch.randn_like(model.encoder.layers[i].self_attn.in_proj.bias)
        state_dict[f"{prefix}.self_attn.out_proj.weight"] = torch.randn_like(model.encoder.layers[i].self_attn.out_proj.weight)
        state_dict[f"{prefix}.self_attn.out_proj.bias"] = torch.randn_like(model.encoder.layers[i].self_attn.out_proj.bias)
        state_dict[f"{prefix}.self_attn_layer_norm.weight"] = torch.randn_like(model.encoder.layers[i].self_attn_layer_norm.weight)
        state_dict[f"{prefix}.self_attn_layer_norm.bias"] = torch.randn_like(model.encoder.layers[i].self_attn_layer_norm.bias)
        # Feed forward
        state_dict[f"{prefix}.fc1.weight"] = torch.randn_like(model.encoder.layers[i].fc1.weight)
        state_dict[f"{prefix}.fc1.bias"] = torch.randn_like(model.encoder.layers[i].fc1.bias)
        state_dict[f"{prefix}.fc2.weight"] = torch.randn_like(model.encoder.layers[i].fc2.weight)
        state_dict[f"{prefix}.fc2.bias"] = torch.randn_like(model.encoder.layers[i].fc2.bias)
        state_dict[f"{prefix}.final_layer_norm.weight"] = torch.randn_like(model.encoder.layers[i].final_layer_norm.weight)
        state_dict[f"{prefix}.final_layer_norm.bias"] = torch.randn_like(model.encoder.layers[i].final_layer_norm.bias)
    
    # Encoder layer norms
    state_dict["encoder.emb_layer_norm.weight"] = torch.randn_like(model.encoder.emb_layer_norm.weight)
    state_dict["encoder.emb_layer_norm.bias"] = torch.randn_like(model.encoder.emb_layer_norm.bias)
    state_dict["encoder.final_layer_norm.weight"] = torch.randn_like(model.encoder.final_layer_norm.weight)
    state_dict["encoder.final_layer_norm.bias"] = torch.randn_like(model.encoder.final_layer_norm.bias)
    
    # GBF layers
    state_dict["gbf_proj.linear1.weight"] = torch.randn_like(model.gbf_proj.linear1.weight)
    state_dict["gbf_proj.linear1.bias"] = torch.randn_like(model.gbf_proj.linear1.bias)
    state_dict["gbf_proj.linear2.weight"] = torch.randn_like(model.gbf_proj.linear2.weight)
    state_dict["gbf_proj.linear2.bias"] = torch.randn_like(model.gbf_proj.linear2.bias)
    
    state_dict["gbf.means.weight"] = torch.randn_like(model.gbf.means.weight)
    state_dict["gbf.stds.weight"] = torch.randn_like(model.gbf.stds.weight)
    state_dict["gbf.mul.weight"] = torch.randn_like(model.gbf.mul.weight)
    state_dict["gbf.bias.weight"] = torch.randn_like(model.gbf.bias.weight)
    
    # Classification head
    head_name = list(model.classification_heads.keys())[0]
    state_dict[f"classification_heads.{head_name}.dense.weight"] = torch.randn_like(model.classification_heads[head_name].dense.weight)
    state_dict[f"classification_heads.{head_name}.dense.bias"] = torch.randn_like(model.classification_heads[head_name].dense.bias)
    state_dict[f"classification_heads.{head_name}.out_proj.weight"] = torch.randn_like(model.classification_heads[head_name].out_proj.weight)
    state_dict[f"classification_heads.{head_name}.out_proj.bias"] = torch.randn_like(model.classification_heads[head_name].out_proj.bias)
    
    return {"model": state_dict} 