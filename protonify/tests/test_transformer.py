import torch
import pytest
from protonify.transformer import (
    SelfMultiheadAttention,
    TransformerEncoderLayer,
    TransformerEncoderWithPair,
    get_activation_fn,
    softmax_dropout
)

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 10

@pytest.fixture
def embed_dim():
    return 64

@pytest.fixture
def num_heads():
    return 4

@pytest.fixture
def sample_input(batch_size, seq_len, embed_dim):
    return torch.randn(batch_size, seq_len, embed_dim)

def test_activation_functions():
    """Test all activation functions are available and work correctly"""
    x = torch.tensor([-1.0, 0.0, 1.0])
    
    # Test each activation function
    assert torch.allclose(get_activation_fn("relu")(x), torch.relu(x))
    assert torch.allclose(get_activation_fn("gelu")(x), torch.nn.functional.gelu(x))
    assert torch.allclose(get_activation_fn("tanh")(x), torch.tanh(x))
    assert torch.allclose(get_activation_fn("linear")(x), x)
    
    # Test invalid activation
    with pytest.raises(RuntimeError):
        get_activation_fn("invalid")

def test_softmax_dropout():
    """Test softmax dropout function with various inputs"""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Test basic softmax dropout
    result = softmax_dropout(x, dropout_prob=0.0, is_training=False)
    expected = torch.nn.functional.softmax(x, dim=-1)
    assert torch.allclose(result, expected)
    
    # Test with mask
    mask = torch.tensor([[0.0, -float('inf')], [0.0, 0.0]])
    result = softmax_dropout(x, dropout_prob=0.0, is_training=False, mask=mask)
    assert result[0, 1] == 0.0  # Masked position should be 0
    
    # Test with bias
    bias = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = softmax_dropout(x, dropout_prob=0.0, is_training=False, bias=bias)
    assert not torch.allclose(result, expected)  # Should be different due to bias

def test_self_multihead_attention(sample_input, embed_dim, num_heads):
    """Test SelfMultiheadAttention module"""
    model = SelfMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1
    )
    
    # Test forward pass
    output = model(sample_input)
    assert output.shape == sample_input.shape
    
    # Test with key padding mask
    key_padding_mask = torch.zeros(sample_input.size(0), sample_input.size(1), dtype=torch.bool)
    output = model(sample_input, key_padding_mask=key_padding_mask)
    assert output.shape == sample_input.shape
    
    # Test with attention bias - reshape to match multi-head attention
    attn_bias = torch.zeros(sample_input.size(0) * num_heads, sample_input.size(1), sample_input.size(1))
    output = model(sample_input, attn_bias=attn_bias)
    assert output.shape == sample_input.shape
    
    # Test return attention weights with bias
    output, attn_weights, attn_probs = model(
        sample_input, 
        attn_bias=attn_bias,  # Provide attn_bias when using return_attn
        return_attn=True
    )
    assert output.shape == sample_input.shape
    assert attn_weights.shape == (sample_input.size(0) * num_heads, sample_input.size(1), sample_input.size(1))
    assert attn_probs.shape == (sample_input.size(0) * num_heads, sample_input.size(1), sample_input.size(1))

def test_transformer_encoder_layer(sample_input, embed_dim, num_heads):
    """Test TransformerEncoderLayer module"""
    model = TransformerEncoderLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=embed_dim * 4,
        attention_heads=num_heads,
        dropout=0.1
    )
    
    # Test forward pass
    output = model(sample_input)
    assert output.shape == sample_input.shape
    
    # Test with attention bias - reshape to match multi-head attention
    attn_bias = torch.zeros(sample_input.size(0) * num_heads, sample_input.size(1), sample_input.size(1))
    output = model(sample_input, attn_bias=attn_bias)
    assert output.shape == sample_input.shape
    
    # Test with padding mask
    padding_mask = torch.zeros(sample_input.size(0), sample_input.size(1), dtype=torch.bool)
    output = model(sample_input, padding_mask=padding_mask)
    assert output.shape == sample_input.shape
    
    # Test return attention with bias
    output, attn_weights, attn_probs = model(
        sample_input, 
        attn_bias=attn_bias,  # Provide attn_bias when using return_attn
        return_attn=True
    )
    assert output.shape == sample_input.shape
    assert attn_weights is not None
    assert attn_probs is not None

def test_transformer_encoder_with_pair(sample_input, embed_dim, num_heads):
    """Test TransformerEncoderWithPair module"""
    model = TransformerEncoderWithPair(
        encoder_layers=2,
        embed_dim=embed_dim,
        ffn_embed_dim=embed_dim * 4,
        attention_heads=num_heads,
        dropout=0.1
    )
    
    # Create attention mask with correct shape for multi-head attention
    attn_mask = torch.zeros(sample_input.size(0) * num_heads, sample_input.size(1), sample_input.size(1))
    
    # Test forward pass
    x, attn, delta_pair_repr, x_norm, delta_pair_repr_norm = model(sample_input, attn_mask=attn_mask)
    assert x.shape == sample_input.shape
    assert attn.shape == (sample_input.size(0), sample_input.size(1), sample_input.size(1), num_heads)
    assert delta_pair_repr.shape == (sample_input.size(0), sample_input.size(1), sample_input.size(1), num_heads)
    assert isinstance(x_norm, torch.Tensor)
    assert isinstance(delta_pair_repr_norm, torch.Tensor)
    
    # Test with padding mask
    padding_mask = torch.zeros(sample_input.size(0), sample_input.size(1), dtype=torch.bool)
    x, attn, delta_pair_repr, x_norm, delta_pair_repr_norm = model(
        sample_input, 
        attn_mask=attn_mask,
        padding_mask=padding_mask
    )
    assert x.shape == sample_input.shape

def test_transformer_post_ln(sample_input, embed_dim, num_heads):
    """Test transformer with post layer normalization"""
    # Test encoder layer with post_ln
    model = TransformerEncoderLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=embed_dim * 4,
        attention_heads=num_heads,
        dropout=0.1,
        post_ln=True
    )
    output = model(sample_input)
    assert output.shape == sample_input.shape
    
    # Test full encoder with post_ln
    model = TransformerEncoderWithPair(
        encoder_layers=2,
        embed_dim=embed_dim,
        ffn_embed_dim=embed_dim * 4,
        attention_heads=num_heads,
        dropout=0.1,
        post_ln=True
    )
    # Create attention mask with correct shape for multi-head attention
    attn_mask = torch.zeros(sample_input.size(0) * num_heads, sample_input.size(1), sample_input.size(1))
    x, attn, delta_pair_repr, x_norm, delta_pair_repr_norm = model(sample_input, attn_mask=attn_mask)
    assert x.shape == sample_input.shape 