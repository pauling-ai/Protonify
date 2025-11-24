# Protonify

Protonation state prediction and microstate distribution for molecules at a given pH.

Given a SMILES string and pH value(s), this tool:
1. Enumerates all possible protonation states
2. Optionally enumerates tautomers
3. Predicts free energy for each microstate using a neural network
4. Calculates Boltzmann distribution at the given pH
5. Returns the most probable protonated SMILES

## Installation

### From GitHub (recommended)

```bash
# Clone the repository
git clone https://github.com/PabloPauling/Protonify.git
cd Protonify

# Build the wheel
pip install build
python -m build --wheel

# Install the package
pip install dist/protonify-0.1.0-py3-none-any.whl
```

Note: Editable installs (`pip install -e .`) require setuptools with PEP 660 support.

### Download Model Weights

The model weights (~500MB) must be downloaded separately:

```bash
# Using the provided script
chmod +x download_model.sh
./download_model.sh

# Or manually download from GitHub Releases and place in protonify/models/
```

### Using Docker

If you prefer to use Docker, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/PabloPauling/Protonify.git
cd Protonify

# 2. Download the model weights
chmod +x download_model.sh
./download_model.sh

# 3. Build the Docker image
chmod +x build_docker.sh
./build_docker.sh
# Or directly: docker build -t protonify:latest .

# 4. Run predictions
docker run --rm protonify --smiles "CCO" --ph 7.4 --template smart

# Multiple pH values
docker run --rm protonify --smiles "NCC(=O)O" --ph "2.0,7.4,10.0" --template smart

# Quiet mode (only SMILES output)
docker run --rm protonify --smiles "CC(=O)O" --ph 7.4 --template smart -q

```

## Quick Start

```bash
# Verify installation
protonify --help

# Basic prediction
protonify --smiles "CCO" --ph 7.4 --template smart
```

## Usage

### Command Line

```bash
# Basic usage (model auto-downloads on first run)
protonify --smiles "CCO" --ph 7.4 --template smart

# Use GPU for faster inference
protonify --smiles "CCO" --ph 7.4 --template smart --gpu

# Multiple pH values
protonify --smiles "CCO" --ph "7.0,7.4,8.0" --template smart

# Enable tautomer enumeration (slower but more thorough)
protonify --smiles "CCO" --ph 7.4 --template smart --enumerate-tautomers

# Use custom model
protonify --smiles "CCO" --ph 7.4 --template smart --model-path /path/to/model.pt

# Quiet mode (only output SMILES, useful for pipelines)
protonify --smiles "CCO" --ph 7.4 --template smart -q
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--smiles` | Yes | SMILES string of the molecule to analyze |
| `--ph` | Yes | pH value(s), single or comma-separated (e.g., "7.4" or "7.0,7.4,8.0") |
| `--template` | Yes | Template type: `simple` or `smart` |
| `--model-path` | No | Path to custom model (auto-downloads default model if not specified) |
| `--enumerate-tautomers` | No | Enable tautomer enumeration (disabled by default for speed) |
| `--gpu` | No | Use GPU for inference (requires CUDA) |
| `--quiet`, `-q` | No | Suppress verbose output, only print final SMILES (useful for pipelines) |

### Output

For each pH value, the tool outputs:
- **Most probable microstate**: The SMILES of the most likely protonation state
- **Charge**: The formal charge of that microstate
- **Probability**: The fraction/probability of that microstate
- **Full distribution**: All microstates with their probabilities

### Python API

```python
from protonify import predict_protonation

# One-line prediction (model auto-downloads on first use)
result = predict_protonation("CCO", ph=7.4)
print(result["smiles"])      # Most probable protonated SMILES
print(result["charge"])      # Formal charge
print(result["probability"]) # Probability

# Multiple pH values
results = predict_protonation("CCO", ph=[7.0, 7.4, 8.0])
for r in results:
    print(f"pH {r['ph']}: {r['smiles']} (charge={r['charge']})")

# Use simple template (faster) or smart template (default, more accurate)
result = predict_protonation("CCO", ph=7.4, template="simple")
result = predict_protonation("CCO", ph=7.4, template="smart")

# With tautomer enumeration (slower, more thorough)
result = predict_protonation("CCO", ph=7.4, skip_tautomers=False)

# Use custom model
result = predict_protonation("CCO", ph=7.4, model_path="/path/to/model.pt")

# Use GPU (faster for large batches)
result = predict_protonation("CCO", ph=7.4, use_gpu=True)

# Verbose mode (show logging, disabled by default)
result = predict_protonation("CCO", ph=7.4, quiet=False)
```

### Testing the API

To verify the Python API works correctly, run the test script:

```bash
python test_api.py
```

This runs several examples including basic predictions, multiple pH values, template comparison, and tautomer enumeration.

## Model

The default model (~500MB) is loaded using the following priority:

1. **Explicit path** - `--model-path` argument or `model_path=` parameter
2. **Environment variable** - `PROTONIFY_MODEL_PATH=/path/to/model.pt`
3. **Bundled model** - If installed with `pip install .` and model is in `protonify/models/`
4. **Cached model** - Previously downloaded to `~/.cache/protonify/`
5. **Auto-download** - Downloads from GitHub Releases on first use

### Model Options

| Method | Description |
|--------|-------------|
| **Bundled** | Model included in package installation |
| **Auto-download** | Model downloads automatically on first use |
| **Custom path** | Use `--model-path` or `model_path=` argument |
| **Environment variable** | Set `PROTONIFY_MODEL_PATH=/path/to/model.pt` |

### Pre-download model

```python
from protonify import download_model
download_model()  # Downloads to ~/.cache/protonify/
```

### Manual download

Download the model weights from [GitHub Releases](https://github.com/pabloPauling/protonify/releases) and place in `~/.cache/protonify/` or set `PROTONIFY_MODEL_PATH`.

## Performance

By default, tautomer enumeration is **disabled** for speed. This significantly reduces computation time while still enumerating all protonation states.

For more thorough analysis (e.g., when accuracy is critical), enable tautomer enumeration:

```bash
# CLI
protonify --smiles "CCO" --ph 7.4 --template smart --enumerate-tautomers

# Python
result = predict_protonation("CCO", ph=7.4, skip_tautomers=False)
```

## Dependencies

- torch >= 2.0.0
- numpy >= 1.21
- pandas >= 1.3
- scipy >= 1.7
- rdkit >= 2023.0.0

## Troubleshooting

### Model not found / HTTP 404 error

If you get an error like `Failed to download model: HTTP Error 404: Not Found`, the automatic download failed. Solutions:

1. **Manual download**: Download the model from [GitHub Releases](https://github.com/PabloPauling/Protonify/releases) and place it in `~/.cache/protonify/`

2. **Set environment variable**:
   ```bash
   export PROTONIFY_MODEL_PATH="/path/to/t_dwar_v_novartis_a_b.pt"
   ```

### Multiple Python installations (conda, pyenv, system Python)

If the CLI works but the Python API fails with a model error, you likely have multiple Python installations with protonify installed in different locations.

**Symptoms:**
- `protonify --smiles "CCO" --ph 7.4 --template smart` works
- `python -c "from protonify import predict_protonation; predict_protonation('CCO', ph=7.4)"` fails with model not found

**Cause:** The CLI uses one Python installation (with the bundled model) while your script uses another (without the model).

**Solution:** Set the environment variable pointing to the existing model:

```bash
# Find where the model is installed
find ~/.local /usr -name "t_dwar_v_novartis_a_b.pt" 2>/dev/null

# Set the environment variable (add to ~/.bashrc for persistence)
export PROTONIFY_MODEL_PATH="/path/found/above/t_dwar_v_novartis_a_b.pt"
```

**Alternative:** Install protonify in your active Python environment:
```bash
# Make sure you're in the correct environment
which python  # Verify this is your intended Python

# Reinstall protonify
pip uninstall protonify
pip install protonify
```

### Verifying your installation

Run this to check if everything is working:

```python
from protonify import predict_protonation
result = predict_protonation("CCO", ph=7.4)
print(f"Success! Result: {result['smiles']}")
```

## About This Project

Protonify is a **wrapper/interface** built on top of the original **UniPKa** project by **DP Technology Corp**.

**Key contribution:** UniPKa visualizes the distribution of microstates vs pH in graphs, but does not offer a direct interface to obtain the most probable SMILES at a specific pH. Protonify adds this functionality: an API and CLI that directly return the most probable protonation state for integration into automated pipelines.

The core prediction model and methodology are from UniPKa - this project adds:
- **pH-to-SMILES interface**: Input pH, get the most probable protonated SMILES
- Simplified CLI interface
- Python API for easy integration
- Automatic model downloading

**Developed by:** Pablo Villanueva Cuñado ([@PabloPauling](https://github.com/PabloPauling)) at **[Pauling AI](https://www.pauling.ai/)**

## Acknowledgments

This project is **based entirely on UniPKa** developed by **DP Technology Corp**. All credit for the scientific methodology and neural network architecture goes to the original authors.

- **[Uni-pKa](https://github.com/dptech-corp/Uni-pKa)**: Neural network-based pKa prediction using Uni-Mol architecture
- **[DP Technology Corp](https://github.com/dptech-corp)**: For developing and open-sourcing the foundational model and methodology
- **Uni-Mol**: The underlying molecular representation framework

We are grateful to DP Technology for making their work open-source.

### References

If you use this software in your research, please cite:

```bibtex
@software{protonify2025,
  author = {Villanueva Cuñado, Pablo},
  title = {Protonify: Protonation State Prediction for Molecules},
  year = {2025},
  organization = {Pauling AI},
  url = {https://github.com/PabloPauling/Protonify}
}

@article{unipka2024,
  title={Bridging Machine Learning and Thermodynamics for Accurate pKa Prediction},
  author={Zhou, Gengmo and others},
  journal={JACS Au},
  year={2024},
  publisher={American Chemical Society},
  url={https://github.com/dptech-corp/Uni-pKa}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is an independent open-source project. It is not affiliated, associated, sponsored, or endorsed by Protonify Corporation, nor by any other entity with a similar name. Any similarity in naming is purely coincidental and does not imply any commercial relationship.
