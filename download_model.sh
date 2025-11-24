#!/bin/bash
# Download Protonify model weights

MODEL_URL="https://github.com/pabloPauling/protonify/releases/download/v0.1.0/t_dwar_v_novartis_a_b.pt"
MODEL_DIR="protonify/models"
MODEL_FILE="$MODEL_DIR/t_dwar_v_novartis_a_b.pt"

# Create directory if not exists
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE"
    echo "Use --force to re-download"
    if [ "$1" != "--force" ]; then
        exit 0
    fi
    echo "Re-downloading..."
fi

echo "Downloading model (~500MB)..."
echo "From: $MODEL_URL"
echo "To: $MODEL_FILE"
echo ""

curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"

if [ $? -eq 0 ]; then
    echo ""
    echo "Download complete!"
    echo "Model saved to: $MODEL_FILE"
else
    echo "Download failed!"
    rm -f "$MODEL_FILE"
    exit 1
fi
