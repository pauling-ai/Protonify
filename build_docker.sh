#!/bin/bash
# Build Protonify Docker image

docker build -t protonify:latest .

echo ""
echo "Build complete! Run with:"
echo "  docker run --rm protonify --smiles \"CCO\" --ph 7.4 --template smart"
