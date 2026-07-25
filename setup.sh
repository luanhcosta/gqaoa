#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch 2.7+ with CUDA 12.8 wheels — required for Blackwell (RTX 5000 series, sm_120)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev,hpo,gpu]"

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "To launch the MLflow UI:"
echo "  mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db"
echo ""
echo "To run an experiment:"
echo "  gqaoa-run"
echo "  gqaoa-stability-check"
echo "  gqaoa-bracket"
