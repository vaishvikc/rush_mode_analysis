#!/bin/bash

echo "Setting up Python virtual environment for Mode Analysis..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment manually, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the marimo notebook, use:"
echo "  ./run.sh"