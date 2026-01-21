#!/bin/bash
echo "Setting up EUR/GBP Predictor..."
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p static/css static/js templates trading_logs
echo "Setup complete!"