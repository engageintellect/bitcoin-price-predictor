#!/usr/bin/env bash

# Navigate to the project directory
cd /home/pi/bitcoin-price-predictor

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
cd /home/pi/bitcoin-price-predictor/app
python3 main.py

# Deactivate the virtual environment (optional, but good practice)
deactivate

