#!/usr/bin/env bash

cd /home/pi/bitcoin-price-predictor
source venv/bin/activate

cd /home/pi/bitcoin-price-predictor/app
python3 api.py
