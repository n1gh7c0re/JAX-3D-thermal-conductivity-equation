#!/bin/bash
set -e

export PYTHONPATH=.

echo "🚀 Быстрый demo-запуск PINN"
echo "--------------------------"
python3 experiments/run_pinn_demo.py
