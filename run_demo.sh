#!/bin/bash
set -e

export PYTHONPATH=.

echo "🚀 Запуск бенчмарка: 3D Уравнение теплопроводности (JAX)"
echo "--------------------------------------------------------"

# 1. Запуск FDM
echo "⏳ [1/3] Решение классическим сеточным методом (FDM)..."
python3 experiments/run_fdm_validation.py

# 2. Запуск PINN
echo "⏳ [2/3] Обучение и инференс PINN..."
python3 experiments/run_pinn_validation.py

# 3. Сравнение с аналитическим решением
echo "📊 [3/3] Генерация отчета и сравнение с аналитическим решением..."
python3 experiments/run_full_validation.py --grid_size 41 --alpha 1.0 --t_max 0.1

echo "✅ Готово! Графики и метрики L2/L_inf сохранены в папке results/"
