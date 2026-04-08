#!/bin/bash

echo "🚀 Запуск бенчмарка: 3D Уравнение теплопроводности (JAX)"
echo "--------------------------------------------------------"

# 1. Запуск FDM
echo "⏳ [1/3] Решение классическим сеточным методом (FDM)..."
python src/experiments/run_fdm_validation.py --config configs/fdm.yaml

# 2. Запуск PINN
echo "⏳ [2/3] Обучение и инференс PINN..."
python src/experiments/run_pinn_validation.py --config configs/pinn.yaml

# 3. Сравнение с аналитическим решением
echo "📊 [3/3] Генерация отчета и сравнение с аналитическим решением..."
python src/experiments/run_comparison.py

echo "✅ Готово! Графики и метрики L2/L_inf сохранены в папке results/"
