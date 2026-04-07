# Validation & Comparison

### Структура

```
src/
├── analytical_solution.py    # Аналитическое решение (JIT-совместимо)
└── validation.py             # Все функции валидации, метрик, графиков

experiments/
├── run_fdm_validation.py     # Запуск FDM
├── run_pinn_validation.py    # Запуск PINN
└── run_full_validation.py    # главный скрипт валидации

docs/
└── validation_report.md      # Научный отчет с таблицами и выводами

results/
├── fdm/
│   ├── fdm_u_num_N41.npy      # Численное решение FDM
│   ├── fdm_u_exact_N41.npy    # Точное решение на сетке
│   ├── metrics.csv            # Таблица сходимости
│   └── fdm_slice_t_0_1.png    # График
├── pinn/
│   ├── params.pkl             # Параметры обученной сети
│   ├── history.json           # История обучения
│   ├── metrics_by_time.json   # Ошибки по времени
│   └── *.png                  # Графики
└── figures/
    ├── comparison/            # Все сравнительные графики
    └── tables/                # Таблицы в Markdown и LaTeX
```

### Запуск валидации

```bash
# Сначала убедитесь, что результаты FDM и PINN вычислены:
python experiments/run_fdm_validation.py
python experiments/run_pinn_validation.py

# Затем запустите ПОЛНУЮ ВАЛИДАЦИЮ:
python -m experiments.run_full_validation --grid_size 41 --alpha 0.1 --t_max 1.0

# С опциями:
python -m experiments.run_full_validation --help
```

### Результаты

Скрипт генерирует:
- **Графики:** `fdm_convergence.png`, `error_vs_time.png`, `snapshots_t_*.png`, `pinn_loss_history.png`
- **Таблицы:** `comparison_table.md`, `comparison_table.tex`
- **Отчет:** [docs/validation_report.md](docs/validation_report.md)

### Ключевые метрики

| Метрика | FDM (N=41) | PINN (21³) |
|---------|-----------|-----------|
| Точность (L²) | $O(10^{-4})$ | $O(10^{-2})$ |
| Время обучения | ~2 сек | ~150 сек |
| Время инференса | ~10 мс | ~1 мс |

### Основные выводы

1. **FDM** показывает превосходную точность для гладких решений на регулярной сетке: вторая ошибка сходимости бесспорна
2. **PINN** обеспечивает гибкость и может обрабатывать нерегулярные геометрии, но требует значительных вычислительных затрат на обучение
3. **Гибридные подходы** могут сочетать преимущества обоих методов
