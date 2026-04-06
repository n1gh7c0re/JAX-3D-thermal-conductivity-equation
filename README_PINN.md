# PINN Solver 3D

**Реализация Physics-Informed Neural Network (PINN)** для решения нестационарного 3D уравнения теплопроводности

$$
u_t = \alpha (u_{xx} + u_{yy} + u_{zz})$$

с нулевыми граничными условиями Дирихле.

### Назначение кода
- Решает тестовую задачу для 3D уравнения теплопроводности с помощью **Physics-Informed Neural Network (PINN)**.
- Использует **JAX** для вычисления производных, невязки уравнения и обучения нейросети.
- Сопоставим с FDM-решателем, так как использует **тот же аналитический валидационный кейс**.
- Содержит усиленную постановку PINN: граничные и начальные условия могут быть **жёстко встроены** в аппроксимацию решения, что улучшает устойчивость обучения и качество результата.
- Выдаёт метрики ошибок $L_2$ и $L_\infty$, кривые loss и графики сравнения с точным решением.

### Описание кода

**src/pinn_solver.py** — основной PINN-солвер:
- `exact_solution` — точное аналитическое решение
- `initial_condition` — начальное условие `sin(πx)·sin(πy)·sin(πz)`
- `init_mlp_params` / `forward_mlp` — инициализация и прямой проход MLP
- `apply_output_transform` — жёсткое наложение начальных и граничных условий
- `sample_residual_points` — генерация точек коллокации внутри области
- `sample_boundary_points` — генерация точек на границе куба
- `sample_initial_points` — генерация точек при `t = 0`
- `pde_residual_single` — вычисляет невязку уравнения теплопроводности
- `loss_terms` — loss-функция PINN
- `train_pinn` — процедура обучения с ресэмплингом точек и scheduler learning rate
- `compute_error_metrics` — расчёт ошибок на регулярной сетке

**experiments/run_pinn_validation.py** — запуск обучения и валидации:
- обучает PINN на аналитическом тестовом кейсе
- сохраняет историю обучения
- строит графики loss и срезы решения
- сохраняет таблицы ошибок для сравнения с FDM

**tests/test_pinn_smoke.py** — быстрый smoke test:
- проверяет импорты
- проверяет, что короткое обучение запускается без падения

### Постановка валидационной задачи

Рассматривается область:

$$(x, y, z) \in [0,1]^3, \qquad t \in [0,T]$$

Начальное условие:

$$u(x,y,z,0)=\sin(\pi x)\sin(\pi y)\sin(\pi z)$$

Граничные условия:

$$u=0 \quad \text{на всех гранях куба}$$

Точное аналитическое решение:

$$u(x,y,z,t)=e^{-3\alpha \pi^2 t}\sin(\pi x)\sin(\pi y)\sin(\pi z)$$

### Результат работы программы

При запуске `python experiments/run_pinn_validation.py` будут получены:

1. История обучения PINN:
   - общий `loss`
   - `pde_loss`
   - при мягких ограничениях также `bc_loss` и `ic_loss`

2. Файлы с результатами:
   - `results/pinn/loss_curves.png`
   - `results/pinn/learning_rate.png`
   - `results/pinn/metrics.csv`
   - `results/pinn/metrics_by_time.json`
   - `results/pinn/history.json`
   - `results/pinn/params.pkl`
   - `results/pinn/slice_comparison_t_*.png`

3. Таблицы ошибок $L_2$ / $L_\infty$ по времени.

### Установка и запуск
```bash
pip install -r requirements_pinn.txt
python tests/test_pinn_smoke.py
python experiments/run_pinn_validation.py
```

