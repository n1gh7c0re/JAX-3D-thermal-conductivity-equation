# PINN Solver 3D

**Реализация Physics-Informed Neural Network (PINN)** для решения нестационарного 3D уравнения теплопроводности

$$
u_t = \alpha (u_{xx} + u_{yy} + u_{zz})
$$

с нулевыми граничными условиями Дирихле.

### Назначение кода
- Решает тестовую задачу для 3D уравнения теплопроводности с помощью **Physics-Informed Neural Network (PINN)**.
- Использует **JAX** для вычисления производных, невязки уравнения и обучения нейросети.
- Проверяет корректность реализации через сравнение с точным аналитическим решением.
- Выдаёт метрики ошибок $L_2$ и $L_\infty$, кривые loss и графики сравнения с точным решением — именно то, что требуется в отчёте курсового проекта.

### Описание кода

**src/pinn_solver.py** — основной PINN-солвер:
- `exact_solution` — задаёт точное аналитическое решение
- `initial_condition` — создаёт начальное условие `sin(πx)·sin(πy)·sin(πz)`
- `init_mlp` / `mlp_forward` — инициализация и прямой проход нейросети
- `sample_residual_points` — генерация точек коллокации внутри области
- `sample_boundary_points` — генерация точек на границе куба
- `sample_initial_points` — генерация точек при `t = 0`
- `pde_residual` — вычисляет невязку уравнения теплопроводности
- `loss_fn` — общая loss-функция: PDE + boundary + initial
- `train_pinn` — процедура обучения PINN
- `evaluate_on_grid` — расчёт решения и ошибок на регулярной сетке

**experiments/run_pinn_validation.py** — запуск обучения и валидации:
- Обучает PINN на аналитическом тестовом кейсе
- Сравнивает предсказание сети с точным решением
- Сохраняет таблицы ошибок, историю loss и графики срезов решения

### Постановка валидационной задачи

Рассматривается область:

$$
(x, y, z) \in [0,1]^3, \qquad t \in [0,T]
$$

Начальное условие:

$$
u(x,y,z,0)=\sin(\pi x)\sin(\pi y)\sin(\pi z)
$$

Граничные условия:

$$
u=0 \quad \text{на всех гранях куба}
$$

Точное аналитическое решение:

$$
u(x,y,z,t)=e^{-3\alpha \pi^2 t}\sin(\pi x)\sin(\pi y)\sin(\pi z)
$$

### Результат работы программы

При запуске `python experiments/run_pinn_validation.py` будут получены:

1. История обучения PINN:
   - общий `loss`
   - `pde_loss`
   - `bc_loss`
   - `ic_loss`

2. Файлы с результатами:
   - `results/pinn/loss_curves.png`
   - `results/pinn/metrics.csv`
   - `results/pinn/metrics_by_time.json`
   - `results/pinn/history.json`
   - `results/pinn/params.pkl`
   - `results/pinn/slice_comparison_t_*.png`

3. Таблицы ошибок $L_2$ / $L_\infty$ по времени, которые можно напрямую использовать в сравнении с FDM-решателем.

### Установка и запуск
```bash
pip install -r requirements_pinn.txt
python experiments/run_pinn_validation.py
```
