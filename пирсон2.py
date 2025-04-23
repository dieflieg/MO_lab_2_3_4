import numpy as np


# Определение функций и их градиентов
def rosenbrock(x):
    """Функция Розенброка (f1): 100*(x₁ - x₀²)² + (1 - x₀)²"""
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_gradient(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2)
    ])


def f3(x):
    """Функция f3: (x₂ - x₁²)² + (1 - x₁)²"""
    return (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f3_gradient(x):
    return np.array([
        -4 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        2 * (x[1] - x[0] ** 2)
    ])


def f4(x):
    """Функция f4: 100*(x₂ - x₁³)² + (1 - x₁)²"""
    return 100 * (x[1] - x[0] ** 3) ** 2 + (1 - x[0]) ** 2


def f4_gradient(x):
    return np.array([
        -600 * x[0] ** 2 * (x[1] - x[0] ** 3) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 3)
    ])


def golden_section_search(f, x, direction, a=0, b=2, epsilon=0.001, delta=0.0001, max_iter=200):
    """Метод золотого сечения для одномерной минимизации"""
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    for _ in range(max_iter):
        fc = f(x + c * direction)
        fd = f(x + d * direction)

        if abs(fc - fd) < delta:
            break

        if fc < fd:
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

        if abs(b - a) < epsilon:
            break

    return (a + b) / 2


def optimize_function(func, gradient_func, x0, epsilon=1e-3, max_iter=1000):
    """Обобщённая функция оптимизации"""
    x = x0.copy().astype(np.float64)
    eta = np.eye(2, dtype=np.float64)
    history = []

    for k in range(max_iter):
        grad = gradient_func(x)
        f_val = func(x)
        grad_norm = np.linalg.norm(grad)

        history.append({
            "Итер": k,
            "x": x.copy(),
            "f(x)": f_val,
            "||∇f||": grad_norm,
            "Останов": "Да" if grad_norm < epsilon else "Нет"
        })

        if grad_norm < epsilon:
            break

        direction = -eta @ grad
        alpha = golden_section_search(func, x, direction)
        x_new = x + alpha * direction

        grad_new = gradient_func(x_new)
        delta_x = x_new - x
        delta_g = grad_new - grad

        denominator = delta_x @ delta_g
        if abs(denominator) > 1e-12:
            numerator = np.outer(delta_x - eta @ delta_g, delta_x)
            eta += numerator / denominator

        if np.trace(eta) < 1e-6 or np.linalg.det(eta) < 1e-12:
            eta = np.eye(2)

        x = x_new

    return history


# Список функций для минимизации
functions = [
    {
        "name": "Розенброка (f1): 100*(x₁ - x₀²)² + (1 - x₀)²",
        "func": rosenbrock,
        "gradient": rosenbrock_gradient
    },
    {
        "name": "f3: (x₂ - x₁²)² + (1 - x₁)²",
        "func": f3,
        "gradient": f3_gradient
    },
    {
        "name": "f4: 100*(x₂ - x₁³)² + (1 - x₁)²",
        "func": f4,
        "gradient": f4_gradient
    }
]

# Начальная точка
x0 = np.array([-1.2, 1.0])

# Последовательно запускаем оптимизацию для каждой функции
for func_info in functions:
    print(f"\n{'-' * 50}")
    print(f"Минимизация функции: {func_info['name']}")
    print(f"{'-' * 50}")

    history = optimize_function(func_info['func'], func_info['gradient'], x0)

    # Вывод последних 10 итераций для краткости
    print(f"{'Итер':<5} | {'x':<35} | {'f(x)':<15} | {'||∇f||':<10} | {'Останов'}")
    print("-" * 85)
    for entry in history[-10:]:
        x_str = f"[{entry['x'][0]:.6f}, {entry['x'][1]:.6f}]"
        f_str = f"{entry['f(x)']:.4e}"
        grad_str = f"{entry['||∇f||']:.4e}"
        print(f"{entry['Итер']:<5} | {x_str:<35} | {f_str:<15} | {grad_str:<10} | {entry['Останов']}")

    final = history[-1]
    print("\nРезультат:")
    print(f"Точка минимума: [{final['x'][0]:.6f}, {final['x'][1]:.6f}]")
    print(f"Значение функции: {final['f(x)']:.2e}")
    print(f"Норма градиента: {final['||∇f||']:.2e}\n")