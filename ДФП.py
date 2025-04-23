import numpy as np

# Функции и их градиенты
def f2(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def f2_gradient(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

def f3(x):
    return (x[1] - x[0]**2)**2 + (1 - x[0])**2

def f3_gradient(x):
    dx1 = -4 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dx2 = 2 * (x[1] - x[0]**2)
    return np.array([dx1, dx2])

def f4(x):
    return 100 * (x[1] - x[0]**3)**2 + (1 - x[0])**2

def f4_gradient(x):
    dx1 = -600 * x[0]**2 * (x[1] - x[0]**3) - 2 * (1 - x[0])
    dx2 = 200 * (x[1] - x[0]**3)
    return np.array([dx1, dx2])

def golden_section_search(f, x, direction, a=0, b=2, epsilon=0.001, delta=0.0001, max_iter=200):
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

def dfp_method(x0, func, grad_func, epsilon=1e-3, max_iter=100):
    """Метод Дэвидона-Флешчера-Пауэлла (DFP)."""
    x = x0.copy().astype(np.float64)
    eta = np.eye(2, dtype=np.float64)
    history = []
    for k in range(max_iter):
        grad = grad_func(x)
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
        grad_new = grad_func(x_new)
        delta_x = (x_new - x).reshape(-1, 1)
        delta_g = (grad_new - grad).reshape(-1, 1)
        numerator1 = delta_x @ delta_x.T
        denominator1 = delta_x.T @ delta_g
        eta_delta_g = eta @ delta_g
        numerator2 = eta_delta_g @ eta_delta_g.T
        denominator2 = delta_g.T @ eta_delta_g
        if abs(denominator1) > 1e-12 and abs(denominator2) > 1e-12:
            eta += (numerator1 / denominator1) - (numerator2 / denominator2)
        if np.trace(eta) < 1e-6 or np.linalg.det(eta) < 1e-12:
            eta = np.eye(2)
        x = x_new
    return history

def print_results(history, func_name):
    print(f"\nРезультаты для {func_name}:")
    print(f"{'Итер':<5} | {'x':<35} | {'f(x)':<15} | {'||∇f||':<10} | {'Останов'}")
    print("-" * 85)
    for entry in history:
        x_str = f"[{entry['x'][0]:.6f}, {entry['x'][1]:.6f}]"
        f_str = f"{entry['f(x)']:.4e}"
        grad_str = f"{entry['||∇f||']:.4e}"
        print(f"{entry['Итер']:<5} | {x_str:<35} | {f_str:<15} | {grad_str:<10} | {entry['Останов']}")
    final = history[-1]
    print(f"\nФинальный результат для {func_name}:")
    print(f"Точка минимума: [{final['x'][0]:.6f}, {final['x'][1]:.6f}]")
    print(f"Значение функции: {final['f(x)']:.2e}")
    print(f"Норма градиента: {final['||∇f||']:.2e}\n")

# Начальная точка
x0 = np.array([-1.2, 1.0])

# Запуск для f2
history_f2 = dfp_method(x0, f2, f2_gradient, epsilon=1e-3, max_iter=1000)
print_results(history_f2, "f2")

# Запуск для f3
history_f3 = dfp_method(x0, f3, f3_gradient, epsilon=1e-3, max_iter=1000)
print_results(history_f3, "f3")

# Запуск для f4
history_f4 = dfp_method(x0, f4, f4_gradient, epsilon=1e-3, max_iter=1000)
print_results(history_f4, "f4")