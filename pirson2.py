import numpy as np


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_gradient(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2)
    ])


def golden_section_search(f, x, direction, a=0, b=2, tol=1e-8, max_iter=200):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    for _ in range(max_iter):
        if f(x + c * direction) < f(x + d * direction):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

        if abs(b - a) < tol:
            break

    return (a + b) / 2


def pearson_method_2(x0, epsilon=1e-8, max_iter=1000, f_tol=1e-12):
    x = x0.copy().astype(np.float64)
    eta = np.eye(2, dtype=np.float64)
    history = []
    prev_f = np.inf

    for k in range(max_iter):
        grad = rosenbrock_gradient(x)
        f_val = rosenbrock(x)
        grad_norm = np.linalg.norm(grad)

        # Критерии останова
        stop_cond = grad_norm < epsilon or abs(f_val - prev_f) < f_tol
        history.append({
            "Итер": k,
            "x": x.copy(),
            "f(x)": f_val,
            "||∇f||": grad_norm,
            "Останов": "Да" if stop_cond else "Нет"
        })

        if stop_cond:
            break
        prev_f = f_val

        # Направление спуска
        direction = -eta @ grad

        # Поиск шага
        alpha = golden_section_search(rosenbrock, x, direction)
        x_new = x + alpha * direction
        grad_new = rosenbrock_gradient(x_new)

        delta_x = x_new - x
        delta_g = grad_new - grad

        # Обновление матрицы η
        denominator = delta_x @ delta_g
        if abs(denominator) > 1e-12:
            numerator = np.outer(delta_x - eta @ delta_g, delta_x)
            eta += numerator / denominator

        # Коррекция матрицы
        if np.trace(eta) < 1e-6 or np.linalg.det(eta) < 1e-12:
            eta = np.eye(2)

        x = x_new

    return history


# Запуск
x0 = np.array([-1.2, 1.0])
history = pearson_method_2(x0, epsilon=1e-8, max_iter=1000)

# Вывод результатов
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
print(f"Норма градиента: {final['||∇f||']:.2e}")