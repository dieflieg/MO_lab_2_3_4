import numpy as np
from tabulate import tabulate  # Для красивого вывода таблиц

def golden_section_search(f, a, b, tol=1e-9):
    """Метод золотого сечения для поиска минимума функции f на интервале [a, b]."""
    # Коэффициенты золотого сечения (1 - φ ≈ 0.618)
    phi1 = 0.381966011  # φ = (3 - √5)/2
    phi2 = 0.618033989  # 1 - φ = (√5 - 1)/2

    # Инициализация точек x1 и x2 внутри интервала
    x1 = a + phi1 * (b - a)
    x2 = a + phi2 * (b - a)
    f1, f2 = f(x1), f(x2)  # Значения функции в точках

    # Сужение интервала до достижения точности tol
    while abs(b - a) > tol:
        if f1 < f2:
            # Минимум в левой части: сдвигаем правую границу
            b, x2, f2 = x2, x1, f1
            x1 = a + phi1 * (b - a)
            f1 = f(x1)
        else:
            # Минимум в правой части: сдвигаем левую границу
            a, x1, f1 = x1, x2, f2
            x2 = a + phi2 * (b - a)
            f2 = f(x2)

    return (a + b) / 2  # Возвращаем середину финального интервала

def find_interval(f, center=0.0, radius=5.0):
    """Находит интервал [a, b], в котором содержится минимум функции f."""
    a, b = center - radius, center + radius  # Начальный интервал

    # Расширяем правую границу, пока функция убывает
    while f(b) < f(b - radius / 2):
        b += radius
        radius *= 2  # Увеличиваем радиус расширения

    # Расширяем левую границу, пока функция убывает
    while f(a) < f(a + radius / 2):
        a -= radius
        radius *= 2

    return a, b

def rosenbrock_method(f, x0, eps=1e-3, delta_eps=1e-9, radius=10.0, max_iter=1000):
    """Реализация метода Розенброка для многомерной оптимизации."""
    n = len(x0)
    S = np.eye(n)
    x = np.array(x0, dtype=float)
    iteration_data = []
    prev_f = f(*x)
    prev_x = x.copy()

    for k in range(max_iter):
        x_current = x.copy()
        lambdas = np.zeros(n)

        # Минимизация по направлениям
        for i in range(n):
            direction = S[i]
            def f_dir(lmbda):
                return f(*(x_current + lmbda * direction))
            a, b = find_interval(f_dir, radius=radius)
            lambda_opt = golden_section_search(f_dir, a, b, eps)
            lambdas[i] = lambda_opt
            x_current += lambda_opt * direction

        # Формирование векторов A и ортогонализация
        A = [sum(lambdas[j] * S[j] for j in range(l, n)) for l in range(n)]
        S_new = []
        for l in range(n):
            B = A[l].copy()
            for m in range(l):
                B -= np.dot(B, S_new[m]) * S_new[m]
            norm = np.linalg.norm(B)
            B = B / norm if norm > eps else np.zeros_like(B)
            S_new.append(B)
        S = np.array(S_new)

        x = x_current.copy()
        current_f = f(*x)

        # Запись данных с округлением
        iteration_data.append([
            k + 1,
            *np.round(x, 6),
            round(current_f, 6),
            np.round(lambdas, 6).tolist(),
            np.round(S, 6).tolist()
        ])

        # Проверка критериев остановки
        delta_f = abs(current_f - prev_f)
        delta_x = np.linalg.norm(x - prev_x)
        if (np.all(np.abs(lambdas) < eps)) or (delta_f < delta_eps) or (delta_x < delta_eps):
            break

        prev_f = current_f
        prev_x = x.copy()

    # Вывод таблицы
    print(tabulate(
        iteration_data,
        headers=["Iter", "x1", "x2", "f(x)", "Lambdas", "Directions"],
        tablefmt="grid",
        floatfmt=".6f"
    ))

    # Точные значения последней итерации
    print("\nТочные значения на последней итерации:")
    print(f"x1 = {x[0]}, x2 = {x[1]}, f(x) = {f(*x)}")
    return x

# Тестовые функции для демонстрации работы метода
def f1(x1, x2):
    """Пример квадратичной функции: 10*(x1 + x2 - 10)^2 + (x1 - x2 + 4)^2."""
    return 10 * (x1 + x2 - 10)**2 + (x1 - x2 + 4)**2

def f2(x1, x2):
    """Функция Розенброка: 100*(x2 - x1^2)^2 + (1 - x1)^2."""
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Пример использования метода
print("Minimizing f1:")
x_opt1 = rosenbrock_method(f1, [0.0, 0.0], delta_eps=1e-7)
print("\nOptimal solution for f1:", x_opt1)

print("\nMinimizing f2:")
x_opt2 = rosenbrock_method(f2, [-1.2, 1.0], delta_eps=1e-7)
print("\nOptimal solution for f2:", x_opt2)