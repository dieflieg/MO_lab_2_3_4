import numpy as np
from tabulate import tabulate


def golden_section_search(f, a, b, tol=1e-9):
    """Метод золотого сечения для одномерной минимизации."""
    phi1 = 0.381966011  # Коэффициент золотого сечения
    phi2 = 0.618033989

    x1 = a + phi1 * (b - a)
    x2 = a + phi2 * (b - a)
    f1, f2 = f(x1), f(x2)

    while abs(b - a) > tol:  # Критерий сходимости
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + phi1 * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi2 * (b - a)
            f2 = f(x2)

    return (a + b) / 2

def hooke_jeeves(f, x0, delta=2, epsilon=1e-3, radius=2.0):
    """Метод Хука и Дживса с выводом всех итераций."""
    x = np.array(x0, dtype=float)
    n = len(x)
    history = []
    total_iter = 0
    successful_iter = 0

    while delta > epsilon:
        total_iter += 1
        x_prev = np.copy(x)
        x_new = np.copy(x)
        improved = False

        # Исследующий поиск
        for i in range(n):
            f_x = f(x_new)
            x_new[i] += delta
            if f(x_new) >= f_x:
                x_new[i] -= 2 * delta
                if f(x_new) >= f_x:
                    x_new[i] += delta
                else:
                    improved = True
            else:
                improved = True

        # Запись данных с округлением
        if improved:
            successful_iter += 1
            direction = x_new - x_prev
            if np.linalg.norm(direction) > 0:
                lambda_opt = golden_section_search(lambda l: f(x_prev + l * direction), -radius, radius)
                x_new = x_prev + lambda_opt * direction
            history.append([
                total_iter,
                "Success",
                *np.round(x_new, 6),
                round(f(x_new), 6),
                round(lambda_opt, 6) if improved else "-",
                round(delta, 6)
            ])
            x = x_new
        else:
            history.append([
                total_iter,
                "Fail",
                *np.round(x, 6),
                round(f(x), 6),
                "-",
                round(delta, 6)
            ])
            delta /= 2

    # Вывод таблицы
    print(tabulate(
        history,
        headers=["Iter", "Type", "x0", "x1", "f(x)", "lambda", "delta"],
        tablefmt="grid",
        floatfmt=".6f"
    ))
    # Точные значения последней итерации
    print("\nТочные значения на последней итерации:")
    print(f"x0 = {x[0]}, x1 = {x[1]}, f(x) = {f(x)}")
    print(f"Количество успешных итераций: {successful_iter}")
    return x

# Функции для тестирования
def f1(x):
    return 10 * (x[0] + x[1] - 10) ** 2 + (x[0] - x[1] + 4) ** 2


def f2(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


# Запуск алгоритма
print("Минимизация первой функции:")
hooke_jeeves(f1, [0, 0], radius=2.0)
print("\nМинимизация второй функции:")
hooke_jeeves(f2, [-1.2, 1], radius=2.0)