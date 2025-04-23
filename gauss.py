# Импорт необходимых библиотек
import numpy as np  # Для работы с массивами и математическими операциями
from tabulate import tabulate  # Для красивого форматирования таблиц

def golden_section_search(f, a, b, tol=1e-9):
    """Метод золотого сечения для поиска минимума функции f на интервале [a, b]."""
    # Коэффициенты золотого сечения (φ ≈ 0.618)
    phi1 = 0.381966011  # (3 - √5)/2 ≈ 0.38197
    phi2 = 0.618033989  # (√5 - 1)/2 ≈ 0.61803

    # Инициализация внутренних точек интервала
    x1 = a + phi1 * (b - a)  # Левая внутренняя точка
    x2 = a + phi2 * (b - a)  # Правая внутренняя точка
    f1, f2 = f(x1), f(x2)     # Значения функции в этих точках

    # Цикл сужения интервала до достижения точности tol
    while abs(b - a) > tol:
        if f1 < f2:
            # Минимум в левой части: сдвигаем правую границу в x2
            b = x2            # Новая правая граница
            x2 = x1           # Перемещаем x2 в x1
            f2 = f1           # Сохраняем значение f1
            x1 = a + phi1 * (b - a)  # Вычисляем новую x1
            f1 = f(x1)        # Обновляем f1
        else:
            # Минимум в правой части: сдвигаем левую границу в x1
            a = x1            # Новая левая граница
            x1 = x2           # Перемещаем x1 в x2
            f1 = f2           # Сохраняем значение f2
            x2 = a + phi2 * (b - a)  # Вычисляем новую x2
            f2 = f(x2)        # Обновляем f2

    return (a + b) / 2  # Возвращаем середину финального интервала

def gauss_method(f, x0, radius=5.0, tol=1e-6, max_iter=5000):
    """Реализация метода Гаусса для многомерной оптимизации."""
    x = np.array(x0, dtype=float)
    n = len(x)
    iteration = 0
    table_data = []

    while iteration < max_iter:
        x_prev = x.copy()
        f_prev = f(x_prev)

        for i in range(n):
            def func_1d(alpha):
                x_temp = x.copy()
                x_temp[i] = alpha
                return f(x_temp)

            a = x[i] - radius
            b = x[i] + radius
            alpha_opt = golden_section_search(func_1d, a, b, tol)
            x[i] = alpha_opt

        f_value = f(x)
        delta = np.linalg.norm(x - x_prev)
        f_delta = abs(f_value - f_prev)

        # Запись данных без округления
        table_data.append([
            iteration + 1,
            x[0],  # Точное значение x1
            x[1],  # Точное значение x2
            f_value,  # Точное значение f(x)
            delta,
            f_delta
        ])

        if delta < tol and f_delta < tol:
            break

        iteration += 1

    # Форматирование таблицы с высокой точностью
    print(tabulate(
        table_data,
        headers=["Iteration", "x1", "x2", "f(x)", "||x_new - x_old||", "|f_new - f_old|"],
        tablefmt="grid",
        floatfmt=(".0f", ".10f", ".10f", ".10f", ".10f", ".10f")  # 10 знаков для всех чисел
    ))
    return x

# Тестовые функции для демонстрации работы метода
def f1(x):
    """Квадратичная функция: 10*(x0 + x1 - 10)^2 + (x0 - x1 + 4)^2."""
    return 10 * (x[0] + x[1] - 10)**2 + (x[0] - x[1] + 4)**2

def f2(x):
    """Функция Розенброка: 100*(x1 - x0^2)^2 + (1 - x0)^2."""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Примеры использования метода Гаусса
print("Minimizing f1:")
x1_opt = gauss_method(f1, [0, 0], radius=2)  # Оптимизация f1 с начальной точкой [0, 0]
print(f"Optimal solution for f1: {x1_opt}\n")

print("Minimizing f2:")
x2_opt = gauss_method(f2, [-1.2, 1], radius=2)  # Оптимизация f2 с начальной точкой [-1.2, 1]
print(f"Optimal solution for f2: {x2_opt}")