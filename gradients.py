# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# --- Методы золотого сечения ---
def golden_section_search(f, a, b, tol=1e-9):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)
    return (a + b) / 2

# --- Автоматический поиск интервала (улучшенная версия) ---
def find_minimum_interval(f, x0, delta=1e-3, max_iter=100):
    # Пробуем оба направления
    for sign in [1, -1]:
        current_delta = sign * delta
        x_prev = x0
        x_curr = x0 + current_delta
        f_prev = f(x_prev)
        f_curr = f(x_curr)

        # Если функция возрастает, пробуем другое направление
        if f_curr > f_prev:
            continue

        # Ищем интервал, где функция снова начнет возрастать
        for _ in range(max_iter):
            current_delta *= 1.5  # Более плавное увеличение шага
            x_next = x_curr + current_delta
            try:
                f_next = f(x_next)
            except:
                break  # Если возникли численные проблемы

            if f_next > f_curr:
                return (x_prev, x_next) if x_prev < x_next else (x_next, x_prev)

            x_prev, x_curr = x_curr, x_next
            f_prev, f_curr = f_curr, f_next

    # Если не нашли подходящий интервал, возвращаем разумные границы
    return (-1.0, 1.0)  # Эмпирически выбранные границы

def conjugate_gradient_method(f, grad_f, x0, tol=1e-4, max_iter=1000):
    x = np.array(x0, dtype=float)
    grad = grad_f(x)
    d = -grad
    table = []
    path = [x.copy()]

    for k in range(max_iter):
        f_line = lambda alpha: f(x + alpha * d)

        try:
            # Автоматический поиск интервала с защитой от ошибок
            a, b = find_minimum_interval(f_line, 0)

            # Защита от слишком больших интервалов
            if abs(b - a) > 1e6:
                a, b = -1.0, 1.0

            # Используем золотое сечение на найденном интервале
            alpha = golden_section_search(f_line, a, b, tol)
        except:
            alpha = 0.1  # Запасной вариант при ошибках

        s = alpha * d
        x_new = x + s

        # Проверка на численные ошибки
        try:
            f_val = f(x_new)
            if np.isnan(f_val) or np.isinf(f_val):
                raise ValueError
        except:
            # Если возникли проблемы, уменьшаем шаг
            alpha = alpha / 2
            s = alpha * d
            x_new = x + s

        x = x_new
        path.append(x.copy())
        grad_new = grad_f(x)

        # Критерий останова
        if (np.linalg.norm(s) < tol or
                np.linalg.norm(grad_new) < tol or
                np.linalg.norm(d) < tol):
            break

        beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
        d = -grad_new + beta * d
        grad = grad_new

        table.append([k + 1, x[0], x[1], f(x), np.linalg.norm(s)])

    print(tabulate(table, headers=["Iter", "x1", "x2", "f(x)", "||s||"], floatfmt=".10f", tablefmt="grid"))
    return x, np.array(path)

# --- Функции и градиенты ---
def f1(x):
    x0, x1 = np.asarray(x)
    return 10 * (x0 + x1 - 10) ** 2 + (x0 - x1 + 4) ** 2


def grad_f1(x):
    df_dx0 = 20 * (x[0] + x[1] - 10) + 2 * (x[0] - x[1] + 4)
    df_dx1 = 20 * (x[0] + x[1] - 10) - 2 * (x[0] - x[1] + 4)
    return np.array([df_dx0, df_dx1])

def f2(x):
    x0, x1 = np.asarray(x)
    try:
        term1 = 100 * (x1 - x0**2)**2
        term2 = (1 - x0)**2
        # Проверка на переполнение
        if abs(term1) > 1e100 or abs(term2) > 1e100:
            return 1e100  # Возвращаем большое число вместо inf
        return term1 + term2
    except:
        return 1e100  # При ошибках возвращаем большое число

def grad_f2(x):
    df_dx0 = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    df_dx1 = 200 * (x[1] - x[0] ** 2)
    return np.array([df_dx0, df_dx1])


# --- Визуализация ---
def plot_3d(f, path, title, xlim=(-3, 5), ylim=(-1, 11)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(*xlim, 100)
    Y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[f([x_, y_]) for x_, y_ in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    path = np.array(path)
    Z_path = [f(p) for p in path]
    ax.plot(path[:, 0], path[:, 1], Z_path, color='r', marker='o')
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.view_init(elev=30, azim=45)
    plt.show(block=False)


def plot_contour(f, path, title, xlim=(-3, 5), ylim=(-1, 11)):
    plt.figure(figsize=(8, 6))
    x = np.linspace(*xlim, 400)
    y = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([x_, y_]) for x_, y_ in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
    levels = [1, 5, 10, 50, 100, 500, 1000]
    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color='red', marker='o', label='Траектория')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show(block=False)


# --- Применение метода к обеим функциям ---
print("\nМинимизация f1:")
res1, path1 = conjugate_gradient_method(f1, grad_f1, [0, 0])
plot_3d(f1, path1, "Минимизация f1 методом сопряжённых градиентов", xlim=(-3, 6), ylim=(-1, 9))
plot_contour(f1, path1, "Линии уровня f1 и траектория оптимизации", xlim=(-2, 8), ylim=(-1, 9))

print("\nМинимизация f2:")
res2, path2 = conjugate_gradient_method(f2, grad_f2, [-1.2, 1])
plot_3d(f2, path2, "Минимизация f2 методом сопряжённых градиентов")
plot_contour(f2, path2, "Линии уровня f2 и траектория оптимизации")

plt.show()