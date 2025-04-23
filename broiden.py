# Импорт необходимых библиотек
import numpy as np  # Работа с массивами и линейной алгеброй
import matplotlib.pyplot as plt  # Построение графиков
from tabulate import tabulate  # Для табличного вывода результатов
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-графиков
from matplotlib import cm  # Цветовые карты для графиков

# Метод золотого сечения для одномерной минимизации
def golden_section_search(f, a, b, tol=1e-6):
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
    resphi = 2 - phi  # Дополнение до 1

    # Начальные точки
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    # Основной цикл поиска
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

    return (a + b) / 2  # Возвращаем найденный минимум

# Метод Бройдена
def broyden_method(f, grad_f, x0, radius=5, tol=1e-4, max_iter=1000):
    x = np.array(x0, dtype=float)  # Начальное приближение
    n = len(x)  # Размерность задачи
    eta = np.eye(n)  # Начальная матрица (единичная)

    table = []  # Для хранения таблицы значений
    path = [x.copy()]  # Путь оптимизации

    for k in range(max_iter):
        grad = grad_f(x)  # Градиент в текущей точке

        # Функция вдоль направления eta * grad
        direction = -eta @ grad
        f_line = lambda lambd: f(x + lambd * direction)

        # Поиск шага методом золотого сечения
        lambd = golden_section_search(f_line, 0, radius, tol)

        # Обновляем точку
        s = lambd * direction
        x_new = x + s
        path.append(x_new.copy())  # Сохраняем путь

        # Проверка критерия останова: если градиент мал, изменения координат и функции тоже малы
        if (
            np.linalg.norm(grad_f(x_new)) < tol or
            np.linalg.norm(x_new - x) < tol or
            abs(f(x_new) - f(x)) < tol
        ):
            break

        # Обновление eta по формуле Бройдена
        delta_x = x_new - x
        delta_g = grad_f(x_new) - grad

        diff = delta_x - eta @ delta_g
        denom = (diff @ delta_g)
        if abs(denom) > 1e-12:  # Защита от деления на ноль
            delta_eta = np.outer(diff, diff) / denom
            eta = eta + delta_eta

        x = x_new  # Переход к следующей итерации

        # Добавление строки в таблицу
        table.append([k + 1, x[0], x[1], f(x), np.linalg.norm(grad_f(x))])

    # Вывод таблицы
    print(tabulate(table, headers=["Iter", "x1", "x2", "f(x)", "||grad||"],
                   floatfmt=".10f", tablefmt="grid"))

    return x, np.array(path)

# Функция f1 и её градиент
def f1(x):
    x0, x1 = np.asarray(x)
    return 10 * (x0 + x1 - 10)**2 + (x0 - x1 + 4)**2

def grad_f1(x):
    df_dx0 = 20 * (x[0] + x[1] - 10) + 2 * (x[0] - x[1] + 4)
    df_dx1 = 20 * (x[0] + x[1] - 10) - 2 * (x[0] - x[1] + 4)
    return np.array([df_dx0, df_dx1])

# Функция Розенброка и её градиент
def f2(x):
    x0, x1 = np.asarray(x)
    return 100 * (x1 - x0**2)**2 + (1 - x0)**2

def grad_f2(x):
    df_dx0 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_dx1 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx0, df_dx1])

# Функция 3D-графика с увеличенными границами (для f1 отдельно, универсально — через параметры)
def plot_3d(f, path, title, xlim=(-3, 3), ylim=(-1, 5)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Сетка значений с новыми границами
    X = np.linspace(*xlim, 100)
    Y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[f([x_, y_]) for x_, y_ in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    # Поверхность функции
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)

    # Траектория
    path = np.array(path)
    Z_path = [f(p) for p in path]
    ax.plot(path[:, 0], path[:, 1], Z_path, color='r', marker='o')

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.view_init(elev=30, azim=45)

    plt.show(block=False)

# Обновлённая функция 2D-графика с явными уровнями
def plot_contour(f, path, title, xlim=(-3, 3), ylim=(-1, 5)):
    plt.figure(figsize=(8, 6))

    # Сетка значений с заданными границами
    x = np.linspace(*xlim, 400)
    y = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([x_, y_]) for x_, y_ in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    # Уровни: заданные вручную
    levels = [1, 5, 10, 50, 100, 500, 1000]

    # Линии уровня
    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    # Траектория оптимизации
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

# --- f1: с расширенными границами ---
print("\nМинимизация f1 методом Бройдена:")
res1, path1 = broyden_method(f1, grad_f1, [0, 0], radius=10)
plot_3d(f1, path1, "Минимизация f1 методом Бройдена", xlim=(-3, 6), ylim=(-1, 9))
plot_contour(f1, path1, "Линии уровня f1 и траектория оптимизации", xlim=(-2, 8), ylim=(-1, 9))

# --- f2: обычные границы, но уровни заданы явно ---
print("\nМинимизация f2 методом Бройдена:")
res2, path2 = broyden_method(f2, grad_f2, [-1.2, 1], radius=2.5)
plot_3d(f2, path2, "Минимизация f2 методом Бройдена")
plot_contour(f2, path2, "Линии уровня f2 и траектория оптимизации")

plt.show()