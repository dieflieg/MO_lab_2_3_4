# Импорт необходимых библиотек
import numpy as np  # Для численных вычислений и работы с массивами
import matplotlib.pyplot as plt  # Для визуализации результатов
from tabulate import tabulate  # Для красивого вывода таблиц
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-графиков
from matplotlib import cm  # Цветовые карты для графиков


# Функция для автоматического поиска интервала, содержащего минимум
def find_minimum_interval(f, x0, delta=1e-3, max_iter=100):
    """
    Находит интервал [a, b], содержащий локальный минимум функции f
    Аргументы:
        f - функция одной переменной
        x0 - начальная точка поиска
        delta - начальный размер шага
        max_iter - максимальное число итераций
    Возвращает:
        Кортеж (a, b) - границы интервала, содержащего минимум
    """
    # Пробуем оба направления (вправо и влево)
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
            current_delta *= 1.5  # Плавно увеличиваем шаг
            x_next = x_curr + current_delta
            try:
                f_next = f(x_next)
                if np.isnan(f_next) or np.isinf(f_next):
                    break  # Обработка численных ошибок
            except:
                break  # Если возникли другие численные проблемы

            # Если функция начала возрастать, возвращаем найденный интервал
            if f_next > f_curr:
                return (x_prev, x_next) if x_prev < x_next else (x_next, x_prev)

            # Переходим к следующей точке
            x_prev, x_curr = x_curr, x_next
            f_prev, f_curr = f_curr, f_next

    # Если не нашли подходящий интервал, возвращаем разумные границы
    return (-1.0, 1.0)


# Метод золотого сечения для одномерной минимизации
def golden_section_search(f, a, b, tol=1e-6):
    """
    Находит минимум функции f на интервале [a, b] методом золотого сечения
    Аргументы:
        f - функция одной переменной
        a, b - границы интервала
        tol - точность поиска
    Возвращает:
        Приближенное значение точки минимума
    """
    phi = (1 + np.sqrt(5)) / 2  # Значение золотого сечения
    resphi = 2 - phi  # Дополнение до 1

    # Инициализация начальных точек
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    # Основной цикл поиска
    while abs(b - a) > tol:
        if f1 < f2:
            # Сужаем интервал справа
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            # Сужаем интервал слева
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)

    return (a + b) / 2  # Возвращаем середину конечного интервала


# Модифицированный метод Бройдена с автоматическим поиском интервала
def broyden_method(f, grad_f, x0, tol=1e-5, max_iter=1000):
    """
    Реализация метода Бройдена для многомерной оптимизации
    Аргументы:
        f - целевая функция
        grad_f - функция вычисления градиента
        x0 - начальная точка
        tol - критерий остановки по точности
        max_iter - максимальное число итераций
    Возвращает:
        x - найденную точку минимума
        path - массив точек пути оптимизации
    """
    x = np.array(x0, dtype=float)  # Начальное приближение
    n = len(x)  # Размерность задачи
    eta = np.eye(n)  # Начальная аппроксимация обратного гессиана (единичная матрица)

    table = []  # Для хранения таблицы значений
    path = [x.copy()]  # Сохраняем историю перемещений

    for k in range(max_iter):
        try:
            grad = grad_f(x)  # Вычисляем градиент
        except:
            grad = np.zeros(n)  # Защита от ошибок вычисления градиента

        # Определяем направление спуска
        direction = -eta @ grad

        # Функция для одномерного поиска вдоль направления
        f_line = lambda lambd: f(x + lambd * direction)

        try:
            # Автоматически находим интервал для поиска
            a, b = find_minimum_interval(f_line, 0)

            # Ограничиваем слишком большие интервалы
            if abs(b - a) > 1e6:
                a, b = -1.0, 1.0

            # Точный поиск минимума на интервале
            lambd = golden_section_search(f_line, a, b, tol)
        except:
            lambd = 0.1  # Запасной вариант при ошибках

        # Делаем шаг
        s = lambd * direction
        x_new = x + s

        # Проверка на численные ошибки
        try:
            f_val = f(x_new)
            if np.isnan(f_val) or np.isinf(f_val):
                raise ValueError
        except:
            # Уменьшаем шаг при проблемах
            lambd = lambd / 2
            s = lambd * direction
            x_new = x + s

        path.append(x_new.copy())  # Сохраняем новую точку

        # Проверка критериев остановки
        grad_norm = np.linalg.norm(grad_f(x_new)) if k % 10 == 0 else np.inf
        if (grad_norm < tol or
                np.linalg.norm(x_new - x) < tol or
                abs(f(x_new) - f(x)) < tol):
            break

        # Обновление матрицы eta по формуле Бройдена
        try:
            delta_x = x_new - x
            delta_g = grad_f(x_new) - grad_f(x)

            diff = delta_x - eta @ delta_g
            denom = (diff @ delta_g)
            if abs(denom) > 1e-12:  # Защита от деления на ноль
                delta_eta = np.outer(diff, diff) / denom
                eta = eta + delta_eta
        except:
            eta = np.eye(n)  # Сброс к единичной матрице при ошибках

        x = x_new  # Переход к следующей точке

        # Запись информации о текущей итерации
        if k % 1 == 0:  # Можно записывать не каждую итерацию для экономии памяти
            try:
                table.append([k + 1, x[0], x[1], f(x), np.linalg.norm(grad_f(x))])
            except:
                table.append([k + 1, x[0], x[1], np.nan, np.nan])

    # Вывод результатов в виде таблицы
    print(tabulate(table, headers=["Iter", "x1", "x2", "f(x)", "||grad||"],
                   floatfmt=".10f", tablefmt="grid"))

    return x, np.array(path)


# Тестовая функция 1
def f1(x):
    """
    Первая тестовая функция:
    f(x0, x1) = 10*(x0 + x1 - 10)^2 + (x0 - x1 + 4)^2
    Имеет минимум в точке (3, 7)
    """
    x0, x1 = np.asarray(x)
    try:
        term1 = 10 * (x0 + x1 - 10) ** 2
        term2 = (x0 - x1 + 4) ** 2
        if abs(term1) > 1e100 or abs(term2) > 1e100:
            return 1e100  # Защита от переполнения
        return term1 + term2
    except:
        return 1e100  # Защита от других ошибок


def grad_f1(x):
    """Градиент функции f1"""
    try:
        df_dx0 = 20 * (x[0] + x[1] - 10) + 2 * (x[0] - x[1] + 4)
        df_dx1 = 20 * (x[0] + x[1] - 10) - 2 * (x[0] - x[1] + 4)
        return np.array([df_dx0, df_dx1])
    except:
        return np.array([0.0, 0.0])  # Защита от ошибок


# Функция Розенброка (тестовая функция 2)
def f2(x):
    """
    Функция Розенброка:
    f(x0, x1) = 100*(x1 - x0^2)^2 + (1 - x0)^2
    Имеет минимум в точке (1, 1)
    """
    x0, x1 = np.asarray(x)
    try:
        term1 = 100 * (x1 - x0 ** 2) ** 2
        term2 = (1 - x0) ** 2
        if abs(term1) > 1e100 or abs(term2) > 1e100:
            return 1e100  # Защита от переполнения
        return term1 + term2
    except:
        return 1e100  # Защита от других ошибок


def grad_f2(x):
    """Градиент функции Розенброка"""
    try:
        df_dx0 = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        df_dx1 = 200 * (x[1] - x[0] ** 2)
        return np.array([df_dx0, df_dx1])
    except:
        return np.array([0.0, 0.0])  # Защита от ошибок


# Функция для построения 3D графика
def plot_3d(f, path, title, xlim=(-3, 3), ylim=(-1, 5)):
    """
    Строит 3D график функции и траекторию оптимизации
    Аргументы:
        f - функция двух переменных
        path - массив точек пути оптимизации
        title - заголовок графика
        xlim, ylim - границы области построения
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Создаем сетку для построения
    X = np.linspace(*xlim, 100)
    Y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(X, Y)

    # Вычисляем значения функции в узлах сетки
    Z = np.array([[f([x_, y_]) for x_, y_ in zip(x_row, y_row)]
                  for x_row, y_row in zip(X, Y)])

    # Рисуем поверхность функции
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)

    # Рисуем траекторию оптимизации
    path = np.array(path)
    Z_path = [f(p) for p in path]
    ax.plot(path[:, 0], path[:, 1], Z_path, color='r', marker='o')

    # Настройки графика
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.view_init(elev=30, azim=45)  # Угол обзора
    plt.show(block=False)


# Функция для построения линий уровня
def plot_contour(f, path, title, xlim=(-3, 3), ylim=(-1, 5)):
    """
    Строит линии уровня функции и траекторию оптимизации
    Аргументы:
        f - функция двух переменных
        path - массив точек пути оптимизации
        title - заголовок графика
        xlim, ylim - границы области построения
    """
    plt.figure(figsize=(8, 6))

    # Создаем сетку для построения
    x = np.linspace(*xlim, 400)
    y = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(x, y)

    # Вычисляем значения функции в узлах сетки
    Z = np.array([[f([x_, y_]) for x_, y_ in zip(x_row, y_row)]
                  for x_row, y_row in zip(X, Y)])

    # Рисуем линии уровня
    levels = [1, 5, 10, 50, 100, 500, 1000]
    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    # Рисуем траекторию оптимизации
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color='red', marker='o', label='Траектория')

    # Настройки графика
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show(block=False)


# Основной блок выполнения
if __name__ == "__main__":
    # Тестирование на первой функции
    print("\nМинимизация f1 методом Бройдена:")
    res1, path1 = broyden_method(f1, grad_f1, [0, 0])
    plot_3d(f1, path1, "Минимизация f1 методом Бройдена",
            xlim=(-3, 6), ylim=(-1, 9))
    plot_contour(f1, path1, "Линии уровня f1 и траектория оптимизации",
                 xlim=(-2, 8), ylim=(-1, 9))

    # Тестирование на функции Розенброка
    print("\nМинимизация f2 методом Бройдена:")
    res2, path2 = broyden_method(f2, grad_f2, [-1.2, 1])
    plot_3d(f2, path2, "Минимизация f2 методом Бройдена")
    plot_contour(f2, path2, "Линии уровня f2 и траектория оптимизации")

    plt.show()  # Показываем все графики