from vector import Vector, Matrix


class Broyden:
    def __init__(self, eps=1e-3):
        self.eps = eps
        self.function_calls = 0
        self.iterations = 0
        self.xk = Vector()
        self.etta = Matrix()

    def objective_function(self, v):
        # Пример функции Розенброка (можно заменить)
        return (1 - v.x) ** 2 + 100 * (v.y - v.x ** 2) ** 2

    def gradient(self, v):
        # Аналитический градиент для Розенброка
        return Vector(
            -2 * (1 - v.x) - 400 * v.x * (v.y - v.x ** 2),
            200 * (v.y - v.x ** 2)
        )

    def line_function(self, lambda_):
        self.function_calls += 1
        direction = self.etta.multiply_vector(self.gradient(self.xk))
        return self.objective_function(self.xk - direction * lambda_)

    def find_segment(self, start, delta=1e-5):
        x = [start]
        h = delta

        if self.line_function(x[0] + delta) < self.line_function(x[0]):
            x.append(x[0] + delta)
            h = delta
        else:
            x.append(x[0] - delta)
            h = -delta

        while self.line_function(x[-1] + h) < self.line_function(x[-1]):
            x.append(x[-1] + h)
            h *= 2

        return (x[-2], x[-1])

    def golden_section(self, a, b, eps=1e-6):
        ratio = 0.618033988749895
        x1 = a + (1 - ratio) * (b - a)
        x2 = a + ratio * (b - a)
        f1 = self.line_function(x1)
        f2 = self.line_function(x2)

        while abs(b - a) > eps:
            if f1 < f2:
                b = x2
                x2, f2 = x1, f1
                x1 = a + (1 - ratio) * (b - a)
                f1 = self.line_function(x1)
            else:
                a = x1
                x1, f1 = x2, f2
                x2 = a + ratio * (b - a)
                f2 = self.line_function(x2)

        return (a + b) / 2

    def minimize(self, start, max_iter=5000):
        self.xk = start
        self.etta = Matrix()
        self.function_calls = 0
        self.iterations = 0

        grad = self.gradient(self.xk)
        converged = False

        while not converged and self.iterations < max_iter:
            self.iterations += 1
            a, b = self.find_segment(0)
            lambda_ = self.golden_section(a, b)

            prev_x = self.xk
            direction = self.etta.multiply_vector(grad)
            self.xk = self.xk - direction * lambda_

            prev_grad = grad
            grad = self.gradient(self.xk)

            delta_grad = grad - prev_grad
            delta_x = self.xk - prev_x

            temp = delta_x - self.etta.multiply_vector(delta_grad)
            denominator = temp.x * delta_grad.x + temp.y * delta_grad.y

            if abs(denominator) > 1e-12:
                update = Matrix(
                    temp.x * temp.x, temp.x * temp.y,
                    temp.x * temp.y, temp.y * temp.y
                )
                update = Matrix(*[val * (1.0 / denominator) for val in update.__dict__.values()])
                self.etta = self.etta + update

            if (self.xk - prev_x).norm() < self.eps:
                converged = True

            if self.iterations % 1000 == 0:
                self.etta = Matrix()
                grad = self.gradient(self.xk)

        return self.xk

    def print_stats(self):
        print(f"Function evaluations: {self.function_calls}")
        print(f"Iterations: {self.iterations}")
        print(f"Final point: {self.xk}")