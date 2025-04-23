import math


class Vector:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __repr__(self):
        return f"({self.x:.8f}, {self.y:.8f})"


class Matrix:
    def __init__(self, a11=1.0, a12=0.0, a21=0.0, a22=1.0):
        self.a11 = a11
        self.a12 = a12
        self.a21 = a21
        self.a22 = a22

    def multiply_vector(self, vec):
        return Vector(
            self.a11 * vec.x + self.a12 * vec.y,
            self.a21 * vec.x + self.a22 * vec.y
        )

    def __add__(self, other):
        return Matrix(
            self.a11 + other.a11,
            self.a12 + other.a12,
            self.a21 + other.a21,
            self.a22 + other.a22
        )


class BroydenOptimizer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.function_calls = 0
        self.iterations = 0
        self.xk = Vector()
        self.etta = Matrix()

    def objective_func(self, v):
        return (1 - v.x) ** 2 + 100 * (v.y - v.x ** 2) ** 2

    def gradient(self, v):
        return Vector(
            -2 * (1 - v.x) - 400 * v.x * (v.y - v.x ** 2),
            200 * (v.y - v.x ** 2)
        )

    def _bracket(self, start, delta=0.0001):
        x = [start]
        h = delta

        if self._line_func(x[0] + delta) < self._line_func(x[0]):
            x.append(x[0] + delta)
            h = delta
        else:
            x.append(x[0] - delta)
            h = -delta

        while self._line_func(x[-1] + h) < self._line_func(x[-1]):
            x.append(x[-1] + h)
            h *= 2

        return (x[-2], x[-1])

    def _line_func(self, lambda_):
        self.function_calls += 1
        direction = self.etta.multiply_vector(self.gradient(self.xk))
        return self.objective_func(self.xk - direction * lambda_)

    def _golden_section(self, a, b, eps=0.001):
        ratio = 0.618033988749895
        x1 = a + (1 - ratio) * (b - a)
        x2 = a + ratio * (b - a)
        f1 = self._line_func(x1)
        f2 = self._line_func(x2)

        while abs(b - a) > eps:
            if f1 < f2:
                b = x2
                x2, f2 = x1, f1
                x1 = a + (1 - ratio) * (b - a)
                f1 = self._line_func(x1)
            else:
                a = x1
                x1, f1 = x2, f2
                x2 = a + ratio * (b - a)
                f2 = self._line_func(x2)

        return (a + b) / 2

    def minimize(self, start_point, max_iter=50000):
        self.xk = start_point
        self.etta = Matrix()
        self.function_calls = 0
        self.iterations = 0
        grad = self.gradient(self.xk)
        converged = False

        print("\nStarting optimization:")
        print(f"{'Iter':<6} | {'X':<25} | {'Y':<25} | {'F(x)':<15}")
        print("-" * 75)

        while not converged and self.iterations < max_iter:
            self.iterations += 1
            current_value = self.objective_func(self.xk)
            print(f"{self.iterations:<6} | {self.xk.x:<25.15f} | {self.xk.y:<25.15f} | {current_value:<15.10f}")

            # 1D optimization
            a, b = self._bracket(0)
            lambda_ = self._golden_section(a, b)

            prev_x = self.xk
            direction = self.etta.multiply_vector(grad)
            self.xk = self.xk - direction * lambda_

            prev_grad = grad
            grad = self.gradient(self.xk)

            # Update matrix (Pearson's formula)
            delta_grad = grad - prev_grad
            delta_x = self.xk - prev_x
            temp = delta_x - self.etta.multiply_vector(delta_grad)
            denominator = delta_x.x * delta_grad.x + delta_x.y * delta_grad.y

            if abs(denominator) > 1e-16:
                update = Matrix(
                    temp.x * delta_x.x, temp.x * delta_x.y,
                    temp.y * delta_x.x, temp.y * delta_x.y
                )
                update = Matrix(*[val * (1.0 / denominator) for val in vars(update).values()])
                self.etta = self.etta + update

            # Convergence check
            grad_norm = grad.norm()
            step_norm = (self.xk - prev_x).norm()
            if grad_norm < self.eps or step_norm < self.eps:
                converged = True

        # Final output
        final_value = self.objective_func(self.xk)
        print("\nOptimization completed:")
        print(f"{self.iterations:<6} | {self.xk.x:<25.15f} | {self.xk.y:<25.15f} | {final_value:<15.10f}")
        return self.xk

    def print_stats(self):
        print(f"\nStatistics:")
        print(f"Function evaluations: {self.function_calls}")
        print(f"Iterations: {self.iterations}")
        print(f"Final point: {self.xk}")
        print(f"Function value: {self.objective_func(self.xk):.12f}")


if __name__ == "__main__":
    optimizer = BroydenOptimizer(eps=1e-3)
    initial_point = Vector(-1.2, 1.0)  # Начальная точка ближе к минимуму
    result = optimizer.minimize(initial_point, max_iter=10000)
    optimizer.print_stats()