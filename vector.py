class Vector:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"({self.x:.6f}, {self.y:.6f})"


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

    def __repr__(self):
        return f"[[{self.a11:.4f} {self.a12:.4f}]\n [{self.a21:.4f} {self.a22:.4f}]]"