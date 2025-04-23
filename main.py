from broyden import Broyden
from vector import Vector  # Добавляем импорт класса Vector

if __name__ == "__main__":
    optimizer = Broyden(eps=1e-6)
    result = optimizer.minimize(Vector(-1.0, -1.0))
    optimizer.print_stats()