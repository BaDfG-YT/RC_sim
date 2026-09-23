"""Минимальный 2D-вектор. Все координаты симулятора — в метрах."""
from __future__ import annotations

import math


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    # --- арифметика ---
    def __add__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec2":
        return Vec2(self.x / k, self.y / k)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return f"Vec2({self.x:.3f}, {self.y:.3f})"

    def __eq__(self, o) -> bool:
        return isinstance(o, Vec2) and self.x == o.x and self.y == o.y

    # --- геометрия ---
    def dot(self, o: "Vec2") -> float:
        return self.x * o.x + self.y * o.y

    def cross(self, o: "Vec2") -> float:
        return self.x * o.y - self.y * o.x

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec2":
        n = self.length()
        return Vec2(self.x / n, self.y / n) if n > 1e-12 else Vec2(0.0, 0.0)

    def dist(self, o: "Vec2") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def rotated(self, ang_rad: float) -> "Vec2":
        c, s = math.cos(ang_rad), math.sin(ang_rad)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def perp(self) -> "Vec2":
        """Поворот на +90°."""
        return Vec2(-self.y, self.x)

    def angle(self) -> float:
        """Угол вектора в радианах (atan2)."""
        return math.atan2(self.y, self.x)

    def angle_to(self, other: "Vec2") -> float:
        """Направление (градусы) из этой точки на другую."""
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))

    def angle_deg(self) -> float:
        return math.degrees(self.angle())

    def copy(self) -> "Vec2":
        return Vec2(self.x, self.y)

    def clamped(self, max_len: float) -> "Vec2":
        n = self.length()
        if n > max_len > 0:
            return self * (max_len / n)
        return self.copy()

    @staticmethod
    def from_angle(ang_rad: float, length: float = 1.0) -> "Vec2":
        return Vec2(math.cos(ang_rad) * length, math.sin(ang_rad) * length)


def wrap_angle(a: float) -> float:
    """Нормализует угол в радианах в диапазон (-pi, pi]."""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def wrap_deg(a: float) -> float:
    """Нормализует угол в градусах в диапазон (-180, 180]."""
    return math.degrees(wrap_angle(math.radians(a)))
