"""Геометрия поля по «Technical Specification For Soccer Fields» RCJ 2026.

Мировая система координат:
  * начало — центр поля;
  * ось X — вдоль длинной стороны (219 см), ось Y — вдоль короткой (158 см);
  * жёлтые ворота слева (x < 0), синие — справа (x > 0);
  * единицы — метры, углы — радианы (внутри), градусы (в API стратегий).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .vec import Vec2

# ---------------------------------------------------------------- размеры (м)
LINE_W = 0.02                 # ширина белой линии 20 мм (п. 4.0.1)
FIELD_L = 2.19                # игровое поле 219 см (по внешнему краю линии, п. 1.0.1)
FIELD_W = 1.58                # 158 см
OUTER = 0.12                  # внешняя зона за линией 12 см
TOTAL_L = FIELD_L + 2 * OUTER  # 243 см
TOTAL_W = FIELD_W + 2 * OUTER  # 182 см
WALL_H = 0.22                 # высота стен (для справки)
WALL_T = 0.02                 # толщина стен (только для отрисовки)

HX = FIELD_L / 2              # 1.095 — внешний край линии поля по X
HY = FIELD_W / 2              # 0.79
WX = TOTAL_L / 2              # 1.215 — внутренняя грань стены по X
WY = TOTAL_W / 2              # 0.91

# Ворота (п. 3): внутреннее пространство 60 см x 74 мм, штанги над белой линией,
# линия находится внутри ворот.
GOAL_W = 0.60
GOAL_D = 0.074
GOAL_H = 0.10
GOAL_T = 0.012                # толщина стенок ворот (конструктив)
GOAL_MOUTH_X = HX - LINE_W    # 1.075 — передний край ворот (внутренний край линии)
GOAL_BACK_X = GOAL_MOUTH_X + GOAL_D  # 1.149 — задняя стенка ворот (внутренняя грань)

# Штрафная (зона вратаря), п. 7: 25 см вглубь поля, 80 см вдоль линии ворот,
# передние углы скруглены R=15 см. Линия — часть зоны.
PEN_DEPTH = 0.25
PEN_LEN = 0.80
PEN_R = 0.15

# Нейтральные точки (п. 5): центр + 4 точки в 45 см от короткой стороны,
# на продолжении боковых сторон штрафных.
NEUTRAL_OFFSET_X = 0.45
NEUTRAL_SPOT_D = 0.01
CENTER_CIRCLE_R = 0.30        # диаметр 60 см (п. 6)

# Клин у внешней стены (п. 1.0.2): основание 10 см, подъём 2 см, в воротах клина нет.
WEDGE_BASE = 0.10
WEDGE_RISE = 0.02

G = 9.81

NEUTRAL_SPOTS = [
    Vec2(0.0, 0.0),
    Vec2(-(HX - NEUTRAL_OFFSET_X), PEN_LEN / 2),
    Vec2(-(HX - NEUTRAL_OFFSET_X), -PEN_LEN / 2),
    Vec2(HX - NEUTRAL_OFFSET_X, PEN_LEN / 2),
    Vec2(HX - NEUTRAL_OFFSET_X, -PEN_LEN / 2),
]


@dataclass(frozen=True)
class Box:
    """Осевой прямоугольник-препятствие (стенки ворот)."""
    x0: float
    y0: float
    x1: float
    y1: float


def _goal_boxes(side: int) -> list[Box]:
    """Стенки ворот. side=-1 — левые (жёлтые), +1 — правые (синие)."""
    s = side
    yl = GOAL_W / 2
    back_x0 = GOAL_BACK_X
    boxes = [
        # задняя стенка — до самой стены поля, чтобы за воротами не было щели
        (back_x0, -yl - GOAL_T, WX + WALL_T, yl + GOAL_T),
        # боковые стенки
        (GOAL_MOUTH_X, yl, GOAL_BACK_X, yl + GOAL_T),
        (GOAL_MOUTH_X, -yl - GOAL_T, GOAL_BACK_X, -yl),
    ]
    out = []
    for x0, y0, x1, y1 in boxes:
        if s > 0:
            out.append(Box(x0, y0, x1, y1))
        else:
            out.append(Box(-x1, y0, -x0, y1))
    return out


GOAL_BOXES = _goal_boxes(-1) + _goal_boxes(+1)


# ------------------------------------------------------------------ SDF-утилиты
def _sd_round_box(px: float, py: float, hx: float, hy: float, r: float) -> float:
    """Знаковое расстояние до прямоугольника (полуразмеры hx, hy) со скруглением r."""
    qx = abs(px) - hx + r
    qy = abs(py) - hy + r
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside - r


def penalty_sdf(p: Vec2, side: int) -> float:
    """Знаковое расстояние от точки до штрафной (<0 — внутри).

    side=-1 — штрафная у левых (жёлтых) ворот, +1 — у правых (синих).
    Форма: прямоугольник 25x80 см от внешнего края линии поля вглубь поля,
    передние (обращённые к центру) углы скруглены R=15 см.
    """
    u = HX - p.x * side          # расстояние от линии ворот вглубь поля
    v = p.y
    # Симметричный скруглённый прямоугольник шириной 2*PEN_DEPTH, обрезанный по u>=0.
    d_box = _sd_round_box(u, v, PEN_DEPTH, PEN_LEN / 2, PEN_R)
    return max(d_box, -u)


def penalty_outline(side: int, inset: float = 0.0, n_arc: int = 16) -> list[Vec2]:
    """Полигон контура штрафной (для отрисовки). inset>0 — внутренний контур."""
    hx = PEN_DEPTH - inset
    hy = PEN_LEN / 2 - inset
    r = PEN_R - inset
    u0 = inset  # задний край (у линии ворот)
    pts_uv: list[tuple[float, float]] = [(u0, -hy)]
    # правый нижний угол (u=hx, v=-hy) скруглён, центр (hx-r, -hy+r)
    cu, cv = hx - r, -hy + r
    for i in range(n_arc + 1):
        a = -math.pi / 2 + (math.pi / 2) * i / n_arc
        pts_uv.append((cu + r * math.cos(a), cv + r * math.sin(a)))
    cu, cv = hx - r, hy - r
    for i in range(n_arc + 1):
        a = 0 + (math.pi / 2) * i / n_arc
        pts_uv.append((cu + r * math.cos(a), cv + r * math.sin(a)))
    pts_uv.append((u0, hy))
    return [Vec2(side * (HX - u), v) for u, v in pts_uv]


def in_goal_mouth_y(y: float) -> bool:
    return abs(y) <= GOAL_W / 2 + GOAL_T


def wedge_accel(p: Vec2, radius: float) -> Vec2:
    """Ускорение мяча от клина у внешних стен (скатывание обратно в поле)."""
    a = G * math.sin(math.atan2(WEDGE_RISE, WEDGE_BASE))
    acc = Vec2()
    # длинные стены (y = ±WY)
    if p.y > WY - WEDGE_BASE:
        acc.y -= a
    elif p.y < -WY + WEDGE_BASE:
        acc.y += a
    # короткие стены (x = ±WX), кроме участка ворот
    if not in_goal_mouth_y(p.y):
        if p.x > WX - WEDGE_BASE:
            acc.x -= a
        elif p.x < -WX + WEDGE_BASE:
            acc.x += a
    return acc


def point_in_field(p: Vec2) -> bool:
    """Точка внутри игрового поля (включая белую линию)."""
    return abs(p.x) <= HX and abs(p.y) <= HY
