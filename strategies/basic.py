"""Пример тактик: нападающий и вратарь.

Все координаты — в системе команды: свои ворота слева (x<0), чужие справа (x>0),
курс 0° смотрит на ворота соперника. Метры, м/с, градусы.
"""
from __future__ import annotations

import math

from rcjsim.api import Strategy, Vec2, angle_diff

# ------------------------------------------------------------------ зоны
# Разрешённые прямоугольники движения для каждого робота: список (x0, y0, x1, y1)
# в системе координат команды (метры). Если список не пуст — роботу разрешено
# находиться ТОЛЬКО внутри объединения этих прямоугольников; остальное поле
# для него запрещено (цель проецируется в ближайший разрешённый прямоугольник).
# Пустой список = ограничения нет. Пример:
#   STRIKER_ZONES = [(-1.0, -0.79, 1.0, 0.79)]     # почти всё поле
#   GOALIE_ZONES  = [(-1.095, -0.5, -0.5, 0.5)]    # только у своих ворот


from rcjsim.field import HX, HY, WX, WY, PEN_DEPTH, PEN_LEN, GOAL_W, GOAL_D, CENTER_CIRCLE_R

FIELD_X0, FIELD_X1 = -HX, HX          # край поля по X (свои ворота … чужие ворота)
FIELD_Y0, FIELD_Y1 = -HY, HY          # край поля по Y (нижняя … верхняя стена линии)
WALL_X0, WALL_X1 = -WX, WX            # внутренняя грань стены по X
WALL_Y0, WALL_Y1 = -WY, WY            # внутренняя грань стены по Y
OWN_PEN_X1 = -HX + PEN_DEPTH          # передняя граница своей штрафной по X
OPP_PEN_X0 = HX - PEN_DEPTH           # передняя граница чужой штрафной по X
PEN_Y0, PEN_Y1 = -PEN_LEN / 2, PEN_LEN / 2  # границы штрафной по Y (обе стороны)

STRIKER_ZONES = [
    (-0.7, FIELD_Y0 - 0.09, 0.85, FIELD_Y1 + 0.09)
]
GOALIE_ZONES = [
    (-1.095, -0.5, -0.5, 0.5),     # своя штрафная и подступы
    (-0.6, -0.79, 0.0, -0.5),      # + узкий коридор вдоль нижней стены
]


def clamp_to_zones(p: Vec2, zones: list[tuple[float, float, float, float]]) -> Vec2:
    """Проецирует точку в ближайшую точку внутри объединения прямоугольников
    zones = [(x0, y0, x1, y1), ...]. Пустой список — точка не меняется."""
    if not zones:
        return p
    best, best_d = None, None
    for x0, y0, x1, y1 in zones:
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        c = Vec2(max(lo_x, min(hi_x, p.x)), max(lo_y, min(hi_y, p.y)))
        d = c.dist(p)
        if best_d is None or d < best_d:
            best, best_d = c, d
    return best


def safe_target(world, p: Vec2, radius: float, own_margin: float = 0.02,
                zones: list[tuple[float, float, float, float]] | None = None) -> Vec2:
    """Сдвигает цель так, чтобы робот не заехал полностью в чужую штрафную,
    не заехал даже частично в свою (чтобы не было двойной защиты) и не касался стен.
    Если передан zones — цель дополнительно, в последнюю очередь (это самое
    жёсткое из ограничений), проецируется в разрешённые прямоугольники."""
    f = world.field
    p = Vec2(p.x, p.y)
    lim_x = f.wall_x - radius - 0.05
    lim_y = f.wall_y - radius - 0.05
    p = Vec2(max(-lim_x, min(lim_x, p.x)), max(-lim_y, min(lim_y, p.y)))
    own_goal_line = f.own_goal.x + 1.5 * radius + 0.01
    # не даём цели уйти за линию своих ворот
    p = Vec2(max(p.x, own_goal_line), p.y)
    for own, need in ((False, -radius + 0.04), (True, radius + own_margin)):
        for _ in range(40):
            d = f.penalty_distance(p, own)
            if d >= need:
                break
            e = 1e-3
            g = Vec2(f.penalty_distance(Vec2(p.x + e, p.y), own) - d,
                     f.penalty_distance(Vec2(p.x, p.y + e), own) - d).normalized()
            if g.length() == 0:
                g = Vec2(1 if own else -1, 0)
            p = p + g * (need - d + 0.002)
    if zones:
        p = clamp_to_zones(p, zones)
    return p


def route(world, start: Vec2, target: Vec2, radius: float) -> Vec2:
    """Если прямой путь проходит через свою штрафную (двойная защита!) —
    сначала выезжаем из неё вперёд, затем едем к цели."""
    f = world.field
    front_x = -(f.half_length - f.penalty_depth) + radius + 0.08
    for i in range(1, 11):
        p = start + (target - start) * (i / 10)
        if f.penalty_distance(p, own=True) < radius + 0.01:
            return Vec2(max(start.x, front_x), start.y if abs(start.y) > 0.45 else
                        (0.5 if start.y >= 0 else -0.5)) if start.x < front_x else Vec2(front_x, target.y)
    return target


class Striker(Strategy):
    # м/с — подъезд к мячу, чтобы он не отскочил от ниши (<0.5)
    APPROACH_SPEED = 0.35
    CRUISE_SPEED = 1.8
    DRIBBLE_SPEED = 1.0

    def step(self, world, robot):
        self._robot = robot
        self._world = world
        self._step(world, _Router(world, robot))

    def _step(self, world, robot):
        me = world.me
        ball = world.ball.pos
        goal = world.field.opp_goal
        R = me.radius

        if me.has_ball:
            # целимся в ворота, чуть в сторону от вратаря соперника
            aim = Vec2(goal.x, 0.0)
            keeper = min(world.opponents,
                         key=lambda o: o.pos.dist(goal), default=None)
            if keeper and keeper.on_field:
                aim = Vec2(goal.x, -0.18 if keeper.pos.y > 0 else 0.18)
            h = me.pos.angle_to(aim)
            t = safe_target(world, aim, R, zones=STRIKER_ZONES)
            robot.move_to(t.x, t.y, speed=self.DRIBBLE_SPEED, heading=h)
            robot.debug(f"с мячом → ({aim.x:.2f},{aim.y:.2f})")
            if me.pos.dist(goal) < 0.75 and abs(angle_diff(me.heading, h)) < 8:
                if robot.kick(1.0):
                    robot.debug("УДАР")
            return

        if world.ball.in_niche and world.ball.owner_team == me.team:
            # мяч у партнёра — открываемся
            t = safe_target(world, Vec2(0.3, -0.3 if ball.y > 0 else 0.3), R,
                            zones=STRIKER_ZONES)
            robot.move_to(t.x, t.y, speed=1.2, heading=0)
            robot.debug("открываюсь")
            return

        # мяч зажат у стены соперником — не лезем (иначе оба в аут), ждём рядом
        f = world.field
        rival = min((o for o in world.opponents if o.on_field),
                    key=lambda o: o.pos.dist(ball), default=None)
        if (f.wall_clearance(ball, 0.0) < 0.15 and rival is not None
                and rival.pos.dist(ball) < rival.radius + 0.06):
            wait = ball + (f.own_goal - ball).normalized() * 0.30
            t = safe_target(world, wait, R, zones=STRIKER_ZONES)
            robot.move_to(t.x, t.y, speed=1.2, heading=me.pos.angle_to(ball))
            robot.debug("мяч у стены под соперником — жду")
            return

        aim = goal
        if f.wall_clearance(ball, 0.0) < R + 0.05:
            # мяч у стены: заходим по диагонали из поля, чтобы не коснуться стены
            aim = Vec2(goal.x, ball.y * 0.3)
        to_goal = (aim - ball).normalized()
        behind = ball - to_goal * (me.seat_dist + 0.06)
        h_goal = ball.angle_to(aim)
        robot.mark(behind.x, behind.y, (0, 255, 255), "за мячом")

        # робот перед мячом (между мячом и воротами) — объезжаем сбоку
        rel = me.pos - ball
        if rel.dot(to_goal) > 0 and abs(rel.cross(to_goal)) < R + 0.7:
            side = 1 if rel.cross(to_goal) <= 0 else -1
            t = ball + to_goal.perp() * (side * (R + 0.12)) - to_goal * 0.05
            t = safe_target(world, t, R, zones=STRIKER_ZONES)
            robot.move_to(t.x, t.y, speed=self.CRUISE_SPEED, heading=h_goal)
            robot.debug("объезд")
            return

        d_behind = me.pos.dist(behind)
        aligned = abs(angle_diff(me.heading, h_goal)) < 12
        if d_behind > 0.04 and not (aligned and me.pos.dist(ball) < me.seat_dist + 0.08):
            t = safe_target(world, behind, R, zones=STRIKER_ZONES)
            speed = min(self.CRUISE_SPEED, 0.4 + d_behind * 3)
            robot.move_to(t.x, t.y, speed=speed, heading=h_goal)
            robot.debug(f"к точке за мячом {d_behind:.2f} м")
        else:
            # медленный заезд: центр ниши на мяч
            t = safe_target(world, ball - to_goal *
                            me.seat_dist + to_goal * 0.01, R, zones=STRIKER_ZONES)
            robot.move_to(t.x, t.y, speed=self.APPROACH_SPEED, heading=h_goal)
            robot.debug("захват мяча")


class _Router:
    """Обёртка над RobotControl: move_to объезжает свою штрафную."""

    def __init__(self, world, robot):
        self._w = world
        self._r = robot

    def move_to(self, x, y, speed=None, heading=None):
        me = self._w.me
        t = route(self._w, me.pos, Vec2(x, y), me.radius)
        # у стены сбрасываем скорость, чтобы по инерции не коснуться её (аут)
        c = self._w.field.wall_clearance(me.pos, me.radius)
        cap = max(0.3, (2 * 2.5 * max(c - 0.02, 0.0)) ** 0.5)
        speed = min(speed if speed is not None else me.max_speed, cap)
        self._r.move_to(t.x, t.y, speed=speed, heading=heading)

    def __getattr__(self, name):
        return getattr(self._r, name)


class Goalie(Strategy):
    # радиус дуги от центра ворот (частично в штрафной, не полностью)
    ARC_R = 0.33
    MAX_ANGLE = 62             # градусы — крайние положения на дуге

    def step(self, world, robot):
        me = world.me
        f = world.field
        ball = world.ball.pos
        R = me.radius
        g = f.own_goal

        if me.has_ball:
            h = me.pos.angle_to(f.opp_goal)
            robot.rotate_to(h)
            if abs(angle_diff(me.heading, h)) < 25:
                robot.kick(1.0)
            robot.debug("выбиваю")
            return

        # медленный мяч рядом со штрафной — выбиваем
        if me.pos.dist(ball) < 0.35 and world.ball.speed < 0.8 and ball.x < -0.45:
            h = me.pos.angle_to(ball)
            # не дальше линии ворот минус свой радиус
            goal_line_x = f.own_goal.x + 1.5 * R + 0.01
            target_ball = Vec2(max(ball.x, goal_line_x), ball.y)
            t = safe_target(world, target_ball, R, own_margin=-2 * R + 0.04,
                            zones=GOALIE_ZONES)
            robot.move_to(t.x, t.y, speed=0.45, heading=h)
            if robot.can_kick:
                robot.kick(1.0)
            robot.debug("отбираю мяч")
            return

        # стоим на дуге между мячом и центром ворот, лицом к мячу
        ang = max(-self.MAX_ANGLE, min(self.MAX_ANGLE, g.angle_to(ball)))
        t = g + Vec2.from_angle(math.radians(ang), self.ARC_R)
        if f.in_penalty_area(t, R, own=True, fully=True):
            t = safe_target(world, t, R, own_margin=-2 *
                            R + 0.03, zones=GOALIE_ZONES)
        robot.move_to(t.x, t.y, speed=1.8, heading=me.pos.angle_to(ball))
        robot.debug(f"дуга {ang:+.0f}°")
