"""API для программирования тактик роботов.

Стратегия видит мир в «системе команды» (если sim.team_frame = true):
    * свои ворота всегда слева (x < 0), ворота соперника — справа (x > 0);
    * курс 0° — в сторону ворот соперника, положительный угол — против часовой;
    * координаты в метрах, скорости в м/с, углы в градусах.
Смена сторон после перерыва для стратегии незаметна.

Минимальный пример:

    from rcjsim.api import Strategy

    class MyStriker(Strategy):
        def step(self, world, robot):
            ball = world.ball.pos
            if world.me.has_ball:
                goal = world.field.opp_goal
                robot.move_to(goal.x, goal.y, speed=1.0,
                              heading=world.me.pos.angle_to(goal))
                if world.me.pos.dist(goal) < 0.6:
                    robot.kick()
            else:
                robot.move_to(ball.x, ball.y, speed=1.5,
                              heading=world.me.pos.angle_to(ball))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from . import field as F
from .entities import Command
from .vec import Vec2, wrap_angle, wrap_deg

if TYPE_CHECKING:
    from .entities import Robot
    from .game import Simulation


# ---------------------------------------------------------------- точки
P = Vec2  # точка/вектор; есть .dist(), .angle_to(), .length(), арифметика


def angle_diff(a_deg: float, b_deg: float) -> float:
    """Разница курсов a−b в диапазоне (−180, 180]."""
    return wrap_deg(a_deg - b_deg)


# ------------------------------------------------------ система координат
class Frame:
    """Преобразование мир ↔ система команды (поворот на 180° для команды, атакующей в −X)."""

    def __init__(self, sign: int, enabled: bool = True):
        self.s = sign if enabled else 1

    def p_in(self, v: Vec2) -> P:
        return Vec2(v.x * self.s, v.y * self.s)

    p_out = p_in

    def h_in(self, h_rad: float) -> float:
        return math.degrees(wrap_angle(h_rad + (0 if self.s > 0 else math.pi)))

    def h_out(self, h_deg: float) -> float:
        return wrap_angle(math.radians(h_deg) + (0 if self.s > 0 else math.pi))


# ------------------------------------------------------------ представления
@dataclass
class RobotInfo:
    id: str
    team: int
    index: int
    name: str
    pos: P
    heading: float          # градусы
    vel: P
    omega: float            # град/с
    radius: float
    niche_width: float
    max_speed: float
    seat_dist: float        # расстояние от центра робота до центра мяча в нише
    has_ball: bool
    on_field: bool
    penalty_left: float     # с до возвращения (0 — на поле)

    @property
    def speed(self) -> float:
        return self.vel.length()

    @property
    def niche_pos(self) -> P:
        """Где находится центр мяча, когда он в нише."""
        return self.pos + Vec2.from_angle(math.radians(self.heading), self.seat_dist)


@dataclass
class BallInfo:
    pos: P
    vel: P
    radius: float
    owner: Optional[str]    # id робота, в нише которого мяч
    in_niche: bool          # мяч в чьей-то нише
    owner_team: Optional[int]

    @property
    def speed(self) -> float:
        return self.vel.length()


class FieldInfo:
    """Геометрия поля в системе команды."""

    def __init__(self, frame: Frame):
        self._frame = frame
        self.half_length = F.HX            # край игрового поля (внешний край линии)
        self.half_width = F.HY
        self.wall_x = F.WX                 # внутренняя грань стен
        self.wall_y = F.WY
        self.line_width = F.LINE_W
        self.goal_width = F.GOAL_W
        self.goal_depth = F.GOAL_D
        self.penalty_depth = F.PEN_DEPTH
        self.penalty_length = F.PEN_LEN
        self.center_circle_r = F.CENTER_CIRCLE_R
        self.own_goal = P(-F.GOAL_MOUTH_X, 0.0)
        self.opp_goal = P(F.GOAL_MOUTH_X, 0.0)
        self.own_goal_back = P(-F.GOAL_BACK_X, 0.0)
        self.opp_goal_back = P(F.GOAL_BACK_X, 0.0)
        self.neutral_spots = [frame.p_in(v) for v in F.NEUTRAL_SPOTS]

    def _side(self, own: bool) -> int:
        # сторона штрафной в мировых координатах
        team_side = -1 if own else 1
        return team_side * self._frame.s

    def penalty_distance(self, p: Vec2, own: bool = True) -> float:
        """Знаковое расстояние до штрафной (<0 — внутри)."""
        return F.penalty_sdf(self._frame.p_out(p), self._side(own))

    def in_penalty_area(self, p: Vec2, radius: float = 0.0, own: bool = True,
                        fully: bool = False) -> bool:
        """Робот радиуса radius в штрафной: частично (fully=False) или полностью."""
        d = self.penalty_distance(p, own)
        return d <= -radius if fully else d < radius

    def wall_clearance(self, p: Vec2, radius: float) -> float:
        """Зазор от круга до ближайшей стены (м). <=0 — касание (аут)."""
        return min(F.WX - abs(p.x), F.WY - abs(p.y)) - radius

    def in_field(self, p: Vec2) -> bool:
        return abs(p.x) <= F.HX and abs(p.y) <= F.HY


@dataclass
class WorldView:
    time: float                      # игровое время, с
    dt: float                        # период вызова стратегии, с
    mode: str                        # match | free | debug
    state: str                       # PLAY | KICKOFF | HALFTIME | FINISHED
    half: int
    time_left: Optional[float]       # до конца тайма (None — без ограничения)
    score_us: int
    score_them: int
    kickoff_us: bool                 # последнее начало с центра — наше
    me: RobotInfo
    teammates: list[RobotInfo]
    opponents: list[RobotInfo]
    ball: BallInfo
    field: FieldInfo
    shared: dict                     # общая память команды (связь между роботами)
    rules: dict

    @property
    def teammate(self) -> Optional[RobotInfo]:
        return self.teammates[0] if self.teammates else None


# ------------------------------------------------------------- управление
class RobotControl:
    """Команды роботу. Вызывайте каждый такт — команда действует до следующей."""

    def __init__(self, sim: "Simulation", robot: "Robot", frame: Frame):
        self._sim = sim
        self._r = robot
        self._f = frame

    # --- движение ---
    def move_to(self, x: float, y: float, speed: Optional[float] = None,
                heading: Optional[float] = None) -> None:
        """Ехать в точку (x, y) со скоростью speed (м/с), держа курс heading (°).

        speed=None — максимальная скорость робота; heading=None — держать текущий курс.
        Робот сам тормозит к точке. Точку, скорость и курс можно менять каждый такт.
        """
        r = self._r
        sp = r.max_speed if speed is None else max(0.0, float(speed))
        sp = min(sp, r.max_speed)            # физический предел робота (config: max_speed)
        r.cmd = Command(mode="target", target=self._f.p_out(Vec2(x, y)), speed=sp,
                        heading=self._heading(heading))

    def move(self, vx: float, vy: float, heading: Optional[float] = None,
             omega: Optional[float] = None) -> None:
        """Ехать с вектором скорости (vx, vy) м/с. Курс: heading (°) или omega (°/с)."""
        r = self._r
        h = self._heading(heading) if heading is not None else None
        if h is None and omega is None:
            h = r.heading
        r.cmd = Command(mode="velocity", vel=self._f.p_out(Vec2(vx, vy)), heading=h,
                        omega=math.radians(omega) if omega is not None and h is None else None)

    def move_dir(self, direction_deg: float, speed: float, heading: Optional[float] = None) -> None:
        """Ехать в направлении direction_deg (°) со скоростью speed."""
        v = Vec2.from_angle(math.radians(direction_deg), speed)
        self.move(v.x, v.y, heading=heading)

    def rotate_to(self, heading: float) -> None:
        """Стоять на месте и повернуться на курс heading (°)."""
        self._r.cmd = Command(mode="target", target=self._r.pos.copy(), speed=0.0,
                              heading=self._heading(heading))

    def stop(self) -> None:
        self._r.cmd = Command(mode="stop", heading=self._r.heading)

    # --- удар ---
    def kick(self, power: float = 1.0) -> bool:
        """Удар (power 0..1 от максимальной силы). True — мяч в зоне удара и кикер готов."""
        r = self._r
        ok = self.can_kick
        r.kick_request = float(power)
        return ok

    @property
    def can_kick(self) -> bool:
        r = self._r
        ready = self._sim.time - r.last_kick_t >= self._sim.physics.p["kick_cooldown_s"]
        return ready and self._sim.physics.kickable(r, self._sim.ball)

    @property
    def has_ball(self) -> bool:
        return self._sim.ball.owner is self._r

    @property
    def kick_speed(self) -> float:
        return self._r.kick_speed

    # --- отладка ---
    def debug(self, text: str) -> None:
        """Строка в панели отладки (видна в режиме debug и при включённом оверлее)."""
        self._r.debug_text.append(str(text))

    def mark(self, x: float, y: float, color=(255, 255, 0), label: str = "") -> None:
        """Нарисовать точку на поле (координаты системы команды)."""
        self._r.debug_marks.append(("pt", self._f.p_out(Vec2(x, y)), color, label))

    def line(self, x1: float, y1: float, x2: float, y2: float, color=(255, 255, 0)) -> None:
        self._r.debug_marks.append(("ln", self._f.p_out(Vec2(x1, y1)),
                                    self._f.p_out(Vec2(x2, y2)), color))

    # --- внутреннее ---
    def _heading(self, h: Optional[float]) -> Optional[float]:
        if h is None:
            return self._r.heading
        return self._f.h_out(float(h))


# ------------------------------------------------------------- стратегия
class Strategy:
    """Базовый класс тактики. Один экземпляр на робота."""

    def setup(self, world: WorldView) -> None:
        """Вызывается один раз перед первым тактом."""

    def on_kickoff(self, world: WorldView, kicking: bool) -> None:
        """Перед каждым началом с центра (после гола, в начале тайма)."""

    def kickoff_position(self, world: WorldView, kicking: bool) -> Optional[tuple[float, float, float]]:
        """Позиция (x, y, курс°) для начала с центра в системе команды или None — по умолчанию.

        Требования правил (п. 2.3): своя половина; не бьющая команда — вне центрального
        круга (>= 30 см от мяча). Недопустимая позиция заменяется позицией по умолчанию.
        """
        return None

    def step(self, world: WorldView, robot: RobotControl) -> None:
        raise NotImplementedError


class Idle(Strategy):
    """Робот стоит на месте."""

    def step(self, world, robot):
        robot.stop()


__all__ = ["Strategy", "Idle", "WorldView", "RobotInfo", "BallInfo", "FieldInfo",
           "RobotControl", "P", "Vec2", "angle_diff"]
