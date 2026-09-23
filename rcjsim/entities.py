"""Мяч и робот: состояние, геометрия ниши, регулятор движения."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from . import field as F
from .config import (NICHE_WIDTH_MIN, ROBOT_RADIUS_MAX, ROBOT_RADIUS_MIN,
                     RULE_BALL_CAPTURE_DEPTH)
from .vec import Vec2, wrap_angle


# ------------------------------------------------------------ геометрия ниши
def niche_seat_x(R: float, w: float, r: float) -> float:
    """Расстояние от центра робота до центра мяча, лежащего в нише.

    Робот — круг радиуса R, спереди прямоугольный вырез шириной w. Мяч радиуса r
    (w < 2r) опирается на два угла выреза (x_c, ±w/2).
    """
    half = w / 2
    x_c = math.sqrt(max(R * R - half * half, 0.0))
    return x_c + math.sqrt(max(r * r - half * half, 0.0))


def niche_penetration(R: float, w: float, r: float) -> float:
    """Глубина захода мяча в выпуклую оболочку робота (правило 6.2.A.2)."""
    return R - (niche_seat_x(R, w, r) - r)


def max_niche_width(R: float, r: float, depth_limit: float = RULE_BALL_CAPTURE_DEPTH) -> float:
    """Максимальная ширина ниши, при которой мяч заходит в оболочку <= 1.5 см."""
    lo, hi = 0.0, 2 * r - 1e-6
    if niche_penetration(R, hi, r) <= depth_limit:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if niche_penetration(R, mid, r) <= depth_limit:
            lo = mid
        else:
            hi = mid
    return lo


def auto_kick_speed(R: float, r: float, decel: float, e_wall: float) -> float:
    """Максимальная скорость удара по процедуре Appendix A правил.

    Робот касается задней стенки своих ворот и бьёт в чужие; после отскока от
    задней стенки чужих ворот мяч не должен докатиться до задней стенки своих.
    При постоянном замедлении a и коэффициенте отскока e:
        e²·(v0² − 2a·d1) < 2a·d2  =>  v0 < sqrt(2a·(d1 + d2/e²))
    Возвращаем 95 % от предела.
    """
    x_start = -F.GOAL_BACK_X + 2 * R          # мяч перед роботом, стоящим у задней стенки
    d1 = (F.GOAL_BACK_X - r) - x_start         # до задней стенки чужих ворот
    d2 = 2 * (F.GOAL_BACK_X - r)               # обратно до задней стенки своих
    v0 = math.sqrt(2 * decel * (d1 + d2 / (e_wall * e_wall)))
    return 0.95 * v0


# ---------------------------------------------------------------------- мяч
@dataclass
class Ball:
    radius: float
    pos: Vec2 = field(default_factory=Vec2)
    vel: Vec2 = field(default_factory=Vec2)
    owner: Optional["Robot"] = None          # робот, в нише которого мяч

    def stop(self):
        self.vel = Vec2()


# -------------------------------------------------------------------- робот
@dataclass
class Command:
    mode: str = "stop"               # stop | target | velocity
    target: Vec2 = field(default_factory=Vec2)
    speed: float = 0.0
    vel: Vec2 = field(default_factory=Vec2)
    heading: Optional[float] = None  # целевой курс, рад (мировой)
    omega: Optional[float] = None    # желаемая угл. скорость, рад/с (режим velocity)


class Robot:
    def __init__(self, rid: str, team: int, index: int, name: str, tcfg: dict,
                 ball_radius: float, warnings: list[str]):
        self.id = rid                 # t1_1, t1_2, t2_1, t2_2
        self.team = team              # 1 или 2
        self.index = index            # 1 или 2 (номер на верхнем маркере)
        self.name = name
        self.color = tuple(tcfg.get("color", (200, 200, 200)))

        R = float(tcfg["radius"])
        if not (ROBOT_RADIUS_MIN <= R <= ROBOT_RADIUS_MAX):
            warnings.append(f"{rid}: радиус {R*100:.1f} см вне диапазона 7..9 см — ограничен")
            R = min(max(R, ROBOT_RADIUS_MIN), ROBOT_RADIUS_MAX)
        self.radius = R

        w_max = max_niche_width(R, ball_radius)
        w = float(tcfg["niche_width"])
        if not (NICHE_WIDTH_MIN <= w <= w_max + 1e-9):
            warnings.append(
                f"{rid}: ширина ниши {w*100:.2f} см вне 1..{w_max*100:.2f} см "
                f"(мяч не должен заходить глубже 1.5 см) — ограничена")
            w = min(max(w, NICHE_WIDTH_MIN), w_max)
        self.niche_width = w
        self.niche_width_max = w_max
        self.seat_x = niche_seat_x(R, w, ball_radius)
        self.corner_x = math.sqrt(R * R - (w / 2) ** 2)
        self.niche_depth = niche_penetration(R, w, ball_radius) + 0.002  # для отрисовки

        self.max_speed = float(tcfg["max_speed"])
        self.max_accel = float(tcfg["max_accel"])
        self.max_omega = math.radians(float(tcfg["max_omega_deg"]))
        self.max_alpha = math.radians(float(tcfg["max_alpha_deg"]))
        self.kick_speed = float(tcfg.get("kick_speed", 0.0))  # 0 → назначается симуляцией

        # состояние
        self.pos = Vec2()
        self.vel = Vec2()
        self.heading = 0.0
        self.omega = 0.0
        self.accel = Vec2()          # фактическое ускорение за последний шаг
        self.alpha = 0.0
        self.cmd = Command()
        self.on_field = True         # False — удалён (штраф)
        self.frozen = False          # режим отладки: остальные роботы заморожены
        self.penalty_until: Optional[float] = None
        self.penalty_reason = ""
        self.out_flag = False        # сейчас в ауте (для режима out_action=none)
        self.last_kick_t = -1e9
        self.no_capture_until = -1e9
        self.kick_request: Optional[float] = None
        self.debug_text: list[str] = []
        self.debug_marks: list[tuple] = []
        self.error: str = ""

    # --- геометрия ---
    def forward(self) -> Vec2:
        return Vec2.from_angle(self.heading)

    def to_local(self, p: Vec2) -> Vec2:
        return (p - self.pos).rotated(-self.heading)

    def to_world(self, lp: Vec2) -> Vec2:
        return self.pos + lp.rotated(self.heading)

    def point_velocity(self, p: Vec2) -> Vec2:
        """Скорость точки p, жёстко связанной с роботом."""
        r = p - self.pos
        return self.vel + r.perp() * self.omega

    def seat_world(self) -> Vec2:
        return self.to_world(Vec2(self.seat_x, 0.0))

    def niche_corners_local(self) -> tuple[Vec2, Vec2]:
        h = self.niche_width / 2
        return Vec2(self.corner_x, h), Vec2(self.corner_x, -h)

    @property
    def active(self) -> bool:
        return self.on_field and not self.frozen

    # --- регулятор движения (голономная база) ---
    def update_motion(self, dt: float, limits: Optional[tuple[float, float, float]] = None):
        """Шаг регулятора. limits=(a_max, omega_max, alpha_max) — доп. ограничения
        (используются, когда мяч в нише и включён auto_limit_accel_with_ball)."""
        c = self.cmd
        a_max = self.max_accel
        w_max = self.max_omega
        al_max = self.max_alpha
        if limits is not None:
            a_max = min(a_max, limits[0])
            w_max = min(w_max, limits[1])
            al_max = min(al_max, limits[2])

        # желаемая линейная скорость
        if c.mode == "target":
            d = c.target - self.pos
            dist = d.length()
            tol = 0.003
            if dist < tol:
                v_des = Vec2()
            else:
                v_lim = min(c.speed, self.max_speed, math.sqrt(2 * a_max * max(dist - tol, 0.0)))
                v_des = d * (v_lim / dist)
        elif c.mode == "velocity":
            v_des = c.vel.clamped(self.max_speed)
        else:
            v_des = Vec2()

        dv = (v_des - self.vel).clamped(a_max * dt)
        new_vel = self.vel + dv
        self.accel = dv / dt
        self.vel = new_vel

        # угловая скорость
        if c.heading is not None:
            err = wrap_angle(c.heading - self.heading)
            w_lim = min(w_max, math.sqrt(2 * al_max * abs(err)))
            w_des = math.copysign(min(w_lim, abs(err) / dt), err)
        elif c.mode == "velocity" and c.omega is not None:
            w_des = max(-w_max, min(w_max, c.omega))
        else:
            w_des = 0.0
        dw = max(-al_max * dt, min(al_max * dt, w_des - self.omega))
        self.alpha = dw / dt
        self.omega += dw

        self.pos = self.pos + self.vel * dt
        self.heading = wrap_angle(self.heading + self.omega * dt)

    def halt(self):
        self.vel = Vec2()
        self.omega = 0.0
        self.accel = Vec2()
        self.alpha = 0.0
        self.cmd = Command()
        self.kick_request = None
