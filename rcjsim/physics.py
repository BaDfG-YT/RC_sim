"""Физика: движение роботов и мяча, столкновения, захват мяча нишей, удар."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from . import field as F
from .entities import Ball, Robot
from .vec import Vec2


@dataclass
class StepContacts:
    """Что произошло за шаг физики (для судьи)."""
    wall_touch: set = field(default_factory=set)          # id роботов, коснувшихся стены
    robot_contacts: set = field(default_factory=set)      # пары (id_a, id_b)
    ball_touch: set = field(default_factory=set)          # id роботов, касающихся мяча
    captures: list = field(default_factory=list)          # id захвативших мяч
    bounces: list = field(default_factory=list)           # (id, отн. скорость) — отскок от ниши
    kicks: list = field(default_factory=list)             # id ударивших
    releases: list = field(default_factory=list)          # (id, причина)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def circle_vs_box(p: Vec2, r: float, b: F.Box) -> Optional[tuple[Vec2, float]]:
    """Возвращает (нормаль наружу, глубина) или None."""
    cx = _clamp(p.x, b.x0, b.x1)
    cy = _clamp(p.y, b.y0, b.y1)
    dx, dy = p.x - cx, p.y - cy
    d2 = dx * dx + dy * dy
    if d2 > 1e-18:
        if d2 >= r * r:
            return None
        d = math.sqrt(d2)
        return Vec2(dx / d, dy / d), r - d
    # центр внутри прямоугольника — выталкиваем по ближайшей грани
    exits = [(p.x - b.x0, Vec2(-1, 0)), (b.x1 - p.x, Vec2(1, 0)),
             (p.y - b.y0, Vec2(0, -1)), (b.y1 - p.y, Vec2(0, 1))]
    dist, n = min(exits, key=lambda e: e[0])
    return n, dist + r


def circle_vs_walls(p: Vec2, r: float) -> list[tuple[Vec2, float]]:
    out = []
    if p.x + r > F.WX:
        out.append((Vec2(-1, 0), p.x + r - F.WX))
    if p.x - r < -F.WX:
        out.append((Vec2(1, 0), -F.WX - (p.x - r)))
    if p.y + r > F.WY:
        out.append((Vec2(0, -1), p.y + r - F.WY))
    if p.y - r < -F.WY:
        out.append((Vec2(0, 1), -F.WY - (p.y - r)))
    return out


def ball_robot_contact(robot: Robot, bp: Vec2, br: float) -> Optional[tuple[Vec2, float, bool]]:
    """Контакт мяча с корпусом робота с учётом ниши.

    Корпус = дуга окружности вне выреза + два угла выреза. Возвращает
    (нормаль от робота к мячу, глубина, контакт_в_зоне_ниши) или None.
    """
    lp = robot.to_local(bp)
    R = robot.radius
    half = robot.niche_width / 2
    best: Optional[tuple[Vec2, float, bool]] = None

    d = lp.length()
    if d > 1e-9:
        # ближайшая точка дуги — в направлении мяча; валидна, если не попадает в вырез
        in_notch = lp.x > 0 and abs(lp.y / d * R) < half
        if not in_notch:
            pen = R + br - d
            if pen > 0:
                best = (lp / d, pen, False)
    for c in robot.niche_corners_local():
        v = lp - c
        dc = v.length()
        pen = br - dc
        if pen > 0 and dc > 1e-9 and (best is None or pen > best[1]):
            best = (v / dc, pen, True)
    if best is None:
        return None
    n_local, pen, niche = best
    return n_local.rotated(robot.heading), pen, niche


class Physics:
    def __init__(self, pcfg: dict):
        self.p = pcfg
        self.ball_r = float(pcfg["ball_radius"])

    # ------------------------------------------------------------ основное
    def step(self, dt: float, t: float, robots: list[Robot], ball: Ball) -> StepContacts:
        c = StepContacts()
        movers = [r for r in robots if r.active]
        solids = [r for r in robots if r.on_field]   # замороженные тоже мешают

        self._process_kicks(t, movers, ball, c)

        # 1) движение роботов
        for r in movers:
            limits = None
            if ball.owner is r and self.p["auto_limit_accel_with_ball"]:
                limits = self._ball_limits(r)
            r.update_motion(dt, limits)

        # 2) столкновения роботов со стенами и воротами
        for r in movers:
            self._robot_static(r, c)

        # 3) робот-робот
        for i in range(len(solids)):
            for j in range(i + 1, len(solids)):
                self._robot_robot(solids[i], solids[j], c)

        # 4) мяч
        if ball.owner is not None:
            self._carry_ball(t, ball, c)
        if ball.owner is None:
            self._free_ball(dt, t, ball, solids, c)
        else:
            # соперник/партнёр коснулся мяча в нише — мяч выбивается
            for r in solids:
                if r is ball.owner:
                    continue
                if ball_robot_contact(r, ball.pos, self.ball_r):
                    c.releases.append((ball.owner.id, f"выбит {r.id}"))
                    ball.owner.no_capture_until = t + self.p["recapture_block_s"]
                    ball.owner = None
                    self._ball_vs_robot(t, ball, r, c)
                    break
        if ball.owner is not None:
            c.ball_touch.add(ball.owner.id)
        return c

    # ------------------------------------------------------------ удар
    def kickable(self, r: Robot, ball: Ball) -> bool:
        if ball.owner is r:
            return True
        if ball.owner is not None:
            return False
        lp = r.to_local(ball.pos)
        tol = float(self.p["kick_contact_tol"])
        lat = float(self.p["capture_lateral_tol"]) * (r.niche_width / 2 + self.ball_r)
        return r.seat_x - 0.002 <= lp.x <= r.seat_x + tol and abs(lp.y) <= lat

    def _process_kicks(self, t: float, movers: list[Robot], ball: Ball, c: StepContacts):
        for r in movers:
            if r.kick_request is None:
                continue
            power = _clamp(r.kick_request, 0.0, 1.0)
            r.kick_request = None
            if t - r.last_kick_t < self.p["kick_cooldown_s"]:
                continue
            if not self.kickable(r, ball):
                continue
            ball.owner = None
            ball.pos = r.seat_world() + r.forward() * 0.001
            ball.vel = r.point_velocity(ball.pos) + r.forward() * (r.kick_speed * power)
            r.last_kick_t = t
            r.no_capture_until = t + self.p["recapture_block_s"]
            c.kicks.append(r.id)

    # ---------------------------------------------------------- роботы
    def _robot_static(self, r: Robot, c: StepContacts):
        for n, pen in circle_vs_walls(r.pos, r.radius):
            r.pos = r.pos + n * pen
            vn = r.vel.dot(n)
            if vn < 0:
                r.vel = r.vel - n * vn
        for b in F.GOAL_BOXES:
            hit = circle_vs_box(r.pos, r.radius, b)
            if hit:
                n, pen = hit
                r.pos = r.pos + n * pen
                vn = r.vel.dot(n)
                if vn < 0:
                    r.vel = r.vel - n * vn
        # касание стены (зазор до 1 мм после разрешения)
        eps = 1e-3
        if (abs(r.pos.x) + r.radius >= F.WX - eps) or (abs(r.pos.y) + r.radius >= F.WY - eps):
            c.wall_touch.add(r.id)

    def _robot_robot(self, a: Robot, b: Robot, c: StepContacts):
        d = b.pos - a.pos
        dist = d.length()
        min_d = a.radius + b.radius
        if dist >= min_d + 0.001:
            return
        c.robot_contacts.add(tuple(sorted((a.id, b.id))))
        if dist >= min_d:
            return
        n = d / dist if dist > 1e-9 else Vec2(1, 0)
        pen = min_d - dist
        ma = 0.0 if not a.active else 1.0
        mb = 0.0 if not b.active else 1.0
        if ma + mb == 0:
            return
        a.pos = a.pos - n * (pen * ma / (ma + mb))
        b.pos = b.pos + n * (pen * mb / (ma + mb))
        rel = (b.vel - a.vel).dot(n)
        if rel < 0:
            e = self.p["robot_robot_restitution"]
            j = -(1 + e) * rel / (ma + mb)
            a.vel = a.vel - n * (j * ma)
            b.vel = b.vel + n * (j * mb)

    def _ball_limits(self, r: Robot) -> tuple[float, float, float]:
        lat = self.p["dribbler_max_lateral_acc"]
        brk = self.p["dribbler_max_brake_acc"]
        x = max(r.seat_x, 1e-3)
        return 0.5 * min(lat, brk), math.sqrt(0.5 * brk / x), 0.5 * lat / x

    # ------------------------------------------------------------ мяч
    def _carry_ball(self, t: float, ball: Ball, c: StepContacts):
        """Мяч в нише едет вместе с роботом, пока дриблер его удерживает."""
        r = ball.owner
        if not r.active:
            ball.owner = None
            return
        # ускорение точки мяча в системе робота
        off = r.seat_world() - r.pos
        a_seat = r.accel + off.perp() * r.alpha - off * (r.omega ** 2)
        a_loc = a_seat.rotated(-r.heading)
        if abs(a_loc.y) > self.p["dribbler_max_lateral_acc"]:
            c.releases.append((r.id, "боковое ускорение"))
            ball.owner = None
            return
        if a_loc.x < -self.p["dribbler_max_brake_acc"]:
            c.releases.append((r.id, "резкое торможение/вращение"))
            ball.owner = None
            return
        ball.pos = r.seat_world()
        ball.vel = r.point_velocity(ball.pos)
        # мяч упирается в стену/ворота — упирается и робот
        fixes: list[tuple[Vec2, float]] = circle_vs_walls(ball.pos, self.ball_r)
        for b in F.GOAL_BOXES:
            hit = circle_vs_box(ball.pos, self.ball_r, b)
            if hit:
                fixes.append(hit)
        for n, pen in fixes:
            r.pos = r.pos + n * pen
            ball.pos = ball.pos + n * pen
            vn = r.vel.dot(n)
            if vn < 0:
                r.vel = r.vel - n * vn
        if fixes:
            ball.vel = r.point_velocity(ball.pos)

    def _free_ball(self, dt: float, t: float, ball: Ball, solids: list[Robot], c: StepContacts):
        # трение качения + клин
        acc = F.wedge_accel(ball.pos, self.ball_r) if self.p["wedge"] else Vec2()
        ball.vel = ball.vel + acc * dt
        sp = ball.vel.length()
        dec = self.p["ball_rolling_decel"] * dt
        if sp <= dec and acc.length_sq() == 0:
            ball.vel = Vec2()
        elif sp > 0:
            ball.vel = ball.vel * (max(sp - dec, 0.0) / sp)
        ball.pos = ball.pos + ball.vel * dt

        e = self.p["ball_wall_restitution"]
        for n, pen in circle_vs_walls(ball.pos, self.ball_r):
            ball.pos = ball.pos + n * pen
            vn = ball.vel.dot(n)
            if vn < 0:
                ball.vel = ball.vel - n * ((1 + e) * vn)
        for b in F.GOAL_BOXES:
            hit = circle_vs_box(ball.pos, self.ball_r, b)
            if hit:
                n, pen = hit
                ball.pos = ball.pos + n * pen
                vn = ball.vel.dot(n)
                if vn < 0:
                    ball.vel = ball.vel - n * ((1 + e) * vn)

        for r in solids:
            if ball.owner is not None:
                break
            self._ball_vs_robot(t, ball, r, c)

    def _ball_vs_robot(self, t: float, ball: Ball, r: Robot, c: StepContacts):
        hit = ball_robot_contact(r, ball.pos, self.ball_r)
        if hit is None:
            # касание «вплотную» — для судьи (толкание, мяч у робота)
            if r.pos.dist(ball.pos) <= r.radius + self.ball_r + 0.003:
                c.ball_touch.add(r.id)
            return
        n, pen, niche = hit
        c.ball_touch.add(r.id)
        v_rel = ball.vel - r.point_velocity(ball.pos)
        vn = v_rel.dot(n)
        fast_niche_hit = None

        if niche and r.active and t >= r.no_capture_until:
            lp = r.to_local(ball.pos)
            lat = self.p["capture_lateral_tol"] * (r.niche_width / 2 + self.ball_r)
            if lp.x >= r.corner_x and abs(lp.y) <= lat:
                speed = v_rel.length()
                if speed <= self.p["capture_max_speed"]:
                    ball.owner = r
                    ball.pos = r.seat_world()
                    ball.vel = r.point_velocity(ball.pos)
                    c.captures.append(r.id)
                    return
                fast_niche_hit = speed

        ball.pos = ball.pos + n * pen
        if vn < 0:
            if fast_niche_hit is not None:
                c.bounces.append((r.id, fast_niche_hit))
            e = self.p["ball_robot_restitution"]
            ball.vel = ball.vel - n * ((1 + e) * vn)
