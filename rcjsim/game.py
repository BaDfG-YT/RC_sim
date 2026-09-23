"""Симуляция: мир, стратегии, судья, режимы, внешние действия (телепорт и т.п.)."""
from __future__ import annotations

import importlib
import math
import random
import sys
import traceback
from collections import deque
from typing import Optional

from . import field as F
from .api import (BallInfo, FieldInfo, Frame, Idle, RobotControl, RobotInfo,
                  Strategy, WorldView)
from .config import rules_for_mode, team_cfg
from .entities import Ball, Robot, auto_kick_speed
from .physics import Physics, StepContacts, circle_vs_box, circle_vs_walls
from .referee import KICKOFF, PLAY, Referee
from .vec import Vec2, wrap_angle

MODES = ("match", "free", "debug")
ROBOT_IDS = ("t1_1", "t1_2", "t2_1", "t2_2")


def load_strategy(spec: str) -> Strategy:
    """'package.module:ClassName' → экземпляр стратегии."""
    if not spec:
        return Idle()
    mod_name, _, cls_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls()


class Simulation:
    def __init__(self, cfg: dict, mode: str = "match", debug_robot: Optional[str] = None):
        if mode not in MODES:
            raise ValueError(f"Неизвестный режим {mode}; доступны: {MODES}")
        self.cfg = cfg
        self.mode = mode
        self.rules = rules_for_mode(cfg, mode)
        self.physics = Physics(cfg["physics"])
        self.warnings: list[str] = []
        self.events: deque[tuple[float, str]] = deque(maxlen=300)
        seed = int(cfg["sim"]["seed"]) or random.randrange(1, 10**9)
        self.rng = random.Random(seed)
        self.time = 0.0
        self.dt = float(cfg["sim"]["physics_dt"])
        self.control_dt = 1.0 / float(cfg["sim"]["control_hz"])
        self._ctrl_acc = 0.0
        self.last_contacts = StepContacts()
        self._last_bounce_log: dict[str, float] = {}

        pr = cfg["physics"]
        self.ball = Ball(radius=float(pr["ball_radius"]))

        # роботы
        self.robots: list[Robot] = []
        self.team_cfgs = {1: team_cfg(cfg, "team1"), 2: team_cfg(cfg, "team2")}
        for team in (1, 2):
            tc = self.team_cfgs[team]
            rlist = tc.get("robots", [])
            for idx in (1, 2):
                rc = rlist[idx - 1] if idx - 1 < len(rlist) else {}
                merged = dict(tc)
                merged.update({k: v for k, v in rc.items() if k not in ("name", "strategy")})
                rb = Robot(f"t{team}_{idx}", team, idx, rc.get("name", f"R{idx}"), merged,
                           self.ball.radius, self.warnings)
                ks_cap = auto_kick_speed(rb.radius, self.ball.radius,
                                         float(pr["ball_rolling_decel"]),
                                         float(pr["ball_wall_restitution"]))
                if rb.kick_speed <= 0:
                    rb.kick_speed = ks_cap
                elif rb.kick_speed > ks_cap:
                    self.warnings.append(
                        f"{rb.id}: kick_speed {rb.kick_speed:.2f} м/с > {ks_cap:.2f} м/с "
                        f"(не пройдёт проверку силы удара, Appendix A)")
                rb.strategy_spec = rc.get("strategy", "")
                self.robots.append(rb)
        self.robot_by_id = {r.id: r for r in self.robots}

        # стороны: _own_side[team] = -1 (свои ворота слева) / +1
        self._own_side = {1: -1, 2: 1}
        tf = bool(cfg["sim"]["team_frame"])
        self.frames = {1: Frame(1, tf), 2: Frame(-1, tf)}
        self.field_infos = {t: FieldInfo(self.frames[t]) for t in (1, 2)}
        self.shared = {1: {}, 2: {}}
        self.strategies: dict[str, Strategy] = {}
        self.controls: dict[str, RobotControl] = {}
        self._setup_done: set[str] = set()

        # режим отладки
        self.debug_robot_id = debug_robot or cfg["modes"]["debug"].get("robot", "t1_1")
        if self.debug_robot_id not in self.robot_by_id:
            raise ValueError(f"Робот {self.debug_robot_id} не найден; есть {list(self.robot_by_id)}")
        self._hidden: set[str] = set()
        if mode == "debug":
            others = cfg["modes"]["debug"].get("others", "hidden")
            for rb in self.robots:
                if rb.id == self.debug_robot_id:
                    continue
                if others == "hidden":
                    self._hidden.add(rb.id)
                    rb.on_field = False
                elif others == "frozen":
                    rb.frozen = True

        self.load_strategies()
        self.referee = Referee(self)
        for w in self.warnings:
            self.log("⚠ " + w)
        self.referee.start_match()

    # ------------------------------------------------------------ утилиты
    def log(self, msg: str):
        self.events.append((self.time, msg))

    def team_name(self, team: int) -> str:
        return str(self.team_cfgs[team].get("name", f"Team {team}"))

    def own_side(self, team: int) -> int:
        return self._own_side[team]

    def team_with_own_side(self, side: int) -> int:
        return 1 if self._own_side[1] == side else 2

    def is_hidden(self, rb: Robot) -> bool:
        return rb.id in self._hidden

    def swap_sides(self):
        self._own_side = {1: self._own_side[2], 2: self._own_side[1]}
        for t in (1, 2):
            self.frames[t].s = -self._own_side[t] if self.cfg["sim"]["team_frame"] else 1
            self.field_infos[t] = FieldInfo(self.frames[t])

    @property
    def state(self) -> str:
        return self.referee.state

    # --------------------------------------------------------- стратегии
    def load_strategies(self, reload: bool = False):
        if reload:
            for name in list(sys.modules):
                if name == "strategies" or name.startswith("strategies."):
                    importlib.reload(sys.modules[name])
        for rb in self.robots:
            try:
                self.strategies[rb.id] = load_strategy(rb.strategy_spec)
                rb.error = ""
            except Exception as e:  # noqa: BLE001
                self.strategies[rb.id] = Idle()
                rb.error = f"load: {e}"
                self.log(f"Ошибка загрузки стратегии {rb.id}: {e}")
            self.controls[rb.id] = RobotControl(self, rb, self.frames[rb.team])
        self._setup_done.clear()
        if reload:
            self.log("Стратегии перезагружены")

    def _info(self, rb: Robot, frame: Frame, noise: bool) -> RobotInfo:
        pos = rb.pos
        hd = rb.heading
        if noise:
            pos, hd = self._noisy(pos, hd)
        left = max(0.0, rb.penalty_until - self.time) if rb.penalty_until else 0.0
        return RobotInfo(
            id=rb.id, team=rb.team, index=rb.index, name=rb.name,
            pos=frame.p_in(pos), heading=frame.h_in(hd), vel=frame.p_in(rb.vel),
            omega=math.degrees(rb.omega), radius=rb.radius, niche_width=rb.niche_width,
            max_speed=rb.max_speed, seat_dist=rb.seat_x, has_ball=self.ball.owner is rb,
            on_field=rb.on_field, penalty_left=left)

    def _noisy(self, pos: Vec2, hd: float) -> tuple[Vec2, float]:
        s = float(self.cfg["sim"]["position_noise"])
        h = math.radians(float(self.cfg["sim"]["heading_noise_deg"]))
        if s > 0:
            pos = Vec2(pos.x + self.rng.gauss(0, s), pos.y + self.rng.gauss(0, s))
        if h > 0:
            hd = hd + self.rng.gauss(0, h)
        return pos, hd

    def world_view(self, rb: Robot) -> WorldView:
        fr = self.frames[rb.team]
        noise = float(self.cfg["sim"]["position_noise"]) > 0 or float(self.cfg["sim"]["heading_noise_deg"]) > 0
        me = self._info(rb, fr, False)
        visible = [r for r in self.robots if not self.is_hidden(r) and r is not rb]
        mates = [self._info(r, fr, noise) for r in visible if r.team == rb.team]
        opps = [self._info(r, fr, noise) for r in visible if r.team != rb.team]
        if noise:
            me = self._info(rb, fr, True)
        b = self.ball
        bpos = b.pos
        if noise:
            bpos, _ = self._noisy(bpos, 0.0)
        ball = BallInfo(pos=fr.p_in(bpos), vel=fr.p_in(b.vel), radius=b.radius,
                        owner=b.owner.id if b.owner else None, in_niche=b.owner is not None,
                        owner_team=b.owner.team if b.owner else None)
        ref = self.referee
        other = 2 if rb.team == 1 else 1
        return WorldView(
            time=self.time, dt=self.control_dt, mode=self.mode, state=ref.state,
            half=ref.half, time_left=ref.half_time_left,
            score_us=ref.score[rb.team], score_them=ref.score[other],
            kickoff_us=ref.last_kickoff_team == rb.team, me=me, teammates=mates,
            opponents=opps, ball=ball, field=self.field_infos[rb.team],
            shared=self.shared[rb.team], rules=dict(self.rules))

    def _call(self, rb: Robot, fn_name: str, *args):
        try:
            return getattr(self.strategies[rb.id], fn_name)(*args)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if rb.error != msg:
                self.log(f"Ошибка стратегии {rb.id}.{fn_name}: {msg}")
                traceback.print_exc()
            rb.error = msg
            rb.cmd.mode = "stop"
            return None

    def _control_tick(self):
        for rb in self.robots:
            if not rb.active:
                continue
            wv = self.world_view(rb)
            if rb.id not in self._setup_done:
                self._setup_done.add(rb.id)
                self._call(rb, "setup", wv)
            rb.debug_text = []
            rb.debug_marks = []
            self._call(rb, "step", wv, self.controls[rb.id])

    # ------------------------------------------------------ начало с центра
    DEFAULT_KICKOFF = {  # система команды: свои ворота слева
        (1, True): (-0.13, 0.0, 0.0),
        (2, True): (-0.70, 0.0, 0.0),
        (1, False): (-0.45, 0.0, 0.0),
        (2, False): (-0.70, 0.0, 0.0),
    }

    def _kickoff_valid(self, rb: Robot, p: Vec2, kicking: bool, placed: list[Robot]) -> bool:
        side = self._own_side[rb.team]
        if p.x * side < rb.radius:               # целиком на своей половине
            return False
        if not kicking and p.length() < F.CENTER_CIRCLE_R + rb.radius:
            return False
        if p.length() < rb.radius + self.ball.radius + 0.005:
            return False
        if circle_vs_walls(p, rb.radius + 0.002):
            return False
        if any(circle_vs_box(p, rb.radius, b) for b in F.GOAL_BOXES):
            return False
        for s in (-1, 1):
            if F.penalty_sdf(p, s) <= -rb.radius:
                return False
        for o in placed:
            if o.pos.dist(p) < o.radius + rb.radius + 0.005:
                return False
        return True

    def place_for_kickoff(self, team: Optional[int]):
        placed: list[Robot] = []
        # сначала бьющая команда (п. 2.3.2)
        order = sorted(self.robots, key=lambda r: 0 if r.team == team else 1)
        for rb in order:
            rb.halt()
            rb.out_flag = False
            rb.debug_marks = []
            rb.debug_text = []
            if not rb.on_field:
                continue
            kicking = team == rb.team
            fr = self.frames[rb.team]
            wv = self.world_view(rb)
            pos = None
            res = self._call(rb, "kickoff_position", wv, kicking)
            if res is not None:
                try:
                    x, y, h = res
                    cand = fr.p_out(Vec2(x, y))
                    if self._kickoff_valid(rb, cand, kicking, placed):
                        pos, hd = cand, fr.h_out(h)
                    else:
                        self.log(f"{rb.id}: недопустимая позиция начала — по умолчанию")
                except (TypeError, ValueError):
                    self.log(f"{rb.id}: kickoff_position должен вернуть (x, y, курс)")
            if pos is None:
                x, y, h = self.DEFAULT_KICKOFF[(rb.index, kicking)]
                pos, hd = fr.p_out(Vec2(x, y)), fr.h_out(h)
                if not self._kickoff_valid(rb, pos, kicking, placed):
                    pos = fr.p_out(Vec2(x, 0.25 if rb.index == 1 else -0.25))
            rb.pos = pos
            rb.heading = hd
            placed.append(rb)
            self._call(rb, "on_kickoff", wv, kicking)

    # -------------------------------------------------------------- шаг
    def step(self, dt: Optional[float] = None):
        dt = dt or self.dt
        if self.referee.state == PLAY:
            self._ctrl_acc += dt
            if self._ctrl_acc >= self.control_dt:
                self._ctrl_acc -= self.control_dt
                self._control_tick()
            c = self.physics.step(dt, self.time, self.robots, self.ball)
        else:
            c = StepContacts()
            if self.referee.state == KICKOFF:
                self._ctrl_acc = self.control_dt  # первый такт сразу после свистка
        self._log_contacts(c)
        self.last_contacts = c
        self.time += dt
        self.referee.update(dt, c)

    def _log_contacts(self, c: StepContacts):
        for rid in c.captures:
            self.log(f"{rid}: мяч в нише")
        for rid, sp in c.bounces:
            # не засоряем журнал серией отскоков при «зажатом» мяче
            if self.time - self._last_bounce_log.get(rid, -1e9) >= 1.0:
                self._last_bounce_log[rid] = self.time
                self.log(f"{rid}: отскок от ниши ({sp:.2f} м/с > {self.physics.p['capture_max_speed']} м/с)")
        for rid, why in c.releases:
            self.log(f"{rid}: потерял мяч ({why})")
        for rid in c.kicks:
            self.log(f"{rid}: удар")

    def advance(self, seconds: float):
        n = int(round(seconds / self.dt))
        for _ in range(n):
            self.step()

    # ------------------------------------------------- действия оператора
    def _fit_inside(self, p: Vec2, r: float) -> Vec2:
        p = Vec2(max(-F.WX + r, min(F.WX - r, p.x)), max(-F.WY + r, min(F.WY - r, p.y)))
        for b in F.GOAL_BOXES:
            hit = circle_vs_box(p, r, b)
            if hit:
                n, pen = hit
                p = p + n * pen
        return p

    def teleport_robot(self, rid: str, p: Vec2, heading: Optional[float] = None):
        rb = self.robot_by_id[rid]
        if self.is_hidden(rb):
            self.log(f"{rid} скрыт в режиме отладки")
            return
        if self.ball.owner is rb:
            self.ball.owner = None
        rb.halt()
        rb.pos = self._fit_inside(p, rb.radius + 0.003)
        if heading is not None:
            rb.heading = wrap_angle(heading)
        if not rb.on_field:
            rb.on_field = True
            rb.penalty_until = None
            rb.penalty_reason = ""
        rb.out_flag = False
        self.log(f"{rid} перенесён в ({rb.pos.x:.2f}, {rb.pos.y:.2f})")

    def rotate_robot(self, rid: str, delta_rad: float):
        rb = self.robot_by_id[rid]
        rb.heading = wrap_angle(rb.heading + delta_rad)
        rb.omega = 0.0

    def teleport_ball(self, p: Vec2):
        b = self.ball
        b.owner = None
        b.pos = self._fit_inside(p, b.radius + 0.001)
        b.stop()
        self.referee._lop_ref = b.pos.copy()
        self.referee._lop_t = self.time
        self.log(f"Мяч перенесён в ({b.pos.x:.2f}, {b.pos.y:.2f})")

    def throw_ball(self, vel: Vec2):
        self.ball.owner = None
        self.ball.vel = vel

    def stop_ball(self):
        self.ball.owner = None
        self.ball.stop()

    def restart_kickoff(self):
        """Ручное нейтральное начало с центра."""
        if self.referee.state in (PLAY, KICKOFF):
            self.referee.kickoff(None)
