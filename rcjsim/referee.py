"""Судья: время, голы, начало с центра, аут, штрафы и другие правила RCJ Soccer 2026."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from . import field as F
from .vec import Vec2

if TYPE_CHECKING:
    from .entities import Robot
    from .game import Simulation
    from .physics import StepContacts

PLAY, KICKOFF, HALFTIME, FINISHED = "PLAY", "KICKOFF", "HALFTIME", "FINISHED"


class Referee:
    def __init__(self, sim: "Simulation"):
        self.sim = sim
        self.state = KICKOFF
        self.timer = 0.0
        self.half = 1
        self.half_time_left: Optional[float] = None
        self.score = {1: 0, 2: 0}
        self.first_kickoff_team = 1
        self.last_kickoff_team: Optional[int] = None
        self._lop_ref = Vec2()
        self._lop_t = 0.0
        self._last_goal_t = -1e9

    # ------------------------------------------------------------ свойства
    @property
    def rules(self) -> dict:
        return self.sim.rules

    @property
    def timed(self) -> bool:
        return self.sim.mode == "match"

    def final_score(self) -> tuple[int, int]:
        """Счёт с урезанием разницы до max_goal_difference (п. 2.1.3)."""
        a, b = self.score[1], self.score[2]
        m = int(self.rules["max_goal_difference"])
        if self.timed and m > 0:
            if a - b > m:
                a = b + m
            elif b - a > m:
                b = a + m
        return a, b

    # -------------------------------------------------------------- старт
    def start_match(self):
        r = self.rules
        toss = r.get("coin_toss", "random")
        if toss == "team1":
            self.first_kickoff_team = 1
        elif toss == "team2":
            self.first_kickoff_team = 2
        else:
            self.first_kickoff_team = self.sim.rng.choice([1, 2])
        self.half = 1
        self.half_time_left = float(r["half_duration_s"]) if self.timed else None
        if self.timed:
            self.sim.log(f"Жеребьёвка: начинает {self.sim.team_name(self.first_kickoff_team)}")
        self.kickoff(self.first_kickoff_team)

    def kickoff(self, team: Optional[int]):
        """Начало с центра. team=None — нейтральное (все вне круга, п. 2.3.A)."""
        sim = self.sim
        self.state = KICKOFF
        self.timer = float(self.rules["kickoff_delay_s"])
        self.last_kickoff_team = team
        sim.ball.owner = None
        sim.ball.pos = Vec2()
        sim.ball.stop()
        # п. 2.3.6 / 2.8.1 — удалённые роботы возвращаются
        for rb in sim.robots:
            if rb.penalty_until is not None and self.rules["return_at_kickoff"]:
                rb.penalty_until = None
                rb.penalty_reason = ""
                if not sim.is_hidden(rb):
                    rb.on_field = True
        sim.place_for_kickoff(team)
        self._lop_ref = Vec2()
        self._lop_t = sim.time
        who = "нейтральное" if team is None else sim.team_name(team)
        sim.log(f"Начало с центра: {who}")

    # ---------------------------------------------------------- основной шаг
    def update(self, dt: float, c: "StepContacts"):
        sim = self.sim
        if self.state == FINISHED:
            return
        if self.state == HALFTIME:
            self.timer -= dt
            if self.timer <= 0:
                self._start_second_half()
            return

        # игровые часы идут без остановок (в т.ч. во время расстановки)
        if self.half_time_left is not None:
            self.half_time_left -= dt
            if self.half_time_left <= 0:
                self._end_half()
                return

        if self.state == KICKOFF:
            self.timer -= dt
            if self.timer <= 0:
                self.state = PLAY
                sim.log("Свисток — игра!")
            return

        self._return_penalized()
        if self._check_goal():
            return
        if self.rules["pushing"]:
            self._check_pushing(c)
        if self.rules["multiple_defense"]:
            self._check_multiple_defense()
        if self.rules["out_of_bounds"]:
            self._check_out(c)
        if self.rules["lack_of_progress"]:
            self._check_lack_of_progress()

    # ---------------------------------------------------------------- тайм
    def _end_half(self):
        sim = self.sim
        if self.half < int(self.rules["halves"]):
            sim.log(f"Конец {self.half}-го тайма")
            self.state = HALFTIME
            self.timer = float(self.rules["halftime_break_s"])
            for rb in sim.robots:
                rb.halt()
            sim.ball.owner = None
            sim.ball.stop()
            if self.timer <= 0:
                self._start_second_half()
        else:
            self.state = FINISHED
            for rb in sim.robots:
                rb.halt()
            a, b = self.final_score()
            sim.log(f"Матч окончен: {a}:{b}")

    def _start_second_half(self):
        sim = self.sim
        self.half += 1
        self.half_time_left = float(self.rules["half_duration_s"])
        sim.swap_sides()
        sim.log(f"{self.half}-й тайм, команды поменялись сторонами")
        # п. 2.2.1 — второй тайм начинает команда, не начинавшая первый
        self.kickoff(2 if self.first_kickoff_team == 1 else 1)

    # ---------------------------------------------------------------- гол
    def _check_goal(self) -> bool:
        sim = self.sim
        b = sim.ball
        if abs(b.pos.y) > F.GOAL_W / 2:
            return False
        if abs(b.pos.x) + b.radius < F.GOAL_BACK_X - 1e-4:
            return False
        side = 1 if b.pos.x > 0 else -1          # в какие ворота (мировая сторона)
        conceded = sim.team_with_own_side(side)
        scorer = 2 if conceded == 1 else 1
        self.score[scorer] += 1
        self._last_goal_t = sim.time
        who = b.owner.id if b.owner else "мяч"
        sim.log(f"ГОЛ! {sim.team_name(scorer)}  {self.score[1]}:{self.score[2]}  ({who})")
        self.kickoff(conceded)                  # п. 2.4.1
        return True

    # ------------------------------------------------------------ штрафы
    def _return_penalized(self):
        sim = self.sim
        for rb in sim.robots:
            if rb.penalty_until is not None and sim.time >= rb.penalty_until:
                rb.penalty_until = None
                rb.penalty_reason = ""
                rb.on_field = True
                self.place_on_neutral_spot(rb)
                sim.log(f"{rb.id} вернулся после штрафа")

    def place_on_neutral_spot(self, rb: "Robot"):
        """Незанятая нейтральная точка, самая дальняя от мяча, лицом к своим воротам (п. 2.8.3)."""
        sim = self.sim
        spot = self.best_spot(sim.ball.pos, furthest=True, for_robot=rb)
        rb.halt()
        rb.pos = spot.copy()
        own = Vec2(sim.own_side(rb.team) * F.GOAL_MOUTH_X, 0.0)
        rb.heading = math.atan2(own.y - spot.y, own.x - spot.x)
        rb.out_flag = False
        if sim.ball.owner is rb:
            sim.ball.owner = None

    def best_spot(self, ref: Vec2, furthest: bool, for_robot: Optional["Robot"] = None,
                  exclude_near: Optional[Vec2] = None) -> Vec2:
        sim = self.sim

        def free(spot: Vec2) -> bool:
            for o in sim.robots:
                if not o.on_field or o is for_robot:
                    continue
                need = o.radius + (for_robot.radius if for_robot else sim.ball.radius) + 0.01
                if o.pos.dist(spot) < need:
                    return False
            if for_robot is not None and sim.ball.pos.dist(spot) < for_robot.radius + sim.ball.radius + 0.01:
                return False
            return True

        spots = [s for s in F.NEUTRAL_SPOTS
                 if exclude_near is None or s.dist(exclude_near) > 0.05]
        cand = [s for s in spots if free(s)] or spots
        key = (lambda s: s.dist(ref))
        return (max(cand, key=key) if furthest else min(cand, key=key)).copy()

    def _move_ball_to(self, spot: Vec2):
        b = self.sim.ball
        b.owner = None
        b.pos = spot.copy()
        b.stop()
        self._lop_ref = spot.copy()
        self._lop_t = self.sim.time

    # ----------------------------------------------------- толкание (2.6.3)
    def _check_pushing(self, c: "StepContacts"):
        sim = self.sim
        byid = sim.robot_by_id
        for a_id, b_id in c.robot_contacts:
            a, b = byid[a_id], byid[b_id]
            if a.team == b.team or not (a.active and b.active):
                continue
            if not (a_id in c.ball_touch or b_id in c.ball_touch):
                continue
            for side in (-1, 1):
                if (F.penalty_sdf(a.pos, side) < a.radius) or (F.penalty_sdf(b.pos, side) < b.radius):
                    spot = self.best_spot(sim.ball.pos, furthest=True)
                    self._move_ball_to(spot)
                    sim.log(f"Толкание ({a_id} ↔ {b_id}) — мяч на нейтральную точку")
                    return

    # --------------------------------------------- двойная защита (2.6.2)
    def _check_multiple_defense(self):
        sim = self.sim
        for team in (1, 2):
            mates = [r for r in sim.robots if r.team == team and r.active]
            if len(mates) < 2:
                continue
            for side in (-1, 1):
                inside = [r for r in mates if F.penalty_sdf(r.pos, side) < r.radius]
                if len(inside) >= 2:
                    far = max(inside, key=lambda r: r.pos.dist(sim.ball.pos))
                    self.place_on_neutral_spot(far)
                    sim.log(f"Двойная защита: {far.id} → нейтральная точка")

    # ------------------------------------------------------------- аут (2.8)
    def out_reason(self, rb: "Robot", c: "StepContacts") -> Optional[str]:
        r = self.rules
        if r["out_on_wall_touch"] and rb.id in c.wall_touch:
            return "касание стены"
        if r["out_on_full_penalty_area"]:
            for side in (-1, 1):
                if F.penalty_sdf(rb.pos, side) <= -rb.radius:
                    return "полностью в штрафной"
        if r["out_on_leaving_field"]:
            if abs(rb.pos.x) - rb.radius > F.HX or abs(rb.pos.y) - rb.radius > F.HY:
                return "полностью за линией поля"
        return None

    def _check_out(self, c: "StepContacts"):
        sim = self.sim
        action = self.rules["out_action"]
        for rb in sim.robots:
            if not rb.active:
                continue
            reason = self.out_reason(rb, c)
            if reason is None:
                rb.out_flag = False
                continue
            if rb.out_flag:
                continue                    # уже зафиксирован (режим none)
            rb.out_flag = True
            if action == "penalty":
                dur = float(self.rules["penalty_duration_s"])
                rb.on_field = False
                rb.halt()
                rb.penalty_until = sim.time + dur
                rb.penalty_reason = reason
                if sim.ball.owner is rb:
                    sim.ball.owner = None
                sim.log(f"АУТ {rb.id}: {reason} — удалён на {dur:.0f} с")
            elif action == "reposition":
                self.place_on_neutral_spot(rb)
                sim.log(f"АУТ {rb.id}: {reason} — на нейтральную точку")
            else:
                sim.log(f"АУТ {rb.id}: {reason} (без наказания)")

    # ------------------------------------------------ нет прогресса (2.7)
    def _check_lack_of_progress(self):
        sim = self.sim
        b = sim.ball
        if b.pos.dist(self._lop_ref) > float(self.rules["lack_of_progress_dist"]):
            self._lop_ref = b.pos.copy()
            self._lop_t = sim.time
            return
        if sim.time - self._lop_t >= float(self.rules["lack_of_progress_s"]):
            spot = self.best_spot(b.pos, furthest=False, exclude_near=b.pos)
            self._move_ball_to(spot)
            sim.log("Нет прогресса — мяч на ближайшую нейтральную точку")
