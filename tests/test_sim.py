"""Проверки физики и правил без окна:  python -m pytest -q"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from rcjsim import field as F  # noqa: E402
from rcjsim.config import load_config  # noqa: E402
from rcjsim.entities import max_niche_width, niche_penetration  # noqa: E402
from rcjsim.game import Simulation  # noqa: E402
from rcjsim.referee import FINISHED, PLAY  # noqa: E402
from rcjsim.api import Strategy  # noqa: E402
from rcjsim.vec import Vec2  # noqa: E402


class Hold(Strategy):
    """Ничего не делает: команды задаёт сам тест."""

    def step(self, world, robot):
        pass


def make_sim(mode="debug", robot="t1_1", **rules):
    cfg = load_config(None)
    cfg["sim"]["seed"] = 7
    for t in ("team1", "team2"):
        cfg[t]["robots"] = [{"name": "A", "strategy": "rcjsim.api:Idle"},
                            {"name": "B", "strategy": "rcjsim.api:Idle"}]
    cfg["modes"][mode] = cfg["modes"].get(mode, {})
    cfg["modes"][mode].setdefault("rules", {}).update(rules)
    sim = Simulation(cfg, mode, robot)
    sim.strategies = {rid: Hold() for rid in sim.strategies}
    sim.advance(1.1)  # пройти расстановку
    assert sim.state == PLAY
    return sim


def place(sim, rid, x, y, h_deg=0.0):
    rb = sim.robot_by_id[rid]
    rb.on_field = True
    rb.halt()
    rb.pos = Vec2(x, y)
    rb.heading = math.radians(h_deg)
    return rb


# ------------------------------------------------------------- геометрия
def test_niche_width_limit_matches_rule():
    for R in (0.07, 0.08, 0.09):
        w = max_niche_width(R, 0.021)
        assert 0.036 < w < 0.042
        assert niche_penetration(R, w, 0.021) == pytest.approx(0.015, abs=1e-5)


def test_config_clamps_radius_and_niche():
    cfg = load_config(None)
    cfg["team1"]["radius"] = 0.12
    cfg["team1"]["niche_width"] = 0.08
    sim = Simulation(cfg, "free")
    rb = sim.robot_by_id["t1_1"]
    assert rb.radius == pytest.approx(0.09)
    assert rb.niche_width == pytest.approx(rb.niche_width_max)
    assert sim.warnings


def test_penalty_area_shape():
    # центр штрафной у правых ворот — внутри; за скруглением угла — снаружи
    assert F.penalty_sdf(Vec2(F.HX - 0.1, 0.0), 1) < 0
    assert F.penalty_sdf(Vec2(F.HX - 0.24, 0.39), 1) > 0      # срезано скруглением R15
    assert F.penalty_sdf(Vec2(F.HX - 0.05, 0.39), 1) < 0
    assert F.penalty_sdf(Vec2(-F.HX + 0.1, 0.0), -1) < 0


# ---------------------------------------------------------- ниша и мяч
def roll_ball_into_niche(speed):
    sim = make_sim()
    rb = place(sim, "t1_1", 0.0, 0.0, 0.0)
    sim.ball.pos = Vec2(0.3, 0.0)
    sim.ball.vel = Vec2(-speed, 0.0)
    sim.advance(1.0)
    return sim, rb


def test_slow_ball_is_captured():
    sim, rb = roll_ball_into_niche(0.45 + 0.3 * 0.35)   # у ниши ≈ 0.4 м/с
    assert sim.ball.owner is rb
    assert sim.ball.pos.dist(rb.seat_world()) < 1e-6


def test_fast_ball_bounces():
    sim, rb = roll_ball_into_niche(1.2)
    assert sim.ball.owner is None
    assert sim.ball.vel.x > 0          # отскочил


def test_robot_carries_and_kicks_ball():
    sim, rb = roll_ball_into_niche(0.5)
    assert sim.ball.owner is rb
    ctl = sim.controls[rb.id]
    # едем вперёд с мячом
    for _ in range(60):
        ctl.move_to(0.5, 0.0, speed=0.8, heading=0.0)
        sim.advance(sim.control_dt)
    assert sim.ball.owner is rb
    assert rb.pos.x > 0.2
    assert ctl.kick(1.0)
    sim.advance(0.01)
    assert sim.ball.owner is None
    assert sim.ball.vel.x > rb.kick_speed * 0.9


def test_kick_power_passes_appendix_a():
    """Робот у задней стенки своих ворот бьёт в чужие: мяч не должен вернуться к своей стенке."""
    sim = make_sim(out_action="none")
    rb = place(sim, "t1_1", -F.GOAL_BACK_X + 0.09 + 0.001, 0.0, 0.0)
    sim.ball.owner = rb
    sim.ball.pos = rb.seat_world()
    ctl = sim.controls[rb.id]
    ctl.kick(1.0)
    ph = sim.physics
    t = sim.time
    ph.step(sim.dt, t, [rb], sim.ball)          # удар
    assert sim.ball.owner is None
    reached_far, back_min = False, 0.0
    for _ in range(int(20 / sim.dt)):           # дальше — только мяч (без судьи и робота)
        t += sim.dt
        ph.step(sim.dt, t, [], sim.ball)
        if sim.ball.pos.x > F.GOAL_BACK_X - 0.03:
            reached_far = True
        if reached_far:
            back_min = min(back_min, sim.ball.pos.x)
    assert reached_far, "мяч должен долететь до чужих ворот"
    assert back_min - sim.ball.radius > -F.GOAL_BACK_X, "мяч вернулся к задней стенке своих ворот"


# --------------------------------------------------------------- правила
def test_wall_touch_is_out_with_one_minute_penalty():
    sim = make_sim(out_action="penalty", mode="free")
    rb = place(sim, "t1_1", 0.3, F.WY - 0.09 - 0.0005)
    sim.advance(0.01)
    assert not rb.on_field
    left = rb.penalty_until - sim.time
    assert 59 < left <= 60
    sim.advance(left + 0.05)
    assert rb.on_field
    assert any(rb.pos.dist(s) < 1e-6 for s in F.NEUTRAL_SPOTS)
    # лицом к своим воротам (team1 — слева)
    own = Vec2(-F.GOAL_MOUTH_X, 0)
    assert abs(math.degrees(rb.heading) - rb.pos.angle_to(own)) < 1


def test_out_action_none_keeps_robot():
    sim = make_sim(out_action="none", out_of_bounds=True)
    rb = place(sim, "t1_1", 0.3, F.WY - 0.09)
    sim.advance(0.01)
    assert rb.on_field and rb.out_flag


def test_out_can_be_disabled():
    sim = make_sim(out_of_bounds=False)
    rb = place(sim, "t1_1", 0.3, F.WY - 0.09)
    sim.advance(0.01)
    assert rb.on_field and not rb.out_flag


def test_fully_inside_penalty_area_is_out_partially_is_not():
    sim = make_sim(out_action="reposition")
    rb = place(sim, "t1_1", F.HX - 0.25 + 0.05, 0.0)   # частично
    sim.advance(0.01)
    assert rb.pos.x == pytest.approx(F.HX - 0.20, abs=1e-3)
    rb = place(sim, "t1_1", F.HX - 0.12, 0.0)          # полностью (R=9: 0.03..0.21 от линии)
    sim.advance(0.01)
    assert any(rb.pos.dist(s) < 1e-6 for s in F.NEUTRAL_SPOTS)


def test_multiple_defense():
    sim = make_sim(mode="free")
    a = place(sim, "t1_1", -F.HX + 0.26, 0.2)
    b = place(sim, "t1_2", -F.HX + 0.26, -0.2)
    sim.ball.pos = Vec2(-0.5, 0.2)
    sim.advance(0.01)
    assert any(b.pos.dist(s) < 1e-6 for s in F.NEUTRAL_SPOTS)   # b дальше от мяча
    assert a.pos.x == pytest.approx(-F.HX + 0.26, abs=1e-3)


def test_goal_and_kickoff_by_conceding_team():
    sim = make_sim(mode="free")
    sim.ball.pos = Vec2(F.GOAL_MOUTH_X + 0.01, 0.0)
    sim.ball.vel = Vec2(1.0, 0.0)
    sim.advance(0.1)
    assert sim.referee.score[1] == 1          # team1 атакует вправо (синие ворота)
    assert sim.referee.last_kickoff_team == 2
    assert sim.ball.pos.length() < 1e-9
    for rb in sim.robots:
        if rb.on_field:
            assert rb.pos.x * sim.own_side(rb.team) >= rb.radius - 1e-9


def test_team_frame_for_team2():
    sim = make_sim(mode="free")
    rb = sim.robot_by_id["t2_1"]
    wv = sim.world_view(rb)
    assert wv.field.own_goal.x < 0
    # мировая позиция робота team2 справа → в системе команды слева
    assert rb.pos.x > 0 and wv.me.pos.x < 0
    ctl = sim.controls[rb.id]
    ctl.move_to(-0.2, 0.0)       # к своим воротам (в мире — вправо)
    assert rb.cmd.target.x == pytest.approx(0.2)


def test_speed_limit():
    sim = make_sim()
    rb = place(sim, "t1_1", -0.8, 0.5)
    ctl = sim.controls[rb.id]
    peak = 0.0
    for _ in range(90):
        ctl.move_to(0.8, 0.5, speed=10.0)
        sim.advance(sim.control_dt)
        peak = max(peak, rb.vel.length())
    assert rb.max_speed * 0.95 <= peak <= rb.max_speed + 1e-9
    assert rb.pos.dist(Vec2(0.8, 0.5)) < 0.01    # доехал и остановился


def test_full_match_halves_and_side_swap():
    cfg = load_config(None)
    cfg["rules"]["half_duration_s"] = 20
    cfg["sim"]["seed"] = 5
    sim = Simulation(cfg, "match")
    assert sim.own_side(1) == -1
    sim.advance(21)
    assert sim.referee.half == 2 and sim.own_side(1) == 1
    sim.advance(21)
    assert sim.state == FINISHED


def test_final_score_trim():
    sim = make_sim(mode="match", robot="t1_1")
    sim.referee.score = {1: 14, 2: 1}
    assert sim.referee.final_score() == (11, 1)
