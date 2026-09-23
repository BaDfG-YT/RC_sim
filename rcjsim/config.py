"""Загрузка конфигурации (TOML) с дефолтами и проверкой ограничений правил."""
from __future__ import annotations

import copy
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore
from pathlib import Path
from typing import Any

# Ограничения из правил RCJ Soccer 2026 (Soccer Vision), п. 6.2.A
RULE_MAX_ROBOT_DIAMETER = 0.18       # робот должен помещаться в цилиндр 18 см
# мяч не может заходить в выпуклую оболочку глубже 1.5 см
RULE_BALL_CAPTURE_DEPTH = 0.015
ROBOT_RADIUS_MIN = 0.07              # по ТЗ: радиус 7..9 см
ROBOT_RADIUS_MAX = RULE_MAX_ROBOT_DIAMETER / 2
NICHE_WIDTH_MIN = 0.01

DEFAULTS: dict[str, Any] = {
    "sim": {
        "physics_dt": 0.002,         # шаг физики, с
        # частота вызова стратегий (цикл робота), Гц
        "control_hz": 60,
        "speed": 1.0,                # множитель скорости симуляции
        "seed": 0,                   # 0 — случайный
        "position_noise": 0.0,       # СКО шума глобального позиционирования, м
        "heading_noise_deg": 0.0,    # СКО шума курса, градусы
        "team_frame": True,          # стратегия видит поле так, будто атакует в +X
    },
    "window": {
        "width": 1400,
        "height": 860,
        "fps": 60,
        "panel_width": 380,
    },
    "physics": {
        "ball_radius": 0.021,        # мяч для гольфа 42 мм
        "ball_rolling_decel": 0.35,  # замедление качения по ковру, м/с²
        "ball_wall_restitution": 0.55,
        "ball_robot_restitution": 0.35,
        "robot_robot_restitution": 0.15,
        "capture_max_speed": 0.5,    # мяч захватывается нишей, если отн. скорость <= 0.5 м/с
        # доля (ширины ниши/2 + r мяча) — боковой допуск захвата
        "capture_lateral_tol": 0.5,
        # м/с² — бок. ускорение, при котором мяч теряется
        "dribbler_max_lateral_acc": 3.0,
        # м/с² — торможение, при котором мяч «уезжает» вперёд
        "dribbler_max_brake_acc": 3.0,
        # с мячом ограничивать ускорение, чтобы не терять
        "auto_limit_accel_with_ball": True,
        "kick_cooldown_s": 0.5,
        "recapture_block_s": 0.25,   # после удара нельзя снова захватить мяч
        "kick_contact_tol": 0.006,   # удар без захвата: мяч в нише с зазором <= 6 мм
        "wedge": True,               # клин у стен
    },
    "rules": {
        "half_duration_s": 600,      # 2 тайма по 10 минут (п. 2.1.1)
        "halves": 2,
        "halftime_break_s": 0,       # по правилам 300 с; в симуляторе по умолчанию 0
        "kickoff_delay_s": 1.0,      # роботы стоят после расстановки, затем «свисток»
        "coin_toss": "random",       # random | team1 | team2 — кто начинает первый тайм
        "out_of_bounds": True,       # п. 2.8
        "out_on_wall_touch": True,
        "out_on_full_penalty_area": True,
        "out_on_leaving_field": False,  # НЕ из правил 2026: аут при полном выходе за линию
        # penalty (удаление) | reposition (сразу на нейтр. точку) | none
        "out_action": "penalty",
        "penalty_duration_s": 60,
        "return_at_kickoff": True,   # п. 2.8.1 — удалённые возвращаются при начале с центра
        "multiple_defense": True,    # п. 2.6.2
        "pushing": True,             # п. 2.6.3
        "lack_of_progress": True,    # п. 2.7
        "lack_of_progress_s": 10.0,
        "lack_of_progress_dist": 0.05,
        "max_goal_difference": 10,   # п. 2.1.3 — итоговая разница урезается до 10
    },
    "modes": {
        "match": {"rules": {}},
        "free": {"rules": {"out_action": "reposition"}},
        "debug": {
            "robot": "t1_1",
            "others": "hidden",      # hidden | frozen | active
            "rules": {"out_action": "none", "lack_of_progress": False,
                      "multiple_defense": False, "pushing": False},
        },
    },
    "team_defaults": {
        "radius": 0.09,
        "niche_width": 0.07,
        # м/с (в правилах 2026 лимита скорости нет)
        "max_speed": 4.0,
        "max_accel": 8.0,            # м/с²
        "max_omega_deg": 720.0,      # град/с
        "max_alpha_deg": 5000.0,     # град/с²
        "kick_speed": 0.0,           # 0 — авто по процедуре Appendix A
        "color": [60, 110, 220],
    },
    "team1": {
        "name": "Team 1",
        "color": [255, 74, 54],
        "robots": [
            {"name": "Striker", "strategy": "strategies.basic:Striker"},
            {"name": "Goalie", "strategy": "strategies.basic:Goalie"},
        ],
    },
    "team2": {
        "name": "Team 2",
        "color": [253, 130, 255],
        "robots": [
            {"name": "Striker", "strategy": "strategies.basic:Striker"},
            {"name": "Goalie", "strategy": "strategies.basic:Goalie"},
        ],
    },
    "hotkeys": {
        "mode_match": "f1",
        "mode_free": "f2",
        "mode_debug": "f3",
        "cycle_mode": "m",
        "teleport_ball": "b",
        "teleport_ball": "b",
        "teleport_selected": "t",
        "teleport_t1_1": "1",
        "teleport_t1_2": "2",
        "teleport_t2_1": "3",
        "teleport_t2_2": "4",
        "rotate_left": "q",
        "rotate_right": "e",
        "select_next": "tab",
        "stop_ball": "x",
        "pause": "space",
        "step": "n",
        "speed_up": "=",
        "speed_down": "-",
        "kickoff": "k",
        "toggle_overlay": "o",
        "toggle_help": "h",
        "reload_strategies": "f5",
        "debug_prev": "[",
        "debug_next": "]",
        "quit": "escape",
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None) -> dict:
    user: dict = {}
    if path:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                user = tomllib.load(f)
        else:
            raise FileNotFoundError(f"Конфиг не найден: {p}")
    return deep_merge(DEFAULTS, user)


def rules_for_mode(cfg: dict, mode: str) -> dict:
    """Базовые правила + переопределения режима."""
    over = cfg.get("modes", {}).get(mode, {}).get("rules", {})
    return deep_merge(cfg["rules"], over)


def team_cfg(cfg: dict, team_key: str) -> dict:
    """Параметры команды с дефолтами."""
    return deep_merge(cfg["team_defaults"], cfg[team_key])
