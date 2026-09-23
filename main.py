"""Запуск симулятора.

    python main.py                         # матч по правилам
    python main.py --mode free             # свободная игра без времени
    python main.py --mode debug --robot t1_1
    python main.py --headless --duration 1200   # без окна, быстрый прогон тактик
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rcjsim.config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="RCJ Soccer Vision simulator")
    ap.add_argument("--mode", choices=["match", "free", "debug"], default="match")
    ap.add_argument("--robot", help="робот для режима debug: t1_1, t1_2, t2_1, t2_2")
    ap.add_argument("--config", default=os.path.join(HERE, "config.toml"))
    ap.add_argument("--headless", action="store_true", help="без окна (для тестов тактик)")
    ap.add_argument("--duration", type=float, default=None,
                    help="headless: сколько секунд симулировать (по умолчанию — весь матч)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config if os.path.exists(args.config) else None)

    if args.headless:
        from rcjsim.game import Simulation
        from rcjsim.referee import FINISHED
        sim = Simulation(cfg, args.mode, args.robot)
        limit = args.duration
        if limit is None:
            r = sim.rules
            limit = r["half_duration_s"] * r["halves"] + r["halftime_break_s"] + 5 \
                if args.mode == "match" else 120
        n = 0
        while sim.time < limit and sim.state != FINISHED:
            sim.step()
            n += 1
        for t, msg in sim.events:
            print(f"{int(t//60):02d}:{t%60:05.2f}  {msg}")
        a, b = sim.referee.final_score()
        print(f"\nИтог: {sim.team_name(1)} {a} : {b} {sim.team_name(2)}")
        return

    from rcjsim.app import App
    App(cfg, args.mode, args.robot).run()


if __name__ == "__main__":
    main()
