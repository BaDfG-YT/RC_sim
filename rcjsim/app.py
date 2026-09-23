"""Окно симулятора: главный цикл, мышь и горячие клавиши (настраиваются в [hotkeys])."""
from __future__ import annotations

import math
import time
from typing import Optional

import pygame

from .game import ROBOT_IDS, Simulation
from .render import Renderer
from .vec import Vec2

SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

ACTION_HELP = {
    "teleport_ball": "перенести мяч под курсор",
    "teleport_selected": "перенести выбранного робота под курсор",
    "teleport_t1_1": "перенести t1_1 под курсор",
    "teleport_t1_2": "перенести t1_2 под курсор",
    "teleport_t2_1": "перенести t2_1 под курсор",
    "teleport_t2_2": "перенести t2_2 под курсор",
    "rotate_left": "повернуть выбранного +15° (Shift: 5°)",
    "rotate_right": "повернуть выбранного −15° (Shift: 5°)",
    "select_next": "выбрать следующего робота",
    "stop_ball": "остановить мяч",
    "pause": "пауза",
    "step": "шаг на такт (на паузе)",
    "speed_up": "быстрее",
    "speed_down": "медленнее",
    "kickoff": "нейтральное начало с центра",
    "toggle_overlay": "оверлей целей/векторов",
    "toggle_help": "эта справка",
    "reload_strategies": "перезагрузить стратегии (код)",
    "debug_prev": "отладка: предыдущий робот",
    "debug_next": "отладка: следующий робот",
    "quit": "выход",
}

MOD_NAMES = {"shift": pygame.KMOD_SHIFT, "ctrl": pygame.KMOD_CTRL, "alt": pygame.KMOD_ALT}


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'b', 'f5', 'space', 'ctrl+t' → (keycode, модификаторы)."""
    parts = [p.strip().lower() for p in str(spec).split("+") if p.strip()]
    mods = 0
    key = parts[-1] if parts else ""
    for p in parts[:-1]:
        if p not in MOD_NAMES:
            raise ValueError(f"Неизвестный модификатор '{p}' в '{spec}'")
        mods |= MOD_NAMES[p]
    try:
        code = pygame.key.key_code(key)
    except ValueError as e:
        raise ValueError(f"Неизвестная клавиша '{key}' в [hotkeys]: {spec}") from e
    return code, mods


class App:
    def __init__(self, cfg: dict, mode: str, debug_robot: Optional[str] = None):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption(f"RCJ Soccer Vision Simulator — {mode}")
        w, h = int(cfg["window"]["width"]), int(cfg["window"]["height"])
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.cfg = cfg
        self.mode = mode
        self.sim = Simulation(cfg, mode, debug_robot)
        self.renderer = Renderer(self.screen, cfg)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.speed = float(cfg["sim"]["speed"])
        self.selected: Optional[str] = self.sim.debug_robot_id if mode == "debug" else "t1_1"
        self.throw_from: Optional[Vec2] = None
        self.bindings: list[tuple[int, int, str]] = []
        for action, spec in cfg["hotkeys"].items():
            if not spec:
                continue
            code, mods = parse_hotkey(spec)
            self.bindings.append((code, mods, action))
        # сначала более специфичные (с модификаторами)
        self.bindings.sort(key=lambda b: -bin(b[1]).count("1"))
        self.help_lines = ["Клавиши (config.toml → [hotkeys]):"] + [
            f"  {spec:>7}  {ACTION_HELP.get(a, a)}" for a, spec in cfg["hotkeys"].items() if spec
        ] + ["Мышь:", "  ЛКМ — выбрать робота", "  ЛКМ-перетаскивание — двигать робота/мяч",
             "  ПКМ-протяжка от мяча — бросить мяч", "  колесо — поворот выбранного"]
        self.drag: Optional[str] = None

    # ------------------------------------------------------------- ввод
    def mouse_world(self) -> Vec2:
        mx, my = pygame.mouse.get_pos()
        return self.renderer.s2w(mx, my)

    def robot_at(self, p: Vec2) -> Optional[str]:
        for rb in self.sim.robots:
            if rb.on_field and rb.pos.dist(p) <= rb.radius:
                return rb.id
        return None

    def action(self, name: str, shift: bool):
        sim = self.sim
        mp = self.mouse_world()
        if name == "quit":
            raise SystemExit
        if name == "pause":
            self.paused = not self.paused
        elif name == "step":
            if self.paused:
                sim.advance(sim.control_dt)
        elif name == "speed_up":
            self.speed = next((s for s in SPEEDS if s > self.speed), SPEEDS[-1])
        elif name == "speed_down":
            self.speed = next((s for s in reversed(SPEEDS) if s < self.speed), SPEEDS[0])
        elif name == "teleport_ball":
            sim.teleport_ball(mp)
        elif name == "stop_ball":
            sim.stop_ball()
        elif name == "teleport_selected":
            if self.selected:
                sim.teleport_robot(self.selected, mp)
        elif name.startswith("teleport_t"):
            rid = name[len("teleport_"):]
            if rid in sim.robot_by_id:
                sim.teleport_robot(rid, mp)
                self.selected = rid
        elif name in ("rotate_left", "rotate_right"):
            if self.selected:
                d = math.radians(5 if shift else 15) * (1 if name == "rotate_left" else -1)
                sim.rotate_robot(self.selected, d)
        elif name == "select_next":
            ids = [r.id for r in sim.robots if not sim.is_hidden(r)]
            if ids:
                i = ids.index(self.selected) if self.selected in ids else -1
                self.selected = ids[(i + 1) % len(ids)]
        elif name == "kickoff":
            sim.restart_kickoff()
        elif name == "toggle_overlay":
            self.renderer.show_overlay = not self.renderer.show_overlay
        elif name == "toggle_help":
            self.renderer.show_help = not self.renderer.show_help
        elif name == "reload_strategies":
            sim.load_strategies(reload=True)
        elif name in ("debug_prev", "debug_next") and self.mode == "debug":
            i = ROBOT_IDS.index(sim.debug_robot_id)
            i = (i + (1 if name == "debug_next" else -1)) % len(ROBOT_IDS)
            self.sim = Simulation(self.cfg, "debug", ROBOT_IDS[i])
            self.selected = ROBOT_IDS[i]

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise SystemExit
            if ev.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
                self.renderer.resize(self.screen)
            elif ev.type == pygame.KEYDOWN:
                mods = ev.mod & (pygame.KMOD_SHIFT | pygame.KMOD_CTRL | pygame.KMOD_ALT)
                for code, need, act in self.bindings:
                    if code == ev.key and (mods & need) == need:
                        self.action(act, bool(ev.mod & pygame.KMOD_SHIFT))
                        break
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                p = self.renderer.s2w(*ev.pos)
                if ev.button == 1:
                    rid = self.robot_at(p)
                    if rid:
                        self.selected = rid
                        self.drag = rid
                    elif self.sim.ball.pos.dist(p) < 0.04:
                        self.drag = "ball"
                elif ev.button == 3 and self.sim.ball.pos.dist(p) < 0.06:
                    self.throw_from = self.sim.ball.pos.copy()
            elif ev.type == pygame.MOUSEMOTION and self.drag:
                p = self.renderer.s2w(*ev.pos)
                if self.drag == "ball":
                    self.sim.teleport_ball(p)
                else:
                    rb = self.sim.robot_by_id[self.drag]
                    rb.halt()
                    rb.pos = self.sim._fit_inside(p, rb.radius + 0.003)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1 and self.drag:
                    if self.drag != "ball":
                        self.sim.teleport_robot(self.drag, self.renderer.s2w(*ev.pos))
                    self.drag = None
                elif ev.button == 3 and self.throw_from is not None:
                    p = self.renderer.s2w(*ev.pos)
                    v = (p - self.sim.ball.pos) * 3.0
                    self.sim.throw_ball(v.clamped(5.0))
                    self.sim.log(f"Мяч брошен: {v.clamped(5.0).length():.2f} м/с")
                    self.throw_from = None
            elif ev.type == pygame.MOUSEWHEEL and self.selected:
                self.sim.rotate_robot(self.selected, math.radians(5) * ev.y)

    # ------------------------------------------------------------ цикл
    def run(self):
        fps = int(self.cfg["window"]["fps"])
        acc = 0.0
        last = time.perf_counter()
        try:
            while True:
                self.handle_events()
                now = time.perf_counter()
                real_dt = min(now - last, 0.1)
                last = now
                if not self.paused and self.drag is None:
                    acc += real_dt * self.speed
                    budget = time.perf_counter() + 0.8 / fps
                    while acc >= self.sim.dt:
                        self.sim.step()
                        acc -= self.sim.dt
                        if time.perf_counter() > budget:
                            acc = 0.0      # не успеваем — пропускаем (замедление)
                            break
                ui = {
                    "selected": self.selected, "paused": self.paused, "speed": self.speed,
                    "mouse_world": self.mouse_world(), "mouse_px": pygame.mouse.get_pos(),
                    "throw_from": self.throw_from, "help_lines": self.help_lines,
                }
                self.renderer.draw(self.sim, ui)
                pygame.display.flip()
                self.clock.tick(fps)
        except SystemExit:
            pass
        finally:
            pygame.quit()
