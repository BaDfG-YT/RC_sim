"""Отрисовка поля, роботов, мяча и панели состояния (pygame)."""
from __future__ import annotations

import math
from typing import Optional

import pygame

from . import field as F
from .game import Simulation
from .referee import FINISHED, HALFTIME, KICKOFF
from .vec import Vec2

CARPET = (22, 130, 58)
CARPET_WEDGE = (18, 112, 50)
WHITE = (245, 245, 245)
BLACK = (12, 12, 12)
YELLOW = (240, 205, 30)
BLUE = (40, 120, 235)
BG = (34, 36, 40)
PANEL_BG = (24, 26, 30)
TEXT = (225, 228, 232)
DIM = (140, 146, 155)
BALL = (255, 120, 20)
RED = (240, 70, 70)


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("segoeui,dejavusans,arial,liberationsans,freesans", size, bold=bold)


def fmt_time(t: Optional[float]) -> str:
    if t is None:
        return "∞"
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class Renderer:
    MARGIN = 0.05  # м вокруг стен

    def __init__(self, screen: pygame.Surface, cfg: dict):
        self.screen = screen
        self.cfg = cfg
        self.panel_w = int(cfg["window"]["panel_width"])
        self.f_small = _font(14)
        self.f_norm = _font(16)
        self.f_big = _font(30, bold=True)
        self.f_mark = _font(11, bold=True)
        self.show_overlay = True
        self.show_help = False
        self._field_cache: Optional[pygame.Surface] = None
        self.resize(screen)

    # ------------------------------------------------------ преобразования
    def resize(self, screen: pygame.Surface):
        self.screen = screen
        w, h = screen.get_size()
        fw = max(200, w - self.panel_w)
        tl = F.TOTAL_L + 2 * (F.WALL_T + self.MARGIN)
        tw = F.TOTAL_W + 2 * (F.WALL_T + self.MARGIN)
        self.ppm = min(fw / tl, h / tw)
        self.cx = fw / 2
        self.cy = h / 2
        self.field_w = fw
        self._field_cache = None
        self.f_mark = _font(max(9, int(0.03 * self.ppm)), bold=True)

    def w2s(self, p: Vec2) -> tuple[int, int]:
        return int(round(self.cx + p.x * self.ppm)), int(round(self.cy - p.y * self.ppm))

    def s2w(self, sx: float, sy: float) -> Vec2:
        return Vec2((sx - self.cx) / self.ppm, (self.cy - sy) / self.ppm)

    def m(self, meters: float) -> int:
        return max(1, int(round(meters * self.ppm)))

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> pygame.Rect:
        a = self.w2s(Vec2(x0, y1))
        b = self.w2s(Vec2(x1, y0))
        return pygame.Rect(a[0], a[1], max(1, b[0] - a[0]), max(1, b[1] - a[1]))

    # ------------------------------------------------------------- поле
    def _build_field(self) -> pygame.Surface:
        surf = pygame.Surface(self.screen.get_size())
        surf.fill(BG)
        # стены
        t = F.WALL_T
        surf.fill(BLACK, self.rect(-F.WX - t, -F.WY - t, F.WX + t, F.WY + t))
        # ковёр
        surf.fill(CARPET, self.rect(-F.WX, -F.WY, F.WX, F.WY))
        # клин (затемнённая полоса 10 см у стен, в воротах клина нет)
        wb = F.WEDGE_BASE
        surf.fill(CARPET_WEDGE, self.rect(-F.WX, F.WY - wb, F.WX, F.WY))
        surf.fill(CARPET_WEDGE, self.rect(-F.WX, -F.WY, F.WX, -F.WY + wb))
        gy = F.GOAL_W / 2 + F.GOAL_T
        for s in (-1, 1):
            x0, x1 = (F.WX - wb, F.WX) if s > 0 else (-F.WX, -F.WX + wb)
            surf.fill(CARPET_WEDGE, self.rect(x0, gy, x1, F.WY))
            surf.fill(CARPET_WEDGE, self.rect(x0, -F.WY, x1, -gy))

        # штрафные: белый контур (линия — часть зоны), внутри ковёр
        for side in (-1, 1):
            outer = [self.w2s(p) for p in F.penalty_outline(side, 0.0)]
            inner = [self.w2s(p) for p in F.penalty_outline(side, F.LINE_W)]
            pygame.draw.polygon(surf, WHITE, outer)
            pygame.draw.polygon(surf, CARPET, inner)

        # граница поля: белая линия 2 см, внешний край = 219 x 158 см
        L = F.LINE_W
        surf.fill(WHITE, self.rect(-F.HX, F.HY - L, F.HX, F.HY))
        surf.fill(WHITE, self.rect(-F.HX, -F.HY, F.HX, -F.HY + L))
        surf.fill(WHITE, self.rect(-F.HX, -F.HY, -F.HX + L, F.HY))
        surf.fill(WHITE, self.rect(F.HX - L, -F.HY, F.HX, F.HY))

        # центральный круг и нейтральные точки — тонкий чёрный маркер
        pygame.draw.circle(surf, BLACK, self.w2s(Vec2()), self.m(F.CENTER_CIRCLE_R), max(1, self.m(0.003)))
        for p in F.NEUTRAL_SPOTS:
            pygame.draw.circle(surf, BLACK, self.w2s(p), max(2, self.m(F.NEUTRAL_SPOT_D / 2)))

        # ворота: жёлтые слева, синие справа (внутренние стенки окрашены)
        for side, col in ((-1, YELLOW), (1, BLUE)):
            gx0, gx1 = sorted((side * F.GOAL_MOUTH_X, side * F.GOAL_BACK_X))
            tint = pygame.Surface(self.rect(gx0, -F.GOAL_W / 2, gx1, F.GOAL_W / 2).size, pygame.SRCALPHA)
            tint.fill((*col, 70))
            surf.blit(tint, self.rect(gx0, -F.GOAL_W / 2, gx1, F.GOAL_W / 2))
        lim = F.GOAL_BACK_X + F.GOAL_T
        for b in F.GOAL_BOXES:
            col = YELLOW if b.x0 < 0 else BLUE
            surf.fill(BLACK, self.rect(max(b.x0, -F.WX), b.y0, min(b.x1, F.WX), b.y1))
            # окрашенная часть конструкции ворот
            cx0, cx1 = max(b.x0, -lim), min(b.x1, lim)
            if cx1 > cx0:
                surf.fill(col, self.rect(cx0, b.y0, cx1, b.y1))
        return surf

    # ------------------------------------------------------------ роботы
    def robot_poly(self, rb) -> list[tuple[int, int]]:
        R = rb.radius
        half = rb.niche_width / 2
        a0 = math.atan2(half, rb.corner_x)
        pts = []
        n = 36
        for i in range(n + 1):
            a = a0 + (2 * math.pi - 2 * a0) * i / n
            pts.append(Vec2(R * math.cos(a), R * math.sin(a)))
        xb = R - rb.niche_depth
        pts += [Vec2(xb, -half), Vec2(xb, half)]
        return [self.w2s(rb.to_world(p)) for p in pts]

    def draw_robot(self, rb, selected: bool, sim: Simulation):
        poly = self.robot_poly(rb)
        col = rb.color
        if rb.frozen:
            col = tuple(int(c * 0.45 + 60) for c in col)
        pygame.draw.polygon(self.screen, col, poly)
        pygame.draw.polygon(self.screen, BLACK, poly, 2)
        c = self.w2s(rb.pos)
        # верхний маркер: белый круг 4 см с номером (п. 3.5)
        pygame.draw.circle(self.screen, WHITE, c, self.m(0.02))
        label = self.f_mark.render(str(rb.index), True, BLACK)
        self.screen.blit(label, label.get_rect(center=c))
        if rb.out_flag:
            pygame.draw.circle(self.screen, RED, c, self.m(rb.radius + 0.012), 3)
        if selected:
            pygame.draw.circle(self.screen, (255, 235, 90), c, self.m(rb.radius + 0.02), 2)
        if rb.error:
            pygame.draw.circle(self.screen, (255, 0, 255), c, self.m(rb.radius + 0.006), 2)

    def draw_overlay(self, rb, full: bool):
        c = self.w2s(rb.pos)
        cmd = rb.cmd
        if cmd.mode == "target":
            t = self.w2s(cmd.target)
            pygame.draw.line(self.screen, (255, 255, 255), c, t, 1)
            s = 5
            pygame.draw.line(self.screen, (255, 255, 255), (t[0] - s, t[1] - s), (t[0] + s, t[1] + s), 2)
            pygame.draw.line(self.screen, (255, 255, 255), (t[0] - s, t[1] + s), (t[0] + s, t[1] - s), 2)
        if cmd.heading is not None:
            e = self.w2s(rb.pos + Vec2.from_angle(cmd.heading, rb.radius + 0.07))
            pygame.draw.line(self.screen, (255, 200, 0), c, e, 1)
        if rb.vel.length() > 0.01:
            e = self.w2s(rb.pos + rb.vel * 0.25)
            pygame.draw.line(self.screen, (0, 230, 255), c, e, 2)
        for mk in rb.debug_marks:
            if mk[0] == "pt":
                _, p, col, lab = mk
                sp = self.w2s(p)
                pygame.draw.circle(self.screen, col, sp, 4, 1)
                if lab and full:
                    self.screen.blit(self.f_small.render(lab, True, col), (sp[0] + 6, sp[1] - 8))
            elif mk[0] == "ln":
                _, a, b, col = mk
                pygame.draw.line(self.screen, col, self.w2s(a), self.w2s(b), 1)
        if full:
            # зона захвата ниши
            pygame.draw.circle(self.screen, (255, 255, 255), self.w2s(rb.seat_world()),
                               self.m(0.021), 1)

    def draw_ball(self, sim: Simulation):
        b = sim.ball
        c = self.w2s(b.pos)
        pygame.draw.circle(self.screen, BALL, c, max(3, self.m(b.radius)))
        pygame.draw.circle(self.screen, (120, 50, 0), c, max(3, self.m(b.radius)), 1)
        if b.vel.length() > 0.02 and self.show_overlay:
            pygame.draw.line(self.screen, BALL, c, self.w2s(b.pos + b.vel * 0.25), 2)

    # ------------------------------------------------------------- кадр
    def draw(self, sim: Simulation, ui: dict):
        if self._field_cache is None:
            self._field_cache = self._build_field()
        self.screen.blit(self._field_cache, (0, 0))

        sel = ui.get("selected")
        dbg = sim.debug_robot_id if sim.mode == "debug" else None
        for rb in sim.robots:
            if rb.on_field:
                self.draw_robot(rb, rb.id == sel, sim)
        self.draw_ball(sim)
        for rb in sim.robots:
            if not rb.active:
                continue
            if rb.id == dbg:
                self.draw_overlay(rb, True)
            elif self.show_overlay:
                self.draw_overlay(rb, rb.id == sel)

        # линия броска мяча (ПКМ)
        if ui.get("throw_from") is not None:
            pygame.draw.line(self.screen, (255, 255, 255), self.w2s(sim.ball.pos), ui["mouse_px"], 2)

        self._draw_banner(sim)
        self._draw_panel(sim, ui)

    def _draw_banner(self, sim: Simulation):
        st = sim.state
        text = None
        if st == KICKOFF:
            text = "Начало с центра"
        elif st == HALFTIME:
            text = f"Перерыв {fmt_time(sim.referee.timer)}"
        elif st == FINISHED:
            a, b = sim.referee.final_score()
            text = f"Матч окончен  {a}:{b}"
        if text:
            s = self.f_big.render(text, True, WHITE)
            r = s.get_rect(center=(self.field_w // 2, 30))
            bg = r.inflate(24, 10)
            pygame.draw.rect(self.screen, (0, 0, 0), bg, border_radius=8)
            self.screen.blit(s, r)

    # ------------------------------------------------------------ панель
    def _draw_panel(self, sim: Simulation, ui: dict):
        W, H = self.screen.get_size()
        x0 = W - self.panel_w
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, self.panel_w, H))
        x = x0 + 14
        y = 10

        max_w = self.panel_w - 24

        def line(txt, col=TEXT, font=None, dy=None):
            nonlocal y
            f = font or self.f_norm
            if f.size(txt)[0] > max_w:
                while txt and f.size(txt + "…")[0] > max_w:
                    txt = txt[:-1]
                txt += "…"
            self.screen.blit(f.render(txt, True, col), (x, y))
            y += dy if dy is not None else f.get_linesize()

        mode_name = {"match": "МАТЧ (правила)", "free": "СВОБОДНАЯ ИГРА",
                     "debug": f"ОТЛАДКА: {sim.debug_robot_id}"}[sim.mode]
        line(mode_name, (255, 220, 120), self.f_norm)
        ref = sim.referee
        tl = fmt_time(ref.half_time_left) if ref.half_time_left is not None else "∞"
        line(f"Тайм {ref.half}   Время {tl}   Игр.время {fmt_time(sim.time)}", DIM, self.f_small)
        sp = ui.get("speed", 1.0)
        line(f"Состояние: {sim.state}   Скорость x{sp:g}{'   ПАУЗА' if ui.get('paused') else ''}",
             DIM, self.f_small)
        y += 4
        # счёт
        n1, n2 = sim.team_name(1), sim.team_name(2)
        s1 = self.f_big.render(str(ref.score[1]), True, sim.team_cfgs[1]["color"])
        s2 = self.f_big.render(str(ref.score[2]), True, sim.team_cfgs[2]["color"])
        self.screen.blit(s1, (x, y))
        self.screen.blit(self.f_big.render(":", True, TEXT), (x + 40, y))
        self.screen.blit(s2, (x + 60, y))
        g1 = "жёлт." if sim.own_side(1) < 0 else "син."
        g2 = "жёлт." if sim.own_side(2) < 0 else "син."
        self.screen.blit(self.f_small.render(f"{n1} (свои ворота: {g1})", True,
                                             sim.team_cfgs[1]["color"]), (x + 110, y + 2))
        self.screen.blit(self.f_small.render(f"{n2} (свои ворота: {g2})", True,
                                             sim.team_cfgs[2]["color"]), (x + 110, y + 20))
        y += 46

        # роботы
        for rb in sim.robots:
            if sim.is_hidden(rb):
                continue
            st = "на поле"
            if rb.penalty_until is not None:
                st = f"штраф {rb.penalty_until - sim.time:4.0f}с"
            elif rb.frozen:
                st = "заморожен"
            ball = " ●" if sim.ball.owner is rb else ""
            col = rb.color if rb.on_field else DIM
            txt = f"{rb.id} {rb.name[:9]:9s} {st}{ball}"
            if rb.error:
                txt += "  ОШИБКА"
            line(txt, col, self.f_small)
        y += 4

        # выбранный робот
        sel = ui.get("selected")
        rb = sim.robot_by_id.get(sel) if sel else None
        if rb is not None:
            fr = sim.frames[rb.team]
            tp = fr.p_in(rb.pos)
            line(f"▶ {rb.id} «{rb.name}»", (255, 235, 90), self.f_norm)
            line(f"мир: ({rb.pos.x:+.3f}, {rb.pos.y:+.3f})  курс {math.degrees(rb.heading):+.0f}°",
                 TEXT, self.f_small)
            line(f"команда: ({tp.x:+.3f}, {tp.y:+.3f})  курс {fr.h_in(rb.heading):+.0f}°",
                 TEXT, self.f_small)
            line(f"v={rb.vel.length():.2f}/{rb.max_speed:.2f} м/с  ω={math.degrees(rb.omega):+.0f}°/с",
                 TEXT, self.f_small)
            line(f"R={rb.radius*100:.1f} см  ниша {rb.niche_width*100:.2f} см "
                 f"(макс {rb.niche_width_max*100:.2f})  удар {rb.kick_speed:.2f} м/с", DIM, self.f_small)
            line(f"стратегия: {rb.strategy_spec}", DIM, self.f_small)
            if rb.error:
                line(rb.error[:52], (255, 120, 255), self.f_small)
            for t in rb.debug_text[:6]:
                line("  " + t[:50], (170, 230, 170), self.f_small)
        y += 4

        mp = ui.get("mouse_world")
        if mp is not None:
            line(f"курсор: ({mp.x:+.3f}, {mp.y:+.3f}) м", DIM, self.f_small)
        b = sim.ball
        line(f"мяч: ({b.pos.x:+.3f}, {b.pos.y:+.3f})  |v|={b.vel.length():.2f} м/с"
             f"{'  в нише ' + b.owner.id if b.owner else ''}", DIM, self.f_small)
        y += 6

        if self.show_help:
            for t in ui.get("help_lines", []):
                line(t, TEXT, self.f_small)
            return
        line("События  (H — справка по клавишам)", (255, 220, 120), self.f_small)
        remain = max(0, (H - y - 8) // self.f_small.get_linesize())
        for t, msg in list(sim.events)[-remain:]:
            col = RED if ("АУТ" in msg or "Ошибка" in msg) else (
                (120, 255, 140) if "ГОЛ" in msg else TEXT)
            line(f"{fmt_time(t)} {msg}", col, self.f_small)
