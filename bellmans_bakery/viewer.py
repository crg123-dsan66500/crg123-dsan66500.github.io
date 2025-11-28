from pathlib import Path
from typing import Dict

import pygame

# Import names and bake constants for light-weight rendering logic
from .env import ITEM_NAMES, ITEM_BAKE_SECONDS


class PastelViewer:
    """
    Lightweight Pygame viewer for Bellman's Bakery.

    Intent: give a fast, cute visual without building a full UI system.
    We draw:
      - Title + header: time, profit, price multiplier
      - Inventory with PNG icons and counts
      - Two ovens with progress bars and tiny item thumbnails
      - Queue as a row of larger pastel circles (first ~12)

    The viewer reads the env's public attributes via `env.unwrapped`.
    """

    def __init__(self, width: int = 900, height: int = 640):
        pygame.init()
        pygame.display.set_caption("Bellman's Bakery — Viewer")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Try Montserrat; fallback to Arial if not installed.
        mont = pygame.font.match_font("montserrat")
        if mont:
            self.font = pygame.font.Font(mont, 24)
            self.small = pygame.font.Font(mont, 18)
            self.title = pygame.font.Font(mont, 28)
        else:
            self.font = pygame.font.SysFont("arial", 24)
            self.small = pygame.font.SysFont("arial", 18)
            self.title = pygame.font.SysFont("arial", 28, bold=True)

        # Pastel palette
        self.COLORS = {
            "bg": (250, 218, 221),  # pink
            "cream": (255, 247, 230),
            "mint": (223, 245, 225),
            "lav": (233, 216, 253),
            "text": (60, 60, 60),
            "queue": (190, 170, 210),
            "queue_border": (160, 140, 190),
        }

        # Load item images; fall back to colored boxes if missing
        root = Path(__file__).resolve().parents[1]
        img_dir = root / "images" / "desserts"
        name_to_file = {
            "mini_red_velvet": "red_velvet.png",
            "raspberry_matcha_roll": "matcha_roll.png",
            "strawberry_cream_slice": "strawberry_cream.png",
            "chocolate_almond_drip_cake": "chocolate_drip.png",
            "chocolate_orange_roll": "orange_roll.png",
        }
        self.icons: Dict[str, pygame.Surface] = {}
        for name, fn in name_to_file.items():
            p = img_dir / fn
            if p.exists():
                img = pygame.image.load(str(p)).convert_alpha()
                self.icons[name] = pygame.transform.smoothscale(img, (64, 64))
            else:
                surf = pygame.Surface((64, 64))
                surf.fill(self.COLORS["cream"])
                self.icons[name] = surf

        # Precompute normalization for progress
        self.max_bake_ticks = max(
            int(ITEM_BAKE_SECONDS[i] // 10) for i in range(len(ITEM_NAMES))
        )

    def draw_header(self, env) -> None:
        s = env.unwrapped
        t = s.t
        mm = int((t * 10) // 60)
        ss = int((t * 10) % 60)
        self._rounded_rect((20, 20, 860, 46), self.COLORS["cream"], radius=14)
        title = self.title.render("Bellman's Bakery", True, self.COLORS["text"])
        self.screen.blit(title, (30, 26))
        header = self.font.render(
            f"Time {mm:02d}:{ss:02d}   Profit ${s.profit:.2f}   Mult x{s.daily_price_multiplier:.2f}",
            True,
            self.COLORS["text"],
        )
        self.screen.blit(header, (340, 28))

    def draw_inventory(self, env) -> None:
        s = env.unwrapped
        x = 20
        y = 80
        for i, name in enumerate(ITEM_NAMES):
            self._rounded_rect((x, y, 168, 88), self.COLORS["lav"], radius=14)
            self.screen.blit(self.icons[name], (x + 10, y + 12))
            count = int(s.inventory[i])
            price = float(s.prices_today[i]) if hasattr(s, "prices_today") else 0.0
            txt = self.small.render(
                f"x{count}  ${price:.2f}", True, self.COLORS["text"]
            )
            self.screen.blit(txt, (x + 82, y + 32))
            x += 176
            if (i + 1) % 5 == 0:
                x = 20
                y += 96

    def draw_ovens(self, env) -> None:
        s = env.unwrapped
        ox = 20
        oy = 260
        for oven in s.ovens:
            self._rounded_rect((ox, oy, 420, 110), self.COLORS["cream"], radius=18)
            if oven:
                max_ticks = max(load[2] for load in oven)
                frac = max(
                    0.0, min(1.0, 1.0 - (max_ticks / max(1, self.max_bake_ticks)))
                )
            else:
                frac = 0.0
            pygame.draw.rect(
                self.screen,
                self.COLORS["mint"],
                (ox + 12, oy + 72, int(396 * frac), 18),
                border_radius=10,
            )
            tx = ox + 16
            for item_idx, _size, _ticks_remaining in oven:
                name = ITEM_NAMES[item_idx]
                thumb = pygame.transform.smoothscale(self.icons[name], (36, 36))
                self.screen.blit(thumb, (tx, oy + 26))
                tx += 40
            ox += 440

    def draw_queue(self, env) -> None:
        s = env.unwrapped
        self._rounded_rect((20, 400, 860, 80), self.COLORS["cream"], radius=18)
        # Current action text
        action_txt = getattr(s, "last_action_str", "idle")
        lbl = self.small.render(
            f"Current action: {action_txt}", True, self.COLORS["text"]
        )
        self.screen.blit(lbl, (32, 410))
        x = 40
        y = 440
        for i, _cust in enumerate(s.queue[:12]):
            pygame.draw.circle(self.screen, self.COLORS["queue"], (x, y), 16)
            pygame.draw.circle(
                self.screen, self.COLORS["queue_border"], (x, y), 16, width=3
            )
            x += 40

    def render(self, env) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        self.screen.fill(self.COLORS["bg"])
        self.draw_header(env)
        self.draw_inventory(env)
        self.draw_ovens(env)
        self.draw_queue(env)
        pygame.display.flip()
        self.clock.tick(30)

    # --------- helpers ----------
    def _rounded_rect(self, rect, color, radius=8):
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)


