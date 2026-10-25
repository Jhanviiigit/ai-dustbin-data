"""
synth.py
--------
Synthetic bin-plate scenes for testing the item detector until real bin
photos exist. Real waste cut-outs (dataset D masks) are pasted onto fake
plates (flat colour + texture + lighting gradient). Every captured frame gets
its own sensor noise and a small exposure change, and items cast soft shadows.

Two kinds of output:
  - one still frame per scene (make_scene)
  - a frame sequence where items are dropped one after another, with a blurred
    "in motion" frame between settled states (make_sequence)
"""

import csv
import os
import random
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps

W, H = 640, 480
PLATES = [(150, 150, 146), (70, 72, 75), (170, 120, 60)]   # light grey, dark grey, blue (BGR)


def long_path(path):
    path = os.path.abspath(path)
    return "\\\\?\\" + path if os.name == "nt" and not path.startswith("\\\\?\\") else path


class ItemPool:
    """Cut-outs of real waste from one split of the cleaned dataset."""

    def __init__(self, clean_dir, masks_dir, split, seed=0):
        self.rng = random.Random(seed)
        self.paths = defaultdict(list)
        for r in csv.DictReader(open(os.path.join(clean_dir, "manifest.csv"), encoding="utf-8")):
            if r["status"] != "keep" or r["split"] != split:
                continue
            mask_path = os.path.join(masks_dir, os.path.splitext(r["new_path"])[0] + ".png")
            if os.path.exists(long_path(mask_path)):
                self.paths[r["label"]].append((os.path.join(clean_dir, r["new_path"]), mask_path))
        self.cache = {}

    def sizes(self):
        return {k: len(v) for k, v in self.paths.items()}

    def _load(self, img_path, mask_path):
        if img_path not in self.cache:
            img = ImageOps.exif_transpose(Image.open(long_path(img_path))).convert("RGB")
            img.thumbnail((512, 512))
            mask = Image.open(long_path(mask_path)).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size)
            a = np.asarray(mask)
            cover = (a > 127).mean()
            ys, xs = np.nonzero(a > 127)
            # Skip failed cut-outs and ones that kept most of the background
            if not (0.05 <= cover <= 0.6) or len(xs) == 0:
                self.cache[img_path] = None
            else:
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)[y0:y1, x0:x1]
                self.cache[img_path] = (rgb, a[y0:y1, x0:x1].astype(np.float32) / 255)
        return self.cache[img_path]

    def pick(self, label):
        while True:
            c = self._load(*self.rng.choice(self.paths[label]))
            if c is not None:
                return c


class SceneMaker:
    def __init__(self, pool, seed=0):
        self.pool = pool
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def plate(self):
        base = np.full((H, W, 3), self.rng.choice(PLATES), np.float32)
        texture = cv2.GaussianBlur(self.np_rng.normal(0, 6, (H, W)).astype(np.float32), (0, 0), 3)[..., None]
        gx = np.linspace(-1, 1, W)[None, :, None]
        gy = np.linspace(-1, 1, H)[:, None, None]
        lighting = 1 + 0.12 * (self.rng.uniform(-1, 1) * gx + self.rng.uniform(-1, 1) * gy)
        return base * lighting + texture

    def snap(self, scene, exposure=1.0):
        """Camera capture: exposure change + per-frame sensor noise."""
        return np.clip(scene * exposure + self.np_rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)

    @staticmethod
    def paste(scene, cutout, cx, cy, size):
        rgb, alpha = cutout
        scale = size / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]))[..., None]
        h, w = rgb.shape[:2]
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xs1 <= xs0 or ys1 <= ys0:
            return
        sub_rgb = rgb[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        sub_a = alpha[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        shadow = cv2.GaussianBlur(sub_a[..., 0], (0, 0), 6)[..., None] * 0.35
        region = scene[min(H, ys0 + 8):min(H, ys1 + 8), min(W, xs0 + 8):min(W, xs1 + 8)]
        region *= 1 - shadow[:region.shape[0], :region.shape[1]]
        scene[ys0:ys1, xs0:xs1] = scene[ys0:ys1, xs0:xs1] * (1 - sub_a) + sub_rgb * sub_a

    def place(self, n_items, touching):
        """Centres and sizes: spread apart, or the first two touching."""
        sizes = [self.rng.randint(140, 200) for _ in range(n_items)]
        if touching:
            cx, cy = self.rng.randint(220, 420), self.rng.randint(180, 300)
            gap = int((sizes[0] + sizes[1]) * 0.33)          # bounding boxes overlap -> objects meet
            return [(cx - gap // 2, cy, sizes[0]), (cx + gap // 2, cy + self.rng.randint(-20, 20), sizes[1])]
        slots = [(160, 140), (480, 140), (160, 340), (480, 340), (320, 240)]
        self.rng.shuffle(slots)
        return [(x + self.rng.randint(-25, 25), y + self.rng.randint(-25, 25), s) for (x, y), s in zip(slots, sizes)]

    def _states(self, labels, touching):
        """Plate, then the scene after each item is added (float images)."""
        plate = self.plate()
        states = [plate]
        scene = plate.copy()
        for (cx, cy, size), label in zip(self.place(len(labels), touching), labels):
            self.paste(scene, self.pool.pick(label), cx, cy, size)
            states.append(scene.copy())
        return states

    def make_scene(self, labels, touching=False):
        """(reference frame, scene frame) with all items present."""
        states = self._states(labels, touching)
        return self.snap(states[0]), self.snap(states[-1], exposure=self.rng.uniform(0.93, 1.07))

    def make_sequence(self, labels, touching=False, together=False, settle=4):
        """(reference frame, list of frames): items arrive one by one, or all at
        once if together=True. Each arrival = one blurred in-motion frame, then
        `settle` still frames. Exposure drifts slowly, as auto-exposure does."""
        states = self._states(labels, touching)
        if together:
            states = [states[0], states[-1]]
        exposure = 1.0
        frames = []
        for k, state in enumerate(states):
            if k > 0:   # in motion: halfway between the previous and new state, blurred
                moving = cv2.GaussianBlur(0.5 * states[k - 1] + 0.5 * state, (0, 0), 4)
                frames.append(self.snap(moving, exposure))
            for _ in range(settle):
                exposure = float(np.clip(exposure + self.rng.uniform(-0.01, 0.01), 0.93, 1.07))
                frames.append(self.snap(state, exposure))
        return self.snap(states[0]), frames
