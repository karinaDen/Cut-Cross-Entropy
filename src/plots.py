"""Графики по results/*.json. Нужны только numpy и Pillow (matplotlib не нужен).

Запуск: python src/plots.py  -> картинки в figures/
"""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"

W, H = 1200, 750
L, R, T, B = 130, 40, 80, 110
COLORS = ["#2a6fdb", "#e4572e", "#29a36a", "#8e44ad", "#f2a541"]
GRID, AXIS, TEXT = "#e3e3e3", "#444444", "#222222"


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


F_T, F_L, F_S = font(34), font(26), font(22)


def nice_ticks(lo, hi, n=6):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.floor(lo / step) * step
    out = []
    while t <= hi + 1e-12:
        if t >= lo - 1e-12:
            out.append(round(t, 10))
        t += step
    return out


def fmt(v):
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if abs(v) >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


class Chart:
    def __init__(self, title, xlabel, ylabel, xlog=False, ylog=False):
        self.im = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.im)
        self.title, self.xl, self.yl = title, xlabel, ylabel
        self.xlog, self.ylog = xlog, ylog

    def setup(self, xmin, xmax, ymin, ymax, xticks=None, xticklabels=None):
        self.tx = (lambda v: math.log10(v)) if self.xlog else (lambda v: v)
        self.ty = (lambda v: math.log10(v)) if self.ylog else (lambda v: v)
        self.x0, self.x1 = self.tx(xmin), self.tx(xmax)
        self.y0, self.y1 = self.ty(ymin), self.ty(ymax)
        d = self.d
        if self.ylog:
            yt = [10 ** e for e in range(math.floor(self.y0), math.ceil(self.y1) + 1)]
        else:
            yt = nice_ticks(ymin, ymax)
        for v in yt:
            if not (self.y0 - 1e-9 <= self.ty(v) <= self.y1 + 1e-9):
                continue
            y = self.py(v)
            d.line([(L, y), (W - R, y)], fill=GRID, width=1)
            d.text((L - 12, y), fmt(v), font=F_S, fill=TEXT, anchor="rm")
        if xticks is None:
            xticks = [10 ** e for e in range(math.floor(self.x0), math.ceil(self.x1) + 1)] if self.xlog else nice_ticks(xmin, xmax)
        for i, v in enumerate(xticks):
            x = self.px(v)
            d.line([(x, T), (x, H - B)], fill=GRID, width=1)
            lab = xticklabels[i] if xticklabels else fmt(v)
            d.text((x, H - B + 14), lab, font=F_S, fill=TEXT, anchor="mt")
        d.line([(L, H - B), (W - R, H - B)], fill=AXIS, width=2)
        d.line([(L, T), (L, H - B)], fill=AXIS, width=2)
        d.text((W / 2, 22), self.title, font=F_T, fill=TEXT, anchor="mt")
        d.text(((L + W - R) / 2, H - 40), self.xl, font=F_L, fill=TEXT, anchor="mm")
        lab = Image.new("RGBA", (H, 50), (255, 255, 255, 0))
        ImageDraw.Draw(lab).text((H / 2 - (T - B) / 2, 25), self.yl, font=F_L, fill=TEXT, anchor="mm")
        self.im.paste(lab.rotate(90, expand=True), (8, 0), lab.rotate(90, expand=True))

    def px(self, v):
        return L + (self.tx(v) - self.x0) / (self.x1 - self.x0) * (W - L - R)

    def py(self, v):
        return H - B - (self.ty(v) - self.y0) / (self.y1 - self.y0) * (H - T - B)

    def line(self, xs, ys, color, label=None, dash=False, marker=True, width=4):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        if dash:
            for (a, b) in zip(pts[:-1], pts[1:]):
                n = max(1, int(math.dist(a, b) / 14))
                for k in range(0, n, 2):
                    p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                    q = (a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n)
                    self.d.line([p, q], fill=color, width=width)
        else:
            self.d.line(pts, fill=color, width=width, joint="curve")
        if marker:
            for x, y in pts:
                self.d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color)
        if label:
            self.legend_items = getattr(self, "legend_items", []) + [(label, color, dash)]

    def bars(self, groups, series, colors, labels, value_fmt=fmt):
        """groups: подписи по оси x; series: список рядов (значения по группам)."""
        n, k = len(groups), len(series)
        slot = (W - L - R) / n
        bw = slot * 0.75 / k
        for g in range(n):
            for s in range(k):
                v = series[s][g]
                x = L + slot * g + slot * 0.125 + bw * s
                y = self.py(v)
                self.d.rectangle([x, y, x + bw - 4, H - B], fill=colors[s])
                self.d.text((x + (bw - 4) / 2, y - 6), value_fmt(v), font=F_S, fill=TEXT, anchor="mb")
            self.d.text((L + slot * (g + 0.5), H - B + 14), groups[g], font=F_S, fill=TEXT, anchor="mt")
        self.legend_items = [(l, c, False) for l, c in zip(labels, colors)]

    def legend(self, where="tl"):
        items = getattr(self, "legend_items", [])
        x = L + 24 if where.endswith("l") else W - R - 470
        y = T + 14 if where.startswith("t") else H - B - 20 - 38 * len(items)
        for lab, c, dash in items:
            self.d.rectangle([x, y + 8, x + 40, y + 16], fill=c)
            self.d.text((x + 52, y + 12), lab, font=F_S, fill=TEXT, anchor="lm")
            y += 38

    def save(self, name):
        FIG.mkdir(exist_ok=True)
        self.im.save(FIG / name)


def plot_memory():
    m = json.load(open(RES / "memory.json"))
    rows = m["scaling"]
    V = [r["V"] for r in rows]
    c = Chart(f"Пиковая память лосса + градиента (N={rows[0]['N']}, D={rows[0]['D']}, fp32)",
              "размер словаря V", "МБ (лог. шкала)", xlog=True, ylog=True)
    ys = [r["base_fwdbwd_mb"] for r in rows] + [r["grads_mb"] for r in rows]
    c.setup(V[0] / 1.3, V[-1] * 1.3, min(ys) / 2, max(ys) * 2, xticks=V, xticklabels=[f"{v // 1024}K" for v in V])
    c.line(V, [r["base_fwdbwd_mb"] for r in rows], COLORS[1], "обычный CE (логиты N×V)")
    c.line(V, [r["logits_mb"] for r in rows], COLORS[1], "размер одной матрицы логитов", dash=True, marker=False)
    c.line(V, [r["cce_fwdbwd_mb"] for r in rows], COLORS[0], "CCE")
    c.line(V, [r["grads_mb"] for r in rows], COLORS[2], "нижняя граница: сами градиенты dE, dC", dash=True, marker=False)
    c.legend("tl")
    c.save("memory_vs_vocab.png")

    c = Chart("Время лосса + градиента на CPU (numpy)", "размер словаря V", "секунды", xlog=True)
    ts = [r["base_fwdbwd_s"] for r in rows] + [r["cce_fwdbwd_s"] for r in rows]
    c.setup(V[0] / 1.3, V[-1] * 1.3, 0, max(ts) * 1.15, xticks=V, xticklabels=[f"{v // 1024}K" for v in V])
    c.line(V, [r["base_fwdbwd_s"] for r in rows], COLORS[1], "обычный CE")
    c.line(V, [r["cce_fwdbwd_s"] for r in rows], COLORS[0], "CCE (без фильтрации)")
    c.legend("tl")
    c.save("time_vs_vocab.png")

    b = m["blocks"]
    c = Chart(f"CCE: память зависит от блока, а не от V (V={rows[-1]['V'] // 1024}K)", "размер блока nb × vb", "МБ")
    c.setup(0, len(b), 0, max(x["peak_mb"] for x in b) * 1.2, xticks=[])
    c.bars([f"{x['nb']}×{x['vb']}" for x in b], [[x["peak_mb"] for x in b], [x["block_mb"] for x in b]],
           [COLORS[0], COLORS[4]], ["пиковая память CCE", "один блок логитов"], value_fmt=lambda v: f"{v:.0f}")
    c.legend("tl")
    c.save("memory_vs_block.png")


def plot_training():
    t = json.load(open(RES / "training.json"))
    c = Chart(f"Обучение LM (V={t['V']}): обычный CE и CCE", "шаг", "лосс")
    hb, hc = t["baseline"]["history"], t["cce"]["history"]
    ys = [h[k] for h in hb + hc for k in ("train", "val")]
    c.setup(0, t["steps"], math.floor(min(ys)) - 0.2, max(ys) + 0.2)
    c.line([h["step"] for h in hb], [h["val"] for h in hb], COLORS[1], "валидация, обычный CE", marker=False, width=8)
    c.line([h["step"] for h in hc], [h["val"] for h in hc], COLORS[0], "валидация, CCE", marker=False, width=3)
    c.line([h["step"] for h in hb], [h["train"] for h in hb], "#f4a582", "обучение, обычный CE", marker=False, width=8)
    c.line([h["step"] for h in hc], [h["train"] for h in hc], "#92c5de", "обучение, CCE", marker=False, width=3)
    c.legend("tr")
    c.save("training_curves.png")


def plot_sparsity():
    s = json.load(open(RES / "sparsity.json"))
    rows = [r for r in s["rows"] if r["model"] == "обученная"]
    groups = [f"β={r['beta']:g}\n{100 * r['frac_above_eps']:.2g}%" for r in rows]
    groups = [g.replace("\n", "  ") for g in groups]
    c = Chart(f"Доля пропущенных блоков (блок {s['nb']}×{s['vb']})", "обострение β и доля элементов softmax ≥ 2^-12", "% блоков")
    c.setup(0, len(rows), 0, 100, xticks=[])
    c.bars(groups, [[100 * r["filter_skipped"] for r in rows], [100 * r["filter_sort_skipped"] for r in rows]],
           [COLORS[4], COLORS[0]], ["фильтрация", "фильтрация + сортировка словаря"], value_fmt=lambda v: f"{v:.0f}")
    c.legend("tl")
    c.save("filter_skipped.png")

    r1 = next(r for r in rows if r["beta"] == 1.0)
    cnt, edges = r1["hist_counts"], r1["hist_edges"]
    tot = sum(cnt)
    k = next(i for i, e in enumerate(edges) if e >= -40)
    cnt, edges = cnt[k:], edges[k:]
    c = Chart("Распределение значений softmax обученной модели", "log2(вероятность)", "% элементов")
    c.setup(edges[0], edges[-1], 0, max(100 * x / tot for x in cnt) * 1.15)
    slot = (W - L - R) / len(cnt)
    for i, x in enumerate(cnt):
        x0 = c.px(edges[i])
        col = COLORS[0] if edges[i] >= -12 else "#b0b0b0"
        c.d.rectangle([x0 + 1, c.py(100 * x / tot), x0 + slot - 1, H - B], fill=col)
    xe = c.px(-12)
    c.d.line([(xe, T), (xe, H - B)], fill=COLORS[1], width=3)
    c.d.text((xe + 10, T + 10), "порог 2^-12", font=F_S, fill=COLORS[1])
    c.save("softmax_hist.png")


if __name__ == "__main__":
    for f in (plot_memory, plot_training, plot_sparsity):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
