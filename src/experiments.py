"""Все эксперименты воспроизведения. Запуск: python src/experiments.py [--quick]

Результаты пишутся в results/*.json, графики строит src/plots.py.
"""
import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

import cce
import lm

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MB = 2 ** 20


def peak_mb(fn, *args, **kw):
    """Пиковая память (МБ), выделенная внутри fn. numpy сообщает свои
    аллокации в tracemalloc, поэтому входные массивы сюда не попадают."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn(*args, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / MB, out


def best_time(fn, *args, repeat=3, **kw):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kw)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def random_problem(N, D, V, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((N, D)).astype(np.float32)
    C = (rng.standard_normal((V, D)) * scale / np.sqrt(D) * 4).astype(np.float32)
    y = rng.integers(0, V, N)
    return E, C, y


# ------------------------------------------------------------- 1. memory
def exp_memory(quick):
    N, D = (2048, 128) if quick else (4096, 256)
    Vs = [8192, 16384, 32768] if quick else [8192, 16384, 32768, 65536, 131072]
    rows = []
    for V in Vs:
        E, C, y = random_problem(N, D, V)
        r = {"N": N, "D": D, "V": V,
             "logits_mb": N * V * 4 / MB,
             "grads_mb": (N * D + V * D) * 4 / MB}
        r["base_fwd_mb"], _ = peak_mb(cce.baseline_forward, E, C, y)
        r["base_fwdbwd_mb"], _ = peak_mb(cce.baseline_loss_and_grad, E, C, y)
        r["cce_fwd_mb"], _ = peak_mb(cce.cce_forward, E, C, y)
        r["cce_fwdbwd_mb"], _ = peak_mb(cce.cce_loss_and_grad, E, C, y, filter_grad=False)
        r["base_fwd_s"] = best_time(cce.baseline_forward, E, C, y)
        r["base_fwdbwd_s"] = best_time(cce.baseline_loss_and_grad, E, C, y)
        r["cce_fwd_s"] = best_time(cce.cce_forward, E, C, y)
        r["cce_fwdbwd_s"] = best_time(cce.cce_loss_and_grad, E, C, y, filter_grad=False)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        rows.append(r)

    # размер блока: память CCE не зависит от V, но растёт с блоком
    V = Vs[-1]
    E, C, y = random_problem(N, D, V)
    blocks = []
    for nb, vb in [(256, 1024), (512, 2048), (1024, 4096), (2048, 8192)]:
        m, _ = peak_mb(cce.cce_loss_and_grad, E, C, y, nb=nb, vb=vb, filter_grad=False)
        t = best_time(cce.cce_loss_and_grad, E, C, y, nb=nb, vb=vb, filter_grad=False, repeat=2)
        blocks.append({"nb": nb, "vb": vb, "block_mb": nb * vb * 4 / MB, "peak_mb": m, "time_s": t})
        print(blocks[-1], flush=True)
    return {"scaling": rows, "blocks": blocks}


# ---------------------------------------------------------- 2. exactness
def exp_exactness():
    out = []
    for dtype in (np.float32, np.float64):
        E, C, y = random_problem(1024, 64, 20000, seed=1)
        E, C = E.astype(dtype), C.astype(dtype)
        l0, dE0, dC0 = cce.baseline_loss_and_grad(E, C, y)
        for name, kw in [("CCE без фильтрации", dict(filter_grad=False)),
                         ("CCE + фильтрация", dict(filter_grad=True)),
                         ("CCE + фильтрация + сортировка", dict(sort_vocab=True))]:
            l, dE, dC = cce.cce_loss_and_grad(E, C, y, nb=256, vb=2048, **kw)
            out.append({"dtype": np.dtype(dtype).name, "variant": name,
                        "loss_abs_err": abs(l - l0),
                        "dE_max_err": float(np.abs(dE - dE0).max()),
                        "dC_max_err": float(np.abs(dC - dC0).max())})
            print(out[-1], flush=True)
    return out


# ------------------------------------------------------------ 3. training
def train(loss_fn, tokens_tr, tokens_va, V, steps, N, seed=0):
    model = lm.MLPLM(V, seed=seed)
    rng = np.random.default_rng(seed)               # одинаковые батчи для обоих лоссов
    rng_va = np.random.default_rng(123)
    ctx_va, y_va = lm.make_batch(tokens_va, model.k, 4096, rng_va)
    hist = []
    t0 = time.perf_counter()
    for s in range(1, steps + 1):
        ctx, y = lm.make_batch(tokens_tr, model.k, N, rng)
        loss = model.step(ctx, y, loss_fn)
        if s % 50 == 0 or s == 1:
            _, E = model.embed(ctx_va)
            vl, _ = cce.baseline_forward(E, model.p["C"], y_va)
            hist.append({"step": s, "train": loss, "val": vl})
    return model, hist, time.perf_counter() - t0


def exp_training(quick):
    tokens, vocab = lm.load_corpus()
    V = len(vocab)
    cut = int(0.9 * len(tokens))
    tr, va = tokens[:cut], tokens[cut:]
    steps = 300 if quick else 600
    N = 1024
    res = {"V": V, "tokens": int(len(tokens)), "steps": steps, "N": N}
    fns = {
        "baseline": cce.baseline_loss_and_grad,
        "cce": lambda E, C, y: cce.cce_loss_and_grad(E, C, y, nb=256, vb=512, sort_vocab=True),
    }
    models = {}
    for name, fn in fns.items():
        model, hist, sec = train(fn, tr, va, V, steps, N)
        models[name] = model
        res[name] = {"history": hist, "seconds": sec}
        print(name, hist[-1], round(sec, 1), "s", flush=True)
    diff = max(float(np.abs(models["baseline"].p[k] - models["cce"].p[k]).max())
               for k in models["cce"].p)
    res["max_param_diff_after_training"] = diff
    return res, models["cce"], va


# --------------------------------------------- 4. sparsity and filtering
def exp_sparsity(model, tokens_va, untrained, nb=16, vb=128):
    """Доля значимых элементов softmax и эффект фильтрации градиента.

    beta - «обострение» распределения: E умножается на beta, то есть логиты
    делятся на температуру 1/beta. Наша маленькая модель далека от уверенности
    большой LLM, поэтому мы смотрим, как фильтрация ведёт себя по мере того,
    как softmax становится всё более разреженным.
    """
    rng = np.random.default_rng(7)
    ctx, y = lm.make_batch(tokens_va, model.k, 4096, rng)
    rows = []
    cases = [("до обучения", untrained, 1.0)] + [("обученная", model, b) for b in (1.0, 2.0, 4.0, 8.0)]
    for tag, m, beta in cases:
        _, E = m.embed(ctx)
        E = (E * beta).astype(np.float32)
        C = m.p["C"]
        logits = E @ C.T
        S = np.exp(logits - logits.max(1, keepdims=True))
        S /= S.sum(1, keepdims=True)
        r = {"model": tag, "beta": beta,
             "loss": float(-np.log(S[np.arange(len(y)), y] + 1e-30).mean()),
             "frac_above_eps": float((S >= cce.EPS).mean()),
             "mean_max_prob": float(S.max(1).mean())}
        if tag == "обученная" and beta == 1.0:
            h = np.histogram(np.log2(np.maximum(S, 2.0 ** -60)).ravel(), bins=np.arange(-60, 2, 2))
            r["hist_counts"], r["hist_edges"] = h[0].tolist(), h[1].tolist()
        _, dE0, dC0 = cce.baseline_loss_and_grad(E, C, y)
        for name, kw in [("no_filter", dict(filter_grad=False)),
                         ("filter", dict(filter_grad=True)),
                         ("filter_sort", dict(filter_grad=True, sort_vocab=True))]:
            st = {}
            _, dE, dC = cce.cce_loss_and_grad(E, C, y, nb=nb, vb=vb, stats=st, **kw)
            tot = st["blocks_done"] + st["blocks_skipped"]
            r[name + "_skipped"] = st["blocks_skipped"] / tot
            r[name + "_time"] = best_time(cce.cce_loss_and_grad, E, C, y, nb=nb, vb=vb, repeat=3, **kw)
            r[name + "_grad_rel_err"] = float(max(np.abs(dE - dE0).max() / np.abs(dE0).max(),
                                                  np.abs(dC - dC0).max() / np.abs(dC0).max()))
        rows.append(r)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("hist")}, flush=True)
    return {"nb": nb, "vb": vb, "N": len(y), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="уменьшенные размеры для быстрой проверки")
    ap.add_argument("--only", default="exactness,memory,training", help="какие эксперименты запускать")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    only = a.only.split(",")
    if "exactness" in only:
        print("== exactness"); json.dump(exp_exactness(), open(RES / "exactness.json", "w"), ensure_ascii=False, indent=1)
    if "memory" in only:
        print("== memory"); json.dump(exp_memory(a.quick), open(RES / "memory.json", "w"), ensure_ascii=False, indent=1)
    if "training" in only:
        print("== training")
        res, model, va = exp_training(a.quick)
        json.dump(res, open(RES / "training.json", "w"), ensure_ascii=False, indent=1)
        np.savez(RES / "model_cce.npz", **model.p)
        print("== sparsity")
        untrained = lm.MLPLM(res["V"], seed=0)
        sp = exp_sparsity(model, va, untrained)
        json.dump(sp, open(RES / "sparsity.json", "w"), ensure_ascii=False, indent=1)
    if "sparsity" in only and "training" not in only:
        # отдельный запуск на сохранённой модели
        tokens, vocab = lm.load_corpus()
        va = tokens[int(0.9 * len(tokens)):]
        model = lm.MLPLM(len(vocab), seed=0)
        model.p = dict(np.load(RES / "model_cce.npz"))
        untrained = lm.MLPLM(len(vocab), seed=0)
        sp = exp_sparsity(model, va, untrained)
        json.dump(sp, open(RES / "sparsity.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
