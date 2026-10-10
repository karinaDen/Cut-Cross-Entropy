"""Маленькая языковая модель на numpy для проверки CCE на «настоящих» логитах.

Модель: эмбеддинги k предыдущих слов -> конкатенация -> Linear -> tanh -> E (N, D)
        -> классификатор C (V, D) -> softmax по всему словарю.
Лосс и его градиент по E и C считает выбранная функция (baseline или CCE),
остальная часть backward одинакова для обеих.
Корпус: «Opticks» Исаака Ньютона (Project Gutenberg, общественное достояние).
"""
import re
import collections
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "Isaac.Newton-Opticks.txt"


def load_corpus(path=DATA, seed=0):
    text = path.read_text(encoding="utf-8").lower()
    words = re.findall(r"[a-z]+|[0-9]+|[^\sa-z0-9]", text)
    counts = collections.Counter(words)
    vocab = [w for w, _ in counts.most_common()]
    # Перемешиваем id слов: в реальных BPE-токенизаторах id не упорядочены
    # по частоте, иначе сортировка словаря была бы уже «сделана» за нас.
    np.random.default_rng(seed).shuffle(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    return np.array([idx[w] for w in words], dtype=np.int64), vocab


class MLPLM:
    def __init__(self, V, k=4, d_emb=32, D=64, seed=0):
        rng = np.random.default_rng(seed)
        self.k = k
        self.p = {
            "emb": (rng.standard_normal((V, d_emb)) * 0.1).astype(np.float32),
            "W1": (rng.standard_normal((k * d_emb, D)) / np.sqrt(k * d_emb)).astype(np.float32),
            "b1": np.zeros(D, dtype=np.float32),
            "C": (rng.standard_normal((V, D)) / np.sqrt(D)).astype(np.float32),
        }
        self.m = {n: np.zeros_like(a) for n, a in self.p.items()}
        self.v = {n: np.zeros_like(a) for n, a in self.p.items()}
        self.t = 0

    def embed(self, ctx):
        p = self.p
        X = p["emb"][ctx].reshape(len(ctx), -1)           # (N, k*d_emb)
        H = np.tanh(X @ p["W1"] + p["b1"])                # (N, D) = E
        return X, H

    def step(self, ctx, y, loss_fn, lr=3e-3, b1=0.9, b2=0.999):
        p = self.p
        X, E = self.embed(ctx)
        loss, dE, dC = loss_fn(E, p["C"], y)
        dpre = dE * (1.0 - E * E)
        g = {"C": dC, "W1": X.T @ dpre, "b1": dpre.sum(0)}
        dX = (dpre @ p["W1"].T).reshape(len(ctx), self.k, -1)
        demb = np.zeros_like(p["emb"])
        np.add.at(demb, ctx, dX)
        g["emb"] = demb
        self.t += 1
        for n in p:                                       # Adam
            self.m[n] = b1 * self.m[n] + (1 - b1) * g[n]
            self.v[n] = b2 * self.v[n] + (1 - b2) * g[n] ** 2
            mh = self.m[n] / (1 - b1 ** self.t)
            vh = self.v[n] / (1 - b2 ** self.t)
            p[n] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return loss


def make_batch(tokens, k, N, rng):
    pos = rng.integers(k, len(tokens), N)
    ctx = np.stack([tokens[pos - k + j] for j in range(k)], axis=1)
    return ctx, tokens[pos]
