"""Cut Cross-Entropy (CCE) на numpy.

Воспроизведение идеи из статьи
  Wijmans et al., "Cut Your Losses in Large-Vocabulary Language Models", ICLR 2025.

Обозначения как в статье:
  E  : (N, D)  эмбеддинги токенов (выход последнего слоя модели)
  C  : (V, D)  матрица классификатора (unembedding)
  y  : (N,)    правильные следующие токены
  loss = mean_i [ logsumexp_j(C_j . E_i) - C_{y_i} . E_i ]

baseline_* строят полную матрицу логитов N x V, как обычный PyTorch.
cce_* никогда не держат в памяти больше одного блока логитов (nb x vb).
"""
import numpy as np

# Порог фильтрации градиента из статьи: 2^-12 (точность bf16).
EPS = 2.0 ** -12


# ---------------------------------------------------------------- baseline
def baseline_forward(E, C, y):
    logits = E @ C.T                                   # (N, V), материализуется целиком
    m = logits.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))[:, 0]
    correct = logits[np.arange(len(y)), y]
    return float((lse - correct).mean()), lse


def baseline_backward(E, C, y):
    N = E.shape[0]
    logits = E @ C.T
    logits -= logits.max(axis=1, keepdims=True)
    S = np.exp(logits)                                 # softmax, (N, V)
    S /= S.sum(axis=1, keepdims=True)
    S[np.arange(N), y] -= 1.0                          # dL/dlogits = softmax - onehot
    S /= N
    return S @ C, S.T @ E                              # dE (N, D), dC (V, D)


def baseline_loss_and_grad(E, C, y):
    loss, _ = baseline_forward(E, C, y)
    dE, dC = baseline_backward(E, C, y)
    return loss, dE, dC


# --------------------------------------------------------------------- CCE
def indexed_dot(E, C, y):
    """Правильный логит C_{y_i} . E_i без построения N x V (indexed matmul)."""
    return np.einsum("nd,nd->n", E, C[y])


def linear_lse(E, C, nb=1024, vb=4096, return_mean_logit=False):
    """logsumexp_j(C_j . E_i) по блокам словаря (linear-log-sum-exp).

    Для каждого блока токенов (nb строк) идём по блокам словаря (vb столбцов)
    и обновляем онлайн-логсумэксп, как во FlashAttention.
    Дополнительно можно вернуть средний логит каждого токена словаря:
    он нужен для сортировки словаря (vocabulary sorting).
    """
    N, V = E.shape[0], C.shape[0]
    lse = np.empty(N, dtype=E.dtype)
    mean_logit = np.zeros(V, dtype=np.float64) if return_mean_logit else None
    for n0 in range(0, N, nb):
        En = E[n0:n0 + nb]
        m = np.full(En.shape[0], -np.inf, dtype=E.dtype)
        s = np.zeros(En.shape[0], dtype=E.dtype)
        for v0 in range(0, V, vb):
            blk = En @ C[v0:v0 + vb].T                 # (nb, vb) - единственный большой буфер
            if return_mean_logit:
                mean_logit[v0:v0 + vb] += blk.sum(axis=0)
            m_new = np.maximum(m, blk.max(axis=1))
            s = s * np.exp(m - m_new) + np.exp(blk - m_new[:, None]).sum(axis=1)
            m = m_new
        lse[n0:n0 + nb] = m + np.log(s)
    if return_mean_logit:
        mean_logit /= N
    return lse, mean_logit


def cce_forward(E, C, y, nb=1024, vb=4096):
    lse, _ = linear_lse(E, C, nb, vb)
    return float((lse - indexed_dot(E, C, y)).mean()), lse


def cce_backward(E, C, y, lse, nb=1024, vb=4096, eps=EPS, filter_grad=True,
                 order=None, stats=None):
    """Градиенты dE, dC блоками, логиты пересчитываются на лету.

    filter_grad : пропускать блоки, где все элементы softmax < eps
                  (gradient filtering из статьи). Блок, где лежит правильный
                  токен, никогда не пропускается, там есть элемент -1.
    order       : перестановка словаря (vocabulary sorting). Если задана,
                  блоки идут в порядке order, чтобы значимые столбцы
                  softmax собрались в небольшое число блоков.
    stats       : dict, сюда пишется число обработанных и пропущенных блоков.
    """
    N, D = E.shape
    V = C.shape[0]
    if order is None:
        Cs, inv = C, None
        ys = y
    else:
        Cs = C[order]                                  # переставленный словарь
        inv = np.empty_like(order)
        inv[order] = np.arange(V)
        ys = inv[y]                                    # индексы целей в новом порядке
    dE = np.zeros_like(E)
    dCs = np.zeros_like(C)
    done = skipped = 0
    for n0 in range(0, N, nb):
        En = E[n0:n0 + nb]
        yn = ys[n0:n0 + nb]
        lse_n = lse[n0:n0 + nb, None]
        for v0 in range(0, V, vb):
            Cb = Cs[v0:v0 + vb]
            S = np.exp(En @ Cb.T - lse_n)              # softmax-блок (nb, vb)
            hit = (yn >= v0) & (yn < v0 + Cb.shape[0])
            if filter_grad and not hit.any() and S.max() < eps:
                skipped += 1                           # градиент блока пренебрежимо мал
                continue
            if hit.any():
                rows = np.nonzero(hit)[0]
                S[rows, yn[rows] - v0] -= 1.0
            S /= N
            dE[n0:n0 + nb] += S @ Cb
            dCs[v0:v0 + vb] += S.T @ En
            done += 1
    if stats is not None:
        stats["blocks_done"] = stats.get("blocks_done", 0) + done
        stats["blocks_skipped"] = stats.get("blocks_skipped", 0) + skipped
    if order is None:
        return dE, dCs
    dC = np.empty_like(dCs)
    dC[order] = dCs
    return dE, dC


def cce_loss_and_grad(E, C, y, nb=1024, vb=4096, filter_grad=True,
                      sort_vocab=False, stats=None):
    lse, mean_logit = linear_lse(E, C, nb, vb, return_mean_logit=sort_vocab)
    loss = float((lse - indexed_dot(E, C, y)).mean())
    order = np.argsort(-mean_logit) if sort_vocab else None
    dE, dC = cce_backward(E, C, y, lse, nb, vb, filter_grad=filter_grad,
                          order=order, stats=stats)
    return loss, dE, dC
