"""
Width-consistency of weight spectra: muP+GD vs idealized Muon.

Exp 1: deep linear network, population GD.
       h1 = W0 x / sqrt(D), h2 = W1 h1 / sqrt(N), f = W2 h2 / (gamma0 * N).
       All entries init N(0,1) (mean-field convention, cf. Lauditi et al.).
       Target y = A x, A in R^{C x D}. Whitened population loss
       L = 0.5 ||Phi - A||_F^2 with Phi the end-to-end map.
       muP+GD: all layers, lr = eta * gamma0^2 * N.
       Idealized Muon (no momentum, exact orthogonalization) on W0, W1:
       dW = -eta_m * (sqrt(fan_in)+sqrt(fan_out)) * msign(grad); readout via GD.
       Gradients have rank <= C: msign computed from thin factors, O(N C^2).

Exp 2: two-layer nonlinear network, online minibatch.
       f = w . phi(W x / sqrt(D)) / (gamma0 * N), phi = tanh.
       Multi-index target y = u1*u2 + 0.5*(u1^2 - 1), u_k = beta_k . x / sqrt(D).
       muP+SGD vs Muon on W (msign(G) = G (G^T G)^{-1/2}, thin, N >= D).

Spectra normalized following plot_resnet_block_spectra.py conventions:
normalized sigma = sigma / sqrt(num_cols * init_var), MP edge = 1 + sqrt(rows/cols).

Usage:
  python3 width_spectra_toy.py lin  <gd|muon> <N> [T]
  python3 width_spectra_toy.py nonlin <gd|muon> <N> [T]   (chunked, resumable)
  python3 width_spectra_toy.py plots
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_width")
os.makedirs(RES, exist_ok=True)

WIDTHS = [128, 256, 512, 1024]
D = 96
C = 8
GAMMA0 = 3.0
LOSS_FRACS = [0.6, 0.25, 0.08]   # snapshot at first crossing of frac * L0


def msign_thin(L, R):
    """Polar factor of G = L @ R.T (rank <= C) from thin factors. O(N C^2)."""
    QL, SL = np.linalg.qr(L)
    QR, SR = np.linalg.qr(R)
    U, _, Vt = np.linalg.svd(SL @ SR.T, full_matrices=False)
    return (QL @ U) @ (Vt @ QR.T)


def msign_tall(G):
    """Polar factor of tall G (N x D, N >= D): G (G^T G)^{-1/2} via thin SVD."""
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


def normalized_svals(W, init_var=1.0):
    rows, cols = W.shape
    s = np.linalg.svd(W, compute_uv=False)
    return s / np.sqrt(cols * init_var), rows / cols


# ===================================================================== Exp 1
def run_lin(opt, N, T=4000, eta_gd=4e-3, eta_m=0.01, seed=0):
    rng = np.random.default_rng(seed)
    sA = np.linspace(1.0, 4.0, C)
    UA = np.linalg.qr(rng.standard_normal((C, C)))[0]
    VA = np.linalg.qr(rng.standard_normal((D, C)))[0]
    A = UA @ np.diag(sA) @ VA.T                       # C x D
    W0 = rng.standard_normal((N, D))
    W1 = rng.standard_normal((N, N))
    W2 = rng.standard_normal((C, N))
    c_out = 1.0 / (GAMMA0 * N * np.sqrt(N * D))       # Phi = c_out W2 W1 W0
    lr = eta_gd * GAMMA0**2 * N
    out, losses, snap_steps = {}, [], []
    L0 = None
    for t in range(T + 1):
        P10 = W1 @ W0                                  # N x D
        Phi = c_out * (W2 @ P10)                       # C x D
        E = Phi - A
        loss = 0.5 * np.sum(E**2)
        losses.append(loss)
        if L0 is None:
            L0 = loss
        # snapshots at loss thresholds
        for f in LOSS_FRACS:
            key = f"snap_{f}"
            if key + "_W1" not in out and loss <= f * L0:
                out[key + "_W0"], _ = normalized_svals(W0)
                out[key + "_W1"], _ = normalized_svals(W1)
                snap_steps.append((f, t))
        if loss <= LOSS_FRACS[-1] * L0 and len(snap_steps) == len(LOSS_FRACS):
            break
        # gradients (exact population)
        G2 = c_out * (E @ P10.T)                       # C x N
        G1 = c_out * (W2.T @ (E @ W0.T))               # N x N, rank <= C
        G0 = c_out * ((W2 @ W1).T @ E)                 # N x D, rank <= C
        if opt == "gd":
            W0 -= lr * G0; W1 -= lr * G1; W2 -= lr * G2
        else:
            W1 -= eta_m * 2 * np.sqrt(N) * msign_thin(W2.T, W0 @ E.T)
            W0 -= eta_m * (np.sqrt(N) + np.sqrt(D)) * msign_thin(
                (W2 @ W1).T, E.T)
            W2 -= lr * G2
    out["losses"] = np.array(losses)
    out["snap_steps"] = np.array([s for _, s in snap_steps])
    np.savez(os.path.join(RES, f"lin_{opt}_N{N}.npz"), **out)
    print(f"lin {opt} N={N}: L0={L0:.3f} final={losses[-1]:.4f} "
          f"snaps at steps {[s for _, s in snap_steps]}")


# ===================================================================== Exp 2
def target_y(X, B1, B2):
    """Multi-index staircase: linear in u1, then u1*u2 and He2(u1)."""
    u1 = B1 @ X / np.sqrt(D)
    u2 = B2 @ X / np.sqrt(D)
    return u1 + u1 * u2 + 0.5 * (u1**2 - 1.0)


def run_nonlin(opt, N, T=3000, B=512, eta_gd=1e-1, eta_m=0.02, seed=0,
               chunk=1200):
    ck_path = os.path.join(RES, f"nonlin_{opt}_N{N}_ckpt.npz")
    fin_path = os.path.join(RES, f"nonlin_{opt}_N{N}.npz")
    if os.path.exists(ck_path):
        ck = np.load(ck_path, allow_pickle=True)
        W, w, B1, B2 = ck["W"], ck["w"], ck["B1"], ck["B2"]
        t0 = int(ck["t0"]); losses = list(ck["losses"])
        L0 = float(ck["L0"]); snaps = dict(ck["snaps"].item())
        rng = np.random.default_rng(seed + 13 * N)
        rng.bit_generator.state = eval(str(ck["rngstate"]))
    else:
        rng = np.random.default_rng(seed + 13 * N)
        B1v = rng.standard_normal(D); B1v /= np.linalg.norm(B1v) / np.sqrt(D)
        B2v = rng.standard_normal(D); B2v /= np.linalg.norm(B2v) / np.sqrt(D)
        B1, B2 = B1v, B2v
        W = rng.standard_normal((N, D))
        w = rng.standard_normal(N)
        t0 = 0; losses = []; L0 = -1.0; snaps = {}
    lr = eta_gd * GAMMA0**2 * N
    t_end = min(t0 + chunk, T)
    for t in range(t0, t_end):
        X = rng.standard_normal((D, B))
        y = target_y(X, B1, B2)
        Z = W @ X / np.sqrt(D)
        H = np.maximum(Z, 0)                           # relu, N x B
        f = w @ H / (GAMMA0 * N)
        err = f - y                                    # B
        loss = 0.5 * np.mean(err**2)
        losses.append(loss)
        if L0 < 0:
            L0 = loss
        for fr in LOSS_FRACS:
            key = f"snap_{fr}"
            if key + "_W" not in snaps and loss <= fr * L0:
                s, _ = normalized_svals(W)
                snaps[key + "_W"] = s
                snaps[key + "_step"] = t
        delta = np.outer(w, err) * (Z > 0) / (GAMMA0 * N)      # N x B
        GW = delta @ X.T / (B * np.sqrt(D))            # N x D
        gw = H @ err / (B * GAMMA0 * N)                # N
        if opt == "gd":
            W -= lr * GW
        else:
            W -= eta_m * (np.sqrt(N) + np.sqrt(D)) * msign_tall(GW)
        w -= lr * gw
    if t_end < T:
        np.savez(ck_path, W=W, w=w, B1=B1, B2=B2, t0=t_end, losses=losses,
                 L0=L0, snaps=np.array(snaps, dtype=object),
                 rngstate=str(rng.bit_generator.state))
        print(f"nonlin {opt} N={N}: checkpoint t={t_end} loss={losses[-1]:.4f}")
    else:
        s, _ = normalized_svals(W)
        snaps["snap_final_W"] = s
        snaps["snap_final_step"] = T
        np.savez(fin_path, losses=np.array(losses), L0=L0,
                 **{k: v for k, v in snaps.items()})
        try:
            os.remove(ck_path)
        except OSError:
            pass
        print(f"nonlin {opt} N={N}: done, L0={L0:.4f} final={losses[-1]:.4f} "
              f"snaps={[k for k in snaps if k.endswith('_step')]}")


# ===================================================================== Exp 3
def run_deep_nonlin(opt, N, T=3000, B=512, eta_gd=1e-1, eta_m=0.02, seed=0,
                    chunk=None):
    if chunk is None:
        chunk = max(100, int((5e5 if opt == "gd" else 1.2e5) / N))
    """Two hidden layers, relu; track the SQUARE middle matrix W1 (N x N,
    fixed aspect across widths -- the conv2 analog). Muon on W0 and W1
    (rank <= B msign via thin factors), muP-SGD on readout."""
    ck_path = os.path.join(RES, f"deep_{opt}_N{N}_ckpt.npz")
    fin_path = os.path.join(RES, f"deep_{opt}_N{N}.npz")
    if os.path.exists(ck_path):
        ck = np.load(ck_path, allow_pickle=True)
        W0, W1, w = ck["W0"], ck["W1"], ck["w"]
        B1, B2 = ck["B1"], ck["B2"]
        t0 = int(ck["t0"]); losses = list(ck["losses"])
        L0 = float(ck["L0"]); snaps = dict(ck["snaps"].item())
        rng = np.random.default_rng(seed + 17 * N)
        rng.bit_generator.state = eval(str(ck["rngstate"]))
    else:
        rng = np.random.default_rng(seed + 17 * N)
        B1 = rng.standard_normal(D); B1 /= np.linalg.norm(B1) / np.sqrt(D)
        B2 = rng.standard_normal(D); B2 /= np.linalg.norm(B2) / np.sqrt(D)
        W0 = rng.standard_normal((N, D))
        W1 = rng.standard_normal((N, N))
        w = rng.standard_normal(N)
        t0 = 0; losses = []; L0 = -1.0; snaps = {}
    lr = eta_gd * GAMMA0**2 * N
    t_end = min(t0 + chunk, T)
    for t in range(t0, t_end):
        X = rng.standard_normal((D, B))
        y = target_y(X, B1, B2)
        Z0 = W0 @ X / np.sqrt(D); H0 = np.maximum(Z0, 0)      # N x B
        Z1 = W1 @ H0 / np.sqrt(N); H1 = np.maximum(Z1, 0)     # N x B
        f = w @ H1 / (GAMMA0 * N)
        err = f - y
        loss = 0.5 * np.mean(err**2)
        losses.append(loss)
        if L0 < 0:
            L0 = loss
        for fr in LOSS_FRACS:
            key = f"snap_{fr}"
            if key + "_W1" not in snaps and loss <= fr * L0:
                snaps[key + "_W0"], _ = normalized_svals(W0)
                snaps[key + "_W1"], _ = normalized_svals(W1)
                snaps[key + "_step"] = t
        d1 = np.outer(w, err) * (Z1 > 0) / (GAMMA0 * N)       # N x B
        d0 = (W1.T @ d1 / np.sqrt(N)) * (Z0 > 0)              # N x B
        if opt == "gd":
            W1 -= lr * (d1 @ H0.T / (B * np.sqrt(N)))
            W0 -= lr * (d0 @ X.T / (B * np.sqrt(D)))
        else:
            W1 -= eta_m * 2 * np.sqrt(N) * msign_thin(d1, H0)
            W0 -= eta_m * (np.sqrt(N) + np.sqrt(D)) * msign_thin(d0, X)
        w -= lr * (H1 @ err / (B * GAMMA0 * N))
    if t_end < T:
        np.savez(ck_path, W0=W0, W1=W1, w=w, B1=B1, B2=B2, t0=t_end,
                 losses=losses, L0=L0, snaps=np.array(snaps, dtype=object),
                 rngstate=str(rng.bit_generator.state))
        print(f"deep {opt} N={N}: checkpoint t={t_end} loss={losses[-1]:.4f}")
    else:
        for layer, Wm in [("W0", W0), ("W1", W1)]:
            s, _ = normalized_svals(Wm)
            snaps[f"snap_final_{layer}"] = s
        snaps["snap_final_step"] = T
        np.savez(fin_path, losses=np.array(losses), L0=L0,
                 **{k: v for k, v in snaps.items()})
        try:
            os.remove(ck_path)
        except OSError:
            pass
        print(f"deep {opt} N={N}: done, L0={L0:.4f} final={losses[-1]:.4f}")


# ===================================================================== plots
def mp_density(xs, aspect):
    """Density of normalized singular values sigma/sqrt(cols) for a
    variance-1 iid rows x cols matrix, aspect = rows/cols >= 1.
    Support [sqrt(aspect)-1, sqrt(aspect)+1], edge 1 + sqrt(aspect)."""
    lo, hi = (np.sqrt(aspect) - 1) ** 2, (np.sqrt(aspect) + 1) ** 2
    x2 = xs**2
    mask = (x2 > lo) & (x2 < hi) & (xs > 0)
    rho = np.zeros_like(xs)
    rho[mask] = np.sqrt((hi - x2[mask]) * (x2[mask] - lo)) / (
        np.pi * xs[mask])
    return rho


def _spectra_panel(fig, axs_col, data_by_width, aspect_by_width, title,
                   bins, xmax=None):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    widths = sorted(data_by_width)
    colors = {n: cmap(i / max(1, len(widths) - 1))
              for i, n in enumerate(widths)}
    ax_top, ax_bot = axs_col
    allv = np.concatenate([data_by_width[n] for n in widths])
    xmax = xmax or allv.max() * 1.05
    xs = np.linspace(1e-3, xmax, 400)
    distinct_aspects = len({round(a, 3) for a in aspect_by_width.values()}) > 1
    for n in widths:
        ax_top.hist(data_by_width[n], bins=np.linspace(0, xmax, bins),
                    density=True, histtype="step", color=colors[n],
                    label=f"W={n}")
        edge = 1 + np.sqrt(aspect_by_width[n])
        ax_top.axvline(edge, color=colors[n], ls="--", lw=0.8)
        ax_bot.axvline(edge, color=colors[n], ls="--", lw=0.8)
        if distinct_aspects:
            ax_top.plot(xs, mp_density(xs, aspect_by_width[n]),
                        color=colors[n], lw=0.8, alpha=0.6)
    if not distinct_aspects:
        ax_top.plot(xs, mp_density(xs, aspect_by_width[widths[-1]]),
                    "k-", lw=1.2, label="MP")
    for i, n in enumerate(widths):
        sv = data_by_width[n]
        edge = 1 + np.sqrt(aspect_by_width[n])
        tail = sv[sv > edge * 1.02]
        ax_bot.plot(tail, np.full_like(tail, i), "o", ms=2.5,
                    color=colors[n])
    ax_bot.set_yticks(range(len(widths)))
    ax_bot.set_yticklabels([f"{n}" for n in widths], fontsize=6)
    ax_bot.set_ylim(-0.7, len(widths) - 0.3)
    ax_bot.set_xlim(ax_top.get_xlim())
    ax_top.set_title(title, fontsize=9)


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Exp 1 / Exp 3 figures: one per (optimizer, layer)
    for pre, stem, desc in [("lin", "exp1", "deep linear"),
                            ("deep", "exp3", "3-layer relu, multi-index")]:
        for opt in ["gd", "muon"]:
            data = {N: np.load(os.path.join(RES, f"{pre}_{opt}_N{N}.npz"))
                    for N in WIDTHS if
                    os.path.exists(os.path.join(RES, f"{pre}_{opt}_N{N}.npz"))}
            if not data:
                continue
            # 'final' column only when all widths share the same final step
            finals_ok = all(f"snap_final_W1" in d.files for d in data.values())
            if finals_ok:
                fsteps = {int(d["snap_final_step"]) for d in data.values()}
                finals_ok = len(fsteps) == 1
            fracs = list(LOSS_FRACS) + (["final"] if finals_ok else [])
            for layer, asp_fn in [("W1", lambda N: 1.0),
                                  ("W0", lambda N: N / D)]:
                fig, axs = plt.subplots(
                    2, len(fracs), figsize=(4.2 * len(fracs), 4.4),
                    gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
                for j, fr in enumerate(fracs):
                    key = f"snap_{fr}_{layer}"
                    dbw = {N: data[N][key] for N in data
                           if key in data[N].files}
                    abw = {N: asp_fn(N) for N in dbw}
                    if not dbw:
                        continue
                    ti = "final" if fr == "final" else f"loss = {fr} L0"
                    _spectra_panel(fig, (axs[0, j], axs[1, j]), dbw, abw,
                                   ti, bins=70)
                axs[0, 0].legend(fontsize=6)
                axs[0, 0].set_ylabel("Density")
                axs[1, 0].set_ylabel("width", fontsize=7)
                for j in range(len(fracs)):
                    axs[1, j].set_xlabel("MP-normalized singular value",
                                         fontsize=8)
                fig.suptitle(f"{stem} {desc}, {opt.upper()}, layer {layer} "
                             f"(D={D}, gamma0={GAMMA0})", fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(RES, f"{stem}_{opt}_{layer}.png"),
                            dpi=150)
                plt.close(fig)

    # ---- Exp 2 figures
    for opt in ["gd", "muon"]:
        data = {N: np.load(os.path.join(RES, f"nonlin_{opt}_N{N}.npz"))
                for N in WIDTHS if
                os.path.exists(os.path.join(RES, f"nonlin_{opt}_N{N}.npz"))}
        if not data:
            continue
        snap_keys = [f"snap_{fr}_W" for fr in LOSS_FRACS] + ["snap_final_W"]
        titles = [f"loss = {fr} L0" for fr in LOSS_FRACS] + ["final"]
        fig, axs = plt.subplots(2, len(snap_keys),
                                figsize=(4.2 * len(snap_keys), 4.4),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex="col")
        for j, (key, ti) in enumerate(zip(snap_keys, titles)):
            dbw = {N: data[N][key] for N in data if key in data[N].files}
            abw = {N: N / D for N in dbw}
            if not dbw:
                continue
            _spectra_panel(fig, (axs[0, j], axs[1, j]), dbw, abw, ti, bins=70)
        axs[0, 0].legend(fontsize=6)
        axs[0, 0].set_ylabel("Density")
        axs[1, 0].set_ylabel("width", fontsize=7)
        for j in range(len(snap_keys)):
            axs[1, j].set_xlabel("MP-normalized singular value", fontsize=8)
        fig.suptitle(f"Exp2 two-layer tanh, multi-index target, {opt.upper()} "
                     f"(D={D}, B=512, gamma0={GAMMA0})", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, f"exp2_{opt}_W.png"), dpi=150)
        plt.close(fig)

    # ---- loss curves
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (exp, pre) in zip(axs, [("lin", "lin"), ("nonlin", "nonlin")]):
        for opt, ls in [("gd", "-"), ("muon", ":")]:
            for N in WIDTHS:
                p = os.path.join(RES, f"{pre}_{opt}_N{N}.npz")
                if os.path.exists(p):
                    l = np.load(p)["losses"]
                    ax.plot(l, ls=ls, lw=1, label=f"{opt} W={N}")
        ax.set_yscale("log"); ax.set_title(exp); ax.set_xlabel("t")
        ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "losses.png"), dpi=150)
    print("plots done")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "lin":
        run_lin(sys.argv[2], int(sys.argv[3]),
                T=int(sys.argv[4]) if len(sys.argv) > 4 else 4000)
    elif cmd == "nonlin":
        run_nonlin(sys.argv[2], int(sys.argv[3]),
                   T=int(sys.argv[4]) if len(sys.argv) > 4 else 3000)
    elif cmd == "deep":
        run_deep_nonlin(sys.argv[2], int(sys.argv[3]),
                        T=int(sys.argv[4]) if len(sys.argv) > 4 else 3000)
    elif cmd == "plots":
        make_plots()
