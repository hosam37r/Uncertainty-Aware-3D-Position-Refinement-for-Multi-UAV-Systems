import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# Text measurement helper (Pillow version-safe)
# =========================================================
def text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Returns (width, height) for text using Pillow version-safe methods.
    """
    # Preferred in newer Pillow:
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        return (x1 - x0, y1 - y0)
    # Older fallback:
    if hasattr(font, "getsize"):
        return font.getsize(text)
    # Last resort (should rarely happen):
    return (len(text) * 6, 10)


# =========================================================
# Core utilities
# =========================================================
def clamp_reflect(pos, vel, low=0.0, high=50.0):
    """Reflect inside cube [low, high]^3."""
    for k in range(3):
        if pos[k] < low:
            pos[k] = low + (low - pos[k])
            vel[k] *= -1
        elif pos[k] > high:
            pos[k] = high - (pos[k] - high)
            vel[k] *= -1
    return pos, vel

def ensure_pd(cov, eps=1e-6):
    """Make covariance numerically safe."""
    return cov + eps * np.eye(cov.shape[0])

def gauss_newton_refine(
    prior_p, prior_cov,
    neighbor_positions, ranges, range_sigmas, omegas,
    max_iter=6, damping=1e-3, tol=1e-4
):
    """
    Solve:
      min_p (p-prior)^T Sigma^{-1} (p-prior)
          + sum_j omega_j * (||p-pj|| - d_j)^2 / sigma_dj^2
    """
    p = prior_p.copy()
    inv_prior = np.linalg.inv(ensure_pd(prior_cov))

    for _ in range(max_iter):
        H = inv_prior.copy()
        g = inv_prior @ (p - prior_p)

        for pj, d_hat, sd, w in zip(neighbor_positions, ranges, range_sigmas, omegas):
            if w <= 0:
                continue
            u = p - pj
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-8:
                continue
            r = norm_u - d_hat
            a = u / norm_u
            ww = w / (sd * sd + 1e-12)
            H += ww * np.outer(a, a)
            g += ww * a * r

        H_damped = H + damping * np.eye(3)
        delta = -np.linalg.solve(H_damped, g)
        p_next = p + delta

        if np.linalg.norm(delta) < tol:
            p = p_next
            break
        p = p_next

    cov = np.linalg.inv(H + damping * np.eye(3))
    return p, cov


# =========================================================
# Projection figure helpers (XY/XZ/YZ) - no matplotlib
# =========================================================
def draw_marker(draw, x, y, kind="o", r=4, width=2, color=(0, 0, 0)):
    if kind == "o":
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=width)
    elif kind == "x":
        draw.line([x-r, y-r, x+r, y+r], fill=color, width=width)
        draw.line([x-r, y+r, x+r, y-r], fill=color, width=width)
    elif kind == "^":
        draw.polygon([(x, y-r), (x-r, y+r), (x+r, y+r)], outline=color)
    else:
        draw.point((x, y), fill=color)

def plot_projection_panel(draw, origin, size, box, truth, local, refined, dims=("x","y"), title=""):
    ox, oy = origin
    w, h = size
    pad = 22

    draw.rectangle([ox, oy, ox+w, oy+h], outline=(0,0,0), width=2)

    ix0, iy0 = ox+pad, oy+pad
    ix1, iy1 = ox+w-pad, oy+h-pad
    draw.rectangle([ix0, iy0, ix1, iy1], outline=(0,0,0), width=1)

    if title:
        draw.text((ox+6, oy+4), title, fill=(0,0,0))

    def idx(c):
        return 0 if c == "x" else (1 if c == "y" else 2)

    a = idx(dims[0])
    b = idx(dims[1])

    def map_pt(p):
        px = ix0 + (p[a] / box) * (ix1 - ix0)
        py = iy1 - (p[b] / box) * (iy1 - iy0)
        return float(px), float(py)

    for i in range(truth.shape[0]):
        tx, ty = map_pt(truth[i])
        rx, ry = map_pt(refined[i])

        # refined -> truth line
        draw.line([rx, ry, tx, ty], fill=(120,120,120), width=1)

        # local -> truth line (if local exists)
        if not np.isnan(local[i,0]):
            lx, ly = map_pt(local[i])
            draw.line([lx, ly, tx, ty], fill=(180,180,180), width=1)

        draw_marker(draw, tx, ty, kind="o")               # truth
        if not np.isnan(local[i,0]):
            draw_marker(draw, lx, ly, kind="x")           # local
        draw_marker(draw, rx, ry, kind="^")               # refined

def save_projection_figure(truth, local, refined, box, malicious_ids, out_path, title=""):
    W, H = 1200, 420
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    header = title if title else "Snapshot at final epoch"
    draw.text((10, 8), header, fill=(0,0,0), font=font)
    draw.text((10, 26), "Legend: o=Truth, x=Local, ^=Refined (lines connect estimates to truth)",
              fill=(0,0,0), font=font)
    draw.text((10, 44), f"Malicious IDs: {malicious_ids.tolist()}",
              fill=(0,0,0), font=font)

    panel_w, panel_h = 380, 340
    y0 = 70
    plot_projection_panel(draw, (10,  y0), (panel_w, panel_h), box, truth, local, refined, dims=("x","y"), title="XY")
    plot_projection_panel(draw, (410, y0), (panel_w, panel_h), box, truth, local, refined, dims=("x","z"), title="XZ")
    plot_projection_panel(draw, (810, y0), (panel_w, panel_h), box, truth, local, refined, dims=("y","z"), title="YZ")

    img.save(out_path)
    return out_path


# =========================================================
# Plot helpers with axis tick numbers (no matplotlib)
# =========================================================
def _nice_ticks(vmin, vmax, n=6):
    """Return ~n 'nice' tick values for [vmin, vmax]."""
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    raw_step = span / max(1, (n - 1))

    mag = 10 ** np.floor(np.log10(raw_step))
    steps = np.array([1, 2, 5, 10], dtype=float) * mag
    step = steps[np.argmin(np.abs(steps - raw_step))]

    start = np.floor(vmin / step) * step
    end = np.ceil(vmax / step) * step
    ticks = np.arange(start, end + 0.5*step, step)
    ticks = ticks[(ticks >= vmin - 1e-9) & (ticks <= vmax + 1e-9)]
    if len(ticks) == 0:
        return [vmin, vmax]
    return ticks.tolist()

def _draw_axes_with_ticks(draw, font, x0, y0, x1, y1, xticks, yticks, xfmt=str, yfmt=lambda v: f"{v:.1f}"):
    """Draw axes box + tick marks + tick labels."""
    draw.rectangle([x0, y0, x1, y1], outline=(0,0,0), width=2)

    # X ticks
    for xval, xpos in xticks:
        draw.line([xpos, y1, xpos, y1+6], fill=(0,0,0), width=1)
        label = xfmt(xval)
        tw, th = text_wh(draw, label, font)
        draw.text((xpos - tw/2, y1 + 8), label, fill=(0,0,0), font=font)

    # Y ticks
    for yval, ypos in yticks:
        draw.line([x0-6, ypos, x0, ypos], fill=(0,0,0), width=1)
        label = yfmt(yval)
        tw, th = text_wh(draw, label, font)
        draw.text((x0 - 10 - tw, ypos - th/2), label, fill=(0,0,0), font=font)


def save_error_plot_single(epochs, local_mean, refined_mean, out_path, title="Mean error vs epoch (single run)"):
    W, H = 1050, 460
    pad_left, pad_right, pad_top, pad_bot = 80, 40, 60, 80

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((10, 10), title, fill=(0,0,0), font=font)

    x0, y0 = pad_left, pad_top
    x1, y1 = W - pad_right, H - pad_bot

    epochs = np.asarray(epochs)
    local_mean = np.asarray(local_mean, dtype=float)
    refined_mean = np.asarray(refined_mean, dtype=float)

    vals = np.concatenate([
        local_mean[~np.isnan(local_mean)],
        refined_mean[~np.isnan(refined_mean)]
    ])
    if len(vals) == 0:
        img.save(out_path)
        return out_path

    ymin, ymax = float(vals.min()), float(vals.max())
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0
    ymin = max(0.0, ymin - 0.05*(ymax-ymin))
    ymax = ymax + 0.10*(ymax-ymin)

    def map_x(e):
        return x0 + (e - epochs[0]) / (epochs[-1] - epochs[0] + 1e-12) * (x1 - x0)

    def map_y(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    # X ticks (show at most ~6 labels)
    xtick_vals = list(range(int(epochs[0]), int(epochs[-1]) + 1))
    if len(xtick_vals) > 6:
        step = max(1, len(xtick_vals)//5)
        xtick_vals = xtick_vals[::step]
        if xtick_vals[-1] != int(epochs[-1]):
            xtick_vals.append(int(epochs[-1]))
    xticks = [(xv, map_x(xv)) for xv in xtick_vals]

    # Y ticks (nice values)
    ytick_vals = _nice_ticks(ymin, ymax, n=6)
    yticks = [(yv, map_y(yv)) for yv in ytick_vals]

    _draw_axes_with_ticks(draw, font, x0, y0, x1, y1, xticks, yticks,
                          xfmt=lambda v: f"{int(v)}", yfmt=lambda v: f"{v:.1f}")

    # Axis titles
    draw.text((x0, y1 + 40), "Epoch", fill=(0,0,0), font=font)
    draw.text((10, y0), "Error (m)", fill=(0,0,0), font=font)

    def polyline(series, color, width=3):
        prev = None
        for e, v in zip(epochs, series):
            if np.isnan(v):
                prev = None
                continue
            p = (map_x(e), map_y(v))
            if prev is not None:
                draw.line([prev, p], fill=color, width=width)
            prev = p

    polyline(local_mean,   color=(160,160,160), width=3)
    polyline(refined_mean, color=(0,0,0),       width=3)

    # Legend
    lx, ly = x1 - 320, y0 + 10
    draw.rectangle([lx, ly, x1 - 10, ly + 70], outline=(0,0,0), width=1)
    draw.line([lx + 12, ly + 22, lx + 52, ly + 22], fill=(160,160,160), width=3)
    draw.text((lx + 62, ly + 15), "Local mean (honest)", fill=(0,0,0), font=font)
    draw.line([lx + 12, ly + 48, lx + 52, ly + 48], fill=(0,0,0), width=3)
    draw.text((lx + 62, ly + 41), "Refined mean (honest)", fill=(0,0,0), font=font)

    img.save(out_path)
    return out_path


def save_error_plot_ensemble(
    epochs,
    local_runs,        # (R, T)
    refined_runs,      # (R, T)
    out_path,
    title="Error vs epoch over 100 runs (mean + 10–90% band)"
):
    epochs = np.asarray(epochs)
    local_runs = np.asarray(local_runs, dtype=float)
    refined_runs = np.asarray(refined_runs, dtype=float)

    local_mean = np.nanmean(local_runs, axis=0)
    ref_mean   = np.nanmean(refined_runs, axis=0)

    local_p10 = np.nanpercentile(local_runs, 10, axis=0)
    local_p90 = np.nanpercentile(local_runs, 90, axis=0)

    ref_p10 = np.nanpercentile(refined_runs, 10, axis=0)
    ref_p90 = np.nanpercentile(refined_runs, 90, axis=0)

    W, H = 1150, 480
    pad_left, pad_right, pad_top, pad_bot = 80, 40, 70, 85

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((10, 10), title, fill=(0,0,0), font=font)
    draw.text((10, 30), "Bands: 10–90 percentile across runs | Lines: mean", fill=(0,0,0), font=font)

    x0, y0 = pad_left, pad_top
    x1, y1 = W - pad_right, H - pad_bot

    vals = np.concatenate([local_runs.reshape(-1), refined_runs.reshape(-1)])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        img.save(out_path)
        return out_path

    ymin = float(vals.min())
    ymax = float(vals.max())
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0
    ymin = max(0.0, ymin - 0.05*(ymax-ymin))
    ymax = ymax + 0.10*(ymax-ymin)

    def map_x(e):
        return x0 + (e - epochs[0]) / (epochs[-1] - epochs[0] + 1e-12) * (x1 - x0)

    def map_y(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    # X ticks (at most ~6 labels)
    xtick_vals = list(range(int(epochs[0]), int(epochs[-1]) + 1))
    if len(xtick_vals) > 6:
        step = max(1, len(xtick_vals)//5)
        xtick_vals = xtick_vals[::step]
        if xtick_vals[-1] != int(epochs[-1]):
            xtick_vals.append(int(epochs[-1]))
    xticks = [(xv, map_x(xv)) for xv in xtick_vals]

    # Y ticks (nice values)
    ytick_vals = _nice_ticks(ymin, ymax, n=6)
    yticks = [(yv, map_y(yv)) for yv in ytick_vals]

    _draw_axes_with_ticks(draw, font, x0, y0, x1, y1, xticks, yticks,
                          xfmt=lambda v: f"{int(v)}", yfmt=lambda v: f"{v:.1f}")

    # Axis titles
    draw.text((x0, y1 + 40), "Epoch", fill=(0,0,0), font=font)
    draw.text((10, y0), "Error (m)", fill=(0,0,0), font=font)

    def draw_band(p10, p90, fill_color):
        top, bot = [], []
        for e, lo, hi in zip(epochs, p10, p90):
            if np.isnan(lo) or np.isnan(hi):
                continue
            top.append((map_x(e), map_y(hi)))
            bot.append((map_x(e), map_y(lo)))
        if len(top) >= 2 and len(bot) >= 2:
            draw.polygon(top + bot[::-1], fill=fill_color)

    def draw_line(series, color, width=3):
        prev = None
        for e, v in zip(epochs, series):
            if np.isnan(v):
                prev = None
                continue
            p = (map_x(e), map_y(v))
            if prev is not None:
                draw.line([prev, p], fill=color, width=width)
            prev = p

    # Bands (local lighter, refined darker)
    draw_band(local_p10, local_p90, fill_color=(230,230,230))
    draw_band(ref_p10,   ref_p90,   fill_color=(200,200,200))

    # Mean lines
    draw_line(local_mean, color=(120,120,120), width=3)
    draw_line(ref_mean,   color=(0,0,0),       width=3)

    # Legend
    lx, ly = x1 - 380, y0 + 10
    draw.rectangle([lx, ly, x1-10, ly+95], outline=(0,0,0), width=1)
    draw.rectangle([lx+12, ly+12, lx+42, ly+28], fill=(230,230,230), outline=(0,0,0))
    draw.text((lx+52, ly+10), "Local band (10–90%)", fill=(0,0,0), font=font)
    draw.rectangle([lx+12, ly+38, lx+42, ly+54], fill=(200,200,200), outline=(0,0,0))
    draw.text((lx+52, ly+36), "Refined band (10–90%)", fill=(0,0,0), font=font)
    draw.line([lx+12, ly+72, lx+42, ly+72], fill=(120,120,120), width=3)
    draw.text((lx+52, ly+64), "Local mean", fill=(0,0,0), font=font)
    draw.line([lx+12, ly+88, lx+42, ly+88], fill=(0,0,0), width=3)
    draw.text((lx+52, ly+80), "Refined mean", fill=(0,0,0), font=font)

    img.save(out_path)
    return out_path


# =========================================================
# Simulation (returns mean error vs epoch)
# =========================================================
def simulate(
    seed: int,
    N=10,
    T=30,
    box=50.0,
    dt=1.0,
    comm_radius=20.0,
    neighbor_budget=6,
    cold_start_steps=10,
    loc_loss_prob=0.05,
    malicious_frac=0.2,
    spoof_offset=20.0,
    base_range_sigma=0.5,
    range_sigma_per_m=0.02,
    trust_lambda=2.0,
    trust_ema_eta=0.8,
    trust_min=0.2,
    print_table=False
):
    rng = np.random.default_rng(seed)

    # True state
    true_p = rng.uniform(0, box, size=(N, 3))
    true_v = np.clip(rng.normal(0, 1.0, size=(N, 3)), -2.0, 2.0)

    # Heterogeneous local noise per drone
    local_sigma_base = rng.uniform(0.8, 4.0, size=N)

    # Refined init
    refined_p = true_p + rng.normal(0, local_sigma_base[:, None], size=(N, 3))
    refined_cov = np.array([np.diag([s*s, s*s, (1.5*s)*(1.5*s)]) for s in local_sigma_base])

    # Trust memory
    trust_ema = np.ones((N, N), dtype=float) * 0.8
    np.fill_diagonal(trust_ema, 1.0)

    # Malicious selection
    M = int(round(malicious_frac * N))
    malicious_ids = rng.choice(N, size=M, replace=False) if M > 0 else np.array([], dtype=int)
    malicious = np.zeros(N, dtype=bool)
    malicious[malicious_ids] = True
    honest = ~malicious

    epochs = np.arange(T)
    mean_local_err = np.full(T, np.nan, dtype=float)
    mean_ref_err   = np.full(T, np.nan, dtype=float)

    truth_final = None
    local_final = None
    refined_final = None

    if print_table:
        print(f"\n=== Simulation seed={seed} | N={N} | box={box} | epochs={T} ===")
        print(f"Malicious IDs: {malicious_ids.tolist()}")
        print("Epoch | Mean Local Err (honest) | Mean Refined Err (honest)")
        print("----- | ---------------------- | ------------------------")

    for t in range(T):
        # Move truth
        true_v += rng.normal(0, 0.2, size=(N, 3))
        true_v = np.clip(true_v, -3.0, 3.0)
        true_p += true_v * dt
        for i in range(N):
            true_p[i], true_v[i] = clamp_reflect(true_p[i], true_v[i], 0.0, box)

        # Local estimate (may be missing)
        local_p = np.empty((N, 3), dtype=float)
        local_cov = np.empty((N, 3, 3), dtype=float)
        local_available = np.ones(N, dtype=bool)

        for i in range(N):
            cold = (t < cold_start_steps)
            loss = (rng.random() < loc_loss_prob) and (t >= cold_start_steps)

            if loss:
                local_available[i] = False
                local_p[i] = np.nan
                local_cov[i] = np.eye(3) * 1e6
                continue

            s = local_sigma_base[i] * (4.0 if cold else 1.0)
            local_p[i] = true_p[i] + rng.normal(0, [s, s, 1.5*s], size=3)
            local_cov[i] = np.diag([s*s, s*s, (1.5*s)*(1.5*s)])

        # Neighbors by true distance
        dist_true = np.linalg.norm(true_p[:, None, :] - true_p[None, :, :], axis=2)

        neighbor_sets = []
        link_quality = np.zeros((N, N), dtype=float)
        for i in range(N):
            neigh = [j for j in range(N) if j != i and dist_true[i, j] <= comm_radius]
            for j in neigh:
                link_quality[i, j] = max(0.0, 1.0 - dist_true[i, j] / comm_radius)
            neigh_sorted = sorted(neigh, key=lambda j: link_quality[i, j], reverse=True)
            neighbor_sets.append(neigh_sorted[:neighbor_budget])

        # Broadcast (spoof malicious)
        broadcast_p = refined_p.copy()
        for j in malicious_ids:
            broadcast_p[j] = true_p[j] + rng.uniform(-spoof_offset, spoof_offset, size=3)

        # Range measurements
        range_meas = {i: {} for i in range(N)}
        range_sigma = {i: {} for i in range(N)}
        for i in range(N):
            for j in neighbor_sets[i]:
                d = dist_true[i, j]
                sd = base_range_sigma + range_sigma_per_m * d
                range_sigma[i][j] = sd
                range_meas[i][j] = d + rng.normal(0, sd)

        # Refinement
        new_refined_p = refined_p.copy()
        new_refined_cov = refined_cov.copy()

        for i in range(N):
            neigh = neighbor_sets[i]
            if len(neigh) == 0:
                if local_available[i]:
                    new_refined_p[i] = local_p[i]
                    new_refined_cov[i] = local_cov[i]
                continue

            pref = local_p[i] if local_available[i] else refined_p[i]

            omegas, neighbor_positions, ranges, sigmas = [], [], [], []
            for j in neigh:
                d_hat = range_meas[i][j]
                sd = range_sigma[i][j]
                pj = broadcast_p[j]

                eps_ij = abs(np.linalg.norm(pref - pj) - d_hat) / (sd + 1e-12)
                s_inst = np.exp(-(eps_ij**2) / (2.0 * trust_lambda**2))
                trust_ema[i, j] = trust_ema_eta * trust_ema[i, j] + (1 - trust_ema_eta) * s_inst

                w = link_quality[i, j] * trust_ema[i, j]
                if trust_ema[i, j] < trust_min:
                    w = 0.0

                neighbor_positions.append(pj)
                ranges.append(d_hat)
                sigmas.append(sd)
                omegas.append(w)

            neighbor_positions = np.array(neighbor_positions)
            ranges = np.array(ranges)
            sigmas = np.array(sigmas)
            omegas = np.array(omegas)

            # Cold start / loss bootstrapping
            if local_available[i]:
                prior_p = local_p[i]
                prior_cov = local_cov[i].copy()
                if t < cold_start_steps:
                    prior_cov *= 50.0
            else:
                prior_p = refined_p[i]
                prior_cov = refined_cov[i] * 200.0

            p_hat, cov_hat = gauss_newton_refine(prior_p, prior_cov, neighbor_positions, ranges, sigmas, omegas)
            new_refined_p[i] = p_hat
            new_refined_cov[i] = cov_hat

        refined_p = new_refined_p
        refined_cov = new_refined_cov

        # Errors (honest drones)
        local_errs = []
        ref_errs = []
        for i in range(N):
            if honest[i]:
                ref_errs.append(np.linalg.norm(refined_p[i] - true_p[i]))
                if local_available[i]:
                    local_errs.append(np.linalg.norm(local_p[i] - true_p[i]))

        mean_local_err[t] = float(np.mean(local_errs)) if len(local_errs) else np.nan
        mean_ref_err[t]   = float(np.mean(ref_errs))   if len(ref_errs)   else np.nan

        if print_table:
            print(f"{t:5d} | {mean_local_err[t]:22.4f} | {mean_ref_err[t]:24.4f}")

        if t == T - 1:
            truth_final = true_p.copy()
            local_final = local_p.copy()
            refined_final = refined_p.copy()

    return {
        "seed": seed,
        "epochs": epochs,
        "mean_local_err": mean_local_err,
        "mean_ref_err": mean_ref_err,
        "truth_final": truth_final,
        "local_final": local_final,
        "refined_final": refined_final,
        "malicious_ids": malicious_ids,
        "box": box
    }


# =========================================================
# Runners
# =========================================================
def run_one_random(out_dir="outputs", T=30):
    os.makedirs(out_dir, exist_ok=True)

    seed = int(np.random.default_rng().integers(0, 1_000_000_000))
    res = simulate(seed=seed, T=T, print_table=True)

    err_path = os.path.join(out_dir, f"error_vs_epoch_seed_{seed}.png")
    snap_path = os.path.join(out_dir, f"snapshot_projections_seed_{seed}.png")

    save_error_plot_single(
        res["epochs"], res["mean_local_err"], res["mean_ref_err"],
        err_path, title=f"Mean error vs epoch (single run, seed={seed})"
    )
    save_projection_figure(
        res["truth_final"], res["local_final"], res["refined_final"],
        res["box"], res["malicious_ids"], snap_path,
        title=f"Final snapshot projections (seed={seed})"
    )

    print(f"\nSaved: {err_path}")
    print(f"Saved: {snap_path}\n")
    return res

def run_ensemble(num_runs=100, out_dir="outputs", T=30, master_seed=None):
    """
    Runs `num_runs` simulations with random seeds and makes ONE ensemble plot:
    epoch vs error (mean + 10–90% band) across runs.
    """
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 1_000_000_000, size=num_runs)

    local_runs = np.zeros((num_runs, T), dtype=float)
    refined_runs = np.zeros((num_runs, T), dtype=float)

    for r, s in enumerate(seeds):
        res = simulate(seed=int(s), T=T, print_table=False)
        local_runs[r, :] = res["mean_local_err"]
        refined_runs[r, :] = res["mean_ref_err"]

    out_path = os.path.join(out_dir, f"error_vs_epoch_{num_runs}runs.png")
    save_error_plot_ensemble(
        epochs=np.arange(T),
        local_runs=local_runs,
        refined_runs=refined_runs,
        out_path=out_path,
        title=f"Epoch vs error over {num_runs} runs (mean + 10–90% band)"
    )

    final_local = float(np.nanmean(local_runs[:, -1]))
    final_ref = float(np.nanmean(refined_runs[:, -1]))
    print(f"[{num_runs} runs] Final-epoch mean local error  : {final_local:.4f} m")
    print(f"[{num_runs} runs] Final-epoch mean refined error: {final_ref:.4f} m")
    print(f"Saved ensemble plot: {out_path}\n")

    return out_path


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    OUT_DIR = "outputs"
    EPOCHS = 30

    # 1) One random run (prints table + saves single-run plots)
    run_one_random(out_dir=OUT_DIR, T=EPOCHS)

    # 2) Ensemble plot over 100 runs with random seeds
    # master_seed=None => different seeds every time you run
    run_ensemble(num_runs=100, out_dir=OUT_DIR, T=EPOCHS, master_seed=None)

    # If you want the SAME 100 seeds every time, set master_seed:
    # run_ensemble(num_runs=100, out_dir=OUT_DIR, T=EPOCHS, master_seed=12345)
