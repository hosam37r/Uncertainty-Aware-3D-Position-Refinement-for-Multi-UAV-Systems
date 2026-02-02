import numpy as np

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
    return cov + eps * np.eye(cov.shape[0])

def gauss_newton_refine(prior_p, prior_cov, neighbor_positions, ranges, range_sigmas, omegas,
                        max_iter=6, damping=1e-3, tol=1e-4):
    """
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

def time_to_recovery_from(series, start_epoch, eps=5.0, W=3):
    """
    First epoch t >= start_epoch such that series[t:t+W] are all <= eps.
    Returns None if never achieved.
    """
    series = np.asarray(series, dtype=float)
    T = len(series)
    for t in range(start_epoch, T - W + 1):
        window = series[t:t+W]
        if np.all(np.isfinite(window)) and np.all(window <= eps):
            return t
    return None


# =========================================================
# One cold-start run
# =========================================================
def simulate_cold_start_run(
    seed: int,
    # world
    N=10,
    T=30,
    box=50.0,
    dt=1.0,
    # comm
    comm_radius=20.0,
    neighbor_budget=6,
    # cold-start cohort
    K=4,
    T_cs=10,                  # epochs 0..T_cs-1 are cold-start for selected UAVs
    cold_mode="missing",      # "missing" or "weak"
    cold_inflate=80.0,        # used for "weak" (inflates std and/or cov)
    # optional random loss for warm nodes (keep 0 for clean cold-start)
    loc_loss_prob=0.0,
    # malicious (keep 0 for cold-start-only)
    malicious_frac=0.0,
    spoof_offset=20.0,
    # range noise
    base_range_sigma=0.5,
    range_sigma_per_m=0.02,
    # trust
    trust_lambda=2.0,
    trust_ema_eta=0.8,
    trust_min=0.2,
    # bootstrapping when local missing
    loss_prior_inflate=200.0,
):
    rng = np.random.default_rng(seed)

    # True state
    true_p = rng.uniform(0, box, size=(N, 3))
    true_v = np.clip(rng.normal(0, 1.0, size=(N, 3)), -2.0, 2.0)

    # Heterogeneous local noise per drone
    local_sigma_base = rng.uniform(0.8, 4.0, size=N)

    # Refined state init (for refined method)
    refined_p = true_p + rng.normal(0, local_sigma_base[:, None], size=(N, 3))
    refined_cov = np.array([np.diag([s*s, s*s, (1.5*s)*(1.5*s)]) for s in local_sigma_base])

    # Trust memory
    trust_ema = np.ones((N, N), dtype=float) * 0.8
    np.fill_diagonal(trust_ema, 1.0)

    # Choose cold-start cohort
    cold_ids = rng.choice(N, size=min(K, N), replace=False)

    # Choose malicious (optional; default 0)
    M = int(round(malicious_frac * N))
    malicious_ids = rng.choice([i for i in range(N) if i not in cold_ids], size=M, replace=False) if M > 0 else np.array([], dtype=int)
    malicious = np.zeros(N, dtype=bool)
    malicious[malicious_ids] = True
    honest = ~malicious

    # Baseline state (local-only): hold last local; initial baseline is a noisy estimate
    baseline_p = true_p + rng.normal(0, local_sigma_base[:, None], size=(N, 3))

    # Track mean error over cold cohort (honest only)
    baseline_cold_err = np.zeros(T, dtype=float)
    refined_cold_err = np.zeros(T, dtype=float)

    for t in range(T):
        # Move truth
        true_v += rng.normal(0, 0.2, size=(N, 3))
        true_v = np.clip(true_v, -3.0, 3.0)
        true_p += true_v * dt
        for i in range(N):
            true_p[i], true_v[i] = clamp_reflect(true_p[i], true_v[i], 0.0, box)

        # Local estimates (may be missing)
        local_p = np.empty((N, 3), dtype=float)
        local_cov = np.empty((N, 3, 3), dtype=float)
        local_available = np.ones(N, dtype=bool)

        for i in range(N):
            is_cold = (i in cold_ids) and (t < T_cs)
            warm_loss = (rng.random() < loc_loss_prob) and (not is_cold)

            if is_cold and cold_mode == "missing":
                local_available[i] = False
                local_p[i] = np.nan
                local_cov[i] = np.eye(3) * 1e6
                continue

            if warm_loss:
                local_available[i] = False
                local_p[i] = np.nan
                local_cov[i] = np.eye(3) * 1e6
                continue

            # noise scale
            s = local_sigma_base[i]
            if is_cold and cold_mode == "weak":
                # inflate std dev (consistent with covariance inflation)
                s = s * np.sqrt(cold_inflate)

            local_p[i] = true_p[i] + rng.normal(0, [s, s, 1.5*s], size=3)
            local_cov[i] = np.diag([s*s, s*s, (1.5*s)*(1.5*s)])

        # ----- Baseline update (local-only) -----
        for i in range(N):
            if local_available[i]:
                baseline_p[i] = local_p[i]
            # else: hold baseline_p[i] (no update)

        # ----- Neighbor discovery (true distance for simulation convenience) -----
        dist_true = np.linalg.norm(true_p[:, None, :] - true_p[None, :, :], axis=2)
        neighbor_sets = []
        link_quality = np.zeros((N, N), dtype=float)

        for i in range(N):
            neigh = [j for j in range(N) if j != i and dist_true[i, j] <= comm_radius]
            for j in neigh:
                link_quality[i, j] = max(0.0, 1.0 - dist_true[i, j] / comm_radius)
            neigh_sorted = sorted(neigh, key=lambda j: link_quality[i, j], reverse=True)
            neighbor_sets.append(neigh_sorted[:neighbor_budget])

        # ----- Broadcast for refinement (spoof malicious if any) -----
        broadcast_p = refined_p.copy()
        for j in malicious_ids:
            broadcast_p[j] = true_p[j] + rng.uniform(-spoof_offset, spoof_offset, size=3)

        # ----- Range observations -----
        range_meas = {i: {} for i in range(N)}
        range_sigma = {i: {} for i in range(N)}
        for i in range(N):
            for j in neighbor_sets[i]:
                d = dist_true[i, j]
                sd = base_range_sigma + range_sigma_per_m * d
                range_sigma[i][j] = sd
                range_meas[i][j] = d + rng.normal(0, sd)

        # ----- Refinement update -----
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

                # trust update (even if no malicious, handles inconsistency)
                eps_ij = abs(np.linalg.norm(pref - pj) - d_hat) / (sd + 1e-12)
                s_inst = np.exp(-(eps_ij**2) / (2.0 * trust_lambda**2))
                trust_ema[i, j] = trust_ema_eta * trust_ema[i, j] + (1 - trust_ema_eta) * s_inst

                trust_val = trust_ema[i, j]
                if trust_val < trust_min:
                    trust_val = 0.0

                w = link_quality[i, j] * trust_val

                neighbor_positions.append(pj)
                ranges.append(d_hat)
                sigmas.append(sd)
                omegas.append(w)

            neighbor_positions = np.array(neighbor_positions)
            ranges = np.array(ranges)
            sigmas = np.array(sigmas)
            omegas = np.array(omegas)

            # bootstrapping logic
            if local_available[i]:
                prior_p = local_p[i]
                prior_cov = local_cov[i].copy()
                if (i in cold_ids) and (t < T_cs) and (cold_mode == "weak"):
                    prior_cov *= cold_inflate
            else:
                prior_p = refined_p[i]
                prior_cov = refined_cov[i] * loss_prior_inflate

            p_hat, cov_hat = gauss_newton_refine(prior_p, prior_cov, neighbor_positions, ranges, sigmas, omegas)
            new_refined_p[i] = p_hat
            new_refined_cov[i] = cov_hat

        refined_p = new_refined_p
        refined_cov = new_refined_cov

        # ----- Evaluate cold cohort (honest only) -----
        cold_honest = [i for i in cold_ids if honest[i]]

        b_err = [np.linalg.norm(baseline_p[i] - true_p[i]) for i in cold_honest]
        r_err = [np.linalg.norm(refined_p[i] - true_p[i]) for i in cold_honest]

        baseline_cold_err[t] = float(np.mean(b_err)) if b_err else np.nan
        refined_cold_err[t] = float(np.mean(r_err)) if r_err else np.nan

    return {
        "seed": seed,
        "cold_ids": cold_ids,
        "baseline_cold_err": baseline_cold_err,
        "refined_cold_err": refined_cold_err,
    }


# =========================================================
# 100-run wrapper + summary
# =========================================================
def run_cold_start_ensemble(
    num_runs=100,
    master_seed=None,
    # experiment parameters
    N=10, T=30, box=50.0,
    K=4, T_cs=10, cold_mode="missing", cold_inflate=80.0,
    eps_recover=5.0, W_recover=3,
    verbose=False
):
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 1_000_000_000, size=num_runs)

    baseline_runs = np.zeros((num_runs, T), dtype=float)
    refined_runs = np.zeros((num_runs, T), dtype=float)

    rec_base = np.full(num_runs, np.nan)
    rec_ref = np.full(num_runs, np.nan)

    for r, s in enumerate(seeds):
        out = simulate_cold_start_run(
            seed=int(s),
            N=N, T=T, box=box,
            K=K, T_cs=T_cs, cold_mode=cold_mode, cold_inflate=cold_inflate,
            malicious_frac=0.0, loc_loss_prob=0.0
        )
        baseline_runs[r] = out["baseline_cold_err"]
        refined_runs[r] = out["refined_cold_err"]

        # recovery defined AFTER cold-start ends
        rec_base[r] = time_to_recovery_from(baseline_runs[r], start_epoch=T_cs, eps=eps_recover, W=W_recover) or np.nan
        rec_ref[r]  = time_to_recovery_from(refined_runs[r],  start_epoch=T_cs, eps=eps_recover, W=W_recover) or np.nan

        if verbose and (r < 5):
            print(f"[run {r}] seed={int(s)} baseline_cold_mean={np.nanmean(baseline_runs[r,:T_cs]):.3f} "
                  f"refined_cold_mean={np.nanmean(refined_runs[r,:T_cs]):.3f}")

    # ---- Aggregate metrics ----
    cold_slice = slice(0, T_cs)
    post_slice = slice(T_cs, T)

    base_cold_mean = np.nanmean(baseline_runs[:, cold_slice], axis=1)
    ref_cold_mean  = np.nanmean(refined_runs[:, cold_slice], axis=1)

    base_post_mean = np.nanmean(baseline_runs[:, post_slice], axis=1)
    ref_post_mean  = np.nanmean(refined_runs[:, post_slice], axis=1)

    # how often refined is better (lower mean error)
    pct_better_cold = 100.0 * np.mean(ref_cold_mean < base_cold_mean)
    pct_better_post = 100.0 * np.mean(ref_post_mean < base_post_mean)

    # recovery stats (ignoring NaN)
    def nan_summary(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return {"count": 0}
        return {
            "count": len(x),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)),
        }

    rec_base_stats = nan_summary(rec_base)
    rec_ref_stats  = nan_summary(rec_ref)

    # ---- Print summary ----
    print("\n=== Cold-start ensemble summary ===")
    print(f"Runs: {num_runs} | N={N} | T={T} | box={box}")
    print(f"Cold cohort: K={K} | cold window T_cs={T_cs} | mode={cold_mode}")
    print(f"Recovery criterion: error <= {eps_recover} m for W={W_recover} consecutive epochs, starting at epoch {T_cs}")
    print("")
    print("Mean error over cold-start UAVs (averaged per run):")
    print(f"  Cold window 0..{T_cs-1}:")
    print(f"    baseline: mean={np.nanmean(base_cold_mean):.3f}  p10={np.nanpercentile(base_cold_mean,10):.3f}  p90={np.nanpercentile(base_cold_mean,90):.3f}")
    print(f"    refined : mean={np.nanmean(ref_cold_mean):.3f}  p10={np.nanpercentile(ref_cold_mean,10):.3f}  p90={np.nanpercentile(ref_cold_mean,90):.3f}")
    print(f"    refined better in {pct_better_cold:.1f}% of runs")
    print(f"  Post window {T_cs}..{T-1}:")
    print(f"    baseline: mean={np.nanmean(base_post_mean):.3f}  p10={np.nanpercentile(base_post_mean,10):.3f}  p90={np.nanpercentile(base_post_mean,90):.3f}")
    print(f"    refined : mean={np.nanmean(ref_post_mean):.3f}  p10={np.nanpercentile(ref_post_mean,10):.3f}  p90={np.nanpercentile(ref_post_mean,90):.3f}")
    print(f"    refined better in {pct_better_post:.1f}% of runs")
    print("")
    print("Time-to-recovery statistics (epoch index, starting from T_cs):")
    print(f"  baseline: {rec_base_stats}")
    print(f"  refined : {rec_ref_stats}")
    print("")

    return {
        "baseline_runs": baseline_runs,
        "refined_runs": refined_runs,
        "rec_base": rec_base,
        "rec_ref": rec_ref,
        "seeds": seeds
    }


if __name__ == "__main__":
    # master_seed=None => different 100 seeds every time you run
    run_cold_start_ensemble(
        num_runs=100,
        master_seed=None,
        N=10, T=30, box=50.0,
        K=4, T_cs=10,
        cold_mode="missing",      # try "weak" too
        cold_inflate=80.0,        # only used for "weak"
        eps_recover=5.0, W_recover=3,
        verbose=False
    )

    # For reproducible 100 runs:
    # run_cold_start_ensemble(num_runs=100, master_seed=12345, K=4, T_cs=10, cold_mode="missing")
