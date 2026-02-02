import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Core simulation utilities
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
# Scenario 4: simulate final-epoch error for one seed
# =========================================================
def simulate_final_error(
    seed: int,
    malicious_frac: float,
    use_trust: bool,
    N=10,
    T=30,
    box=50.0,
    dt=1.0,
    comm_radius=20.0,
    neighbor_budget=6,
    cold_start_steps=10,
    loc_loss_prob=0.05,
    spoof_offset=20.0,
    base_range_sigma=0.5,
    range_sigma_per_m=0.02,
    trust_lambda=2.0,
    trust_ema_eta=0.8,
    trust_min=0.2,
):
    rng = np.random.default_rng(seed)

    # True state
    true_p = rng.uniform(0, box, size=(N, 3))
    true_v = np.clip(rng.normal(0, 1.0, size=(N, 3)), -2.0, 2.0)

    # Heterogeneous local noise
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

        # Neighbors by true distance (simulation convenience)
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

                if use_trust:
                    eps_ij = abs(np.linalg.norm(pref - pj) - d_hat) / (sd + 1e-12)
                    s_inst = np.exp(-(eps_ij**2) / (2.0 * trust_lambda**2))
                    trust_ema[i, j] = trust_ema_eta * trust_ema[i, j] + (1 - trust_ema_eta) * s_inst
                    trust_val = trust_ema[i, j]
                    if trust_val < trust_min:
                        trust_val = 0.0
                else:
                    trust_val = 1.0  # baseline

                w = link_quality[i, j] * trust_val

                neighbor_positions.append(pj)
                ranges.append(d_hat)
                sigmas.append(sd)
                omegas.append(w)

            neighbor_positions = np.array(neighbor_positions)
            ranges = np.array(ranges)
            sigmas = np.array(sigmas)
            omegas = np.array(omegas)

            # Bootstrapping
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

    # Final-epoch mean error over honest drones
    errs = [np.linalg.norm(refined_p[i] - true_p[i]) for i in range(N) if honest[i]]
    return float(np.mean(errs)) if len(errs) else float("nan")


# =========================================================
# Scenario 4: run sweep + plot with matplotlib
# =========================================================
def run_scenario4(
    out_dir="outputs",
    num_runs_per_point=100,
    fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    T=30,
    master_seed=None,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(master_seed)

    fractions = np.array(list(fractions), dtype=float)
    R = int(num_runs_per_point)

    trust_errors = np.zeros((len(fractions), R), dtype=float)
    notrust_errors = np.zeros((len(fractions), R), dtype=float)

    print("\nScenario 4: Final-epoch honest error vs malicious fraction")
    print(f"Runs per point: {R} | epochs={T}")
    print("rho(%) | mean(no-trust) | mean(trust) | p10..p90(trust)")

    for i, rho in enumerate(fractions):
        seeds = rng.integers(0, 1_000_000_000, size=R)
        for k, s in enumerate(seeds):
            trust_errors[i, k] = simulate_final_error(int(s), rho, use_trust=True,  T=T)
            notrust_errors[i, k] = simulate_final_error(int(s), rho, use_trust=False, T=T)

        print(f"{int(round(100*rho)):>5d} |"
              f" {np.mean(notrust_errors[i]):>13.3f} |"
              f" {np.mean(trust_errors[i]):>10.3f} |"
              f" {np.percentile(trust_errors[i],10):.3f}..{np.percentile(trust_errors[i],90):.3f}")

    # Save raw results for later
    np.savez(
        os.path.join(out_dir, "scenario4_results.npz"),
        fractions=fractions,
        trust_errors=trust_errors,
        notrust_errors=notrust_errors,
        num_runs_per_point=R,
        T=T,
        master_seed=master_seed if master_seed is not None else -1,
    )

    # ---- Plot 1: mean + 10–90 band ----
    x = fractions * 100.0

    trust_mean = trust_errors.mean(axis=1)
    trust_p10 = np.percentile(trust_errors, 10, axis=1)
    trust_p90 = np.percentile(trust_errors, 90, axis=1)

    not_mean = notrust_errors.mean(axis=1)
    not_p10 = np.percentile(notrust_errors, 10, axis=1)
    not_p90 = np.percentile(notrust_errors, 90, axis=1)

    plt.figure(figsize=(9, 5.2))
    plt.grid(True, which="both", linewidth=0.5, alpha=0.6)

    plt.fill_between(x, not_p10, not_p90, alpha=0.25, label="No trust (10–90%)")
    plt.fill_between(x, trust_p10, trust_p90, alpha=0.35, label="With trust (10–90%)")

    plt.plot(x, not_mean, marker="o", linewidth=2.5, label="No trust (mean)")
    plt.plot(x, trust_mean, marker="o", linewidth=2.5, label="With trust (mean)")

    plt.xlabel("Malicious nodes (%)")
    plt.ylabel("Final-epoch mean error of honest UAVs (m)")
    plt.title("Scenario 4: Robustness vs % Malicious Nodes")
    plt.legend()
    out1 = os.path.join(out_dir, "scenario4_mean_band.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    # ---- Plot 2: grouped boxplots ----
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.grid(True, which="both", linewidth=0.5, alpha=0.6)

    pos = np.arange(len(x))
    width = 0.32

    ax.boxplot(
        notrust_errors.T,
        positions=pos - width/2,
        widths=0.28,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(alpha=0.4),
        medianprops=dict(linewidth=2),
    )
    ax.boxplot(
        trust_errors.T,
        positions=pos + width/2,
        widths=0.28,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(alpha=0.7),
        medianprops=dict(linewidth=2),
    )

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{int(v)}%" for v in x])
    ax.set_xlabel("Malicious nodes (%)")
    ax.set_ylabel("Final-epoch mean error of honest UAVs (m)")
    ax.set_title("Scenario 4: Distribution over seeds (boxplots)")

    # manual legend
    ax.plot([], [], color="C0", label="No trust")
    ax.plot([], [], color="C1", label="With trust")
    ax.legend()

    out2 = os.path.join(out_dir, "scenario4_boxplots.png")
    fig.tight_layout()
    fig.savefig(out2, dpi=200)
    plt.close(fig)

    print(f"\nSaved plots:\n  {out1}\n  {out2}")
    print(f"Saved raw results:\n  {os.path.join(out_dir, 'scenario4_results.npz')}\n")
    return out1, out2


if __name__ == "__main__":
    run_scenario4(
        out_dir="outputs",
        num_runs_per_point=100,
        fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        T=30,
        master_seed=None,     # set e.g. 12345 for repeatable seeds
    )
