#!/usr/bin/env python3
"""EP05 "10,000 Double Pendulums": how fast does a tiny difference become chaos?

usage: python run.py [stage ...] [--force]
  validate  V1 energy, V2 small-swing normal modes vs (2 -+ sqrt 2) g/l, V3 step size and DOP853 vs our RK4,
            V4 Lyapunov exponent two ways                                              -> results/stages/validate.json
  fan       E1: 10,000 pendulums from 120/120 deg, lower arms spread over 0.0001 deg, 60 s (+ determinism rerun)
                                                                                         -> results/stages/fan.json
  dial      E2: pairs differing by 1e-1 ... 1e-15 deg (hero start + 1,000 random starts at the same energy);
            film mini-fans                                                               -> results/stages/dial.json
  energy    E3: release-angle dial (fans of 2,000) and Lyapunov exponent vs energy      -> results/stages/energy.json
  flipmap   E4 CPU part: float64 flip times on a 270 x 270 subsample of the film's 1080 x 1080 GPU grid
                                                                                         -> results/stages/flipmap.json
  friction  E5: the hero fan with a simple friction term (limits card)                   -> results/stages/friction.json
  gpucheck  V5: GPU float32 (gpu/pend.js) vs CPU float64; needs results/gpu/probe.jsonl (see README)
                                                                                         -> results/stages/gpucheck.json
  results   RESULTS.json, RESULTS.md, run_meta.json
  all       everything above except gpucheck, in order (finished stages are skipped; --force recomputes)
Units: SI (m, s, rad); energies in m g l = 9.81 J, measured from the lowest state. Set GIT_COMMIT=<sha> outside a git checkout.
"""
import json, math, os, sys, time, platform, hashlib, subprocess, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = pathlib.Path(os.environ.get("DP_RESULTS", HERE / "results"))
ST, CACHE, FILM, GPU = RES / "stages", RES / "cache", RES / "film", RES / "gpu"
for d in (ST, CACHE, FILM, GPU):
    d.mkdir(parents=True, exist_ok=True)
DP = HERE / "dp" / "dp"
FORCE = "--force" in sys.argv
DEG = math.pi / 180
G, L = 9.81, 1.0
DT = 1 / 2400            # s, fans and pairs
DT_L = 1 / 1200          # s, Lyapunov sweep, flip map (same as the GPU)
FPS = 60
HERO = (120.0, 120.0)    # release angles (deg), from rest
SPREAD = 1e-4            # deg, total spread of the hero fan (first to last pendulum)
N_FAN = 10_000
T_FAN = 60.0
SPLIT_M = 0.20           # m: "split" = a lower weight 20 cm from pendulum 0's
SEED_DIAL = 5_000_201
SEED_LYAP = 5_000_301    # + energy index
FRICTION_C = 0.02        # 1/s, E5 only (energy e-folds in about 1/(2c) = 25 s)


def dump(path, obj):
    path = pathlib.Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float)); os.replace(tmp, path)


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def done(name):
    return (ST / f"{name}.json").exists() and not FORCE


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dp(*args, env=None):
    if not DP.exists():
        subprocess.run(["clang", "-O2", "-fblocks", "-o", str(DP), str(HERE / "dp/dp.c"), "-lm"], check=True)
    t0 = time.time()
    subprocess.run([str(DP), *map(str, args)], check=True, capture_output=True, env={**os.environ, **(env or {})})
    return time.time() - t0


def energy0(t1, t2):
    """E/(m g l) of a release from rest, zero at the bottom"""
    return 3 - 2 * np.cos(t1) - np.cos(t2)


def bob(th):
    """lower weight position (m) from angles [..., 2]"""
    return L * np.sin(th[..., 0]) + L * np.sin(th[..., 1]), -L * np.cos(th[..., 0]) - L * np.cos(th[..., 1])


def fan_states(a1, a2, n, spread_deg):
    s = np.zeros((n, 4)); s[:, 0] = a1 * DEG; s[:, 1] = a2 * DEG + spread_deg * DEG * np.arange(n) / (n - 1)
    return s


def run_traj(states, T, dt, fps, tag):
    n = len(states); inp = CACHE / f"{tag}_in.f64"; states.astype(np.float64).tofile(inp)
    every = int(round(1 / (fps * dt)))
    secs = dp("traj", inp, n, T, dt, every, CACHE / f"{tag}.f64", CACHE / f"{tag}_E.f64")
    th = np.fromfile(CACHE / f"{tag}.f64").reshape(-1, n, 2)
    return th, np.fromfile(CACHE / f"{tag}_E.f64"), secs


def wrap16(th):
    """angles -> uint16 (2 pi / 65536 = 9.6e-5 rad steps; < 0.2 mm at the lower weight)"""
    return np.round((np.mod(th, 2 * np.pi)) / (2 * np.pi) * 65536).astype(np.int64).astype(np.uint16)


def random_releases(E, n, rng):
    """n releases from rest with energy E (m g l), theta1 uniform over the allowed range, sign of theta2 random"""
    t1, t2 = [], []
    while len(t1) < n:
        a = rng.uniform(-np.pi, np.pi); c = 3 - 2 * np.cos(a) - E
        if abs(c) <= 1:
            t1.append(a); t2.append(np.arccos(c) * rng.choice([-1, 1]))
    s = np.zeros((n, 4)); s[:, 0] = t1; s[:, 1] = t2
    return s


def run_pairs(a, b, T, dt, tag):
    n = len(a); s = np.stack([a, b], 1).reshape(-1, 4); inp = CACHE / f"{tag}_in.f64"; s.tofile(inp)
    dp("pairs", inp, n, T, dt, CACHE / f"{tag}.f64")
    W = 5 + int(round(T / 0.05)) + 1
    o = np.fromfile(CACHE / f"{tag}.f64").reshape(n, W)
    return o[:, :5], o[:, 5:]      # first-crossing times for 1, 10, 20, 50, 100 cm; log10 distance every 0.05 s


THR_IDX = {0.01: 0, 0.1: 1, 0.2: 2, 0.5: 3, 1.0: 4}


def run_lyap(states, T, dt, d0, tau, tag):
    inp = CACHE / f"{tag}_in.f64"; states.astype(np.float64).tofile(inp)
    dp("lyap", inp, len(states), T, dt, d0, tau, CACHE / f"{tag}.f64")
    return np.fromfile(CACHE / f"{tag}.f64").reshape(-1, 3)


# ----------------------------------------------------------------------------------------------- stages
def stage_fan():
    if done("fan"): return
    log("fan: 10,000 pendulums, 60 s")
    s = fan_states(*HERO, N_FAN, SPREAD)
    th, ed, secs = run_traj(s, T_FAN, DT, FPS, "fan")
    E0 = float(energy0(*(np.array(HERO) * DEG)))
    x, y = bob(th)
    d = np.hypot(x - x[:, :1], y - y[:, :1])
    dmax = d.max(1); dmed = np.median(d, 1)
    t = np.arange(len(th)) / FPS
    split_i = int(np.argmax(dmax > SPLIT_M)); split_t = float(t[split_i])
    # interpolate the crossing inside the frame for a cleaner number (log-linear)
    if split_i > 0:
        a0, a1 = math.log(dmax[split_i - 1]), math.log(dmax[split_i]); f = (math.log(SPLIT_M) - a0) / (a1 - a0); split_t = float(t[split_i - 1] + f / FPS)
    # coverage: share of 4 cm cells inside the 2 m disk visited by any lower weight in a frame, relative to all cells ever visited (60 s)
    cell = 0.04; nb = int(4.0 / cell)
    ix = np.clip(((x + 2) / cell).astype(int), 0, nb - 1); iy = np.clip(((y + 2) / cell).astype(int), 0, nb - 1)
    ever = np.zeros((nb, nb), bool); ever[ix.ravel(), iy.ravel()] = True
    cov = np.array([np.unique(ix[f] * nb + iy[f]).size for f in range(len(th))]) / ever.sum()
    # the cloud's median distance from pendulum 0 after the split (mean over 20-60 s), and the bob angle distribution
    late = (t >= 20)
    phi = np.arctan2(x, -y)
    # growth rate of the spread: fit log10(max distance) vs t over the clean exponential part (1e-5 ... 1e-1 m)
    m = (dmax > 1e-5) & (dmax < 1e-1) & (t > 1)
    k_dec, b_dec = np.polyfit(t[m], np.log10(dmax[m]), 1)
    # times the spread first passes 1 mm, 1 cm, 10 cm
    pas = {str(v): float(t[np.argmax(dmax > v)]) for v in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)}
    # deltas relative to pendulum 0 for the magnified view (first 12 s): float32 (th - th0)
    dl = (th[: int(12 * FPS) + 1] - th[: int(12 * FPS) + 1, :1]).astype(np.float32)
    dl.tofile(FILM / "fan_delta.f32")
    wrap16(th).tofile(FILM / "fan.u16")
    h1 = sha(CACHE / "fan.f64")
    # determinism: the same run again must be bit-identical
    th2, _, _ = run_traj(s, T_FAN, DT, FPS, "fan_rerun")
    h2 = sha(CACHE / "fan_rerun.f64"); del th2
    (CACHE / "fan_rerun.f64").unlink()
    # 600 s energy check on pendulum 0 (V1)
    _, ed600, _ = run_traj(s[:1], 600.0, DT, 1, "fan600")
    out = dict(n=N_FAN, release_deg=HERO, spread_deg=SPREAD, E_mgl=E0, dt=DT, T=T_FAN, fps=FPS, cpu_seconds=secs,
               split_m=SPLIT_M, split_t=split_t, spread_passes=pas, decades_per_s=float(k_dec), s_per_decade=float(1 / k_dec),
               spread_max_m=dmax.tolist()[::6], spread_med_m=dmed.tolist()[::6], spread_t=t.tolist()[::6],
               coverage=cov.tolist()[::6], coverage_at={str(s_): float(cov[int(s_ * FPS)]) for s_ in (5, 10, 11, 12, 13, 15, 20, 30, 60)},
               late_median_dist_m=float(np.median(d[late])), energy_err_max_mgl=float(ed.max()), energy_err_rel=float(ed.max() / E0),
               energy_err_600s_mgl=float(ed600.max()), sha256_run1=h1, sha256_run2=h2, deterministic=h1 == h2,
               bob_angle_hist_late=np.histogram(phi[late], bins=36, range=(-np.pi, np.pi))[0].tolist(),
               film=dict(angles="film/fan.u16 uint16 [frames][10000][2], angle = v / 65536 * 2 pi", delta="film/fan_delta.f32 float32 [721][10000][2] th - th(pendulum 0)",
                         frames=len(th)))
    (CACHE / "fan.f64").unlink()
    dump(ST / "fan.json", out); log(f"fan: split {split_t:.2f} s, {1 / k_dec:.2f} s per factor 10, energy err {ed.max():.2e} mgl, deterministic {h1 == h2}")


def stage_dial():
    if done("dial"): return
    decs = list(range(-1, -16, -1))
    rng = np.random.default_rng(SEED_DIAL)
    base = random_releases(energy0(*(np.array(HERO) * DEG)), 1000, rng)
    hero = np.array([[HERO[0] * DEG, HERO[1] * DEG, 0, 0]])
    res = {}
    for e in decs:
        a = np.concatenate([hero, base]); b = a.copy(); b[:, 1] += 10.0 ** e * DEG
        thr, _ = run_pairs(a, b, 120.0, DT, f"dial_{-e}")
        res[e] = thr
        log(f"dial 1e{e}: hero {thr[0, 2]:.2f} s, median {np.median(np.where(thr[1:, 2] > 0, thr[1:, 2], np.inf)):.2f} s")
    rows = []
    for e in decs:
        t = res[e][1:, 2]; ok = t > 0; tt = np.where(ok, t, np.inf)
        rows.append(dict(delta_deg=10.0 ** e, log10=e, hero_t=float(res[e][0, 2]), hero_t_by_thr=res[e][0].tolist(), frac_diverged=float(ok.mean()),
                         median=float(np.median(tt)) if ok.mean() > 0.5 else None, p25=float(np.percentile(tt, 25)) if ok.mean() > 0.25 else None,
                         p75=float(np.percentile(tt, 75)) if ok.mean() > 0.75 else None))
    fit = [r for r in rows if -10 <= r["log10"] <= -2]
    k, c = np.polyfit([r["log10"] for r in fit], [r["median"] for r in fit], 1)
    kh, ch = np.polyfit([r["log10"] for r in fit], [r["hero_t"] for r in fit], 1)
    m2, m10 = [r["median"] for r in rows if r["log10"] == -2][0], [r["median"] for r in rows if r["log10"] == -10][0]
    # float64 floor: the offset is below the spacing of 64-bit numbers near 120 deg (2^-51 rad = 2.5e-14 deg)
    ulp = math.ulp(HERO[1] * DEG) / DEG
    # film mini-fans: 1,000 pendulums at the hero start, total spread 1e-2 ... 1e-10 deg, 30 s
    films = {}
    for e in (-2, -4, -6, -8, -10):
        s = fan_states(*HERO, 1000, 10.0 ** e)
        th, _, _ = run_traj(s, 30.0, DT, FPS, f"mini_{-e}")
        wrap16(th).tofile(FILM / f"mini_{-e}.u16")
        x, y = bob(th); d = np.hypot(x - x[:, :1], y - y[:, :1]).max(1)
        films[str(e)] = dict(file=f"film/mini_{-e}.u16", split_t=float(np.argmax(d > SPLIT_M) / FPS), frames=len(th))
        (CACHE / f"mini_{-e}.f64").unlink()
    out = dict(threshold_m=SPLIT_M, T=120.0, dt=DT, n_random=1000, seed=SEED_DIAL, rows=rows,
               s_per_decade_median=float(-k), s_per_decade_hero=float(-kh), gain_1e2_to_1e10=float(m10 - m2),
               lambda_from_slope=float(math.log(10) / -k), float64_ulp_deg=ulp, films=films)
    dump(ST / "dial.json", out); log(f"dial: {-k:.2f} s per factor 10 (median), +{m10 - m2:.1f} s from 1e-2 to 1e-10 deg")


def stage_energy():
    if done("energy"): return
    fans = []
    for A in range(10, 180, 10):
        s = fan_states(A, A, 2000, SPREAD)
        th, ed, _ = run_traj(s, T_FAN, DT, FPS, f"en_{A}")
        x, y = bob(th); d = np.hypot(x - x[:, :1], y - y[:, :1]).max(1)
        sp = float(np.argmax(d > SPLIT_M) / FPS) if (d > SPLIT_M).any() else None
        fans.append(dict(A=A, E=float(energy0(A * DEG, A * DEG)), split_t=sp, spread_at_60=float(d[-1]), max_spread=float(d.max()), energy_err=float(ed.max())))
        if A in (20, 60, 90, 120, 150):
            wrap16(th).tofile(FILM / f"energy_{A}.u16")
        (CACHE / f"en_{A}.f64").unlink()
        log(f"energy fan {A}: split {sp}, spread at 60 s {d[-1]:.3g} m")
    lam = []
    for i, E in enumerate([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.9]):
        rng = np.random.default_rng(SEED_LYAP + i)
        s = random_releases(E, 200, rng)
        Lx = run_lyap(s, 400.0, DT_L, 1e-8, 0.1, f"lyap_{i}")
        lam.append(dict(E=E, median=float(np.median(Lx[:, 0])), p25=float(np.percentile(Lx[:, 0], 25)), p75=float(np.percentile(Lx[:, 0], 75)),
                        frac_chaotic=float(np.mean(Lx[:, 0] > 0.1)), median_last_quarter=float(np.median(Lx[:, 2]))))
        log(f"lyap E={E}: median {lam[-1]['median']:.3f} /s, chaotic {lam[-1]['frac_chaotic']:.2f}")
    dump(ST / "energy.json", dict(fans=fans, lyapunov=lam, n_per_energy=200, T=400.0, dt=DT_L, d0=1e-8, tau=0.1, seed0=SEED_LYAP,
                                  film={str(A): f"film/energy_{A}.u16 uint16 [3601][2000][2]" for A in (20, 60, 90, 120, 150)}))


def stage_flipmap():
    if done("flipmap"): return
    M, sub, T = 1080, 4, 60.0
    idx = np.arange(1, M, sub)                                   # pixels 1, 5, 9, ... (270 per axis)
    J, I = np.meshgrid(idx, idx, indexing="ij")
    # row j = theta2, column i = theta1 (the texel order of the GPU grid)
    s = np.zeros((idx.size ** 2, 4)); s[:, 0] = -np.pi + (I.ravel() + 0.5) * 2 * np.pi / M
    s[:, 1] = -np.pi + (J.ravel() + 0.5) * 2 * np.pi / M
    inp = CACHE / "flip_in.f64"; s.tofile(inp)
    secs = dp("flip", inp, len(s), T, DT_L, CACHE / "flip_cpu.f32")
    f = np.fromfile(CACHE / "flip_cpu.f32", np.float32)
    np.save(CACHE / "flip_cpu_idx.npy", np.stack([I.ravel(), J.ravel()], 1))
    forbid = 2 * np.cos(s[:, 0]) + np.cos(s[:, 1]) > 1
    dump(ST / "flipmap.json", dict(M=M, sub=sub, n=int(len(s)), T=T, dt=DT_L, cpu_seconds=secs, frac_flipped=float(np.mean(f > 0)),
                                    forbidden_frac=float(forbid.mean()), flips_in_forbidden=int(np.sum((f > 0) & forbid)),
                                    median_flip_t=float(np.median(f[f > 0])), frac_flip_lt_10s=float(np.mean((f > 0) & (f < 10)))))
    log(f"flipmap: {np.mean(f > 0):.3f} flipped, {np.sum((f > 0) & forbid)} flips in the forbidden region")


def stage_friction():
    if done("friction"): return
    s = fan_states(*HERO, 2000, SPREAD)
    inp = CACHE / "fric_in.f64"; s.tofile(inp)
    dp("traj", inp, 2000, T_FAN, DT, int(round(1 / (FPS * DT))), CACHE / "fric.f64", CACHE / "fric_E.f64", env={"DP_DAMP": str(FRICTION_C)})
    th = np.fromfile(CACHE / "fric.f64").reshape(-1, 2000, 2)
    wrap16(th).tofile(FILM / "friction.u16")
    x, y = bob(th); d = np.hypot(x - x[:, :1], y - y[:, :1]).max(1)
    # energy of pendulum 0 vs time needs velocities: estimate from the angle series (central differences)
    w = np.gradient(th[:, 0, :], 1 / FPS, axis=0)
    t1, t2, w1, w2 = th[:, 0, 0], th[:, 0, 1], w[:, 0], w[:, 1]
    E = (w1 ** 2 + 0.5 * w2 ** 2 + w1 * w2 * np.cos(t1 - t2)) / G + 3 - 2 * np.cos(t1) - np.cos(t2)
    split = float(np.argmax(d > SPLIT_M) / FPS) if (d > SPLIT_M).any() else None
    (CACHE / "fric.f64").unlink()
    dump(ST / "friction.json", dict(c=FRICTION_C, n=2000, split_t=split, E_start=float(E[2]), E_end=float(E[-3]), E_series=E[::30].tolist(),
                                     spread_end=float(d[-1]), film="film/friction.u16"))
    log(f"friction: split {split}, E {E[2]:.2f} -> {E[-3]:.2f} mgl")


# ----------------------------------------------------------------------------------------------- validation
def rhs_shinbrot(t, y):
    """independent right-hand side (Shinbrot et al. 1992 eqs. B5-B6, mu = 1 + m1/m2 = 2), vectorised over pendulums"""
    y = y.reshape(-1, 4); a, b, wa, wb = y[:, 0], y[:, 1], y[:, 2], y[:, 3]; mu = 2.0; d = a - b
    den = mu - np.cos(d) ** 2
    aa = (G * (np.sin(b) * np.cos(d) - mu * np.sin(a)) - (L * wb ** 2 + L * wa ** 2 * np.cos(d)) * np.sin(d)) / (L * den)
    ab = (G * mu * (np.sin(a) * np.cos(d) - np.sin(b)) + (mu * L * wa ** 2 + L * wb ** 2 * np.cos(d)) * np.sin(d)) / (L * den)
    return np.stack([wa, wb, aa, ab], 1).ravel()


def stage_validate():
    if done("validate"): return
    from scipy.integrate import solve_ivp
    out = {}
    # V2 normal modes
    w1a, w2a = math.sqrt((2 - math.sqrt(2)) * G / L), math.sqrt((2 + math.sqrt(2)) * G / L)
    amp = 1.0 * DEG
    s = np.array([[amp, amp * math.sqrt(2), 0, 0], [amp, -amp * math.sqrt(2), 0, 0], [amp, 0, 0, 0]])
    T = 600.0; th, _, _ = run_traj(s, T, DT_L, 50, "modes")
    sig = th[:, :, 0]; n = len(sig); win = np.hanning(n)
    F = np.abs(np.fft.rfft((sig - sig.mean(0)) * win[:, None], axis=0)); fr = np.fft.rfftfreq(n, 1 / 50) * 2 * np.pi

    def peak(col, near):
        m = (fr > near * 0.7) & (fr < near * 1.3); i = np.argmax(np.where(m, F[:, col], 0))
        a, b, c = np.log(F[i - 1:i + 2, col]); p = 0.5 * (a - c) / (a - 2 * b + c)
        return float(fr[i] + p * (fr[1] - fr[0]))
    m1, m2 = peak(0, w1a), peak(1, w2a); mx1, mx2 = peak(2, w1a), peak(2, w2a)
    out["V2_modes"] = dict(amp_deg=1.0, T=T, omega_slow_analytic=w1a, omega_fast_analytic=w2a, omega_slow=m1, omega_fast=m2,
                          mixed_slow=mx1, mixed_fast=mx2, period_slow_s=2 * math.pi / m1, period_fast_s=2 * math.pi / m2,
                          rel_err_slow=abs(m1 / w1a - 1), rel_err_fast=abs(m2 / w2a - 1), note="1 deg swings; the small-angle formula is exact only in the limit")
    (CACHE / "modes.f64").unlink()
    # V1 energy: from the fan stage
    fan = load(ST / "fan.json")
    out["V1_energy"] = dict(fan_max_err_mgl=fan["energy_err_max_mgl"], fan_rel=fan["energy_err_rel"], one_600s_mgl=fan["energy_err_600s_mgl"], E=fan["E_mgl"])
    # V3 integrator: 200 hero-fan pendulums (every 50th), dt vs dt/2 vs DOP853
    sel = np.arange(0, N_FAN, 50); s = fan_states(*HERO, N_FAN, SPREAD)[sel]
    a, _, _ = run_traj(s, 40.0, DT, FPS, "v3a"); b, _, _ = run_traj(s, 40.0, DT / 2, FPS, "v3b")
    t_eval = np.arange(0, 40.0 + 1e-9, 1 / FPS)
    sol = solve_ivp(rhs_shinbrot, (0, 40.0), s.ravel(), method="DOP853", rtol=1e-12, atol=1e-12, t_eval=t_eval)
    c = sol.y.T.reshape(len(t_eval), -1, 4)[:, :, :2]

    def agree(p, q, tol=1e-3):
        xp, yp = bob(p); xq, yq = bob(q); d = np.hypot(xp - xq, yp - yq)
        return np.array([np.argmax(col > tol) / FPS if (col > tol).any() else 40.0 for col in d.T])

    def split(p):
        x, y = bob(p); d = np.hypot(x - x[:, :1], y - y[:, :1]).max(1); return float(np.argmax(d > SPLIT_M) / FPS)
    ag_dt, ag_dop = agree(a, b), agree(a, c)
    from scipy.stats import ks_2samp
    late = slice(int(30 * FPS), None)
    phiA = np.arctan2(*bob(a[late])[::1]); phiC = np.arctan2(*bob(c[late]))
    out["V3_integrator"] = dict(n=len(sel), T=40.0, dt=DT, agree_1mm_dt_half_median=float(np.median(ag_dt)), agree_1mm_dt_half_min=float(ag_dt.min()),
                                agree_1mm_dop853_median=float(np.median(ag_dop)), agree_1mm_dop853_min=float(ag_dop.min()),
                                split_rk4=split(a), split_rk4_half=split(b), split_dop853=split(c),
                                ks_late_bob_angle_rk4_vs_dop853=float(ks_2samp(phiA.ravel(), phiC.ravel()).statistic), dop853_nfev=int(sol.nfev))
    for f in ("v3a", "v3b"): (CACHE / f"{f}.f64").unlink()
    # V4 Lyapunov at the hero start, two settings, 2,000 s
    hs = np.array([[HERO[0] * DEG, HERO[1] * DEG, 0, 0]])
    l1 = run_lyap(hs, 2000.0, DT_L, 1e-8, 0.1, "lyh1")[0]; l2 = run_lyap(hs, 2000.0, DT_L, 1e-9, 0.2, "lyh2")[0]
    out["V4_lyapunov"] = dict(hero_lambda=float(l1[0]), hero_lambda_alt=float(l2[0]), hero_lambda_last_quarter=float(l1[2]), T=2000.0,
                              settings=["d0 1e-8, tau 0.1 s", "d0 1e-9, tau 0.2 s"])
    dump(ST / "validate.json", out)
    log("validate:", json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items()})[:900])


def stage_gpucheck():
    p = GPU / "probe.jsonl"
    if not p.exists(): sys.exit("results/gpu/probe.jsonl missing: run gpu/gpucheck.html first (README)")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    fanp = [r for r in rows if r.get("probe", r).get("kind") == "fan" or r.get("kind") == "fan"]
    mapp = [r for r in rows if r.get("probe", r).get("kind") == "map" or r.get("kind") == "map"]
    fanp = fanp[0].get("probe", fanp[0]); mapp = mapp[0].get("probe", mapp[0])
    out = {}
    # fan: float32 vs float64
    g = np.array(fanp["frames"], np.float64).reshape(len(fanp["frames"]), N_FAN, 2)
    s = fan_states(*HERO, N_FAN, SPREAD)
    c, _, _ = run_traj(s, 30.0, DT_L, 2, "gpucmp")        # same step as the GPU (1/1200 s), every 0.5 s
    (CACHE / "gpucmp.f64").unlink()
    xg, yg = bob(g); xc, yc = bob(c); tt = np.arange(len(g)) * 0.5
    dgc = np.hypot(xg - xc, yg - yc)                       # GPU vs CPU, same pendulum
    dg = np.hypot(xg - xg[:, :1], yg - yg[:, :1]).max(1)   # spread inside the GPU fan
    dc = np.hypot(xc - xc[:, :1], yc - yc[:, :1]).max(1)
    first = lambda arr, v: float(tt[np.argmax(arr > v)]) if (arr > v).any() else None
    distinct_end = len(np.unique(np.round(g[-1], 6).view([("a", np.float64), ("b", np.float64)])))
    out["fan"] = dict(distinct_starts_float32=fanp["distinct"], distinct_at_30s=distinct_end, split_gpu=first(dg, SPLIT_M), split_cpu=first(dc, SPLIT_M),
                      track_20cm_median=float(np.median([first(col, SPLIT_M) or 30.0 for col in dgc.T])),
                      track_1cm_median=float(np.median([first(col, 0.01) or 30.0 for col in dgc.T])), every=0.5, dt=fanp["dt"])
    # flip map
    M = mapp["M"]; fg = np.array(mapp["flip"], np.float32).reshape(M, M)        # [row j (theta2)][column i (theta1)]
    np.save(CACHE / "flip_gpu.npy", fg)
    fc = np.fromfile(CACHE / "flip_cpu.f32", np.float32); ij = np.load(CACHE / "flip_cpu_idx.npy")
    fgs = fg[ij[:, 1], ij[:, 0]]
    same_status = (fgs > 0) == (fc > 0); both = (fgs > 0) & (fc > 0)
    rel = np.abs(fgs[both] - fc[both]) / fc[both]
    th1 = -np.pi + (ij[:, 0] + 0.5) * 2 * np.pi / M; th2 = -np.pi + (ij[:, 1] + 0.5) * 2 * np.pi / M
    early = both & (fc < 5)
    out["map"] = dict(M=M, n_compared=int(len(fc)), same_status=float(same_status.mean()), flip_time_within_1pct=float(np.mean(rel < 0.01)),
                      flip_time_within_1pct_of_all=float(np.sum(rel < 0.01) / len(fc)), early_flips_within_1pct=float(np.mean(np.abs(fgs[early] - fc[early]) / fc[early] < 0.01)),
                      gpu_frac_flipped=float(np.mean(fg > 0)), gpu_flips_in_forbidden=int(np.sum((fg > 0) & (2 * np.cos(-np.pi + (np.arange(M)[None, :] + 0.5) * 2 * np.pi / M) + np.cos(-np.pi + (np.arange(M)[:, None] + 0.5) * 2 * np.pi / M) > 1))))
    # film product: the GPU flip map itself is recomputed live in the film; keep a compact copy for charts / thumbnails
    np.round(np.where(fg > 0, fg, -1), 3).astype(np.float32).tofile(FILM / "flipmap_gpu.f32")
    dump(ST / "gpucheck.json", out); log("gpucheck:", json.dumps(out)[:600])


# ----------------------------------------------------------------------------------------------- results
def stage_results():
    S = {n: load(ST / f"{n}.json") for n in ("fan", "dial", "energy", "flipmap", "friction", "validate") if (ST / f"{n}.json").exists()}
    if (ST / "gpucheck.json").exists(): S["gpucheck"] = load(ST / "gpucheck.json")
    P = {}

    def put(k, v, src, unit=""):
        P[k] = dict(value=v, source=src, unit=unit)
    f, d, e, v = S["fan"], S["dial"], S["energy"], S["validate"]
    put("N_FAN", f["n"], "fan.n"); put("SPREAD_DEG", f["spread_deg"], "fan.spread_deg", "deg"); put("RELEASE_DEG", f["release_deg"][0], "fan.release_deg", "deg")
    put("E_HERO", f["E_mgl"], "fan.E_mgl", "m g l"); put("SPLIT_T", f["split_t"], "fan.split_t", "s"); put("SPLIT_CM", 100 * f["split_m"], "fan.split_m", "cm")
    put("FAN_S_PER_DECADE", f["s_per_decade"], "fan.s_per_decade", "s"); put("FAN_PASS_1CM", f["spread_passes"]["0.01"], "fan.spread_passes", "s")
    put("FAN_PASS_1MM", f["spread_passes"]["0.001"], "fan.spread_passes", "s"); put("FAN_PASS_1M", f["spread_passes"]["1.0"], "fan.spread_passes", "s")
    put("FAN_COVER_12", f["coverage_at"]["12"], "fan.coverage_at", "share"); put("FAN_COVER_20", f["coverage_at"]["20"], "fan.coverage_at", "share")
    cv = np.array(f["coverage"]); ct = np.array(f["spread_t"]); full = float(cv[ct >= 20].mean())
    put("FAN_COVER_FULL", full, "mean of fan.coverage over 20-60 s", "share")
    put("FAN_FILL_T", float(ct[np.argmax(cv >= 0.9 * full)]), "fan.coverage: first t with >= 90% of the 20-60 s mean", "s")
    if (ST / "timing.json").exists():
        tm = load(ST / "timing.json"); put("RUN_WALL_S", sum(tm.values()), "stages/timing.json (sum of stage wall times)", "s")
    put("FAN_DETERMINISTIC", f["deterministic"], "fan.sha256_run1 == run2"); put("ENERGY_ERR_REL", f["energy_err_rel"], "fan.energy_err_rel")
    put("ENERGY_ERR_600", v["V1_energy"]["one_600s_mgl"], "validate.V1_energy", "m g l")
    put("DIAL_S_PER_DECADE", d["s_per_decade_median"], "dial.s_per_decade_median", "s"); put("DIAL_S_PER_DECADE_HERO", d["s_per_decade_hero"], "dial.s_per_decade_hero", "s")
    put("DIAL_GAIN_2_TO_10", d["gain_1e2_to_1e10"], "dial.gain_1e2_to_1e10", "s"); put("DIAL_LAMBDA", d["lambda_from_slope"], "dial.lambda_from_slope", "1/s")
    for r in d["rows"]:
        put(f"DIAL_MED_{-r['log10']}", r["median"], f"dial.rows[1e{r['log10']}].median", "s"); put(f"DIAL_HERO_{-r['log10']}", r["hero_t"], f"dial.rows[1e{r['log10']}].hero_t", "s")
        put(f"DIAL_FRAC_{-r['log10']}", r["frac_diverged"], f"dial.rows[1e{r['log10']}].frac_diverged", "share")
    md = {r["log10"]: r["median"] for r in d["rows"]}
    put("DIAL_GAIN_4_TO_12", md[-12] - md[-4], "dial.rows median(1e-12) - median(1e-4)", "s")
    put("FLOAT64_ULP_DEG", d["float64_ulp_deg"], "dial.float64_ulp_deg", "deg"); put("DIAL_N", d["n_random"], "dial.n_random")
    put("LAMBDA_HERO", v["V4_lyapunov"]["hero_lambda"], "validate.V4_lyapunov.hero_lambda", "1/s"); put("LAMBDA_HERO_ALT", v["V4_lyapunov"]["hero_lambda_alt"], "validate.V4_lyapunov", "1/s")
    for r in e["lyapunov"]:
        put(f"LAMBDA_E{r['E']:g}", r["median"], f"energy.lyapunov[E={r['E']}].median", "1/s"); put(f"CHAOTIC_E{r['E']:g}", r["frac_chaotic"], f"energy.lyapunov[E={r['E']}].frac_chaotic", "share")
    for r in e["fans"]:
        put(f"EN_SPLIT_{r['A']}", r["split_t"], f"energy.fans[A={r['A']}].split_t", "s"); put(f"EN_SPREAD60_{r['A']}", r["spread_at_60"], f"energy.fans[A={r['A']}].spread_at_60", "m")
    m = v["V2_modes"]
    put("MODE_SLOW_T", m["period_slow_s"], "validate.V2_modes", "s"); put("MODE_FAST_T", m["period_fast_s"], "validate.V2_modes", "s")
    put("MODE_SLOW_T_TH", 2 * math.pi / m["omega_slow_analytic"], "formula", "s"); put("MODE_FAST_T_TH", 2 * math.pi / m["omega_fast_analytic"], "formula", "s")
    put("MODE_ERR_MAX", max(m["rel_err_slow"], m["rel_err_fast"]), "validate.V2_modes", "relative")
    i3 = v["V3_integrator"]
    put("V3_AGREE_DOP", i3["agree_1mm_dop853_median"], "validate.V3_integrator", "s"); put("V3_AGREE_DT", i3["agree_1mm_dt_half_median"], "validate.V3_integrator", "s")
    put("V3_SPLIT_DOP", i3["split_dop853"], "validate.V3_integrator", "s"); put("V3_KS", i3["ks_late_bob_angle_rk4_vs_dop853"], "validate.V3_integrator")
    fm = S["flipmap"]; put("MAP_N_GPU", fm["M"] ** 2, "flipmap.M^2"); put("MAP_FRAC_FLIPPED_CPU", fm["frac_flipped"], "flipmap.frac_flipped", "share")
    put("MAP_FLIPS_FORBIDDEN_CPU", fm["flips_in_forbidden"], "flipmap.flips_in_forbidden"); put("MAP_T", fm["T"], "flipmap.T", "s")
    if "friction" in S: put("FRICTION_SPLIT", S["friction"]["split_t"], "friction.split_t", "s"); put("FRICTION_C", S["friction"]["c"], "friction.c", "1/s")
    if "gpucheck" in S:
        g = S["gpucheck"]
        put("GPU_DISTINCT", g["fan"]["distinct_starts_float32"], "gpucheck.fan.distinct_starts_float32"); put("GPU_SPLIT", g["fan"]["split_gpu"], "gpucheck.fan.split_gpu", "s")
        put("GPU_TRACK_20CM", g["fan"]["track_20cm_median"], "gpucheck.fan.track_20cm_median", "s"); put("GPU_TRACK_1CM", g["fan"]["track_1cm_median"], "gpucheck.fan.track_1cm_median", "s")
        put("GPU_MAP_SAME", g["map"]["same_status"], "gpucheck.map.same_status", "share"); put("GPU_MAP_1PCT", g["map"]["flip_time_within_1pct"], "gpucheck.map.flip_time_within_1pct", "share")
        put("GPU_MAP_FRAC_FLIPPED", g["map"]["gpu_frac_flipped"], "gpucheck.map.gpu_frac_flipped", "share"); put("GPU_MAP_FORBIDDEN", g["map"]["gpu_flips_in_forbidden"], "gpucheck.map.gpu_flips_in_forbidden")
    R = dict(episode="ep05-double-pendulums", generated=time.strftime("%Y-%m-%dT%H:%M:%S"), placeholders=P,
             checks=dict(validate=v, gpucheck=S.get("gpucheck"), flipmap=fm), stages=list(S))
    dump(RES / "RESULTS.json", R)
    md = ["# EP05 results (generated by run.py results; do not edit)", "", "| key | value | unit | source |", "|---|---|---|---|"]
    for k, x in P.items():
        val = x["value"]; val = f"{val:.6g}" if isinstance(val, float) else str(val)
        md.append(f"| {k} | {val} | {x['unit']} | {x['source']} |")
    (RES / "RESULTS.md").write_text("\n".join(md) + "\n")
    commit = os.environ.get("GIT_COMMIT")
    if not commit:
        try: commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception: commit = "unknown"
    meta = dict(commit=commit, generated=R["generated"], python=platform.python_version(), numpy=np.__version__, machine=platform.machine(),
                platform=platform.platform(), cpu=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor(),
                dt=DT, dt_lyap=DT_L, seeds=dict(dial=SEED_DIAL, lyap=SEED_LYAP), model=dict(m1=1, m2=1, l1=1, l2=1, g=G),
                film_files={p.name: sha(p) for p in sorted(FILM.glob("*")) if p.is_file()})
    dump(RES / "run_meta.json", meta); log(f"results: {len(P)} placeholders, commit {commit}")


STAGES = dict(fan=stage_fan, dial=stage_dial, energy=stage_energy, flipmap=stage_flipmap, friction=stage_friction,
              validate=stage_validate, gpucheck=stage_gpucheck, results=stage_results)
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")] or ["all"]
    if args == ["all"]:
        args = ["fan", "dial", "energy", "flipmap", "friction", "validate", "results"]
    for a in args:
        t0 = time.time(); skipped = done(a) and a not in ("results", "gpucheck"); STAGES[a](); dt_ = time.time() - t0; log(f"{a} done in {dt_:.0f} s")
        if not skipped and a != "results":
            tp = ST / "timing.json"; tm = load(tp) if tp.exists() else {}; tm[a] = round(dt_, 1); dump(tp, tm)
