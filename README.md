# 10,000 double pendulums start 0.0001° apart — simulation code

Source code for episode 5 of **Computed Curiosity**. Every number in the video comes from `results/RESULTS.json`, produced by
`run.py`. Every swinging pendulum in the video (except the 32-bit comparison) is drawn from the 64-bit trajectories written
here; the flip-time map is computed live by the film's GPU code (`gpu/pend.js`) and checked against this code (check V5).

**The question:** how fast does a tiny difference become total chaos, and what does chaos actually mean?

## What is computed

* **The model:** an ideal double pendulum — two 1 kg point weights on two weightless rigid 1 m rods, frictionless hinges,
  no air, g = 9.81 m/s². Equations of motion as in myPhysicsLab and Shinbrot et al. 1992 (Am. J. Phys. 60, 491), eqs. B5–B6.
* **Integrator:** classical 4th-order Runge–Kutta, fixed step 1/2400 s (1/1200 s for the Lyapunov sweep and the flip map),
  64-bit floats, `dp/dp.c` (C, Grand Central Dispatch for the parallel loop).
* **E1, the fan:** 10,000 pendulums released from rest with both arms at 120° (energy 4.5 m g l above the lowest state);
  the lower arm of pendulum k is offset by k × 0.0001°/9,999, so the first and the last start 0.0001° apart. 60 s.
* **E2, the dial:** pairs of pendulums whose lower arms differ by 10⁻¹ … 10⁻¹⁵ degrees, at the same start and at 1,000 random
  releases from rest with the same energy; time until the lower weights are 20 cm apart.
* **E3, energy:** fans of 2,000 released from 10° … 170°; the largest Lyapunov exponent (Benettin method) for 200 random
  releases at each of 13 energies.
* **E4, flip map:** 1080 × 1080 starts from rest over all angles, first time either arm goes over the top (GPU, 32-bit;
  a 270 × 270 subsample recomputed here in 64-bit).
* **E5, friction (illustration of a limit):** the fan with a simple friction term (angular accelerations −0.02 ω).

## Checks (`python run.py validate`, `gpucheck`; see `results/RESULTS.md`)

| | check | result key in RESULTS.json |
|---|---|---|
| V1 | energy of every pendulum over 60 s, and of one pendulum over 600 s | `ENERGY_ERR_REL`, `ENERGY_ERR_600` |
| V2 | 1° swings: periods vs the textbook normal modes ω² = (2 ∓ √2) g/l | `MODE_*` |
| V3 | step halved, and scipy's DOP853 (8th order, adaptive, separately written equations): single pendulums agree for ~14 s, the split time and the late cloud agree | `V3_*` |
| V4 | Lyapunov exponent at the 120° start, Benettin with two settings vs the slope of the dial (ln 10 / seconds per factor 10) | `LAMBDA_HERO`, `DIAL_LAMBDA` |
| V5 | GPU 32-bit vs CPU 64-bit: the 10,000 starts become 8 distinct 32-bit numbers; flip map agreement | `GPU_*` |
| — | the fan run twice is bit-identical (SHA-256) | `FAN_DETERMINISTIC` |
| — | no start inside 2 cos θ1 + cos θ2 > 1 ever flips (energy conservation) | `MAP_FLIPS_FORBIDDEN_CPU`, `GPU_MAP_FORBIDDEN` |

No published table of Lyapunov exponents exists for this exact pendulum (equal point masses and lengths) that we could open;
our λ(E) rises from ~0 to above 1 /s around E ≈ 2 m g l, where flipping first becomes possible, in line with Cabrera et al. 2023
(arXiv:2312.13436). Shinbrot et al. 1992 measured 7.5 ± 1.5 /s on a smaller compound pendulum (not directly comparable).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all      # = clang + python run.py all — about 5 minutes on a 10-core Apple M4
```
Stages are cached in `results/stages/` and resume (`--force` recomputes). `results/cache/`, `results/film/` (angles for the
film, ~1 GB) and `results/gpu/probe.jsonl` are not in git. Outside a git checkout set `GIT_COMMIT=<sha>` so that
`results/run_meta.json` records the code version. `dp.c` uses Grand Central Dispatch (macOS); on Linux replace
`dispatch_apply` with an OpenMP loop.

**GPU check (V5).** Serve the `gpu/` folder with a local web server, open `gpu/gpucheck.html` in a WebGL2 browser and call
`window.__probe(0)` (fan) and `window.__probe(1)` (flip map); the episode used `capture.mjs --times 0,1 --probe` in headless
Chrome on an Apple M4. Save the JSON lines as `results/gpu/probe.jsonl`, then `python run.py gpucheck results`.

## Seeds
dial 5000201 · Lyapunov sweep 5000301 + energy index. The fans and the flip map have no random part.

## Known limits
Point weights on weightless rigid rods, no friction, no air, ideal hinges; a real double pendulum (heavy bars, bearings, a
stand) loses energy and is usually smaller and faster. Single trajectories cannot be trusted beyond ~14 s (two methods
disagree after that); the statistics of the cloud can.

## Sources
Shinbrot, Grebogi, Wisdom & Yorke 1992, Am. J. Phys. 60, 491 · myPhysicsLab "Double Pendulum" (E. Neumann) · Heyl 2008
"The double pendulum fractal" · Cabrera, Leonel & Marti 2023, arXiv:2312.13436 · Lorenz 1963, J. Atmos. Sci. 20, 130 ·
Zhang et al. 2019, J. Atmos. Sci. 76, 1077. Full list with notes in the episode's `FACTS.md`.

Code: MIT licence (see `LICENSE`). Data and figures: CC BY 4.0. Written with AI assistance and checked against the sources above.
