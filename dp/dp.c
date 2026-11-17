/* dp.c — ideal double pendulum (two point masses on massless rigid rods, frictionless hinges, uniform gravity),
   double precision, classical 4th-order Runge-Kutta with a fixed step. Many independent pendulums in parallel (GCD).
   Equations of motion: myPhysicsLab "Double Pendulum" (E. Neumann), identical to Shinbrot et al. 1992 eqs. (B5)-(B6).
   Angles from the downward vertical, counter-clockwise positive. State per pendulum: th1, th2, w1, w2 (rad, rad/s).

   Modes (all files are raw little-endian arrays; N = number of pendulums; see run.py for the layout):
     dp traj  in.f64 N T dt every out.f64 energy.f64     angles (th1,th2) every `every` steps, float64 [frames][N][2];
                                                            energy.f64 = per pendulum max |E(t)-E(0)| / (m g l)
     dp pairs in.f64 N T dt out.f64                       in = 2N states (pair i = rows 2i, 2i+1); out = [N][NTHR] first
                                                            time the two lower bobs are farther apart than THR[k] metres
                                                            (-1 = never within T), plus [N][NLOG] log10 bob distance
                                                            sampled every 0.05 s
     dp lyap  in.f64 N T dt d0 tau out.f64                Benettin: shadow orbit at state distance d0, renormalised every
                                                            tau s; out = [N][3]: lambda (1/s) over T, over T/2, over T/4
     dp flip  in.f64 N T dt out.f32                       first time either arm passes over the top (|th| > pi), -1 = none
   Options (env): DP_DAMP=c adds a simple friction term (angular accelerations -c*w; dE/dt = -c w.M.w < 0); 0 = ideal (default).
   Parameters (env, defaults): DP_G=9.81 DP_L1=1 DP_L2=1 DP_M1=1 DP_M2=1. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dispatch/dispatch.h>

static double G = 9.81, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0, DAMP = 0.0;

static inline void rhs(const double *s, double *d) {
  double t1 = s[0], t2 = s[1], w1 = s[2], w2 = s[3];
  double dl = t1 - t2, sd = sin(dl), cd = cos(dl);
  double den = 2 * M1 + M2 - M2 * cos(2 * dl);
  d[0] = w1; d[1] = w2;
  d[2] = (-G * (2 * M1 + M2) * sin(t1) - M2 * G * sin(t1 - 2 * t2) - 2 * sd * M2 * (w2 * w2 * L2 + w1 * w1 * L1 * cd)) / (L1 * den);
  d[3] = (2 * sd * (w1 * w1 * L1 * (M1 + M2) + G * (M1 + M2) * cos(t1) + w2 * w2 * L2 * M2 * cd)) / (L2 * den);
  if (DAMP != 0) { d[2] -= DAMP * w1; d[3] -= DAMP * w2; }   /* generalised force -c M w: energy only decreases */
}

static inline void rk4(double *s, double h) {
  double k1[4], k2[4], k3[4], k4[4], t[4];
  rhs(s, k1); for (int i = 0; i < 4; i++) t[i] = s[i] + 0.5 * h * k1[i];
  rhs(t, k2); for (int i = 0; i < 4; i++) t[i] = s[i] + 0.5 * h * k2[i];
  rhs(t, k3); for (int i = 0; i < 4; i++) t[i] = s[i] + h * k3[i];
  rhs(t, k4); for (int i = 0; i < 4; i++) s[i] += h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

/* energy / (m g l) with zero at the bottom (for M1=M2, L1=L2 this is 3 - 2cos th1 - cos th2 + kinetic) */
static double energy(const double *s) {
  double t1 = s[0], t2 = s[1], w1 = s[2], w2 = s[3];
  double T = 0.5 * (M1 + M2) * L1 * L1 * w1 * w1 + 0.5 * M2 * L2 * L2 * w2 * w2 + M2 * L1 * L2 * w1 * w2 * cos(t1 - t2);
  double V = (M1 + M2) * G * L1 * (1 - cos(t1)) + M2 * G * L2 * (1 - cos(t2));
  return (T + V) / (M1 * G * L1);
}
static inline void bob(const double *s, double *x, double *y) { *x = L1 * sin(s[0]) + L2 * sin(s[1]); *y = -L1 * cos(s[0]) - L2 * cos(s[1]); }

static double *readf64(const char *p, size_t n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  double *a = malloc(n * sizeof(double)); if (fread(a, sizeof(double), n, f) != n) { fprintf(stderr, "short read %s\n", p); exit(1); }
  fclose(f); return a;
}
static void writef(const char *p, const void *a, size_t sz, size_t n) {
  FILE *f = fopen(p, "wb"); if (!f) { perror(p); exit(1); } fwrite(a, sz, n, f); fclose(f);
}
static double envd(const char *k, double d) { const char *v = getenv(k); return v ? atof(v) : d; }

#define NTHR 5
static const double THR[NTHR] = {0.01, 0.1, 0.2, 0.5, 1.0};

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "see header\n"); return 2; }
  G = envd("DP_G", 9.81); L1 = envd("DP_L1", 1); L2 = envd("DP_L2", 1); M1 = envd("DP_M1", 1); M2 = envd("DP_M2", 1); DAMP = envd("DP_DAMP", 0);
  const char *mode = argv[1];
  dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  if (!strcmp(mode, "traj")) {
    const char *in = argv[2]; long N = atol(argv[3]); double T = atof(argv[4]), dt = atof(argv[5]); long every = atol(argv[6]);
    long nsteps = (long)llround(T / dt), nfr = nsteps / every + 1;
    double *s0 = readf64(in, 4 * N); double *out = malloc(sizeof(double) * 2 * N * nfr); double *ed = calloc(N, sizeof(double));
    dispatch_apply((size_t)N, q, ^(size_t i) {
      double s[4]; memcpy(s, s0 + 4 * i, sizeof s); double e0 = energy(s), md = 0;
      for (long k = 0; k <= nsteps; k++) {
        if (k % every == 0) { long f = k / every; out[(f * N + i) * 2] = s[0]; out[(f * N + i) * 2 + 1] = s[1];
          double de = fabs(energy(s) - e0); if (de > md) md = de; }
        if (k < nsteps) rk4(s, dt);
      }
      ed[i] = md;
    });
    writef(argv[7], out, sizeof(double), 2 * N * nfr); writef(argv[8], ed, sizeof(double), N);
    printf("traj N=%ld frames=%ld\n", N, nfr);
  } else if (!strcmp(mode, "pairs")) {
    const char *in = argv[2]; long N = atol(argv[3]); double T = atof(argv[4]), dt = atof(argv[5]);
    long nsteps = (long)llround(T / dt), le = (long)llround(0.05 / dt), NLOG = nsteps / le + 1;
    double *s0 = readf64(in, 8 * N); long W = NTHR + NLOG; double *out = malloc(sizeof(double) * W * N);
    dispatch_apply((size_t)N, q, ^(size_t i) {
      double a[4], b[4]; memcpy(a, s0 + 8 * i, sizeof a); memcpy(b, s0 + 8 * i + 4, sizeof b);
      double *o = out + W * i; for (int k = 0; k < NTHR; k++) o[k] = -1;
      for (long k = 0; k <= nsteps; k++) {
        double xa, ya, xb, yb; bob(a, &xa, &ya); bob(b, &xb, &yb); double d = hypot(xa - xb, ya - yb);
        for (int j = 0; j < NTHR; j++) if (o[j] < 0 && d > THR[j]) o[j] = k * dt;
        if (k % le == 0) o[NTHR + k / le] = log10(d > 1e-300 ? d : 1e-300);
        if (k < nsteps) { rk4(a, dt); rk4(b, dt); }
      }
    });
    writef(argv[6], out, sizeof(double), W * N); printf("pairs N=%ld nlog=%ld\n", N, NLOG);
  } else if (!strcmp(mode, "lyap")) {
    const char *in = argv[2]; long N = atol(argv[3]); double T = atof(argv[4]), dt = atof(argv[5]), d0 = atof(argv[6]), tau = atof(argv[7]);
    long nsteps = (long)llround(T / dt), rn = (long)llround(tau / dt);
    double *s0 = readf64(in, 4 * N); double *out = malloc(sizeof(double) * 3 * N);
    dispatch_apply((size_t)N, q, ^(size_t i) {
      double a[4], b[4]; memcpy(a, s0 + 4 * i, sizeof a);
      /* initial shadow offset: along (1,1,1,1)/2 in (th1, th2, w1*sqrt(l/g), w2*sqrt(l/g)) */
      double sc = sqrt(L1 / G); b[0] = a[0] + d0 / 2; b[1] = a[1] + d0 / 2; b[2] = a[2] + d0 / 2 / sc; b[3] = a[3] + d0 / 2 / sc;
      double sum = 0, s2 = 0, s4 = 0;
      for (long k = 1; k <= nsteps; k++) {
        rk4(a, dt); rk4(b, dt);
        if (k % rn == 0) {
          double dv[4] = {b[0] - a[0], b[1] - a[1], (b[2] - a[2]) * sc, (b[3] - a[3]) * sc};
          double d = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3]); double l = log(d / d0);
          sum += l; if (k > nsteps / 2) s2 += l; if (k > 3 * nsteps / 4) s4 += l;
          b[0] = a[0] + dv[0] * d0 / d; b[1] = a[1] + dv[1] * d0 / d; b[2] = a[2] + dv[2] / sc * d0 / d; b[3] = a[3] + dv[3] / sc * d0 / d;
        }
      }
      out[3 * i] = sum / T; out[3 * i + 1] = s2 / (T / 2); out[3 * i + 2] = s4 / (T / 4);
    });
    writef(argv[8], out, sizeof(double), 3 * N); printf("lyap N=%ld\n", N);
  } else if (!strcmp(mode, "flip")) {
    const char *in = argv[2]; long N = atol(argv[3]); double T = atof(argv[4]), dt = atof(argv[5]);
    long nsteps = (long)llround(T / dt);
    double *s0 = readf64(in, 4 * N); float *out = malloc(sizeof(float) * N);
    dispatch_apply((size_t)N, q, ^(size_t i) {
      double s[4]; memcpy(s, s0 + 4 * i, sizeof s); out[i] = -1;
      for (long k = 1; k <= nsteps; k++) { rk4(s, dt); if (fabs(s[0]) > M_PI || fabs(s[1]) > M_PI) { out[i] = (float)(k * dt); break; } }
    });
    writef(argv[6], out, sizeof(float), N); printf("flip N=%ld\n", N);
  } else { fprintf(stderr, "unknown mode %s\n", mode); return 2; }
  return 0;
}
