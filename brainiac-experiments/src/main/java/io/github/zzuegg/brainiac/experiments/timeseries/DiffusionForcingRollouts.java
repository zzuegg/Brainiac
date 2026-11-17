package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChaoticSeries;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.DiffusionForcing;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Long rollouts of the Lorenz system: a teacher-forced autoregressive MLP (next state from
 * the last {@code W}, mean squared error), full-sequence diffusion rolled out by
 * replacement inpainting, and {@link DiffusionForcing} conditioning on its past clean or
 * re-noised to a small level (<a href="https://arxiv.org/abs/2407.01392">arXiv:2407.01392</a>).
 * Measured on rollouts from held-out states: the error over the first steps against the true
 * trajectory, and, since a chaotic rollout must decorrelate, whether the late rollout still
 * lies on the attractor (mean distance to the nearest of many true states) and the share of
 * rollouts that blew up.
 */
public final class DiffusionForcingRollouts {
    private static final int WINDOW = 4;
    private static final int DIM = 3;

    private DiffusionForcingRollouts() {
    }

    /**
     * One method's rollout statistics.
     *
     * @param shortError root mean squared error over the first 10 steps (normalized units)
     * @param attractorDistance mean distance of the last half of the rollout to the attractor
     * @param diverged share of rollouts that left the box {@code |x| < 5}
     */
    public record Stats(double shortError, double attractorDistance, double diverged) {
    }

    private static float[][] normalized(int n, double dt, int discard) {
        double[][] raw = ChaoticSeries.lorenz(n, dt, discard);
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            out[i] = new float[] {(float) (raw[i][0] / 10), (float) (raw[i][1] / 10), (float) ((raw[i][2] - 25) / 10)};
        }
        return out;
    }

    private static float[] window(float[][] series, int start, int length) {
        float[] w = new float[length * DIM];
        for (int k = 0; k < length; k++) {
            System.arraycopy(series[start + k], 0, w, k * DIM, DIM);
        }
        return w;
    }

    /**
     * Trains every model and rolls them out.
     *
     * @param iterations training steps of 64 windows for each model
     * @param horizon rollout length
     * @param rollouts how many held-out starts
     * @param seed the seed
     * @return statistics per method
     */
    public static Map<String, Stats> run(int iterations, int horizon, int rollouts, long seed) {
        float[][] train = normalized(4000, 0.02, 500);
        float[][] test = normalized(6000 + horizon, 0.02, 500);
        float[][] reference = java.util.Arrays.copyOfRange(test, 0, 4000);
        SplittableRandom r = new SplittableRandom(seed);
        SimpleMlp ar = SimpleMlp.tanh(seed, WINDOW * DIM, 128, 128, DIM);
        DiffusionForcing df = DiffusionForcing.of(DIM, WINDOW, DiffusionForcing.Mode.INDEPENDENT, 50, 128, 3, seed);
        DiffusionForcing full = DiffusionForcing.of(DIM, WINDOW, DiffusionForcing.Mode.FULL_SEQUENCE, 50, 128, 3,
                seed);
        Optimizer arOpt = Optimizer.adam(1e-3f);
        Optimizer dfOpt = Optimizer.adam(1e-3f);
        Optimizer fullOpt = Optimizer.adam(1e-3f);
        for (int it = 0; it < iterations; it++) {
            float[][] w = new float[64][];
            float[] x = new float[64 * WINDOW * DIM];
            float[] y = new float[64 * DIM];
            for (int b = 0; b < 64; b++) {
                int start = r.nextInt(train.length - WINDOW - 1);
                w[b] = window(train, start, WINDOW + 1);
                System.arraycopy(w[b], 0, x, b * WINDOW * DIM, WINDOW * DIM);
                for (int c = 0; c < DIM; c++) {
                    y[b * DIM + c] = w[b][WINDOW * DIM + c] - w[b][(WINDOW - 1) * DIM + c];
                }
            }
            Var arLoss = ar.forward(Var.of(Tensor.of(Shape.of(64, WINDOW * DIM), x)))
                    .sub(Var.of(Tensor.of(Shape.of(64, DIM), y))).square().mean();
            arLoss.backward();
            arOpt.step(ar.parameters());
            df.loss(w, r).backward();
            dfOpt.step(df.parameters());
            full.loss(w, r).backward();
            fullOpt.step(full.parameters());
        }
        int[] starts = new int[rollouts];
        for (int i = 0; i < rollouts; i++) {
            starts[i] = 4000 + i * (2000 / rollouts);
        }
        Map<String, Stats> out = new LinkedHashMap<>();
        out.put("teacher-forced AR", stats(rolloutAr(ar, test, starts, horizon), test, starts, reference));
        out.put("full-sequence diffusion", stats(rollout(full, 0, test, starts, horizon, seed), test, starts,
                reference));
        out.put("diffusion forcing, clean past", stats(rollout(df, 0, test, starts, horizon, seed), test, starts,
                reference));
        out.put("diffusion forcing, noisy past", stats(rollout(df, 3, test, starts, horizon, seed), test, starts,
                reference));
        return out;
    }

    private static float[][][] rolloutAr(SimpleMlp ar, float[][] series, int[] starts, int horizon) {
        int n = starts.length;
        float[][][] traj = new float[n][horizon][];
        float[][] ctx = new float[n][];
        for (int i = 0; i < n; i++) {
            ctx[i] = window(series, starts[i], WINDOW);
        }
        for (int s = 0; s < horizon; s++) {
            float[] flat = new float[n * WINDOW * DIM];
            for (int i = 0; i < n; i++) {
                System.arraycopy(ctx[i], 0, flat, i * WINDOW * DIM, WINDOW * DIM);
            }
            float[] d = ar.predict(Tensor.of(Shape.of(n, WINDOW * DIM), flat)).toArray();
            for (int i = 0; i < n; i++) {
                float[] next = new float[DIM];
                for (int c = 0; c < DIM; c++) {
                    next[c] = clamp(ctx[i][(WINDOW - 1) * DIM + c] + d[i * DIM + c]);
                }
                traj[i][s] = next;
                shift(ctx[i], next);
            }
        }
        return traj;
    }

    private static float[][][] rollout(DiffusionForcing model, int contextLevel, float[][] series, int[] starts,
            int horizon, long seed) {
        int n = starts.length;
        SplittableRandom r = new SplittableRandom(seed + 5);
        float[][][] traj = new float[n][horizon][];
        float[][] ctx = new float[n][];
        for (int i = 0; i < n; i++) {
            ctx[i] = window(series, starts[i], WINDOW);
        }
        for (int s = 0; s < horizon; s++) {
            float[][] next = model.next(ctx, contextLevel, 10, r);
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < DIM; c++) {
                    next[i][c] = clamp(next[i][c]);
                }
                traj[i][s] = next[i];
                shift(ctx[i], next[i]);
            }
        }
        return traj;
    }

    private static float clamp(float v) {
        return Float.isNaN(v) ? 100 : Math.max(-100, Math.min(100, v));
    }

    private static void shift(float[] ctx, float[] next) {
        System.arraycopy(ctx, DIM, ctx, 0, (WINDOW - 1) * DIM);
        System.arraycopy(next, 0, ctx, (WINDOW - 1) * DIM, DIM);
    }

    private static Stats stats(float[][][] traj, float[][] series, int[] starts, float[][] reference) {
        double shortErr = 0;
        int shortCount = 0;
        double attractor = 0;
        int attractorCount = 0;
        int diverged = 0;
        for (int i = 0; i < traj.length; i++) {
            boolean blew = false;
            for (int s = 0; s < traj[i].length; s++) {
                float[] p = traj[i][s];
                if (s < 10) {
                    float[] truth = series[starts[i] + WINDOW + s];
                    for (int c = 0; c < DIM; c++) {
                        shortErr += (p[c] - truth[c]) * (p[c] - truth[c]);
                        shortCount++;
                    }
                }
                for (float v : p) {
                    blew |= Math.abs(v) > 5;
                }
                if (s >= traj[i].length / 2 && s % 5 == 0) {
                    double best = Double.MAX_VALUE;
                    for (float[] q : reference) {
                        double d = 0;
                        for (int c = 0; c < DIM; c++) {
                            d += (p[c] - q[c]) * (p[c] - q[c]);
                        }
                        best = Math.min(best, d);
                    }
                    attractor += Math.sqrt(best);
                    attractorCount++;
                }
            }
            diverged += blew ? 1 : 0;
        }
        return new Stats(Math.sqrt(shortErr / shortCount), attractor / attractorCount, (double) diverged / traj.length);
    }
}
