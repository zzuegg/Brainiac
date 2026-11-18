package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Reliable {@code ℓ∞} robustness evaluation after Croce and Hein (ICML 2020,
 * <a href="https://arxiv.org/abs/2003.01690">arXiv:2003.01690</a>).
 *
 * <p>{@link #apgd}: auto-PGD with momentum {@code α = 0.75}, step size {@code η = 2ε} halved at
 * the checkpoints {@code w_j = ⌈p_j N⌉} ({@code p_0 = 0}, {@code p_1 = 0.22},
 * {@code p_{j+1} = p_j + max(p_j - p_{j-1} - 0.03, 0.06)}) when fewer than {@code ρ = 0.75}
 * of the steps since the last checkpoint increased the loss, or when neither the step size
 * nor the best loss changed; after a halving the iterate restarts from the best point. The
 * losses are cross-entropy, the scale-invariant difference of logits ratio
 * {@code DLR = -(z_y - max_{i≠y} z_i) / (z_π1 - z_π3)}, and its targeted version
 * {@code -(z_y - z_t) / (z_π1 - (z_π3 + z_π4)/2)}, whose denominators (sorted logits) are
 * treated as constants in the gradient, which leaves the sign step unchanged.
 *
 * <p>{@link #evaluate}: the standard ensemble, run in sequence on the examples still robust:
 * APGD-CE, targeted APGD-DLR against the three most likely wrong classes, and the
 * black-box Square Attack ({@link BlackBoxAttacks#square}); an example counts as robust only
 * if every attack fails. FAB is not included.
 */
public final class AutoAttack {

    /** A loss maximised by APGD. */
    public enum Loss {
        /** cross-entropy */
        CE,
        /** difference of logits ratio */
        DLR,
        /** targeted difference of logits ratio */
        DLR_TARGETED
    }

    /**
     * Robust accuracy after each stage of the ensemble.
     *
     * @param clean clean accuracy
     * @param afterCe after APGD-CE
     * @param afterDlr after targeted APGD-DLR
     * @param afterSquare after Square, the AutoAttack robust accuracy
     */
    public record Report(double clean, double afterCe, double afterDlr, double afterSquare) {
    }

    private AutoAttack() {
    }

    /**
     * APGD on a batch.
     *
     * @param model the logits function
     * @param x clean inputs
     * @param y true classes
     * @param loss the loss
     * @param targets the target class per row (for the targeted loss), else {@code null}
     * @param epsilon the {@code ℓ∞} budget
     * @param iterations the iterations {@code N}
     * @param broken receives, per row, whether a misclassified iterate was found
     * @param seed the seed of the random start
     * @return per row, the first misclassified iterate found, else the best-loss point
     */
    public static float[][] apgd(UnaryOperator<Var> model, float[][] x, int[] y, Loss loss, int[] targets,
            float epsilon, int iterations, boolean[] broken, long seed) {
        int n = x.length;
        int d = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        float[][] cur = new float[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                cur[i][j] = x[i][j] + (float) r.nextDouble(-epsilon / 2, epsilon / 2);
            }
        }
        double[] eta = new double[n];
        java.util.Arrays.fill(eta, 2 * epsilon);
        List<Integer> checkpoints = checkpoints(iterations);
        float[][] best = BlackBoxAttacks.copy(cur);
        float[][] found = new float[n][];
        double[] fBest = new double[n];
        float[][] grad = new float[n][];
        double[] f = evaluate(model, cur, y, loss, targets, grad, found);
        System.arraycopy(f, 0, fBest, 0, n);
        float[][] prev = BlackBoxAttacks.copy(cur);
        int[] increases = new int[n];
        double[] etaAtLast = eta.clone();
        double[] fBestAtLast = fBest.clone();
        int last = 0;
        for (int k = 0; k < iterations; k++) {
            float[][] next = new float[n][d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    float z = clip(cur[i][j] + (float) (eta[i] * Math.signum(grad[i][j])), x[i][j], epsilon);
                    float v = k == 0 ? z : cur[i][j] + 0.75f * (z - cur[i][j]) + 0.25f * (cur[i][j] - prev[i][j]);
                    next[i][j] = clip(v, x[i][j], epsilon);
                }
            }
            double[] fNext = evaluate(model, next, y, loss, targets, grad, found);
            for (int i = 0; i < n; i++) {
                if (fNext[i] > f[i]) {
                    increases[i]++;
                }
                if (fNext[i] > fBest[i]) {
                    fBest[i] = fNext[i];
                    best[i] = next[i].clone();
                }
            }
            prev = cur;
            cur = next;
            f = fNext;
            if (checkpoints.contains(k + 1)) {
                int span = k + 1 - last;
                boolean recompute = false;
                for (int i = 0; i < n; i++) {
                    boolean slow = increases[i] < 0.75 * span;
                    boolean stuck = etaAtLast[i] == eta[i] && fBestAtLast[i] == fBest[i];
                    etaAtLast[i] = eta[i];
                    fBestAtLast[i] = fBest[i];
                    if (slow || stuck) {
                        eta[i] /= 2;
                        cur[i] = best[i].clone();
                        prev[i] = best[i].clone();
                        recompute = true;
                    }
                    increases[i] = 0;
                }
                last = k + 1;
                if (recompute) {
                    f = evaluate(model, cur, y, loss, targets, grad, found);
                }
            }
        }
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            broken[i] = found[i] != null;
            out[i] = found[i] != null ? found[i] : best[i];
        }
        return out;
    }

    /**
     * The full ensemble.
     *
     * @param model the logits function
     * @param x clean inputs
     * @param y true classes
     * @param side the image side (for Square)
     * @param epsilon the {@code ℓ∞} budget
     * @param iterations APGD iterations
     * @param squareQueries Square's query budget per example
     * @param seed the seed
     * @return robust accuracy after each stage
     */
    public static Report evaluate(UnaryOperator<Var> model, float[][] x, int[] y, int side, float epsilon,
            int iterations, int squareQueries, long seed) {
        int n = x.length;
        float[][] z = logits(model, x);
        boolean[] robust = new boolean[n];
        for (int i = 0; i < n; i++) {
            robust[i] = QueryOracle.argmax(z[i]) == y[i];
        }
        double clean = share(robust);
        attackRemaining(model, x, y, robust, Loss.CE, null, epsilon, iterations, seed);
        double afterCe = share(robust);
        for (int t = 1; t <= 3; t++) {
            int[] targets = new int[n];
            for (int i = 0; i < n; i++) {
                targets[i] = rankedWrong(z[i], y[i], t);
            }
            attackRemaining(model, x, y, robust, Loss.DLR_TARGETED, targets, epsilon, iterations, seed + t);
        }
        double afterDlr = share(robust);
        List<Integer> left = indices(robust);
        if (!left.isEmpty()) {
            float[][] sx = new float[left.size()][];
            int[] sy = new int[left.size()];
            for (int k = 0; k < sx.length; k++) {
                sx[k] = x[left.get(k)];
                sy[k] = y[left.get(k)];
            }
            BlackBoxAttacks.Result sq = BlackBoxAttacks.square(new QueryOracle(model, sx.length), sx, sy, side, true,
                    epsilon, squareQueries, 0.3, seed + 7);
            for (int k = 0; k < sx.length; k++) {
                if (sq.success()[k]) {
                    robust[left.get(k)] = false;
                }
            }
        }
        return new Report(clean, afterCe, afterDlr, share(robust));
    }

    private static void attackRemaining(UnaryOperator<Var> model, float[][] x, int[] y, boolean[] robust, Loss loss,
            int[] targets, float epsilon, int iterations, long seed) {
        List<Integer> left = indices(robust);
        if (left.isEmpty()) {
            return;
        }
        float[][] sx = new float[left.size()][];
        int[] sy = new int[left.size()];
        int[] st = targets == null ? null : new int[left.size()];
        for (int k = 0; k < sx.length; k++) {
            sx[k] = x[left.get(k)];
            sy[k] = y[left.get(k)];
            if (st != null) {
                st[k] = targets[left.get(k)];
            }
        }
        boolean[] broken = new boolean[sx.length];
        apgd(model, sx, sy, loss, st, epsilon, iterations, broken, seed);
        for (int k = 0; k < sx.length; k++) {
            if (broken[k]) {
                robust[left.get(k)] = false;
            }
        }
    }

    /**
     * APGD's checkpoints.
     *
     * @param iterations {@code N}
     * @return the iterations {@code w_j} at which the step size is reconsidered
     */
    public static List<Integer> checkpoints(int iterations) {
        List<Integer> out = new ArrayList<>();
        double pPrev = 0;
        double p = 0.22;
        while (p < 1) {
            out.add((int) Math.ceil(p * iterations));
            double next = p + Math.max(p - pPrev - 0.03, 0.06);
            pPrev = p;
            p = next;
        }
        return out;
    }

    /** Per-row losses (to maximise), their input gradients, and the first misclassified iterate. */
    private static double[] evaluate(UnaryOperator<Var> model, float[][] x, int[] y, Loss loss, int[] targets,
            float[][] gradOut, float[][] found) {
        int n = x.length;
        Var in = Var.param(Tensor.of(x));
        Var z = model.apply(in);
        int k = z.shape().dim(1);
        float[] zv = z.value().toArray();
        float[] ids = new float[n];
        for (int i = 0; i < n; i++) {
            ids[i] = y[i];
        }
        Tensor idT = Tensor.of(Shape.of(n), ids);
        Var perRow;
        if (loss == Loss.CE) {
            perRow = Functional.crossEntropyPerRow(z, idT);
        } else {
            float[] mask = new float[n * k];
            float[] other = new float[n * k];
            float[] scale = new float[n];
            for (int i = 0; i < n; i++) {
                float[] row = new float[k];
                System.arraycopy(zv, i * k, row, 0, k);
                float[] sorted = row.clone();
                java.util.Arrays.sort(sorted);
                double den = loss == Loss.DLR ? sorted[k - 1] - sorted[k - 3]
                        : sorted[k - 1] - 0.5 * (sorted[k - 3] + sorted[Math.max(0, k - 4)]);
                scale[i] = (float) (1 / Math.max(den, 1e-12));
                for (int c = 0; c < k; c++) {
                    mask[i * k + c] = c == y[i] ? -1e4f : 0f;
                }
                if (loss == Loss.DLR_TARGETED) {
                    other[i * k + targets[i]] = 1;
                }
            }
            Var zy = z.takeAlong(Tensor.of(Shape.of(n, 1), ids)).reshape(n);
            Var zo = loss == Loss.DLR ? z.add(Tensor.of(Shape.of(n, k), mask)).max(1, false)
                    : z.mul(Tensor.of(Shape.of(n, k), other)).sum(1, false);
            perRow = zo.sub(zy).mul(Tensor.of(Shape.of(n), scale));
        }
        Tensor g = Var.gradients(perRow.sum(), List.of(in), false).get(0).value();
        float[] gv = g.toArray();
        float[] lv = perRow.value().toArray();
        int d = x[0].length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            gradOut[i] = new float[d];
            System.arraycopy(gv, i * d, gradOut[i], 0, d);
            out[i] = lv[i];
            float[] row = new float[k];
            System.arraycopy(zv, i * k, row, 0, k);
            if (found[i] == null && QueryOracle.argmax(row) != y[i]) {
                found[i] = x[i].clone();
            }
        }
        return out;
    }

    private static int rankedWrong(float[] z, int y, int rank) {
        boolean[] used = new boolean[z.length];
        used[y] = true;
        int pick = -1;
        for (int t = 0; t < rank; t++) {
            pick = -1;
            for (int c = 0; c < z.length; c++) {
                if (!used[c] && (pick < 0 || z[c] > z[pick])) {
                    pick = c;
                }
            }
            used[pick] = true;
        }
        return pick;
    }

    private static float[][] logits(UnaryOperator<Var> model, float[][] x) {
        Tensor z = model.apply(Var.of(Tensor.of(x))).value();
        int k = z.shape().dim(1);
        float[] v = z.toArray();
        float[][] out = new float[x.length][k];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(v, i * k, out[i], 0, k);
        }
        return out;
    }

    private static List<Integer> indices(boolean[] b) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < b.length; i++) {
            if (b[i]) {
                out.add(i);
            }
        }
        return out;
    }

    private static double share(boolean[] b) {
        return indices(b).size() / (double) b.length;
    }

    private static float clip(float v, float centre, float epsilon) {
        return Math.clamp(v, centre - epsilon, centre + epsilon);
    }
}
