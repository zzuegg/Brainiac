package io.github.zzuegg.brainiac.experimental.security;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Score-based black-box attacks on a {@link QueryOracle} that returns logits, all untargeted
 * and minimising the margin {@code z_y - max_{j≠y} z_j} until it turns negative, inside an
 * {@code ℓ∞} (or, for Square, {@code ℓ2}) ball of radius {@code ε} around single-channel
 * {@code side x side} images:
 *
 * <ul>
 *   <li>{@link #nes}: natural-evolution-strategies gradient estimates from {@code q}
 *       antithetic Gaussian pairs, followed by signed PGD steps (Ilyas et al., ICML 2018,
 *       <a href="https://arxiv.org/abs/1804.08598">arXiv:1804.08598</a>).</li>
 *   <li>{@link #simba}: SimBA tries one orthonormal direction at a time, pixel or low-frequency
 *       DCT, keeping {@code +α q} or {@code -α q} if the margin drops (Guo et al., ICML 2019,
 *       <a href="https://arxiv.org/abs/1905.07121">arXiv:1905.07121</a>).</li>
 *   <li>{@link #square}: the Square Attack starts from vertical stripes at the corners of the
 *       {@code ℓ∞} ball and resamples random squares of shrinking size, keeping improvements
 *       (Andriushchenko et al., ECCV 2020, <a href="https://arxiv.org/abs/1912.00049">arXiv:1912.00049</a>);
 *       the {@code ℓ2} version adds random square bumps and projects back onto the sphere,
 *       a simplification of the paper's paired-window update.</li>
 *   <li>{@link #randomNoise}: one random corner of the {@code ℓ∞} ball, the control.</li>
 * </ul>
 *
 * <p>All attacks run on a batch of examples at once; every candidate row is one query
 * charged to its example, and an example stops querying once it is misclassified.
 */
public final class BlackBoxAttacks {

    /**
     * The outcome.
     *
     * @param adversarial the final inputs
     * @param success which examples are misclassified
     * @param queries the queries each example used
     */
    public record Result(float[][] adversarial, boolean[] success, long[] queries) {

        /** @return the success rate */
        public double successRate() {
            int s = 0;
            for (boolean b : success) {
                s += b ? 1 : 0;
            }
            return s / (double) success.length;
        }

        /** @return the median queries of the successful examples, or -1 */
        public double medianQueries() {
            List<Long> q = new ArrayList<>();
            for (int i = 0; i < success.length; i++) {
                if (success[i]) {
                    q.add(queries[i]);
                }
            }
            if (q.isEmpty()) {
                return -1;
            }
            q.sort(null);
            int n = q.size();
            return n % 2 == 1 ? q.get(n / 2) : 0.5 * (q.get(n / 2 - 1) + q.get(n / 2));
        }
    }

    /** The search basis of SimBA. */
    public enum Basis {
        /** single pixels */
        PIXEL,
        /** low-frequency 2-D DCT vectors first */
        DCT
    }

    private BlackBoxAttacks() {
    }

    /**
     * NES with antithetic sampling.
     *
     * @param model the oracle
     * @param x clean inputs
     * @param y true classes
     * @param epsilon the {@code ℓ∞} budget
     * @param budget the query budget per example
     * @param pairs antithetic pairs per estimate
     * @param sigma the sampling scale
     * @param seed the seed
     * @return the result
     */
    public static Result nes(QueryOracle model, float[][] x, int[] y, float epsilon, int budget, int pairs,
            float sigma, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        int d = x[0].length;
        float[][] adv = copy(x);
        boolean[] done = check(model, adv, y, all(n), new boolean[n]);
        float lr = epsilon / 4;
        while (true) {
            List<Integer> active = active(model, done, budget, 2 * pairs + 1);
            if (active.isEmpty()) {
                break;
            }
            float[][] rows = new float[active.size() * 2 * pairs][];
            int[] owner = new int[rows.length];
            float[][][] noise = new float[active.size()][pairs][];
            int k = 0;
            for (int a = 0; a < active.size(); a++) {
                int i = active.get(a);
                for (int p = 0; p < pairs; p++) {
                    float[] u = gaussian(d, r);
                    noise[a][p] = u;
                    float[] plus = new float[d];
                    float[] minus = new float[d];
                    for (int j = 0; j < d; j++) {
                        plus[j] = adv[i][j] + sigma * u[j];
                        minus[j] = adv[i][j] - sigma * u[j];
                    }
                    owner[k] = i;
                    rows[k++] = plus;
                    owner[k] = i;
                    rows[k++] = minus;
                }
            }
            float[][] z = model.logits(rows, owner);
            int[] ids = new int[active.size()];
            for (int a = 0; a < active.size(); a++) {
                int i = active.get(a);
                ids[a] = i;
                double[] g = new double[d];
                for (int p = 0; p < pairs; p++) {
                    double diff = QueryOracle.margin(z[(a * pairs + p) * 2], y[i])
                            - QueryOracle.margin(z[(a * pairs + p) * 2 + 1], y[i]);
                    for (int j = 0; j < d; j++) {
                        g[j] += diff * noise[a][p][j];
                    }
                }
                for (int j = 0; j < d; j++) {
                    adv[i][j] = clip(adv[i][j] - lr * (float) Math.signum(g[j]), x[i][j], epsilon);
                }
            }
            check(model, adv, y, ids, done);
        }
        return result(model, adv, done);
    }

    /**
     * SimBA.
     *
     * @param model the oracle
     * @param x clean inputs
     * @param y true classes
     * @param side the image side
     * @param basis pixel or DCT directions
     * @param epsilon the {@code ℓ∞} budget (the step is {@code ε} along a pixel, or
     *     {@code ε side / 2} along a unit DCT vector, then clipped)
     * @param budget the query budget per example
     * @param seed the seed
     * @return the result
     */
    public static Result simba(QueryOracle model, float[][] x, int[] y, int side, Basis basis, float epsilon,
            int budget, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        int d = x[0].length;
        float[][] directions = basis == Basis.PIXEL ? pixelBasis(d) : dctBasis(side);
        int[][] order = new int[n][];
        for (int i = 0; i < n; i++) {
            order[i] = basis == Basis.PIXEL ? permutation(d, r) : frequencyOrder(side, r);
        }
        float step = basis == Basis.PIXEL ? epsilon : epsilon * side / 2f;
        float[][] adv = copy(x);
        float[] margin = new float[n];
        float[][] z0 = model.logits(adv, all(n));
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            margin[i] = QueryOracle.margin(z0[i], y[i]);
            done[i] = margin[i] < 0;
        }
        int[] position = new int[n];
        while (true) {
            List<Integer> active = active(model, done, budget, 2);
            if (active.isEmpty()) {
                break;
            }
            boolean[] improved = new boolean[n];
            for (int sign = 1; sign >= -1; sign -= 2) {
                List<Integer> trying = new ArrayList<>();
                List<float[]> rows = new ArrayList<>();
                for (int i : active) {
                    if (improved[i]) {
                        continue;
                    }
                    float[] q = directions[order[i][position[i] % d]];
                    float[] cand = new float[d];
                    for (int j = 0; j < d; j++) {
                        cand[j] = clip(adv[i][j] + sign * step * q[j], x[i][j], epsilon);
                    }
                    trying.add(i);
                    rows.add(cand);
                }
                if (trying.isEmpty()) {
                    continue;
                }
                int[] owner = trying.stream().mapToInt(Integer::intValue).toArray();
                float[][] cands = rows.toArray(new float[0][]);
                float[][] z = model.logits(cands, owner);
                for (int t = 0; t < owner.length; t++) {
                    int i = owner[t];
                    float m = QueryOracle.margin(z[t], y[i]);
                    if (m < margin[i]) {
                        margin[i] = m;
                        adv[i] = cands[t];
                        done[i] = m < 0;
                        improved[i] = true;
                    }
                }
            }
            for (int i : active) {
                position[i]++;
            }
        }
        return result(model, adv, done);
    }

    /**
     * The Square Attack.
     *
     * @param model the oracle
     * @param x clean inputs
     * @param y true classes
     * @param side the image side
     * @param linf {@code ℓ∞} if true, else {@code ℓ2}
     * @param epsilon the budget
     * @param budget the query budget per example
     * @param pInit the initial share of pixels changed, such as 0.3
     * @param seed the seed
     * @return the result
     */
    public static Result square(QueryOracle model, float[][] x, int[] y, int side, boolean linf, float epsilon,
            int budget, double pInit, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        int d = x[0].length;
        float[][] delta = new float[n][d];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < side; c++) {
                float s = r.nextBoolean() ? epsilon : -epsilon;
                for (int row = 0; row < side; row++) {
                    delta[i][row * side + c] = s;
                }
            }
            if (!linf) {
                rescale(delta[i], epsilon);
            }
        }
        float[][] adv = new float[n][];
        for (int i = 0; i < n; i++) {
            adv[i] = add(x[i], delta[i]);
        }
        float[][] z0 = model.logits(adv, all(n));
        float[] margin = new float[n];
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            margin[i] = QueryOracle.margin(z0[i], y[i]);
            done[i] = margin[i] < 0;
        }
        int iteration = 0;
        while (true) {
            List<Integer> active = active(model, done, budget, 1);
            if (active.isEmpty()) {
                break;
            }
            double p = pInit * schedule(iteration++, budget);
            int h = (int) Math.max(1, Math.min(side - 1, Math.round(Math.sqrt(p * d))));
            float[][] rows = new float[active.size()][];
            float[][] cand = new float[active.size()][];
            int[] owner = new int[active.size()];
            for (int a = 0; a < active.size(); a++) {
                int i = active.get(a);
                owner[a] = i;
                float[] nd = delta[i].clone();
                int r0 = r.nextInt(side - h + 1);
                int c0 = r.nextInt(side - h + 1);
                if (linf) {
                    float s = r.nextBoolean() ? epsilon : -epsilon;
                    for (int rr = r0; rr < r0 + h; rr++) {
                        for (int cc = c0; cc < c0 + h; cc++) {
                            nd[rr * side + cc] = s;
                        }
                    }
                } else {
                    float s = (float) ((r.nextBoolean() ? 1 : -1) * epsilon / Math.sqrt(2.0));
                    double centre = (h - 1) / 2.0;
                    double norm = 0;
                    double[] bump = new double[h * h];
                    for (int rr = 0; rr < h; rr++) {
                        for (int cc = 0; cc < h; cc++) {
                            double dist = Math.max(Math.abs(rr - centre), Math.abs(cc - centre));
                            bump[rr * h + cc] = 1 / (dist + 1);
                            norm += bump[rr * h + cc] * bump[rr * h + cc];
                        }
                    }
                    for (int rr = 0; rr < h; rr++) {
                        for (int cc = 0; cc < h; cc++) {
                            nd[(r0 + rr) * side + c0 + cc] += (float) (s * bump[rr * h + cc] / Math.sqrt(norm));
                        }
                    }
                    rescale(nd, epsilon);
                }
                cand[a] = nd;
                rows[a] = add(x[i], nd);
            }
            float[][] z = model.logits(rows, owner);
            for (int a = 0; a < owner.length; a++) {
                int i = owner[a];
                float m = QueryOracle.margin(z[a], y[i]);
                if (m < margin[i]) {
                    margin[i] = m;
                    delta[i] = cand[a];
                    adv[i] = rows[a];
                    done[i] = m < 0;
                }
            }
        }
        return result(model, adv, done);
    }

    /**
     * One random corner of the {@code ℓ∞} ball per example.
     *
     * @param model the oracle
     * @param x clean inputs
     * @param y true classes
     * @param epsilon the budget
     * @param seed the seed
     * @return the result
     */
    public static Result randomNoise(QueryOracle model, float[][] x, int[] y, float epsilon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] adv = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            adv[i] = new float[x[i].length];
            for (int j = 0; j < adv[i].length; j++) {
                adv[i][j] = x[i][j] + (r.nextBoolean() ? epsilon : -epsilon);
            }
        }
        boolean[] done = check(model, adv, y, all(x.length), new boolean[x.length]);
        return result(model, adv, done);
    }

    /** Square's schedule of the changed share, halving at the paper's checkpoints rescaled to the budget. */
    static double schedule(int it, int budget) {
        double t = it * 10000.0 / Math.max(1, budget);
        int[] points = {10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000};
        double p = 1;
        for (int c : points) {
            if (t > c) {
                p /= 2;
            }
        }
        return p;
    }

    static float[][] pixelBasis(int d) {
        float[][] b = new float[d][d];
        for (int i = 0; i < d; i++) {
            b[i][i] = 1;
        }
        return b;
    }

    /**
     * The orthonormal 2-D DCT-II basis of {@code side x side} images, frequency {@code (u, v)}
     * at index {@code u side + v}.
     *
     * @param side the image side
     * @return {@code side²} unit vectors
     */
    public static float[][] dctBasis(int side) {
        float[][] b = new float[side * side][side * side];
        for (int u = 0; u < side; u++) {
            for (int v = 0; v < side; v++) {
                double au = u == 0 ? Math.sqrt(1.0 / side) : Math.sqrt(2.0 / side);
                double av = v == 0 ? Math.sqrt(1.0 / side) : Math.sqrt(2.0 / side);
                for (int py = 0; py < side; py++) {
                    for (int px = 0; px < side; px++) {
                        b[u * side + v][py * side + px] = (float) (au * av
                                * Math.cos(Math.PI * (2 * py + 1) * u / (2.0 * side))
                                * Math.cos(Math.PI * (2 * px + 1) * v / (2.0 * side)));
                    }
                }
            }
        }
        return b;
    }

    private static int[] frequencyOrder(int side, SplittableRandom r) {
        Integer[] idx = new Integer[side * side];
        double[] key = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
            key[i] = Math.max(i / side, i % side) + r.nextDouble() * 0.5;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(key[a], key[b]));
        int[] out = new int[idx.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    static int[] permutation(int d, SplittableRandom r) {
        int[] p = new int[d];
        for (int i = 0; i < d; i++) {
            p[i] = i;
        }
        for (int i = d - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    static List<Integer> active(QueryOracle model, boolean[] done, int budget, int cost) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < done.length; i++) {
            if (!done[i] && model.queries(i) + cost <= budget) {
                out.add(i);
            }
        }
        return out;
    }

    static boolean[] check(QueryOracle model, float[][] adv, int[] y, int[] ids, boolean[] done) {
        float[][] rows = new float[ids.length][];
        for (int k = 0; k < ids.length; k++) {
            rows[k] = adv[ids[k]];
        }
        float[][] z = model.logits(rows, ids);
        for (int k = 0; k < ids.length; k++) {
            done[ids[k]] = QueryOracle.margin(z[k], y[ids[k]]) < 0;
        }
        return done;
    }

    static Result result(QueryOracle model, float[][] adv, boolean[] done) {
        long[] q = new long[done.length];
        for (int i = 0; i < q.length; i++) {
            q[i] = model.queries(i);
        }
        return new Result(adv, done.clone(), q);
    }

    static int[] all(int n) {
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) {
            ids[i] = i;
        }
        return ids;
    }

    static float[][] copy(float[][] x) {
        float[][] out = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].clone();
        }
        return out;
    }

    static float[] gaussian(int d, SplittableRandom r) {
        float[] u = new float[d];
        for (int j = 0; j < d; j++) {
            u[j] = (float) r.nextGaussian();
        }
        return u;
    }

    static float clip(float v, float centre, float epsilon) {
        return Math.clamp(v, centre - epsilon, centre + epsilon);
    }

    static float[] add(float[] a, float[] b) {
        float[] out = new float[a.length];
        for (int j = 0; j < a.length; j++) {
            out[j] = a[j] + b[j];
        }
        return out;
    }

    static void rescale(float[] v, float radius) {
        double s = 0;
        for (float f : v) {
            s += f * (double) f;
        }
        double norm = Math.sqrt(s);
        if (norm > radius) {
            float k = (float) (radius / norm);
            for (int j = 0; j < v.length; j++) {
                v[j] *= k;
            }
        }
    }
}
