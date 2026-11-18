package io.github.zzuegg.brainiac.experimental.security;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Pre-aggregation steps that make robust rules work under heterogeneous (non-IID) clients,
 * each an {@link Aggregator} composing with an inner rule from {@link RobustAggregators}:
 *
 * <ul>
 *   <li>{@link #bucketing}: average random groups of {@code s} updates before the inner rule,
 *       so honest inputs look more alike (Karimireddy, He and Jaggi, ICLR 2022,
 *       <a href="https://arxiv.org/abs/2006.09365">arXiv:2006.09365</a>).</li>
 *   <li>{@link #centeredClip}: clip every update to a radius {@code τ} around the previous
 *       aggregate and iterate, with worker momentum so the history of every slot is
 *       remembered (Karimireddy, He and Jaggi, ICML 2021,
 *       <a href="https://arxiv.org/abs/2012.10333">arXiv:2012.10333</a>).</li>
 *   <li>{@link #nnm}: nearest-neighbour mixing replaces every update by the mean of its
 *       {@code n - f} nearest neighbours (itself included) before the inner rule (Allouah et
 *       al., AISTATS 2023, <a href="https://arxiv.org/abs/2302.01772">arXiv:2302.01772</a>).</li>
 * </ul>
 */
public final class PreAggregation {

    private PreAggregation() {
    }

    /**
     * Bucketing.
     *
     * @param s the bucket size
     * @param inner the rule applied to the bucket means
     * @param seed the seed of the random grouping
     * @return the aggregator
     */
    public static Aggregator bucketing(int s, Aggregator inner, long seed) {
        Objects.requireNonNull(inner);
        SplittableRandom r = new SplittableRandom(seed);
        return u -> {
            int n = u.length;
            int[] perm = new int[n];
            for (int i = 0; i < n; i++) {
                perm[i] = i;
            }
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }
            int buckets = (n + s - 1) / s;
            float[][] means = new float[buckets][];
            for (int b = 0; b < buckets; b++) {
                int to = Math.min(n, (b + 1) * s);
                float[][] group = new float[to - b * s][];
                for (int i = b * s; i < to; i++) {
                    group[i - b * s] = u[perm[i]];
                }
                means[b] = RobustAggregators.mean(group);
            }
            return inner.aggregate(means);
        };
    }

    /**
     * Nearest-neighbour mixing.
     *
     * @param f tolerated malicious clients
     * @param inner the rule applied to the mixed updates
     * @return the aggregator
     */
    public static Aggregator nnm(int f, Aggregator inner) {
        Objects.requireNonNull(inner);
        return u -> inner.aggregate(mix(u, f));
    }

    /**
     * The mixing step of {@link #nnm}.
     *
     * @param u the updates
     * @param f tolerated malicious clients
     * @return every update replaced by the mean of its {@code n - f} nearest neighbours
     */
    public static float[][] mix(float[][] u, int f) {
        int n = u.length;
        double[][] d = RobustAggregators.pairwise(u);
        int k = Math.max(1, n - f);
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            Integer[] order = new Integer[n];
            for (int j = 0; j < n; j++) {
                order[j] = j;
            }
            double[] row = d[i];
            Arrays.sort(order, (a, b) -> Double.compare(row[a], row[b]));
            float[][] near = new float[k][];
            for (int j = 0; j < k; j++) {
                near[j] = u[order[j]];
            }
            out[i] = RobustAggregators.mean(near);
        }
        return out;
    }

    /**
     * Centred clipping with worker momentum: each slot keeps {@code m_i ← β m_i + (1-β) Δ_i},
     * and the server iterates {@code v ← v + mean_i clip(m_i - v, τ)} from the previous
     * aggregate {@code v}.
     *
     * @param tau the clipping radius, or {@code <= 0} for an adaptive radius: the median
     *     distance of this round's (momentum) updates to the previous aggregate
     * @param iterations clipping iterations per round, 1 to 3
     * @param momentum the worker momentum {@code β}, 0 for none
     * @return a stateful aggregator, one per run
     */
    public static Aggregator centeredClip(double tau, int iterations, double momentum) {
        return new Aggregator() {
            private float[] center;
            private float[][] memory;

            @Override
            public float[] aggregate(float[][] u) {
                int n = u.length;
                int d = u[0].length;
                if (center == null) {
                    center = new float[d];
                    memory = new float[n][];
                }
                float[][] m = new float[n][];
                for (int i = 0; i < n; i++) {
                    if (memory[i] == null || momentum == 0) {
                        memory[i] = u[i].clone();
                    } else {
                        for (int j = 0; j < d; j++) {
                            memory[i][j] = (float) (momentum * memory[i][j] + (1 - momentum) * u[i][j]);
                        }
                    }
                    m[i] = memory[i];
                }
                float[] v = center.clone();
                double radius = tau;
                if (radius <= 0) {
                    double[] dist = new double[n];
                    for (int i = 0; i < n; i++) {
                        dist[i] = Math.sqrt(RobustAggregators.distance2(m[i], v));
                    }
                    Arrays.sort(dist);
                    radius = dist[n / 2];
                }
                for (int t = 0; t < iterations; t++) {
                    double[] step = new double[d];
                    for (float[] mi : m) {
                        double norm = Math.sqrt(RobustAggregators.distance2(mi, v));
                        double scale = Math.min(1, radius / Math.max(norm, 1e-12));
                        for (int j = 0; j < d; j++) {
                            step[j] += scale * (mi[j] - v[j]) / n;
                        }
                    }
                    for (int j = 0; j < d; j++) {
                        v[j] += (float) step[j];
                    }
                }
                center = v;
                return v.clone();
            }
        };
    }
}
