package io.github.zzuegg.brainiac.experimental.dpmech;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * The sparse vector technique: answer a stream of threshold queries "is {@code q_i(D) >= T}?"
 * paying privacy only for the positive answers. {@link Variant#CORRECT} is AboveThreshold
 * with {@code c} positives as in Lyu, Su and Li's Algorithm 1 (VLDB 2017,
 * <a href="https://arxiv.org/abs/1603.01699">arXiv:1603.01699</a>): {@code ρ ~ Lap(Δ/ε₁)}
 * once, {@code ν_i ~ Lap(2cΔ/ε₂)} per query, {@code ε₁ = ε₂ = ε/2}, stop after {@code c}
 * positives. The known-broken variants of that paper's Figure 1 are kept as negative
 * controls:
 *
 * <ul>
 *   <li>{@link Variant#NO_QUERY_NOISE} (their Algorithm 5): no noise on the queries.</li>
 *   <li>{@link Variant#UNBOUNDED} (their Algorithm 6): query noise {@code Lap(Δ/ε₂)} and no
 *       limit on positives.</li>
 * </ul>
 *
 * <p>{@link #empiricalEpsilon} estimates the privacy loss of an output pattern on a pair of
 * neighbouring query vectors by simulation, which exposes the broken variants.
 */
public final class SparseVector {

    /** An algorithm variant. */
    public enum Variant {
        /** AboveThreshold with correct noise scales */
        CORRECT,
        /** no query noise (broken) */
        NO_QUERY_NOISE,
        /** too little query noise and no cut-off (broken) */
        UNBOUNDED
    }

    private SparseVector() {
    }

    /**
     * Runs the mechanism.
     *
     * @param variant the variant
     * @param queries the query answers {@code q_i(D)}
     * @param threshold {@code T}
     * @param epsilon the claimed budget
     * @param sensitivity {@code Δ}
     * @param c positives allowed
     * @param random the random source
     * @return one answer per query: 1 above, 0 below, -1 not answered (stopped)
     */
    public static int[] run(Variant variant, double[] queries, double threshold, double epsilon, double sensitivity,
            int c, SplittableRandom random) {
        Objects.requireNonNull(variant);
        double e1 = epsilon / 2;
        double e2 = epsilon / 2;
        double rho = LocalDp.laplace(sensitivity / e1, random);
        int[] out = new int[queries.length];
        int positives = 0;
        for (int i = 0; i < queries.length; i++) {
            if (variant != Variant.UNBOUNDED && positives >= c) {
                out[i] = -1;
                continue;
            }
            double nu = switch (variant) {
                case CORRECT -> LocalDp.laplace(2 * c * sensitivity / e2, random);
                case NO_QUERY_NOISE -> 0;
                case UNBOUNDED -> LocalDp.laplace(sensitivity / e2, random);
            };
            if (queries[i] + nu >= threshold + rho) {
                out[i] = 1;
                positives++;
            }
        }
        return out;
    }

    /**
     * The empirical privacy loss: the largest {@code |ln P[o | D] / P[o | D']|} over output
     * patterns {@code o} seen at least {@code minCount} times under one of the datasets, with
     * add-one smoothing.
     *
     * @param variant the variant
     * @param q the queries on {@code D}
     * @param qPrime the queries on a neighbour {@code D'} (each differs by at most {@code Δ})
     * @param threshold {@code T}
     * @param epsilon the claimed budget
     * @param sensitivity {@code Δ}
     * @param c positives allowed
     * @param trials runs per dataset
     * @param minCount the smallest count of a pattern considered
     * @param seed the seed
     * @return the estimated privacy loss
     */
    public static double empiricalEpsilon(Variant variant, double[] q, double[] qPrime, double threshold,
            double epsilon, double sensitivity, int c, int trials, int minCount, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<String, int[]> counts = new HashMap<>();
        for (int t = 0; t < trials; t++) {
            counts.computeIfAbsent(key(run(variant, q, threshold, epsilon, sensitivity, c, r)), k -> new int[2])[0]++;
            counts.computeIfAbsent(key(run(variant, qPrime, threshold, epsilon, sensitivity, c, r)),
                    k -> new int[2])[1]++;
        }
        double worst = 0;
        for (int[] v : counts.values()) {
            if (Math.max(v[0], v[1]) < minCount) {
                continue;
            }
            worst = Math.max(worst, Math.abs(Math.log((v[0] + 1.0) / (v[1] + 1.0))));
        }
        return worst;
    }

    private static String key(int[] out) {
        StringBuilder b = new StringBuilder(out.length);
        for (int o : out) {
            b.append(o < 0 ? '-' : (char) ('0' + o));
        }
        return b.toString();
    }
}
