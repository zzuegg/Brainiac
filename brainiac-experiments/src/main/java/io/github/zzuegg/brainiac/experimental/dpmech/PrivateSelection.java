package io.github.zzuegg.brainiac.experimental.dpmech;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Differentially private selection of the best of {@code k} candidates with utilities
 * {@code u_i} of sensitivity {@code Δ}, every mechanism {@code ε}-DP:
 *
 * <ul>
 *   <li>{@link Mechanism#EXPONENTIAL}: sample {@code i} with probability
 *       {@code ∝ exp(ε u_i / 2Δ)} (McSherry and Talwar, FOCS 2007,
 *       <a href="https://doi.org/10.1109/FOCS.2007.66">doi:10.1109/FOCS.2007.66</a>).</li>
 *   <li>{@link Mechanism#NOISY_MAX_LAPLACE}: argmax of {@code u_i + Lap(2Δ/ε)}.</li>
 *   <li>{@link Mechanism#NOISY_MAX_GUMBEL}: argmax of {@code u_i + Gumbel(2Δ/ε)}, which is
 *       distributed exactly like the exponential mechanism.</li>
 *   <li>{@link Mechanism#PERMUTE_AND_FLIP}: visit the candidates in random order and accept
 *       each with probability {@code exp(ε (u_i - u*) / 2Δ)}; it is never worse than the
 *       exponential mechanism in expected utility (McKenna and Sheldon, NeurIPS 2020,
 *       <a href="https://arxiv.org/abs/2010.12603">arXiv:2010.12603</a>).</li>
 * </ul>
 */
public final class PrivateSelection {

    /** A selection mechanism. */
    public enum Mechanism {
        /** the exponential mechanism */
        EXPONENTIAL,
        /** report-noisy-max with Laplace noise */
        NOISY_MAX_LAPLACE,
        /** report-noisy-max with Gumbel noise */
        NOISY_MAX_GUMBEL,
        /** permute-and-flip */
        PERMUTE_AND_FLIP
    }

    private PrivateSelection() {
    }

    /**
     * Selects one candidate.
     *
     * @param mechanism the mechanism
     * @param utility the candidates' utilities
     * @param epsilon the budget
     * @param sensitivity the utilities' sensitivity {@code Δ}
     * @param random the random source
     * @return the chosen index
     */
    public static int select(Mechanism mechanism, double[] utility, double epsilon, double sensitivity,
            SplittableRandom random) {
        Objects.requireNonNull(mechanism);
        int k = utility.length;
        double scale = 2 * sensitivity / epsilon;
        switch (mechanism) {
            case EXPONENTIAL -> {
                double[] p = exponentialProbabilities(utility, epsilon, sensitivity);
                double u = random.nextDouble();
                double acc = 0;
                for (int i = 0; i < k; i++) {
                    acc += p[i];
                    if (u < acc) {
                        return i;
                    }
                }
                return k - 1;
            }
            case NOISY_MAX_LAPLACE, NOISY_MAX_GUMBEL -> {
                int best = 0;
                double bestV = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < k; i++) {
                    double noise = mechanism == Mechanism.NOISY_MAX_LAPLACE ? LocalDp.laplace(scale, random)
                            : -scale * Math.log(-Math.log(Math.max(random.nextDouble(), 1e-300)));
                    double v = utility[i] + noise;
                    if (v > bestV) {
                        bestV = v;
                        best = i;
                    }
                }
                return best;
            }
            case PERMUTE_AND_FLIP -> {
                double max = Double.NEGATIVE_INFINITY;
                for (double u : utility) {
                    max = Math.max(max, u);
                }
                while (true) {
                    int[] order = permutation(k, random);
                    for (int i : order) {
                        if (random.nextDouble() < Math.exp((utility[i] - max) / scale)) {
                            return i;
                        }
                    }
                }
            }
            default -> throw new IllegalStateException();
        }
    }

    /**
     * The exponential mechanism's selection probabilities.
     *
     * @param utility the utilities
     * @param epsilon the budget
     * @param sensitivity the sensitivity
     * @return the probabilities
     */
    public static double[] exponentialProbabilities(double[] utility, double epsilon, double sensitivity) {
        double max = Double.NEGATIVE_INFINITY;
        for (double u : utility) {
            max = Math.max(max, u);
        }
        double[] p = new double[utility.length];
        double z = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp(epsilon * (utility[i] - max) / (2 * sensitivity));
            z += p[i];
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= z;
        }
        return p;
    }

    /**
     * The expected utility gap {@code u* - E[u_chosen]} by simulation (exact for the
     * exponential mechanism).
     *
     * @param mechanism the mechanism
     * @param utility the utilities
     * @param epsilon the budget
     * @param sensitivity the sensitivity
     * @param trials simulations
     * @param seed the seed
     * @return the expected gap
     */
    public static double expectedGap(Mechanism mechanism, double[] utility, double epsilon, double sensitivity,
            int trials, long seed) {
        double max = Double.NEGATIVE_INFINITY;
        for (double u : utility) {
            max = Math.max(max, u);
        }
        if (mechanism == Mechanism.EXPONENTIAL) {
            double[] p = exponentialProbabilities(utility, epsilon, sensitivity);
            double g = 0;
            for (int i = 0; i < p.length; i++) {
                g += p[i] * (max - utility[i]);
            }
            return g;
        }
        SplittableRandom r = new SplittableRandom(seed);
        double g = 0;
        for (int t = 0; t < trials; t++) {
            g += (max - utility[select(mechanism, utility, epsilon, sensitivity, r)]) / trials;
        }
        return g;
    }

    static int[] permutation(int k, SplittableRandom r) {
        int[] p = new int[k];
        for (int i = 0; i < k; i++) {
            p[i] = i;
        }
        for (int i = k - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
