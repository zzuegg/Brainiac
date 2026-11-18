package io.github.zzuegg.brainiac.experimental.security;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Model-poisoning attacks by malicious federated clients, crafted in update space:
 *
 * <ul>
 *   <li>{@link Attack#SIGN_FLIP}: send {@code -scale Δ}.</li>
 *   <li>{@link Attack#GAUSSIAN}: send Gaussian noise whose norm is {@code scale} times the
 *       mean honest norm ({@code scale = 1} is the "random update of the same norm"
 *       control).</li>
 *   <li>{@link Attack#ALIE}: "a little is enough" moves every coordinate of the mean by
 *       {@code z} standard deviations of the honest updates, with
 *       {@code z = Φ⁻¹((n - s)/n)}, {@code s = ⌊n/2 + 1⌋ - f}, so the values stay inside the
 *       median's acceptance region (Baruch et al., NeurIPS 2019,
 *       <a href="https://arxiv.org/abs/1902.06156">arXiv:1902.06156</a>); {@code scale}
 *       multiplies {@code z}.</li>
 *   <li>{@link Attack#IPM}: inner-product manipulation sends {@code -scale μ}, so the
 *       aggregate's inner product with the true direction turns negative (Xie, Koyejo and
 *       Gupta, UAI 2020, <a href="https://arxiv.org/abs/1903.03936">arXiv:1903.03936</a>).</li>
 *   <li>{@link Attack#MODEL_REPLACEMENT}: the client's own (backdoored) update scaled by
 *       {@code scale = n/η}, so it survives averaging (Bagdasaryan et al., AISTATS 2020,
 *       <a href="https://arxiv.org/abs/1807.00459">arXiv:1807.00459</a>).</li>
 * </ul>
 *
 * <p>An omniscient attacker computes its statistics from all honest updates; a
 * non-omniscient one only from the genuine updates its own clients computed on their data.
 */
public final class ModelPoisoning {

    /** An attack. */
    public enum Attack {
        /** honest behaviour */
        NONE,
        /** negated update */
        SIGN_FLIP,
        /** Gaussian noise */
        GAUSSIAN,
        /** a little is enough */
        ALIE,
        /** inner-product manipulation */
        IPM,
        /** boosted backdoor update */
        MODEL_REPLACEMENT
    }

    private ModelPoisoning() {
    }

    /**
     * Crafts the malicious updates of one round.
     *
     * @param attack the attack
     * @param honest every honest client's update (used by an omniscient attacker)
     * @param own the malicious clients' genuine (or, for model replacement, backdoored)
     *     updates computed on their own data
     * @param n the total number of clients
     * @param omniscient whether the attacker sees the honest updates
     * @param scale the attack's strength, see the class comment
     * @param random the noise source
     * @return one update per malicious client
     */
    public static float[][] craft(Attack attack, float[][] honest, float[][] own, int n, boolean omniscient,
            double scale, SplittableRandom random) {
        Objects.requireNonNull(attack);
        int f = own.length;
        float[][] out = new float[f][];
        if (f == 0) {
            return out;
        }
        float[][] known = omniscient && honest.length > 0 ? honest : own;
        int d = own[0].length;
        switch (attack) {
            case NONE -> {
                for (int i = 0; i < f; i++) {
                    out[i] = own[i].clone();
                }
            }
            case SIGN_FLIP -> {
                for (int i = 0; i < f; i++) {
                    out[i] = scaled(own[i], -scale);
                }
            }
            case GAUSSIAN -> {
                double norm = 0;
                for (float[] v : known) {
                    norm += Math.sqrt(RobustAggregators.distance2(v, new float[d])) / known.length;
                }
                for (int i = 0; i < f; i++) {
                    float[] v = new float[d];
                    for (int j = 0; j < d; j++) {
                        v[j] = (float) (random.nextGaussian() * scale * norm / Math.sqrt(d));
                    }
                    out[i] = v;
                }
            }
            case ALIE -> {
                float[] mu = RobustAggregators.mean(known);
                double[] sd = new double[d];
                for (float[] v : known) {
                    for (int j = 0; j < d; j++) {
                        sd[j] += (v[j] - mu[j]) * (double) (v[j] - mu[j]) / known.length;
                    }
                }
                int s = n / 2 + 1 - f;
                double z = scale * inverseNormal(Math.clamp((n - s) / (double) n, 0.5, 0.999));
                float[] v = new float[d];
                for (int j = 0; j < d; j++) {
                    v[j] = (float) (mu[j] - z * Math.sqrt(sd[j]) * Math.signum(mu[j]));
                }
                for (int i = 0; i < f; i++) {
                    out[i] = v.clone();
                }
            }
            case IPM -> {
                float[] v = scaled(RobustAggregators.mean(known), -scale);
                for (int i = 0; i < f; i++) {
                    out[i] = v.clone();
                }
            }
            case MODEL_REPLACEMENT -> {
                for (int i = 0; i < f; i++) {
                    out[i] = scaled(own[i], scale);
                }
            }
        }
        return out;
    }

    private static float[] scaled(float[] v, double s) {
        float[] out = new float[v.length];
        for (int j = 0; j < v.length; j++) {
            out[j] = (float) (v[j] * s);
        }
        return out;
    }

    /**
     * The standard normal quantile (Acklam's rational approximation, relative error about
     * {@code 1e-9}).
     *
     * @param p the probability in {@code (0, 1)}
     * @return {@code Φ⁻¹(p)}
     */
    public static double inverseNormal(double p) {
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
                -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
                -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
                4.374664141464968e+00, 2.938163982698783e+00};
        double[] e = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
        double low = 0.02425;
        if (p < low) {
            double q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((e[0] * q + e[1]) * q + e[2]) * q + e[3]) * q + 1);
        }
        if (p > 1 - low) {
            return -inverseNormal(1 - p);
        }
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
}
