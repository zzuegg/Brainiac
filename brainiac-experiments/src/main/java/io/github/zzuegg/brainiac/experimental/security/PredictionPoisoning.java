package io.github.zzuegg.brainiac.experimental.security;

import java.util.Objects;

/**
 * Defences of a prediction {@link ModelExtraction.Api} against model extraction that never
 * change the top-1 answer, so honest users who read only the label lose nothing:
 *
 * <ul>
 *   <li>{@link Method#TOP1_PRESERVING}: prediction poisoning moves the returned distribution
 *       toward a misleading one that puts its mass on the least likely classes,
 *       {@code p' = (1 - λ) p + λ q}, reducing {@code λ} when needed so the argmax survives.
 *       This misleads distillation on the returned probabilities, a simplified, model-free
 *       form of Orekondy, Schiele and Fritz's maximising-angular-deviation perturbation
 *       (ICLR 2020, <a href="https://arxiv.org/abs/1906.10908">arXiv:1906.10908</a>).</li>
 *   <li>{@link Method#ROUNDING}: probabilities rounded to a few decimals, the countermeasure
 *       Tramèr et al. evaluate against equation solving.</li>
 * </ul>
 */
public final class PredictionPoisoning {

    /** A defence. */
    public enum Method {
        /** argmax-preserving perturbation toward the least likely classes */
        TOP1_PRESERVING,
        /** rounding to a number of decimals */
        ROUNDING
    }

    private PredictionPoisoning() {
    }

    /**
     * Wraps a service with a defence.
     *
     * @param api the service
     * @param method the defence
     * @param strength {@code λ} in {@code [0, 1]}, or the number of decimals
     * @return the defended service
     */
    public static ModelExtraction.Api defend(ModelExtraction.Api api, Method method, double strength) {
        Objects.requireNonNull(method);
        return rows -> {
            float[][] p = api.query(rows);
            float[][] out = new float[p.length][];
            for (int i = 0; i < p.length; i++) {
                out[i] = method == Method.ROUNDING ? round(p[i], (int) strength) : perturb(p[i], strength);
            }
            return out;
        };
    }

    /**
     * The argmax-preserving perturbation of one distribution.
     *
     * @param p the distribution
     * @param lambda the largest mixing weight
     * @return the perturbed distribution, same argmax
     */
    public static float[] perturb(float[] p, double lambda) {
        int k = p.length;
        int top = QueryOracle.argmax(p);
        double max = p[top];
        double[] q = new double[k];
        double z = 0;
        for (int c = 0; c < k; c++) {
            q[c] = c == top ? 0 : Math.pow(max - p[c] + 1e-6, 4);
            z += q[c];
        }
        for (int c = 0; c < k; c++) {
            q[c] /= z;
        }
        double l = lambda;
        for (int t = 0; t < 40; t++) {
            float[] cand = new float[k];
            for (int c = 0; c < k; c++) {
                cand[c] = (float) ((1 - l) * p[c] + l * q[c]);
            }
            if (QueryOracle.argmax(cand) == top && cand[top] > secondLargest(cand, top) + 1e-4) {
                return cand;
            }
            l *= 0.8;
        }
        return p.clone();
    }

    private static double secondLargest(float[] v, int top) {
        double s = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < v.length; c++) {
            if (c != top) {
                s = Math.max(s, v[c]);
            }
        }
        return s;
    }

    /**
     * Rounds a distribution (the argmax is kept even when entries tie after rounding).
     *
     * @param p the distribution
     * @param decimals the decimals kept
     * @return the rounded distribution
     */
    public static float[] round(float[] p, int decimals) {
        double f = Math.pow(10, decimals);
        int top = QueryOracle.argmax(p);
        float[] out = new float[p.length];
        for (int c = 0; c < p.length; c++) {
            out[c] = (float) (Math.round(p[c] * f) / f);
        }
        out[top] += (float) (0.1 / f);
        return out;
    }

    /**
     * The mean {@code ℓ1} distortion a defence adds, its cost to users who read probabilities.
     *
     * @param clean undefended answers
     * @param defended defended answers
     * @return the mean {@code ℓ1} distance
     */
    public static double distortion(float[][] clean, float[][] defended) {
        double s = 0;
        for (int i = 0; i < clean.length; i++) {
            for (int c = 0; c < clean[i].length; c++) {
                s += Math.abs(clean[i][c] - defended[i][c]) / clean.length;
            }
        }
        return s;
    }
}
