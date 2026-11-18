package io.github.zzuegg.brainiac.experimental.security;

import java.util.Arrays;

/**
 * Aggregation weights learned jointly with the global model, after "Byzantine-robust
 * federated learning with learnable aggregation weights" (ICLR 2026,
 * <a href="https://arxiv.org/abs/2511.03529">arXiv:2511.03529</a>). The server solves
 * {@code min_{θ, w} Σ_i w_i f_i(θ)} over the capped sparse simplex
 * {@code {Σ w = 1, 0 <= w_i <= t, ||w||_0 <= s}} by alternating minimisation: a trial model
 * step {@code θ̃ = θ - α G w}, a weight step
 * {@code h = w + αβ Gᵀ G̃ w - β f̃}, a projection that keeps the {@code s} largest entries of
 * {@code h} and projects them onto the capped simplex, then the model step with the new
 * weights. Honest but unusual clients keep weight as long as their losses and gradients fit
 * the model, instead of being discarded by distance alone.
 *
 * <p>Simplifications, documented for the drop-in: the client "gradients" are the negated
 * local updates {@code -Δ_i}; the trial-point quantities are the first-order Taylor
 * estimates {@code G̃ ≈ G} and {@code f̃_i ≈ f_i - α g_iᵀ G w}; and the two terms of the
 * weight step are normalised by the mean squared update norm and the mean loss, so a single
 * {@code β} works at any scale. With the paper's Byzantine setting {@code t = 1/(n - f)},
 * {@code s = n - f}, the projection gives equal weight to the {@code n - f} kept clients.
 */
public final class LearnedAggregationWeights implements Aggregator {

    private final int keep;
    private final double cap;
    private final double beta;
    private double[] w;

    /**
     * Creates the rule.
     *
     * @param keep the sparsity {@code s}: clients kept each round
     * @param cap the weight cap {@code t}, at least {@code 1/s}
     * @param beta the weight step size
     */
    public LearnedAggregationWeights(int keep, double cap, double beta) {
        this.keep = keep;
        this.cap = cap;
        this.beta = beta;
    }

    /**
     * The paper's Byzantine setting.
     *
     * @param n clients
     * @param f tolerated malicious clients
     * @param beta the weight step size
     * @return the rule with {@code s = n - f} and a cap of {@code 1.5/(n - f)}
     */
    public static LearnedAggregationWeights byzantine(int n, int f, double beta) {
        return new LearnedAggregationWeights(n - f, 1.5 / (n - f), beta);
    }

    /** @return the current weights, or {@code null} before the first round */
    public double[] weights() {
        return w == null ? null : w.clone();
    }

    @Override
    public float[] aggregate(float[][] updates) {
        return aggregate(updates, new double[updates.length]);
    }

    @Override
    public float[] aggregate(float[][] updates, double[] losses) {
        int n = updates.length;
        int d = updates[0].length;
        if (w == null) {
            w = new double[n];
            Arrays.fill(w, 1.0 / n);
        }
        double[] gw = combine(updates, w, d);
        double meanNorm = 0;
        double meanLoss = 0;
        for (int i = 0; i < n; i++) {
            meanNorm += RobustAggregators.distance2(updates[i], new float[d]) / n;
            meanLoss += Math.abs(losses[i]) / n;
        }
        double[] h = new double[n];
        for (int i = 0; i < n; i++) {
            double align = 0;
            for (int j = 0; j < d; j++) {
                align += updates[i][j] * gw[j];
            }
            align /= Math.max(meanNorm, 1e-12);
            double trialLoss = losses[i] / Math.max(meanLoss, 1e-12) - align;
            h[i] = w[i] + beta * (align - trialLoss);
        }
        w = project(h, keep, cap);
        double[] agg = combine(updates, w, d);
        float[] out = new float[d];
        for (int j = 0; j < d; j++) {
            out[j] = (float) agg[j];
        }
        return out;
    }

    private static double[] combine(float[][] u, double[] w, int d) {
        double[] out = new double[d];
        for (int i = 0; i < u.length; i++) {
            if (w[i] == 0) {
                continue;
            }
            for (int j = 0; j < d; j++) {
                out[j] += w[i] * u[i][j];
            }
        }
        return out;
    }

    /**
     * The projection onto {@code {Σ w = 1, 0 <= w <= t, ||w||_0 <= s}}: keep the {@code s}
     * largest entries, then project them onto the capped simplex by bisection on the shift.
     *
     * @param h the point
     * @param s the sparsity
     * @param t the cap, at least {@code 1/s}
     * @return the projection
     */
    public static double[] project(double[] h, int s, double t) {
        int n = h.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(h[b], h[a]));
        int k = Math.min(s, n);
        double cap = Math.max(t, 1.0 / k);
        double lo = h[order[0]] - cap - 1;
        double hi = h[order[0]] + 1;
        for (int it = 0; it < 100; it++) {
            double mid = 0.5 * (lo + hi);
            double sum = 0;
            for (int j = 0; j < k; j++) {
                sum += Math.clamp(h[order[j]] - mid, 0, cap);
            }
            if (sum > 1) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        double shift = 0.5 * (lo + hi);
        double[] out = new double[n];
        double sum = 0;
        for (int j = 0; j < k; j++) {
            out[order[j]] = Math.clamp(h[order[j]] - shift, 0, cap);
            sum += out[order[j]];
        }
        if (sum <= 0) {
            for (int j = 0; j < k; j++) {
                out[order[j]] = 1.0 / k;
            }
        }
        return out;
    }
}
