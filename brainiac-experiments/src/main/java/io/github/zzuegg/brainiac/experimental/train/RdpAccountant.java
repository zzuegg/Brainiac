package io.github.zzuegg.brainiac.experimental.train;

/**
 * Rényi-DP accounting for the sampled Gaussian mechanism, which is what DP-SGD runs: each
 * step samples every example with probability {@code q}, clips per-example gradients to
 * norm {@code C} and adds Gaussian noise of standard deviation {@code sigma * C}.
 *
 * <p>For integer orders {@code α >= 2} the step's RDP is computed exactly (Mironov, Talwar
 * and Zhang 2019, "Rényi Differential Privacy of the Sampled Gaussian Mechanism",
 * <a href="https://arxiv.org/abs/1908.10530">arXiv:1908.10530</a>):
 *
 * <pre>ε(α) = log( Σ_k C(α,k) (1-q)^(α-k) q^k exp((k² - k) / (2σ²)) ) / (α - 1)</pre>
 *
 * <p>RDP adds up over steps, and the total is turned into {@code (ε, δ)} with the
 * conversion of Balle et al. 2020 ({@code ε = rdp + log((α-1)/α) - (log δ + log α)/(α-1)}),
 * minimised over orders 2 to 256.
 */
public final class RdpAccountant {

    private static final int MAX_ORDER = 256;

    private final double samplingRate;
    private final double noiseMultiplier;
    private int steps;

    private RdpAccountant(double samplingRate, double noiseMultiplier) {
        if (!(samplingRate > 0 && samplingRate <= 1)) {
            throw new IllegalArgumentException("sampling rate must be in (0, 1]: " + samplingRate);
        }
        if (!(noiseMultiplier >= 0)) {
            throw new IllegalArgumentException("noise multiplier must be non-negative: " + noiseMultiplier);
        }
        this.samplingRate = samplingRate;
        this.noiseMultiplier = noiseMultiplier;
    }

    /**
     * An accountant with no steps yet.
     *
     * @param samplingRate the per-example sampling probability {@code q}
     * @param noiseMultiplier the noise standard deviation over the clipping norm
     * @return the accountant
     */
    public static RdpAccountant of(double samplingRate, double noiseMultiplier) {
        return new RdpAccountant(samplingRate, noiseMultiplier);
    }

    /**
     * Records steps.
     *
     * @param n the number of steps taken
     */
    public void step(int n) {
        steps += n;
    }

    /** @return the steps recorded */
    public int steps() {
        return steps;
    }

    /**
     * @param delta the failure probability
     * @return the {@code ε} spent so far
     */
    public double epsilon(double delta) {
        return epsilon(samplingRate, noiseMultiplier, steps, delta);
    }

    /**
     * The RDP of one step at an integer order.
     *
     * @param q the sampling rate
     * @param sigma the noise multiplier
     * @param alpha the order, at least 2
     * @return {@code ε(α)} of one step
     */
    public static double rdp(double q, double sigma, int alpha) {
        if (alpha < 2) {
            throw new IllegalArgumentException("order must be at least 2: " + alpha);
        }
        if (sigma == 0) {
            return Double.POSITIVE_INFINITY;
        }
        if (q == 1) {
            return alpha / (2 * sigma * sigma);
        }
        double[] terms = new double[alpha + 1];
        double logBinom = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (int k = 0; k <= alpha; k++) {
            if (k > 0) {
                logBinom += Math.log(alpha - k + 1) - Math.log(k);
            }
            terms[k] = logBinom + (alpha - k) * Math.log1p(-q) + (k == 0 ? 0 : k * Math.log(q))
                    + (k * (double) k - k) / (2 * sigma * sigma);
            max = Math.max(max, terms[k]);
        }
        double sum = 0;
        for (double t : terms) {
            sum += Math.exp(t - max);
        }
        return (max + Math.log(sum)) / (alpha - 1);
    }

    /**
     * The {@code ε} of a whole DP-SGD run.
     *
     * @param q the sampling rate
     * @param sigma the noise multiplier
     * @param steps the steps
     * @param delta the failure probability
     * @return the smallest {@code ε} over orders 2 to 256; infinite without noise
     */
    public static double epsilon(double q, double sigma, int steps, double delta) {
        if (steps == 0) {
            return 0;
        }
        double best = Double.POSITIVE_INFINITY;
        for (int a = 2; a <= MAX_ORDER; a++) {
            double total = steps * rdp(q, sigma, a);
            double eps = total + Math.log((a - 1.0) / a) - (Math.log(delta) + Math.log(a)) / (a - 1);
            best = Math.min(best, eps);
        }
        return Math.max(best, 0);
    }

    /**
     * The smallest noise multiplier that keeps a run within a privacy budget (bisection).
     *
     * @param targetEpsilon the budget
     * @param q the sampling rate
     * @param steps the steps
     * @param delta the failure probability
     * @return the noise multiplier
     */
    public static double noiseMultiplierFor(double targetEpsilon, double q, int steps, double delta) {
        double lo = 0.01;
        double hi = 1;
        while (epsilon(q, hi, steps, delta) > targetEpsilon) {
            hi *= 2;
            if (hi > 1e4) {
                throw new IllegalArgumentException("budget " + targetEpsilon + " is out of reach");
            }
        }
        for (int i = 0; i < 50; i++) {
            double mid = (lo + hi) / 2;
            if (epsilon(q, mid, steps, delta) > targetEpsilon) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return hi;
    }

    @Override
    public String toString() {
        return "RdpAccountant(q " + samplingRate + ", sigma " + noiseMultiplier + ", steps " + steps + ")";
    }
}
