package io.github.zzuegg.brainiac.experimental.anomaly;

import java.util.ArrayList;
import java.util.List;

/**
 * Bayesian online change-point detection (Adams and MacKay 2007,
 * <a href="https://arxiv.org/abs/0710.3742">arXiv:0710.3742</a>). The posterior over the
 * run length {@code r_t} (steps since the last change) is updated in one pass: every run
 * length grows by one with probability {@code 1 - H} or drops to zero with the hazard
 * {@code H = 1 / lambda}, each weighted by the predictive probability of the new value under
 * that run's sufficient statistics. The observation model is a Gaussian with unknown mean and
 * variance under a Normal-Gamma prior, whose predictive is a Student-t, so changes of the mean
 * <em>and</em> of the variance are detected. Run lengths with negligible mass are pruned.
 *
 * <p>Because the change mass {@code P(r_t = 0)} equals {@code H} under a constant hazard, a
 * change at {@code s} is reported when the posterior has moved its mass to short runs: when
 * {@code P(r_t <= t - s)} first exceeds a threshold a few steps after {@code s} (the delay
 * buys confidence).
 */
public final class BayesianOnlineChangePoint {

    private final double hazard;
    private final double mu0;
    private final double kappa0;
    private final double alpha0;
    private final double beta0;

    /**
     * Creates the detector.
     *
     * @param lambda the expected run length (the hazard is {@code 1 / lambda})
     * @param mu0 the prior mean
     * @param kappa0 the prior strength of the mean
     * @param alpha0 the Gamma shape of the precision prior
     * @param beta0 the Gamma rate of the precision prior
     */
    public BayesianOnlineChangePoint(double lambda, double mu0, double kappa0, double alpha0, double beta0) {
        this.hazard = 1 / lambda;
        this.mu0 = mu0;
        this.kappa0 = kappa0;
        this.alpha0 = alpha0;
        this.beta0 = beta0;
    }

    /**
     * The result of a pass.
     *
     * @param mapRunLength the most probable run length after each step
     * @param shortRunMass {@code P(r_t < delay)} after each step
     * @param changePoints the reported change points (first step of each new segment)
     * @param detectedAt the step at which each change point was reported
     */
    public record Result(int[] mapRunLength, double[] shortRunMass, int[] changePoints, int[] detectedAt) {
    }

    /**
     * Runs the detector over a series.
     *
     * @param y the series
     * @param delay the decision delay in steps
     * @param threshold the short-run mass that reports a change, for example 0.5
     * @return run-length summaries and change points
     */
    public Result run(double[] y, int delay, double threshold) {
        int n = y.length;
        // per run length: probability and posterior parameters
        double[] p = {1};
        double[] mu = {mu0};
        double[] kappa = {kappa0};
        double[] alpha = {alpha0};
        double[] beta = {beta0};
        int[] map = new int[n];
        double[] shortMass = new double[n];
        List<Integer> cps = new ArrayList<>();
        List<Integer> at = new ArrayList<>();
        int lastReported = -delay - 1;
        for (int t = 0; t < n; t++) {
            int r = p.length;
            double[] pred = new double[r];
            for (int i = 0; i < r; i++) {
                pred[i] = studentT(y[t], mu[i], beta[i] * (kappa[i] + 1) / (alpha[i] * kappa[i]), 2 * alpha[i]);
            }
            double[] np = new double[r + 1];
            double cp = 0;
            for (int i = 0; i < r; i++) {
                double w = p[i] * pred[i];
                np[i + 1] = w * (1 - hazard);
                cp += w * hazard;
            }
            np[0] = cp;
            double z = 0;
            for (double v : np) {
                z += v;
            }
            double[] nmu = new double[r + 1];
            double[] nk = new double[r + 1];
            double[] na = new double[r + 1];
            double[] nb = new double[r + 1];
            nmu[0] = mu0;
            nk[0] = kappa0;
            na[0] = alpha0;
            nb[0] = beta0;
            for (int i = 0; i < r; i++) {
                nmu[i + 1] = (kappa[i] * mu[i] + y[t]) / (kappa[i] + 1);
                nk[i + 1] = kappa[i] + 1;
                na[i + 1] = alpha[i] + 0.5;
                nb[i + 1] = beta[i] + kappa[i] * (y[t] - mu[i]) * (y[t] - mu[i]) / (2 * (kappa[i] + 1));
            }
            // prune negligible run lengths, keeping index 0
            int keep = r + 1;
            while (keep > 1 && np[keep - 1] / z < 1e-8) {
                keep--;
            }
            p = new double[keep];
            for (int i = 0; i < keep; i++) {
                p[i] = np[i] / z;
            }
            mu = java.util.Arrays.copyOf(nmu, keep);
            kappa = java.util.Arrays.copyOf(nk, keep);
            alpha = java.util.Arrays.copyOf(na, keep);
            beta = java.util.Arrays.copyOf(nb, keep);
            int best = 0;
            double s = 0;
            for (int i = 0; i < keep; i++) {
                if (p[i] > p[best]) {
                    best = i;
                }
                if (i <= delay) {
                    s += p[i];
                }
            }
            map[t] = best;
            shortMass[t] = s;
            if (t >= delay && s > threshold && t - lastReported > 2 * delay) {
                // the most probable recent change position
                int cut = t - mostProbable(p, delay) + 1;
                if (cut > 0 && (cps.isEmpty() || cut - cps.get(cps.size() - 1) > delay)) {
                    cps.add(cut);
                    at.add(t);
                    lastReported = t;
                }
            }
        }
        return new Result(map, shortMass, cps.stream().mapToInt(Integer::intValue).toArray(),
                at.stream().mapToInt(Integer::intValue).toArray());
    }

    private static int mostProbable(double[] p, int upTo) {
        int best = 0;
        for (int i = 1; i <= Math.min(upTo, p.length - 1); i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }

    /** The Student-t density with location, squared scale and degrees of freedom. */
    private static double studentT(double x, double loc, double scale2, double nu) {
        double z = (x - loc) * (x - loc) / scale2;
        double logC = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI * scale2);
        return Math.exp(logC - (nu + 1) / 2 * Math.log1p(z / nu));
    }

    /** Lanczos approximation of {@code log Gamma(x)} for {@code x > 0}. */
    static double lgamma(double x) {
        double[] g = {676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
        if (x < 0.5) {
            return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
        }
        x -= 1;
        double a = 0.99999999999980993;
        double t = x + 7.5;
        for (int i = 0; i < 8; i++) {
            a += g[i] / (x + i + 1);
        }
        return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
    }
}
