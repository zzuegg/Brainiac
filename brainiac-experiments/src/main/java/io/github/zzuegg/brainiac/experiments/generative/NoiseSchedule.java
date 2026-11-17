package io.github.zzuegg.brainiac.experiments.generative;

/**
 * A minimal discrete diffusion noise schedule for the toy models in this package: the
 * betas and the cumulative products {@code alphaBar[t] = prod_{s<=t} (1 - beta[s])}, index
 * 0 meaning "no noise" and index {@code steps} the noisiest.
 */
final class NoiseSchedule {
    final int steps;
    final double[] beta;
    final double[] alphaBar;

    private NoiseSchedule(double[] beta) {
        this.steps = beta.length - 1;
        this.beta = beta;
        this.alphaBar = new double[beta.length];
        alphaBar[0] = 1;
        for (int t = 1; t <= steps; t++) {
            alphaBar[t] = alphaBar[t - 1] * (1 - beta[t]);
        }
    }

    /** Linearly increasing betas (Ho et al. 2020), scaled to the number of steps. */
    static NoiseSchedule linear(int steps) {
        double[] b = new double[steps + 1];
        double lo = 1e-4 * 1000.0 / steps;
        double hi = 0.02 * 1000.0 / steps;
        for (int t = 1; t <= steps; t++) {
            b[t] = Math.min(0.999, lo + (hi - lo) * (t - 1) / Math.max(1, steps - 1));
        }
        return new NoiseSchedule(b);
    }

    /** The cosine schedule of Nichol and Dhariwal (arXiv 2102.09672): gentler at both ends. */
    static NoiseSchedule cosine(int steps) {
        double s = 0.008;
        double[] b = new double[steps + 1];
        for (int t = 1; t <= steps; t++) {
            double a1 = f(t, steps, s) / f(0, steps, s);
            double a0 = f(t - 1, steps, s) / f(0, steps, s);
            b[t] = Math.min(0.999, 1 - a1 / a0);
        }
        return new NoiseSchedule(b);
    }

    private static double f(int t, int steps, double s) {
        double c = Math.cos(((double) t / steps + s) / (1 + s) * Math.PI / 2);
        return c * c;
    }

    /** Signal-to-noise ratio {@code alphaBar / (1 - alphaBar)} at step t. */
    double snr(int t) {
        return alphaBar[t] / (1 - alphaBar[t]);
    }
}
